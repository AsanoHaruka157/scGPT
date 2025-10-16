import time
from typing import List, Optional, Tuple, Union
from layers.configs import TrajGPTConfig
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import logging
from layers.Xpos import XPOS
from layers.snippets import *
import gc


class Selective_Recurrent_Attention(nn.Module):
    def __init__(self, config: TrajGPTConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model,
                             config.qk_dim * 2 + config.v_dim,
                             bias=config.use_bias_in_sra)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(config.d_model, config.v_dim, bias=False)
        self.proj = nn.Linear(config.v_dim, config.d_model, bias=config.use_bias_in_sra_out)
        self.gn = nn.GroupNorm(num_groups=config.num_heads, num_channels=config.v_dim, affine=False)
        self.xpos = XPOS(config.qk_dim)

        # initialize gamma, found
        if config.use_default_gamma:
            gamma = 1 - 2**(-5 - torch.arange(0, config.num_heads, dtype=torch.float))
        else:
            # you should update the gamma value based on your dataset, if the time span is large, make gamma close to 1
            # s = torch.log(torch.tensor(1 / 32))
            s = torch.log(torch.tensor(1 / 64))
            # s = torch.log(torch.tensor(1 / 128))
            e = torch.log(torch.tensor(1 / 512))
            gamma = 1 - torch.exp(torch.linspace(s, e, config.num_heads))  # [h,]
        self.decay = nn.Parameter(gamma, requires_grad=False) # tensor([0.9688, 0.9876, 0.9951, 0.9980])

        # Define head-wise projection to compute \gamma
        self.gamma_proj = nn.Linear(
            config.d_model,
            config.num_heads,
            bias=True
        )
        self.tau = config.tau  # temperature for the gamma

    def get_parallel_decay_mask(self, t, decay, retention_mask=None):
        B, T, H = decay.shape
        # compute cumulative decay product b_n
        b = torch.cumprod(decay, dim=1)
        # pairwise ratio b_n / b_m
        ratio = b.unsqueeze(2) / b.unsqueeze(1)
        D = ratio.permute(0, 3, 1, 2)  # (B, H, T, T)
        # Apply lower triangular mask to ensure causality
        decay_mask = torch.tril(D, diagonal=0)  # [batch_size, num_heads, seq_len, seq_len]
        if retention_mask is not None:
            # Expand the retention mask to match the decay mask dimensions
            retention_mask = retention_mask.float().view(-1, 1, 1, T)
            decay_mask = decay_mask * retention_mask
        else:
            decay_mask = decay_mask

        return decay_mask

    def get_data_dependent_decay(self, hidden_states):
        gamma = self.gamma_proj(hidden_states)
        decay = torch.sigmoid(gamma) ** (1.0 / self.tau)
        return decay

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_heads * len * qk_dim
        k,  # bsz * num_heads * len * qk_dim
        v,  # bsz * num_heads * len * v_dim
        decay_mask,  # (1 or bsz) * num_heads * len * len
        """
        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2) * k.size(-1)**-0.5  # (scaled dot-product) found
        retention = retention * decay_mask
        output = retention @ v #

        # kv cache
        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        # [bsz, num_heads, qk_dim, v_dim]
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        return output, current_kv, retention

    def recurrent_retention(self, q, k, v, past_key_value=None, decay=None, retention_mask=None):
        """
        q, k, v, # bsz * num_heads * 1 * qkv_dim
        past_key_value, # bsz * num_heads * qk_dim * v_dim
        decay # num_heads * 1 * 1
        retention_mask # bsz * 1
        """
        B, H, _, _ = q.shape
        past_key_value = past_key_value if past_key_value is not None else 0
        decay = decay if decay is not None else 0
        decay = decay.view(B, H, 1, 1)
        retention_mask = retention_mask.view(B, 1, 1, 1) if retention_mask is not None else 1
        current_kv = decay * past_key_value + retention_mask * (k.transpose(-1, -2) @ v)
        output = q @ current_kv * k.size(-1)**-0.5  # (b, h, 1, d_v)
        return output, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        sequence_offset: Optional[int] = 0,
        output_retentions: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, d = hidden_states.size()

        q, k, v = self.qkv(hidden_states).split(
            [self.config.qk_dim, self.config.qk_dim, self.config.v_dim], dim=-1)

        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset) # xpos for q and k
        q, k, v = split_heads((q, k, v), B, T, self.config.num_heads)
        decay = self.get_data_dependent_decay(hidden_states)
        # SRA recurrent attention computation
        if forward_impl == 'parallel':
            decay_mask = self.get_parallel_decay_mask(t, decay, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v,
                                                                                decay_mask=decay_mask)
            torch.cuda.empty_cache()
            gc.collect()
        elif forward_impl == 'recurrent':
            for n in range(T):
                gamma_n = decay[:, n, :]
                retention_mask_n = retention_mask[:, n]         # (B,)
                retention_out, curr_kv = self.recurrent_retention(q, k, v,
                                                              past_key_value=past_key_value,
                                                              decay=gamma_n,
                                                              retention_mask=retention_mask_n)
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        # concat heads
        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.config.v_dim)
        # group norm (merge batch, length dimension -> group norm -> split back)
        normed = self.gn(retention_out.view(B * T, self.config.v_dim))
        normed = normed.view(B, T, self.config.v_dim)
        # out gate & proj
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.proj(out), curr_kv) # project v_dim to d_model
        if output_retentions:
            outputs += (retention_weights,) if forward_impl == 'parallel' else (None,)
        return outputs


class SRA_Block(nn.Module):
    '''
    An SRA block consists of an SRA module and an MLP layer
    '''
    def __init__(self, config: TrajGPTConfig):
        super().__init__()
        self.config = config

        # multi-scale retention, output shape is [bsz, seq_len, d_model]
        self.sra = Selective_Recurrent_Attention(config)

        # 2-layer MLP, output shape is [bsz, seq_len, d_model]
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_proj_size, bias=config.use_bias_in_mlp),
            nn.GELU(),
            nn.Linear(config.ffn_proj_size, config.d_model, bias=config.use_bias_in_mlp),
        )

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        sequence_offset: Optional[int] = 0,
        output_retentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        sra_outs = self.sra(self.ln1(hidden_states),
                            t=t,
                            retention_mask=retention_mask,
                            past_key_value=past_key_value,
                            forward_impl=forward_impl,
                            sequence_offset=sequence_offset,
                            output_retentions=output_retentions)
        # Original SRA layer with residual connection
        sra = sra_outs[0]
        curr_kv = sra_outs[1]
        x = hidden_states + sra

        # MLP layer
        y = x + self.ffn(self.ln2(x))

        outputs = (y, curr_kv)

        if output_retentions:
            outputs += (sra_outs[2],)
        return outputs



