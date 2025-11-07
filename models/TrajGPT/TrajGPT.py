import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
# from layers.Retention_layers import RetNetBlock
from layers.SRA_layers import SRA_Block
from layers.heads import *
from layers.Embed import TokenEmbeddingFixed
from layers.snippets import *
from sklearn.metrics import roc_auc_score, average_precision_score


class TrajGPT(nn.Module):
    '''
    TrajGPT leverages Selective Recurrent Attention (SRA) module for irregularly-sampled time series
    '''
    def __init__(self, configs, head_type='pretrain', forecast_method='time_specific', n_output=2, use_continuous_input=False):
        super(TrajGPT, self).__init__()

        # load parameters
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.d_model = configs.d_model
        self.qk_dim = configs.qk_dim
        self.v_dim = configs.v_dim if configs.v_dim else self.qk_dim
        self.dropout = configs.dropout
        self.use_continuous_input = use_continuous_input

        # Initialize embedding layer
        if use_continuous_input:
            # For continuous input (e.g., single-cell data), use a linear projection
            self.token_embedding = None
            self.input_projection = nn.Linear(configs.d_model, configs.d_model)
        else:
            # Initialize TokenEmbeddingFixed with pre-trained embeddings
            embedding_path = './data/phecode_embeddings.csv'
            self.token_embedding = TokenEmbeddingFixed(embedding_path)
            self.input_projection = None
        
        # the start token for shifted right
        self.sos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.sos)

        # The stacked SRA blocks, each consists of an SRA module and a MLP layer
        self.SRA_blocks = nn.ModuleList([SRA_Block(configs) for _ in range(self.num_layers)])

        # output layer
        self.ln_f = nn.LayerNorm(self.d_model)  # Layer Normalization
        self.head_type = head_type
        self.forecast_method = forecast_method
        if self.head_type == "pretrain":
            if use_continuous_input:
                # For continuous input, output dimension equals d_model
                self.n_output = configs.d_model
                self.head = nn.Linear(self.d_model, self.n_output)
            else:
                self.n_output = self.token_embedding.get_num_tokens()  # Update n_output to the number of unique PheCodes
                self.head = PretrainHead(self.d_model, self.n_output) # the token is [batch_size x seq_len x c_in]
        elif self.head_type == "forecast":
            if use_continuous_input:
                self.n_output = configs.d_model
                self.head = nn.Linear(self.d_model, self.n_output)
            else:
                self.n_output = self.token_embedding.get_num_tokens()  # Update n_output to the number of unique PheCodes
                self.head = ForecastHead(self.d_model, self.n_output)
        elif self.head_type == "clf":
            self.n_output = n_output # Update the number of output classes
            self.head = ClfHead(self.d_model, self.n_output)
        else:
            raise ValueError("Invalid head_type provided.")
        self.gradient_checkpointing = configs.use_grad_ckp

    def forward(self,
                X, input_time,
                y=None, target_time=None,
                retention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                forward_impl: Optional[str] = 'parallel',
                sequence_offset: Optional[int] = 0,
                output_retentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ):
        if self.use_continuous_input:
            # For continuous input: X is (batch_size, seq_len, d_model)
            batch_size, seq_len, _ = X.shape
            # Apply linear projection
            hidden_states = self.input_projection(X)
            # Add the SOS token to the input sequence
            sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
            hidden_states = torch.cat([sos_token, hidden_states[:, :-1, :]], dim=1)  # Shift right
        else:
            # For token input: X is (batch_size, seq_len)
            batch_size, seq_len = X.shape
            # transform a sequence of tokens to embeddings
            hidden_states = torch.zeros(batch_size, seq_len, self.d_model).to(X.device) # batch_size x seq_len x d_model
            for i in range(batch_size):
                # Use token_embedding to get embeddings for the entire sequence for batch i
                hidden_states[i] = self.token_embedding(X[i])
            # Add the SOS token to the input sequence
            sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
            hidden_states = torch.cat([sos_token, hidden_states[:, :-1, :]], dim=1)  # Shift right and drop the last value to maintain original length

        if retention_mask is None:
            retention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=X.device) # batch_size x token_num
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        present_key_values = ()  # To store current key-value pairs
        for i, block in enumerate(self.SRA_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training: # Use gradient checkpointing for the forward pass
                def custom_forward(*inputs):
                    return block(*inputs)
                block_outputs = torch.utils.checkpoint.checkpoint(
                                custom_forward,
                                hidden_states,
                                input_time,
                                retention_mask,
                                forward_impl,
                                past_key_value,
                                sequence_offset,
                                output_retentions
                                )
            else:
                block_outputs = block(hidden_states,
                                      input_time,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      output_retentions=output_retentions
                                      )
            # outputs two variables if output_retentions is False: output hidden states (self.proj(out)), present key values (curr_kv)
            hidden_states = block_outputs[0]
            present_key_values += (block_outputs[1],)

            torch.cuda.empty_cache()
            gc.collect()

            # calculate memory usage after processing the current layer
            # gpu_mem_usage = get_gpu_memory_usage()
            # print("GPU memory usage after %d th layer:" % i)
            # print("Total GPU Memory: {} MiB".format(gpu_mem_usage['total']))
            # print("Used GPU Memory: {} MiB".format(gpu_mem_usage['used']))
            # print("Free GPU Memory: {} MiB".format(gpu_mem_usage['free']))

            # if output_retentions is True, we have an extra variable in block_outputs: retentions (visualization analysis)
            if output_retentions:
                all_retentions += (block_outputs[2],)

        # add hidden states from the last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states)

        # Apply the custom head on the hidden states for output
        outputs = self.ln_f(hidden_states)
        if self.head_type == 'pretrain':
            logits = self.head(outputs)
            if self.use_continuous_input:
                # For continuous input, return the output directly
                return logits
            else:
                # For token input, compute and return loss
                return self.compute_pretrain_loss(logits, y)
        elif self.head_type == "forecast":
            logits_seq = self._run_forecast(hidden_states, input_time, target_time, y)
            logits = torch.stack(logits_seq, dim=1) # stack into (B, M, vocab_size) and compute loss
            if self.use_continuous_input:
                return logits
            else:
                return self.compute_forecast_loss(logits, y)
        elif self.head_type == 'clf':
            logits = self.head(outputs)
            return self.compute_classify_loss(logits, y) # return classification loss
        elif self.head_type == 'multi_target_clf': # return AUPRC
            logits = self.head(outputs)
            return self.compute_multi_target_clf_loss(logits, y) # return multi-target classification loss

    def _run_forecast(self,
                      hidden_states: torch.Tensor,
                      input_time:    torch.Tensor,
                      target_time:   torch.Tensor
                     ) -> List[torch.Tensor]:
        """
        Performs M-step forecasting using SRA.recurrent_retention.
        Returns a list of `M` per-step logits of shape (B, n_output).
        """
        B, T, D     = hidden_states.shape
        M = target_time.size(1) if target_time is not None else 1
        qkv = self.SRA_blocks[-1].sra.qkv(hidden_states)

        q, k, v = torch.split(qkv,
                              [self.qk_dim, self.qk_dim, self.v_dim],
                              dim=-1)
        # split into heads: each becomes (B, H, T, *)
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        # only keep the last time step
        qN, kN, vN = q[:, :, -1:], k[:, :, -1:], v[:, :, -1:]
        state = hidden_states[:, -1, :]

        if self.forecast_method == "time_specific":
            # Obtain the data‐dependent decay at time step N from SRA
            gamma = self.SRA_blocks[-1].sra.get_data_dependent_decay(hidden_states)  # (B, T, H)
            gamma_N = gamma[:, -1, :].view(B, self.num_heads, 1, 1)
            last_t = input_time[:, -1]  # shape (B,)
            dt = target_time - last_t.unsqueeze(1)  # shape (B, M)
        else:
            gamma_N = None
            dt = torch.ones(B, M, device=hidden_states.device)

        # loop M steps of recurrence and projection
        logits_seq = []
        for m in range(M):
            if self.forecast_method == "time_specific":
                decay_m = gamma_N ** dt[:, m].view(B, self.num_heads, 1, 1) # For time-specific inference, it will the decay rate at the time N
            else:
                decay_m = None  # For auto-regressive inference, it will the previous decay rate
            # recurrent update: (qN,kN,vN) are each (B,H,1,*)
            out, S = self.SRA_blocks[-1].sra.recurrent_retention(
                qN, kN, vN,
                past_key_value=state,
                decay=decay_m,
                retention_mask=None
            )
            hidden_out = out.permute(0, 2, 1, 3).reshape(B, -1)
            logits_seq.append(self.head(hidden_out))

        return logits_seq


    def compute_pretrain_loss(self, logits, targets):
        """
        Compute the loss of the pre-training task (next token prediction)
        """
        # Define CrossEntropyLoss with ignore_index for padding token
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.token_embedding.padding_idx)
        # logits: [batch_size, seq_len, num_classes], targets: [batch_size, seq_len]
        logits = logits.view(-1, logits.size(-1))
        # Convert PheCode targets to indices
        targets = self.token_embedding.map_phecodes_to_indices(targets.view(-1)).to(logits.device)
        token_loss = self.ce_loss(logits, targets)
        return token_loss

    def compute_forecast_loss(self, logits, targets):
        """
        Compute the cross entropy loss for the forecasting head
        """
        # Define CrossEntropyLoss with ignore_index for padding token
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.token_embedding.padding_idx)
        # logits: [batch_size, seq_len, num_classes], targets: [batch_size, seq_len]
        # logits: [batch_size, seq_len, num_classes], targets: [batch_size, seq_len]
        logits = logits.view(-1, logits.size(-1))
        # Convert PheCode targets to indices
        targets = self.token_embedding.map_phecodes_to_indices(targets.view(-1)).to(logits.device)
        token_loss = self.ce_loss(logits, targets)
        return token_loss

    def compute_classify_loss(self, logits, targets):
        """
        Compute the loss of classification task
        """
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Apply a threshold of 0.5 to get binary predictions
        predicted = (probs > 0.5).float()
        # Compute accuracy
        correct = (predicted == targets).float()
        accuracy = correct.mean() * 100.0
        return accuracy

    def compute_multi_target_clf_loss(self, logits, targets):
        """
        Compute the loss of multi-target classification task
        """
        # compute binary cross entropy with logits loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss = self.bce_with_logits_loss(logits, targets)

        # Threshold logits to get predicted labels
        predicted = (logits > 0.5).float()

        # compute Precision, Recall
        TP = (predicted * targets).sum().float()
        FP = (predicted * (1 - targets)).sum().float()
        FN = ((1 - predicted) * targets).sum().float()

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)

        # AUPRC
        y_true = targets.cpu().numpy()
        y_scores = logits.cpu().numpy()
        auprc = average_precision_score(y_true, y_scores)

        return loss, precision, recall, auprc
