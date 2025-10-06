import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
    matplotlib.rcParams['font.family'] = "sans-serif"
except ImportError:
    print("Matplotlib not found, skipping font configuration.")

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.evaluation import globalEvaluation
from plotting.PlottingUtils import umapWithPCA
from plotting.visualization import plotPredTestTime

# -------------------------------------------------------------------
# 1. 升级后的 TrajGPT 模型定义 (RoPE, Multi-Head, Parallel SRA)
# -------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, t):
        if t.ndim == 1: t = t.unsqueeze(0)
        t = t.unsqueeze(1) # -> (batch, 1, seq_len)
        freqs = torch.einsum("bij, d -> bidj", t.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb, sin_emb = emb.cos(), emb.sin()
        x_rotated = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return x * cos_emb + x_rotated * sin_emb

class MultiHeadSRALayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.decay_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, t):
        # 并行模式，用于高效训练
        batch_size, seq_len, _ = x.shape
        
        Q, K, V = [proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        
        Q = self.rope(Q, t)
        K = self.rope(K, t)
        
        gamma = torch.sigmoid(self.decay_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 构建因果衰减矩阵 D (vectorized)
        gamma_cum = torch.cumprod(gamma, dim=2)
        D_numerator = gamma_cum.unsqueeze(3)
        D_denominator = gamma_cum.unsqueeze(2)
        D_full = D_numerator / D_denominator
        causal_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).to(x.device)
        D_full = D_full * causal_mask.view(1, 1, seq_len, seq_len, 1)
        
        attn_scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K)
        # 逐元素相乘
        decayed_attn = attn_scores * D_full.mean(dim=-1) # Average decay over head_dim
        
        output = torch.einsum("bhqk, bhvd -> bhqd", decayed_attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)

    def get_final_state(self, x, t):
        # 循环模式，用于计算最终状态S以进行时间特定推理
        batch_size, seq_len, _ = x.shape
        Q, K, V = [proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        
        Q = self.rope(Q, t)
        K = self.rope(K, t)
        
        gamma = torch.sigmoid(self.decay_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        S = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim).to(x.device)
        for i in range(seq_len):
            k_i = K[:, :, i, :].unsqueeze(3) # (batch, n_heads, head_dim, 1)
            v_i = V[:, :, i, :].unsqueeze(2) # (batch, n_heads, 1, head_dim)
            gamma_i = gamma[:, :, i, :].unsqueeze(2) # (batch, n_heads, 1, head_dim)
            S = gamma_i.transpose(2, 3) * S + (k_i @ v_i)
        
        return S

class TrajGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.sra_layers = nn.ModuleList([MultiHeadSRALayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        x_mean = x.mean(dim=2) # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        h = self.input_layer(x_mean)
        for layer in self.sra_layers:
            h = layer(h, t) + h
        h = self.feed_forward(h) + h
        return self.output_layer(h)
    
    def get_final_state(self, x, t):
        x_mean = x.mean(dim=2)
        h = self.input_layer(x_mean)
        states = []
        for layer in self.sra_layers:
            h = layer(h, t) + h
            states.append(layer.get_final_state(h.detach(), t))
        return states

# -------------------------------------------------------------------
# 2. 加载数据并准备序列
# -------------------------------------------------------------------
print("=" * 70)
print("1. Loading and preparing data...")
data_name, split_type = "zebrafish", "three_interpolation"
ann_data, cell_tps, _, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps_idx, test_tps_idx = tpSplitInd(data_name, split_type)
data = ann_data.X

all_tps = np.arange(n_tps)
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t + 1)[0], :]) for t in all_tps]
train_data_list, _ = splitBySpec(traj_data, train_tps_idx, test_tps_idx)

min_cells = min(d.shape[0] for d in train_data_list)
train_data_resampled = [d[torch.randperm(d.shape[0])[:min_cells]] for d in train_data_list]
train_seqs = torch.stack(train_data_resampled).unsqueeze(0)
train_times = torch.FloatTensor(train_tps_idx).unsqueeze(0)
print(f"Data resampled to {min_cells} cells per timepoint for sequence training.")
print("-" * 70)

# -------------------------------------------------------------------
# 3. 模型训练
# -------------------------------------------------------------------
print("2. Training TrajGPT model with parallel SRA...")
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim, hidden_dim, num_heads, num_layers = n_genes, 128, 4, 3
epochs, lr = 100, 3e-4

model = TrajGPT(input_dim, hidden_dim, num_heads, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

input_sequence, target_sequence = train_seqs.to(device), torch.roll(train_seqs, shifts=-1, dims=1).to(device)
input_times = train_times.to(device)

pbar = tqdm(range(epochs), desc="Training Progress")
for epoch in pbar:
    optimizer.zero_grad()
    predicted_mean_sequence = model(input_sequence, input_times)
    loss = criterion(predicted_mean_sequence[:, :-1, :], target_sequence.mean(dim=2)[:, 1:, :])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    pbar.set_postfix({"Latent MSE Loss": f"{loss.item():.6f}"})
print("Training finished.")
print("-" * 70)

# -------------------------------------------------------------------
# 4. 新的预测函数：时间特定推理
# -------------------------------------------------------------------
def predict_time_specific(model, history_x, history_t, target_t, n_sim_cells):
    model.eval()
    with torch.no_grad():
        last_obs_time = history_t[0, -1].item()
        last_obs_data = history_x[0, -1, :, :]
        delta_t = target_t - last_obs_time
        
        # 1. 计算历史序列的最终隐藏状态 S
        final_states = model.get_final_state(history_x.to(device), history_t.to(device))
        
        # 2. 演化状态
        h_last = model.input_layer(last_obs_data.mean(dim=0).to(device))
        gamma_last = torch.sigmoid(model.sra_layers[-1].decay_proj(h_last))
        
        evolved_states = []
        for s in final_states:
            gamma_s = torch.sigmoid(model.sra_layers[final_states.index(s)].decay_proj(h_last)).view(1, num_heads, 1, model.sra_layers[0].head_dim)
            evolved_s = (gamma_s.transpose(2,3) ** delta_t) * s
            evolved_states.append(evolved_s)

        # 3. 生成目标时刻的Query并解码
        h_final = torch.zeros_like(h_last)
        for i, layer in enumerate(model.sra_layers):
            dummy_q = layer.q_proj(h_last).view(1, num_heads, layer.head_dim)
            rotated_q = layer.rope(dummy_q.unsqueeze(1), torch.tensor([target_t]).to(device)).squeeze(1)
            
            # Output O = Q_target @ S_evolved
            out_h = torch.einsum("bhd, bhde -> bhe", rotated_q, evolved_states[i])
            out_h = out_h.contiguous().view(1, hidden_dim)
            h_final = layer.out_proj(out_h) + h_final

        h_final = model.feed_forward(h_final) + h_final
        pred_mean = model.output_layer(h_final)
        
        # 从预测的均值生成细胞群
        noise = torch.randn(n_sim_cells, n_genes, device=device) * last_obs_data.std(dim=0) * 0.5
        return (pred_mean + noise).cpu().numpy()

print("3. Generating predictions using time-specific inference...")
predictions = [traj_data[0].cpu().numpy()]
history_x, history_t = train_seqs[:, :-1, :, :], train_times[:, :-1]
n_sim_cells = min_cells

for t_val in all_tps[1:]:
     if t_val in train_tps_idx:
         # 对于训练中存在的时间点，我们用真实值（或可以预测并比较）
         pred = traj_data[train_tps_idx.index(t_val)].cpu().numpy()
     else:
         # 对于需要预测的时间点（包括测试点）
         pred = predict_time_specific(model, history_x, history_t, t_val, n_sim_cells)
     predictions.append(pred)

print("Prediction finished.")
print("-" * 70)

# -------------------------------------------------------------------
# 5. 评估与可视化
# -------------------------------------------------------------------
print("4. Evaluating and visualizing results...")
print("Quantitative Evaluation on Test Timepoints:")
for t in test_tps_idx:
    true_test_data, pred_test_data = traj_data[t].cpu().numpy(), predictions[t]
    if np.isnan(pred_test_data).any() or not np.isfinite(pred_test_data).all():
        print(f"!!! Warning: NaNs or Infs detected for t={t}. Skipping.")
        continue
    metrics = globalEvaluation(true_test_data, pred_test_data)
    print(f"\n----- Timepoint t={t} -----")
    print(f"Wasserstein Distance (OT): {metrics['ot']:.4f}")
    print(f"L2 Distance: {metrics['l2']:.4f}")
    print(f"Correlation Distance: {metrics['corr']:.4f}")

print("\nGenerating UMAP visualization...")
true_data_all = [d.cpu().numpy() for d in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data_all)])
pred_data_all_safe = [p for p in predictions if p.ndim == 2 and p.shape[0] > 0]
pred_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(pred_data_all_safe)])
true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data_all, axis=0))
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(pred_data_all_safe, axis=0)))
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, np.array(test_tps_idx))

print("="*70)
print("Done.")