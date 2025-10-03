import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root) # Change the current working directory to the project root

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 解决matplotlib字体警告 (全局设置) ---
try:
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
    matplotlib.rcParams['font.family'] = "sans-serif"
except ImportError:
    print("Matplotlib not found, skipping font configuration.")
# ---------------------------

# -------------------------------------------------------------------
# 从提供的文件中导入所需的工具函数
# 假设所有文件都在正确的路径下
# -------------------------------------------------------------------
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.evaluation import globalEvaluation
from plotting.PlottingUtils import umapWithPCA
from plotting.visualization import plotPredTestTime
# -------------------------------------------------------------------
# 导入SinkhornLoss
# -------------------------------------------------------------------
import geomloss

def SinkhornLoss(X, Y, blur=0.05, scaling=0.5): # <--- 调整 scaling 参数
    '''
    X, Y are clouds of points of shape (n_points, n_features)
    '''
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    return ot_solver(X, Y)
# -------------------------------------------------------------------
# 1. TrajGPT 模型定义 (根据论文描述构建)
# -------------------------------------------------------------------
class SRALayer(nn.Module):
    """
    Selective Recurrent Attention Layer based on TrajGPT paper.
    """
    def __init__(self, embed_dim, rnn_mode=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.rnn_mode = rnn_mode
        
        # 定义Q, K, V和衰减向量的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.decay_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 数据依赖的衰减 (data-dependent decay)
        gamma = torch.sigmoid(self.decay_proj(x)) # shape: (batch_size, seq_len, embed_dim)
        
        # 使用并行方式计算以提高效率
        # 初始化状态 S
        S = torch.zeros(batch_size, self.embed_dim, self.embed_dim).to(x.device)
        hidden_states = []

        for i in range(seq_len):
            q_i = Q[:, i, :].unsqueeze(1) # (batch, 1, dim)
            k_i = K[:, i, :].unsqueeze(2) # (batch, dim, 1)
            v_i = V[:, i, :].unsqueeze(1) # (batch, 1, dim)
            gamma_i = gamma[:, i, :].unsqueeze(1) # (batch, 1, dim)

            # 更新状态 S_n = gamma_n * S_{n-1} + K_n @ V_n
            # 这里 K 和 V 都是向量，所以是外积
            S = gamma_i.transpose(1, 2) * S + (k_i @ v_i)
            
            # 计算输出 O_n = Q_n @ S_n
            out_i = q_i @ S
            hidden_states.append(out_i)

        O = torch.cat(hidden_states, dim=1)
        return self.out_proj(O)


class TrajGPT(nn.Module):
    """
    Simplified TrajGPT model for trajectory prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.sra_layers = nn.ModuleList([SRALayer(hidden_dim) for _ in range(num_layers)])
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.num_layers = num_layers

    def forward(self, x):
        # x shape: (batch_size, input_dim), where batch_size is the number of cells
        # We process the whole batch of cells as a single "sequence" of length 1
        # to fit the SRALayer's expectation of (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1) # -> (n_cells, 1, input_dim)

        x = self.input_layer(x)
        for layer in self.sra_layers:
            x = layer(x) + x # Residual connection
        
        x = self.feed_forward(x) + x # Residual connection
        output = self.output_layer(x)
        return output.squeeze(1) # -> (n_cells, input_dim)

# -------------------------------------------------------------------
# 2. 加载数据并设置为插值任务
# -------------------------------------------------------------------
print("=" * 70)
print("1. Loading and preparing data...")
data_name = "zebrafish"
split_type = "three_interpolation" # 指定为插值任务
ann_data, cell_tps, _, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps_idx, test_tps_idx = tpSplitInd(data_name, split_type)
data = ann_data.X

traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
train_data, _ = splitBySpec(traj_data, train_tps_idx, test_tps_idx)

print(f"Dataset: {data_name}")
print(f"Task: {split_type}")
print(f"Training timepoints: {train_tps_idx}")
print(f"Testing (interpolation) timepoints: {test_tps_idx}")
print("-" * 70)


# -------------------------------------------------------------------
# 3. 模型训练
# -------------------------------------------------------------------
print("2. Training TrajGPT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# 模型参数
input_dim = n_genes
hidden_dim = 64 # 减少维度以加快训练
print(f"Input dimension (n_genes): {input_dim}")
print(f"Hidden dimension: {hidden_dim}")
num_layers = 3
epochs = 70 # 增加训练轮数以获得更好效果
lr = 1e-4 # <--- 降低学习率
# batch_size = 64 # 定义batch_size  <--- 不再使用minibatch

model = TrajGPT(input_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 我们使用Sinkhorn Loss代替MSE
# criterion = nn.MSELoss()

loss_history = [] # <--- 初始化loss记录列表

pbar = tqdm(range(epochs), desc="Training Progress")
for epoch in pbar:
    epoch_loss = 0.0
    num_transitions = 0
    # 在每个时间点转换上进行训练
    for i in range(len(train_data) - 1):
        source_data = train_data[i].to(device)
        target_data = train_data[i+1].to(device)

        optimizer.zero_grad()

        # 模型接收所有源细胞，并预测目标细胞群
        predicted_batch = model(source_data)
        
        # 使用Sinkhorn Loss比较两个细胞群（数量可以不同）
        loss = SinkhornLoss(predicted_batch, target_data)
        loss.backward()
        
        # --- 梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        num_transitions += 1
    
    if num_transitions > 0:
        avg_loss = epoch_loss / num_transitions
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
        loss_history.append(avg_loss) # <--- 保存每个epoch的loss

print("Training finished.")
print("-" * 70)

# -------------------------------------------------------------------
# 绘制Loss曲线
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label="Training Loss")
plt.title("TrajGPT Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Sinkhorn Loss")
plt.legend()
plt.grid(True)
plt.show()
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 4. 模型预测
# -------------------------------------------------------------------
print("3. Generating predictions...")
model.eval()
# n_sim_cells = 500 # 定义要生成的细胞数量 <--- 改为使用平均细胞数
# --- 计算平均细胞数 ---
n_cells_per_tp = [d.shape[0] for d in traj_data]
avg_n_cells = int(np.mean(n_cells_per_tp))
n_sim_cells = avg_n_cells
print(f"Using average number of cells for simulation: {n_sim_cells}")
# --------------------
with torch.no_grad():
    # 使用第一个时间点的数据作为起点进行自回归预测
    # 我们随机采样n_sim_cells个细胞作为起点
    start_indices = torch.randperm(traj_data[0].shape[0])[:n_sim_cells]
    current_input = traj_data[0][start_indices].to(device)
    predictions = [current_input.cpu().numpy()] # 存储所有时间点的预测结果

    for t in range(n_tps - 1):
        # 预测下一个时间点
        next_state = model(current_input)
        predictions.append(next_state.cpu().numpy())
        # 更新输入，为下一次预测做准备
        current_input = next_state
print("Prediction finished.")
print("-" * 70)

# -------------------------------------------------------------------
# 5. 评估与可视化
# -------------------------------------------------------------------
print("4. Evaluating and visualizing results...")

# 评估
print("Quantitative Evaluation on Test Timepoints:")
for t in test_tps_idx:
    print(f"\n----- Timepoint t={t} -----")
    true_test_data = traj_data[t].cpu().numpy()
    pred_test_data = predictions[t]
    
    # -- 增加 NaN 和 inf 检查 --
    if np.isnan(pred_test_data).any() or not np.isfinite(pred_test_data).all():
        print(f"!!! Warning: NaNs or Infs detected in predictions for timepoint t={t}. Skipping evaluation.")
        continue

    # 调用评估函数
    metrics = globalEvaluation(true_test_data, pred_test_data)
    print(f"Wasserstein Distance (OT): {metrics['ot']:.4f}")
    print(f"L2 Distance: {metrics['l2']:.4f}")
    print(f"Correlation Distance: {metrics['corr']:.4f}")

# 可视化
print("\nGenerating UMAP visualization...")

# --- 解决matplotlib字体警告 ---
# ... (代码块被移动到文件顶部) ...
# ---------------------------

# 准备绘图数据
true_data_all = [d.cpu().numpy() for d in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data_all)])
pred_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(predictions)])

# 使用 scNODE 的工具函数进行 UMAP 降维
true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data_all, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(predictions, axis=0)))

# 调用 scNODE 的工具函数绘图
plotPredTestTime(
    true_umap_traj,
    pred_umap_traj,
    true_cell_tps,
    pred_cell_tps,
    np.array(test_tps_idx)
)
print("="*70)
print("Done.")