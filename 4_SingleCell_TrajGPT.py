<<<<<<< HEAD
# file: run_single_cell_trajgpt.py
# Description: Uses the TrajGPT library to solve the single-cell trajectory
# modeling problem, following the pipeline from scNODE and usage from the notebook.

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
# 设置字体，避免警告
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import time

# --- 1. 导入 TrajGPT 库和配置 ---
# 将项目根目录添加到 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TrajGPT.TrajGPT import TrajGPT
from TrajGPT.configs import TrajGPTConfig

# --- 2. 导入工具函数 (来自 scNODE 项目) ---
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.evaluation import globalEvaluation
from plotting.PlottingUtils import umapWithPCA

def plotPredTestTime(true_umap, pred_umap, true_tps, pred_tps, test_tps_list, save_path=None):
    """
    Plot UMAP visualization comparing true and predicted data at test timepoints.
    Left: True data (all timepoints in gray, test timepoints highlighted)
    Right: Predicted data (all timepoints in gray, test timepoints highlighted)
    """
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    gray_color = "#D3D3D3"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: True data
    ax1.set_title("True Data", fontsize=15)
    ax1.scatter(true_umap[:, 0], true_umap[:, 1], c=gray_color, s=20, alpha=0.5, label="other")
    for i, t in enumerate(test_tps_list):
        mask = true_tps == t
        if np.any(mask):
            ax1.scatter(true_umap[mask, 0], true_umap[mask, 1], 
                       c=colors[i % len(colors)], s=30, alpha=1.0, label=f"t={int(t)}")
    ax1.set_xlabel("UMAP 1"), ax1.set_ylabel("UMAP 2")
    ax1.legend(loc="best")
    
    # Right plot: Predicted data
    ax2.set_title("Predictions", fontsize=15)
    ax2.scatter(true_umap[:, 0], true_umap[:, 1], c=gray_color, s=20, alpha=0.5, label="other")
    for i, t in enumerate(test_tps_list):
        mask = pred_tps == t
        if np.any(mask):
            ax2.scatter(pred_umap[mask, 0], pred_umap[mask, 1],
                       c=colors[i % len(colors)], s=30, alpha=1.0, label=f"t={int(t)}")
    ax2.set_xlabel("UMAP 1"), ax2.set_ylabel("UMAP 2")
    ax2.legend(loc="best")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


print("="*70)
print("1. CONFIGURATION")
# --- 配置 ---
DATA_NAME, SPLIT_TYPE = "zebrafish", "three_interpolation"
LATENT_DIM = 50
TIME_LIMIT = 300  # 训练时间限制（秒）
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Time limit: {TIME_LIMIT} seconds")

print("-"*70)
print("2. DATA LOADING & PREPROCESSING")
# --- 数据加载与预处理 ---
ann_data, cell_tps, cell_types, n_genes, n_tps, all_tps = loadSCData(DATA_NAME, SPLIT_TYPE)
train_tps_idx, test_tps_idx = tpSplitInd(DATA_NAME, SPLIT_TYPE)
data = ann_data.X

# Convert to torch tensors (cell_tps ranges from 1 to n_tps)
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
all_tps = list(all_tps)  # Convert to list
train_data, test_data = splitBySpec(traj_data, train_tps_idx, test_tps_idx)
n_cells = [each.shape[0] for each in traj_data]

print(f"# timepoints={n_tps}, # genes={n_genes}")
print(f"# cells per timepoint: {n_cells}")
print(f"Train timepoints: {train_tps_idx}")
print(f"Test timepoints: {test_tps_idx}")

# 降维
tsvd = TruncatedSVD(n_components=LATENT_DIM, random_state=42).fit(np.concatenate([d.numpy() for d in train_data]))
print(f"Loaded simulated data with {len(ann_data)} cells, {n_genes} genes, across {n_tps} timepoints.")
print(f"TruncatedSVD model fitted on training data.")

# 计算训练序列 (细胞群均值)
train_latent_means = [torch.tensor(tsvd.transform(d.numpy()).mean(axis=0), dtype=torch.float32) for d in train_data]
input_seq = torch.stack(train_latent_means).unsqueeze(0).to(DEVICE)
target_seq = torch.roll(input_seq, shifts=-1, dims=1)
train_times = torch.tensor(train_tps_idx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
print(f"Training input sequence shape: {input_seq.shape}")

print("-"*70)
print("3. MODEL TRAINING")
# --- 模型实例化 (调用 TrajGPT 库) ---
# Note: num_heads must divide v_dim for GroupNorm in SRA layers
# 50 is divisible by 5, so we use num_heads=5
config = TrajGPTConfig(d_model=LATENT_DIM, num_heads=5, num_layers=3, qk_dim=LATENT_DIM, v_dim=LATENT_DIM)
model = TrajGPT(configs=config, head_type='pretrain', use_continuous_input=True).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --- 训练（时间限制） ---
print(f"Starting time-limited training ({TIME_LIMIT} seconds)...")
train_start_time = time.time()

epoch = 0
while (time.time() - train_start_time) < TIME_LIMIT:
    epoch += 1
    model.train()
    optimizer.zero_grad()
    predicted_seq = model(X=input_seq, input_time=train_times)
    loss = criterion(predicted_seq[:, :-1, :], target_seq[:, :-1, :])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        elapsed = time.time() - train_start_time
        remaining = TIME_LIMIT - elapsed
        print(f"Epoch {epoch}, Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s, MSE Loss: {loss.item():.6f}")

# End timing
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print("="*70)
print(f"Training completed! Total epochs: {epoch}")
print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
print("="*70)
    
print("-"*70)
print("4. PREDICTION")
# --- 预测所有时间点 ---
print("Predicting all timepoints...")
model.eval()
n_sim_cells = 3000  # Number of cells to generate per timepoint

with torch.no_grad():
    # Prepare latent representations of all training data
    all_latent_data = []
    for t_idx in range(len(train_data)):
        latent = tsvd.transform(train_data[t_idx].numpy())
        all_latent_data.append(latent)
    
    # Create input sequence with all training timepoints
    train_latent_means = [torch.tensor(ld.mean(axis=0), dtype=torch.float32) for ld in all_latent_data]
    full_input_seq = torch.stack(train_latent_means).unsqueeze(0).to(DEVICE)  # (1, len(train_tps), latent_dim)
    full_train_times = torch.tensor(train_tps_idx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Get model predictions for training sequence
    pred_seq = model(X=full_input_seq, input_time=full_train_times)  # (1, len(train_tps), latent_dim)
    
    # Reconstruct all timepoints (including test timepoints via interpolation/extrapolation)
    all_recon_obs = []
    for t_idx in all_tps:
        if t_idx in train_tps_idx:
            # For training timepoints, use model output
            train_idx = train_tps_idx.index(t_idx)
            latent_mean = pred_seq[0, train_idx, :].cpu().numpy()
        else:
            # For test timepoints, use interpolation or last prediction
            # Simple strategy: use the prediction from nearest training timepoint
            # More sophisticated: train a decoder or use ODE-based interpolation
            closest_train_idx = min(range(len(train_tps_idx)), 
                                    key=lambda i: abs(train_tps_idx[i] - t_idx))
            latent_mean = pred_seq[0, closest_train_idx, :].cpu().numpy()
        
        # Reconstruct to gene space
        recon_mean = tsvd.inverse_transform(latent_mean.reshape(1, -1))
        # Add noise to generate population
        noise = np.random.normal(0, np.concatenate([d.numpy() for d in train_data]).std(0), 
                                (n_sim_cells, n_genes))
        recon_cells = recon_mean + noise
        all_recon_obs.append(recon_cells)

all_recon_obs = np.array(all_recon_obs)  # (n_tps, n_sim_cells, n_genes)
print(f"Reconstructed data shape: {all_recon_obs.shape}")

print("-"*70)
print("5. EVALUATION & VISUALIZATION")
# --- Evaluate predictions on test timepoints ---
print("\nEvaluation metrics for test timepoints:")
for t_idx in test_tps_idx:
    true_data_t = traj_data[t_idx].numpy()
    pred_data_t = all_recon_obs[t_idx]
    metrics = globalEvaluation(true_data_t, pred_data_t)
    print(f"  t={t_idx}: OT={metrics['ot']:.4f}, L2={metrics['l2']:.4f}, CorrDist={metrics['corr']:.4f}")

# --- UMAP Visualization ---
print("\nGenerating UMAP visualization...")
# Concatenate all true data
true_all = np.concatenate([d.numpy() for d in traj_data], axis=0)
# Concatenate all predicted data
pred_all = np.concatenate([all_recon_obs[t] for t in all_tps], axis=0)
# Combine for fitting UMAP
combined_data = np.concatenate([true_all, pred_all], axis=0)

# Fit UMAP on combined data
umap_embedding, umap_model, pca_model = umapWithPCA(combined_data, n_neighbors=50, min_dist=0.1, pca_pcs=50)

# Split back into true and pred
n_true = true_all.shape[0]
true_umap = umap_embedding[:n_true]
pred_umap = umap_embedding[n_true:]

# Create timepoint labels
true_tps = np.concatenate([np.repeat(t_idx, traj_data[t_idx].shape[0]) for t_idx in all_tps])
pred_tps = np.concatenate([np.repeat(t_idx, all_recon_obs[t_idx].shape[0]) for t_idx in all_tps])

# Plot
plotPredTestTime(true_umap, pred_umap, true_tps, pred_tps, test_tps_idx, save_path="TrajGPT_Results.png")
print("="*70)
=======
# file: run_single_cell_trajgpt.py
# Description: Uses the TrajGPT library to solve the single-cell trajectory
# modeling problem, following the pipeline from scNODE and usage from the notebook.

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
# 设置字体，避免警告
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import time

# --- 1. 导入 TrajGPT 库和配置 ---
# 将项目根目录添加到 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TrajGPT.TrajGPT import TrajGPT
from TrajGPT.configs import TrajGPTConfig

# --- 2. 导入工具函数 (来自 scNODE 项目) ---
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.evaluation import globalEvaluation
from plotting.PlottingUtils import umapWithPCA

def plotPredTestTime(true_umap, pred_umap, true_tps, pred_tps, test_tps_list, save_path=None):
    """
    Plot UMAP visualization comparing true and predicted data at test timepoints.
    Left: True data (all timepoints in gray, test timepoints highlighted)
    Right: Predicted data (all timepoints in gray, test timepoints highlighted)
    """
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    gray_color = "#D3D3D3"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: True data
    ax1.set_title("True Data", fontsize=15)
    ax1.scatter(true_umap[:, 0], true_umap[:, 1], c=gray_color, s=20, alpha=0.5, label="other")
    for i, t in enumerate(test_tps_list):
        mask = true_tps == t
        if np.any(mask):
            ax1.scatter(true_umap[mask, 0], true_umap[mask, 1], 
                       c=colors[i % len(colors)], s=30, alpha=1.0, label=f"t={int(t)}")
    ax1.set_xlabel("UMAP 1"), ax1.set_ylabel("UMAP 2")
    ax1.legend(loc="best")
    
    # Right plot: Predicted data
    ax2.set_title("Predictions", fontsize=15)
    ax2.scatter(true_umap[:, 0], true_umap[:, 1], c=gray_color, s=20, alpha=0.5, label="other")
    for i, t in enumerate(test_tps_list):
        mask = pred_tps == t
        if np.any(mask):
            ax2.scatter(pred_umap[mask, 0], pred_umap[mask, 1],
                       c=colors[i % len(colors)], s=30, alpha=1.0, label=f"t={int(t)}")
    ax2.set_xlabel("UMAP 1"), ax2.set_ylabel("UMAP 2")
    ax2.legend(loc="best")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


print("="*70)
print("1. CONFIGURATION")
# --- 配置 ---
DATA_NAME, SPLIT_TYPE = "zebrafish", "three_interpolation"
LATENT_DIM = 50
TIME_LIMIT = 300  # 训练时间限制（秒）
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Time limit: {TIME_LIMIT} seconds")

print("-"*70)
print("2. DATA LOADING & PREPROCESSING")
# --- 数据加载与预处理 ---
ann_data, cell_tps, cell_types, n_genes, n_tps, all_tps = loadSCData(DATA_NAME, SPLIT_TYPE)
train_tps_idx, test_tps_idx = tpSplitInd(DATA_NAME, SPLIT_TYPE)
data = ann_data.X

# Convert to torch tensors (cell_tps ranges from 1 to n_tps)
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
all_tps = list(all_tps)  # Convert to list
train_data, test_data = splitBySpec(traj_data, train_tps_idx, test_tps_idx)
n_cells = [each.shape[0] for each in traj_data]

print(f"# timepoints={n_tps}, # genes={n_genes}")
print(f"# cells per timepoint: {n_cells}")
print(f"Train timepoints: {train_tps_idx}")
print(f"Test timepoints: {test_tps_idx}")

# 降维
tsvd = TruncatedSVD(n_components=LATENT_DIM, random_state=42).fit(np.concatenate([d.numpy() for d in train_data]))
print(f"Loaded simulated data with {len(ann_data)} cells, {n_genes} genes, across {n_tps} timepoints.")
print(f"TruncatedSVD model fitted on training data.")

# 计算训练序列 (细胞群均值)
train_latent_means = [torch.tensor(tsvd.transform(d.numpy()).mean(axis=0), dtype=torch.float32) for d in train_data]
input_seq = torch.stack(train_latent_means).unsqueeze(0).to(DEVICE)
target_seq = torch.roll(input_seq, shifts=-1, dims=1)
train_times = torch.tensor(train_tps_idx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
print(f"Training input sequence shape: {input_seq.shape}")

print("-"*70)
print("3. MODEL TRAINING")
# --- 模型实例化 (调用 TrajGPT 库) ---
# Note: num_heads must divide v_dim for GroupNorm in SRA layers
# 50 is divisible by 5, so we use num_heads=5
config = TrajGPTConfig(d_model=LATENT_DIM, num_heads=5, num_layers=3, qk_dim=LATENT_DIM, v_dim=LATENT_DIM)
model = TrajGPT(configs=config, head_type='pretrain', use_continuous_input=True).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --- 训练（时间限制） ---
print(f"Starting time-limited training ({TIME_LIMIT} seconds)...")
train_start_time = time.time()

epoch = 0
while (time.time() - train_start_time) < TIME_LIMIT:
    epoch += 1
    model.train()
    optimizer.zero_grad()
    predicted_seq = model(X=input_seq, input_time=train_times)
    loss = criterion(predicted_seq[:, :-1, :], target_seq[:, :-1, :])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        elapsed = time.time() - train_start_time
        remaining = TIME_LIMIT - elapsed
        print(f"Epoch {epoch}, Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s, MSE Loss: {loss.item():.6f}")

# End timing
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print("="*70)
print(f"Training completed! Total epochs: {epoch}")
print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
print("="*70)
    
print("-"*70)
print("4. PREDICTION")
# --- 预测所有时间点 ---
print("Predicting all timepoints...")
model.eval()
n_sim_cells = 3000  # Number of cells to generate per timepoint

with torch.no_grad():
    # Prepare latent representations of all training data
    all_latent_data = []
    for t_idx in range(len(train_data)):
        latent = tsvd.transform(train_data[t_idx].numpy())
        all_latent_data.append(latent)
    
    # Create input sequence with all training timepoints
    train_latent_means = [torch.tensor(ld.mean(axis=0), dtype=torch.float32) for ld in all_latent_data]
    full_input_seq = torch.stack(train_latent_means).unsqueeze(0).to(DEVICE)  # (1, len(train_tps), latent_dim)
    full_train_times = torch.tensor(train_tps_idx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Get model predictions for training sequence
    pred_seq = model(X=full_input_seq, input_time=full_train_times)  # (1, len(train_tps), latent_dim)
    
    # Reconstruct all timepoints (including test timepoints via interpolation/extrapolation)
    all_recon_obs = []
    for t_idx in all_tps:
        if t_idx in train_tps_idx:
            # For training timepoints, use model output
            train_idx = train_tps_idx.index(t_idx)
            latent_mean = pred_seq[0, train_idx, :].cpu().numpy()
        else:
            # For test timepoints, use interpolation or last prediction
            # Simple strategy: use the prediction from nearest training timepoint
            # More sophisticated: train a decoder or use ODE-based interpolation
            closest_train_idx = min(range(len(train_tps_idx)), 
                                    key=lambda i: abs(train_tps_idx[i] - t_idx))
            latent_mean = pred_seq[0, closest_train_idx, :].cpu().numpy()
        
        # Reconstruct to gene space
        recon_mean = tsvd.inverse_transform(latent_mean.reshape(1, -1))
        # Add noise to generate population
        noise = np.random.normal(0, np.concatenate([d.numpy() for d in train_data]).std(0), 
                                (n_sim_cells, n_genes))
        recon_cells = recon_mean + noise
        all_recon_obs.append(recon_cells)

all_recon_obs = np.array(all_recon_obs)  # (n_tps, n_sim_cells, n_genes)
print(f"Reconstructed data shape: {all_recon_obs.shape}")

print("-"*70)
print("5. EVALUATION & VISUALIZATION")
# --- Evaluate predictions on test timepoints ---
print("\nEvaluation metrics for test timepoints:")
for t_idx in test_tps_idx:
    true_data_t = traj_data[t_idx].numpy()
    pred_data_t = all_recon_obs[t_idx]
    metrics = globalEvaluation(true_data_t, pred_data_t)
    print(f"  t={t_idx}: OT={metrics['ot']:.4f}, L2={metrics['l2']:.4f}, CorrDist={metrics['corr']:.4f}")

# --- UMAP Visualization ---
print("\nGenerating UMAP visualization...")
# Concatenate all true data
true_all = np.concatenate([d.numpy() for d in traj_data], axis=0)
# Concatenate all predicted data
pred_all = np.concatenate([all_recon_obs[t] for t in all_tps], axis=0)
# Combine for fitting UMAP
combined_data = np.concatenate([true_all, pred_all], axis=0)

# Fit UMAP on combined data
umap_embedding, umap_model, pca_model = umapWithPCA(combined_data, n_neighbors=50, min_dist=0.1, pca_pcs=50)

# Split back into true and pred
n_true = true_all.shape[0]
true_umap = umap_embedding[:n_true]
pred_umap = umap_embedding[n_true:]

# Create timepoint labels
true_tps = np.concatenate([np.repeat(t_idx, traj_data[t_idx].shape[0]) for t_idx in all_tps])
pred_tps = np.concatenate([np.repeat(t_idx, all_recon_obs[t_idx].shape[0]) for t_idx in all_tps])

# Plot
plotPredTestTime(true_umap, pred_umap, true_tps, pred_tps, test_tps_idx, save_path="TrajGPT_Results.png")
print("="*70)
>>>>>>> 79adf705c1bc4c71af40b3b22de696eaf1c9a4f4
print("COMPLETE!")