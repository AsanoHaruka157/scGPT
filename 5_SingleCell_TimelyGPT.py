# file: run_single_cell_timelygpt.py
# Description: Uses the TimelyGPT architecture (with fixed decay vectors) to solve 
# the single-cell trajectory modeling problem, following the scNODE algorithm structure:
# 1. Fast Pre-training: Train VAE encoder/decoder only (MSE reconstruction, no KL divergence)
# 2. Dynamic Training: Sample from first timepoint, use Sinkhorn OT + latent trajectory smoothing
# 3. Prediction: Start from real t0 cells (not prior sampling)
# 4. Visualization: Fit UMAP only on real data, then transform predictions

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import warnings
# 彻底禁用matplotlib的字体警告
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# 简单设置，不指定特定字体
plt.rcParams['font.size'] = 10
import time
import torch.nn.functional as F
import copy
from tqdm import tqdm

# --- 1. 导入 TimelyGPT 库和配置 ---
# 将项目根目录添加到 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 TimelyGPT CTS 版本的组件
sys.path.insert(0, os.path.join(project_root, 'TimelyGPT_CTS'))
from layers.configs import RetNetConfig
from layers.Retention_layers import RetNetBlock

# --- 2. 导入工具函数 (来自 scNODE 项目) ---
sys.path.insert(0, project_root)
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.evaluation import globalEvaluation
from plotting.PlottingUtils import umapWithPCA


# --- 3. 定义适配的 TimelyGPT 模型（用于细胞级别预测） ---
class TimelyGPT_CellLevel(nn.Module):
    """
    TimelyGPT for cell-level trajectory prediction.
    Input: multiple cells at initial timepoint
    Output: predicted states of these cells at future timepoints
    Uses FIXED decay vectors (stable for small datasets).
    """
    def __init__(self, config, latent_dim):
        super(TimelyGPT_CellLevel, self).__init__()
        
        # 参数
        self.num_layers = config.num_layers
        self.d_model = config.d_model
        self.latent_dim = latent_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(latent_dim, config.d_model)
        
        # 时间编码（学习的）
        self.time_embedding = nn.Embedding(20, config.d_model)  # 支持最多20个时间点
        
        # 堆叠的 RetNet blocks（使用固定衰减）
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(self.num_layers)])
        
        # 输出层
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, latent_dim)
        
        self.gradient_checkpointing = config.use_grad_ckp if hasattr(config, 'use_grad_ckp') else False
    
    def forward(self, cells_latent, time_indices, forward_impl='parallel'):
        """
        Predict future states for a batch of cells.
        
        Args:
            cells_latent: [n_cells, latent_dim] - initial cell states in latent space
            time_indices: [n_timepoints] - indices of timepoints to predict
            forward_impl: 'parallel' or 'recurrent'
        Returns:
            predictions: [n_cells, n_timepoints, latent_dim] - predicted cell states
        """
        n_cells = cells_latent.shape[0]
        n_timepoints = len(time_indices)
        
        # Project cells to model dimension
        cell_embeddings = self.input_projection(cells_latent)  # [n_cells, d_model]
        
        # Get time embeddings
        time_emb = self.time_embedding(time_indices)  # [n_timepoints, d_model]
        
        # Create input: each cell at each timepoint
        # [n_cells, n_timepoints, d_model]
        cell_expanded = cell_embeddings.unsqueeze(1).expand(-1, n_timepoints, -1)
        time_expanded = time_emb.unsqueeze(0).expand(n_cells, -1, -1)
        hidden_states = cell_expanded + time_expanded
        
        # 通过 RetNet blocks
        for block in self.blocks:
            block_outputs = block(
                hidden_states,
                retention_mask=None,
                forward_impl=forward_impl,
                past_key_value=None,
                sequence_offset=0,
                chunk_size=None,
                output_retentions=False
            )
            hidden_states = block_outputs[0]  # [n_cells, n_timepoints, d_model]
        
        # 输出投影
        outputs = self.ln_f(hidden_states)
        predictions = self.output_projection(outputs)  # [n_cells, n_timepoints, latent_dim]
        
        return predictions


# --- 3a. VAE Model Definition ---
class Encoder(nn.Module):
    def __init__(self, n_genes, latent_dim, hidden_dims=[128, 128]):
        super(Encoder, self).__init__()
        layers = []
        in_dim = n_genes
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder_net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_genes, hidden_dims=[128, 128]):
        super(Decoder, self).__init__()
        layers = []
        in_dim = latent_dim
        # Use a reversed architecture for the decoder
        reversed_hidden_dims = list(reversed(hidden_dims))
        for h_dim in reversed_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(reversed_hidden_dims[-1], n_genes))
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, z):
        # Handle 3D input for trajectory decoding
        if z.dim() == 3:
            n_cells, n_timepoints, latent_dim = z.shape
            z_flat = z.reshape(-1, latent_dim)
            recon_flat = self.decoder_net(z_flat)
            return recon_flat.reshape(n_cells, n_timepoints, -1)
        else:
            return self.decoder_net(z)

class VAE(nn.Module):
    def __init__(self, n_genes, latent_dim, enc_hidden_dims=[128, 128], dec_hidden_dims=[128, 128]):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_genes, latent_dim, enc_hidden_dims)
        self.decoder = Decoder(latent_dim, n_genes, dec_hidden_dims)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, z, mu, log_var

# --- 3b. Combined VAE-TimelyGPT Model ---
class VAETimelyGPT(nn.Module):
    def __init__(self, vae, timely_gpt):
        super(VAETimelyGPT, self).__init__()
        self.vae = vae
        self.timely_gpt = timely_gpt

    def forward(self, initial_cells_obs, time_indices, forward_impl='parallel'):
        """
        Forward pass matching scNODE style:
        1. Encode initial cells to latent space
        2. Predict future latent states with TimelyGPT (replacing ODE)
        3. Decode predicted latent states back to gene space
        Returns: recon_obs, first_latent_dist (mu, log_var), first_tp_data, latent_seq
        """
        # 1. Encode initial cells to latent space
        mu, log_var = self.vae.encoder(initial_cells_obs)
        z_initial = self.vae.reparameterize(mu, log_var)

        # 2. Predict future latent states with TimelyGPT
        # z_predictions (latent_seq): [n_cells, n_timepoints, latent_dim]
        latent_seq = self.timely_gpt(z_initial, time_indices, forward_impl)

        # 3. Decode predicted latent states back to gene space
        recon_obs = self.vae.decoder(latent_seq)

        # Return format matching scNODE: (recon_obs, first_latent_dist, first_tp_data, latent_seq)
        first_latent_dist = (mu, log_var)
        first_tp_data = initial_cells_obs
        
        return recon_obs, first_latent_dist, first_tp_data, latent_seq


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
    ax2.set_title("Predictions (VAE + TimelyGPT)", fontsize=15)
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
    plt.close()

# --- 配置 ---
DATA_NAME, SPLIT_TYPE = "zebrafish", "three_interpolation"
LATENT_DIM = 64
LATENT_COEFF = 1.0  # Regularization coefficient for latent trajectory smoothing (beta)
N_PRED_CELLS = 5000  # Number of cells to predict
MAX_EPOCHS = 20  # Maximum number of epochs
EARLY_STOP_PATIENCE = 10  # Stop if best model is not updated for this many epochs
BATCH_SIZE = 32
PRETRAIN_LR = 3e-4
PRETRAIN_ITERS = 500  # Number of pre-training iterations
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Early stopping: Patience of {EARLY_STOP_PATIENCE} epochs without improvement.")
print(f"Learning rate: {LR} with exponential decay (gamma=0.99)")
print(f"Pre-training iterations: {PRETRAIN_ITERS}")
print(f"Latent smoothing coefficient: {LATENT_COEFF}")

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

print("-"*70)
print("3. MODEL INITIALIZATION")
# --- 模型实例化 (VAE + TimelyGPT) ---
# 1. VAE model
vae = VAE(n_genes, LATENT_DIM).to(DEVICE)

# 2. TimelyGPT model (operates on latent space)
# 关键设置：use_default_gamma=True → 使用固定衰减向量
timely_gpt_config = RetNetConfig(
    num_layers=3,
    num_heads=8,  # 必须能整除v_dim
    d_model=LATENT_DIM,
    qk_dim=LATENT_DIM,
    v_dim=LATENT_DIM,
    ffn_proj_size=200,
    use_bias_in_msr=False,
    use_bias_in_mlp=True,
    use_bias_in_msr_out=False,
    use_default_gamma=True,  # ★ 关键：使用固定衰减（gamma = 1 - 2^(-5-i)）
    forward_impl='parallel'
)
timely_gpt_config.use_grad_ckp = False

timely_gpt_model = TimelyGPT_CellLevel(
    config=timely_gpt_config,
    latent_dim=LATENT_DIM
)

# 3. Combined model
model = VAETimelyGPT(vae, timely_gpt_model).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

import geomloss

print(f"VAE-TimelyGPT model initialized.")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("-"*70)
print("4. TRAINING (Following scNODE Algorithm)")
print("-"*70)

# Import necessary modules
import geomloss
import itertools

# Start timing
print(f"Starting training (max {MAX_EPOCHS} epochs)...")
train_start_time = time.time()

# Prepare time indices for training
train_time_indices = torch.LongTensor(train_tps_idx).to(DEVICE)
train_tps_tensor = torch.FloatTensor(train_tps_idx).to(DEVICE)

# ======================================================
# Phase 1: Fast Pre-training - Train only VAE Encoder and Decoder
# ======================================================
print("\n[Phase 1] Fast Pre-training: Training VAE Encoder and Decoder only...")
latent_encoder = model.vae.encoder
obs_decoder = model.vae.decoder
all_train_data = torch.cat(train_data, dim=0).to(DEVICE)

if PRETRAIN_ITERS > 0:
    # Only train encoder and decoder parameters
    dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
    dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=PRETRAIN_LR, betas=(0.95, 0.99))
    pretrain_scheduler = torch.optim.lr_scheduler.ExponentialLR(dim_reduction_optimizer, gamma=0.99)
    latent_encoder.train()
    obs_decoder.train()
    
    best_pretrain_loss = float('inf')
    best_pretrain_state = None
    
    pbar = tqdm(range(PRETRAIN_ITERS), desc="Pre-training VAE")
    for i in pbar:
        # Sample random batch from all training data
        rand_idx = np.random.choice(all_train_data.shape[0], size=BATCH_SIZE, replace=False)
        batch_data = all_train_data[rand_idx, :]
        
        dim_reduction_optimizer.zero_grad()
        
        # Encode -> sample -> decode (NO KL divergence term)
        latent_mu, latent_log_var = latent_encoder(batch_data)
        latent_std = torch.exp(0.5 * latent_log_var)
        latent_sample = latent_mu + torch.randn_like(latent_std) * latent_std
        recon_obs = obs_decoder(latent_sample)
        
        # Reconstruction MSE loss only (no KL divergence)
        recon_loss = torch.mean((recon_obs - batch_data) ** 2)
        recon_loss.backward()
        dim_reduction_optimizer.step()
        pretrain_scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{recon_loss.item():.6f}"})
        
        # Save the best model
        if recon_loss.item() < best_pretrain_loss:
            best_pretrain_loss = recon_loss.item()
            best_pretrain_state = {
                'encoder': copy.deepcopy(latent_encoder.state_dict()),
                'decoder': copy.deepcopy(obs_decoder.state_dict())
            }

    print(f"Pre-training completed. Best loss: {best_pretrain_loss:.6f}")
    
    # Load the best pre-trained model
    if best_pretrain_state:
        print("Loading best pre-trained model state...")
        latent_encoder.load_state_dict(best_pretrain_state['encoder'])
        obs_decoder.load_state_dict(best_pretrain_state['decoder'])

# ======================================================
# Phase 2: Dynamic Training - Train full model with Sinkhorn + Latent Smoothing
# ======================================================
print("\n[Phase 2] Dynamic Training: Training full model (VAE + TimelyGPT)...")
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=(0.95, 0.99))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
loss_list = []
iters_per_epoch = 100

# Early stopping variables
best_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_losses = []
    for iter_idx in range(iters_per_epoch):
        # Sample mini-batch from FIRST timepoint only (scNODE style)
        # According to scNODE: "Only data from the first time point is fed to the model"
        rand_idx = np.random.choice(train_data[0].shape[0], size=BATCH_SIZE, replace=False)
        batch_data = train_data[0][rand_idx, :].to(DEVICE)
        
        optimizer.zero_grad()
        model.train()
        
        # Forward pass: Only feed data from first timepoint
        # Model expects: forward(data_at_first_tp, time_indices)
        recon_obs, first_latent_dist, first_tp_data, latent_seq = model(
            batch_data, train_time_indices, forward_impl='parallel'
        )
        
        # Compute loss: Sinkhorn OT + Latent Trajectory Smoothing
        ot_loss = 0.0
        for t_idx, t in enumerate(train_tps_idx):
            pred_x = recon_obs[:, t_idx, :]  # [batch_size, n_genes]
            true_x = train_data[t_idx].to(DEVICE)  # [n_cells_at_t, n_genes]
            
            # Subsample for efficiency (matching scNODE)
            subsample_size = min(200, true_x.shape[0])
            subsample_idx = np.random.choice(true_x.shape[0], subsample_size, replace=False)
            ot_loss += ot_solver(pred_x, true_x[subsample_idx])
        
        # Latent Trajectory Smoothing Regularizer: ||z_{t+1} - z_t||^2
        latent_drift_loss = torch.mean((latent_seq[:, 1:, :] - latent_seq[:, :-1, :]) ** 2)
        
        # Combined loss
        loss = ot_loss + LATENT_COEFF * latent_drift_loss
        
        loss.backward()
        optimizer.step()
        
        loss_list.append((loss.item(), ot_loss.item(), latent_drift_loss.item()))
        epoch_losses.append(loss.item())
    
    # Calculate average epoch loss and check for improvement
    avg_epoch_loss = np.mean(epoch_losses)
    elapsed = time.time() - train_start_time
    
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch}/{MAX_EPOCHS}, Elapsed: {elapsed:.1f}s, Loss: {avg_epoch_loss:.4f} (New best model)")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch}/{MAX_EPOCHS}, Elapsed: {elapsed:.1f}s, Loss: {avg_epoch_loss:.4f} ({epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs without improvement)")

    # Check early stopping
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}: No improvement in loss for {EARLY_STOP_PATIENCE} consecutive epochs.")
        break

    scheduler.step()

# End timing
train_end_time = time.time()
train_duration = train_end_time - train_start_time

print("="*70)
print(f"Training completed! Total epochs: {epoch}")
print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
print(f"Best loss: {best_loss:.6f}")
print("="*70)

# Load the best model state for prediction
if best_model_state:
    print("Loading best performing model for prediction...")
    model.load_state_dict(best_model_state)

# Visualization - loss curve (matching scNODE)
if len(loss_list) > 0:
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.title("Loss")
    plt.plot([each[0] for each in loss_list])
    plt.subplot(3, 1, 2)
    plt.title("OT Term")
    plt.plot([each[1] for each in loss_list])
    plt.subplot(3, 1, 3)
    plt.title("Dynamic Reg")
    plt.plot([each[2] for each in loss_list])
    plt.xlabel("Dynamic Learning Iter")
    plt.tight_layout()
    loss_plot_path = os.path.join(current_dir, "TimelyGPT_Loss.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

print("-"*70)
print("5. PREDICTION (Generative Augmentation of t0 Cells)")
# --- 预测：从t0细胞的潜在分布中采样生成新细胞 ---
print(f"Generatively augmenting t0 cells to create {N_PRED_CELLS} initial states...")
model.eval()

with torch.no_grad():
    # 1. Encode all real t0 cells to get their latent distributions
    t0_real_cells = traj_data[0].to(DEVICE)
    mus, log_vars = model.vae.encoder(t0_real_cells)
    stds = torch.exp(0.5 * log_vars)
    num_real_t0_cells = t0_real_cells.shape[0]
    print(f"Created a latent GMM from {num_real_t0_cells} real t0 cells.")

    # 2. Sample N_PRED_CELLS latent variables from this GMM
    # To sample from the GMM, we first pick a component, then sample from it.
    component_indices = np.random.choice(num_real_t0_cells, size=N_PRED_CELLS, replace=True)
    selected_mus = mus[component_indices]
    selected_stds = stds[component_indices]
    
    # Reparameterization trick to sample
    eps = torch.randn_like(selected_stds)
    z_initial_sampled = selected_mus + eps * selected_stds  # Shape: [N_PRED_CELLS, LATENT_DIM]
    print(f"Sampled {N_PRED_CELLS} initial latent states.")

    # 3. Use these sampled z as input to TimelyGPT
    all_time_indices = torch.LongTensor(all_tps).to(DEVICE)
    
    # Directly use timely_gpt and decoder, bypassing the encoder for prediction
    # predictions_latent: [N_PRED_CELLS, n_all_tps, latent_dim]
    predictions_latent = model.timely_gpt(z_initial_sampled, all_time_indices, forward_impl='parallel')
    
    # 4. Decode back to gene space
    # all_recon_obs_tensor: [N_PRED_CELLS, n_tps, n_genes]
    all_recon_obs_tensor = model.vae.decoder(predictions_latent)
    
    # Reshape for evaluation: [n_tps, n_cells, n_genes]
    all_recon_obs = all_recon_obs_tensor.permute(1, 0, 2).cpu().numpy()


print(f"Predicted data shape: {all_recon_obs.shape}")
print(f"Predicted cells at {len(all_tps)} timepoints")

print("-"*70)
print("6. EVALUATION")
# --- Evaluate predictions on all timepoints ---
print("\n★ Evaluation metrics for ALL timepoints:")
print(f"{'Time':<6} {'Type':<6} {'OT':<10} {'L2':<10} {'CorrDist':<10}")
print("-"*70)

for t_idx in all_tps:
    true_data_t = traj_data[t_idx].numpy()
    pred_data_t = all_recon_obs[t_idx]
    
    # Standard metrics
    metrics = globalEvaluation(true_data_t, pred_data_t)
    
    # Determine if this is a training or test timepoint
    tp_type = "TRAIN" if t_idx in train_tps_idx else "TEST"
    
    # Print results
    print(f"t={t_idx:<4} {tp_type:<6} {metrics['ot']:<10.4f} {metrics['l2']:<10.4f} "
          f"{metrics['corr']:<10.4f}")

print("\n★ Summary for TEST timepoints only:")
test_metrics = {'ot': [], 'l2': [], 'corr': []}
for t_idx in test_tps_idx:
    true_data_t = traj_data[t_idx].numpy()
    pred_data_t = all_recon_obs[t_idx]
    metrics = globalEvaluation(true_data_t, pred_data_t)
    test_metrics['ot'].append(metrics['ot'])
    test_metrics['l2'].append(metrics['l2'])
    test_metrics['corr'].append(metrics['corr'])

print(f"Average OT: {np.mean(test_metrics['ot']):.4f} ± {np.std(test_metrics['ot']):.4f}")
print(f"Average L2: {np.mean(test_metrics['l2']):.4f} ± {np.std(test_metrics['l2']):.4f}")
print(f"Average CorrDist: {np.mean(test_metrics['corr']):.4f} ± {np.std(test_metrics['corr']):.4f}")

# --- UMAP Visualization (Following scNODE: Fit only on real data, then transform predictions) ---
print("\nGenerating UMAP visualization...")
print("Following scNODE approach: Fit UMAP only on real data, then transform predictions")

# Concatenate all true data
true_data_list = [each.detach().numpy() if isinstance(each, torch.Tensor) else each for each in traj_data]
true_all = np.concatenate(true_data_list, axis=0)

# Fit UMAP ONLY on real data (matching scNODE)
print("Fitting UMAP on real data only...")
true_umap_traj, umap_model, pca_model = umapWithPCA(true_all, n_neighbors=50, min_dist=0.1, pca_pcs=50)

# Transform predicted data using pre-fitted UMAP (avoiding coordinate drift)
print("Transforming predicted data using pre-fitted UMAP...")
pred_data_list = [all_recon_obs[t_idx] for t_idx in all_tps]
pred_all = np.concatenate(pred_data_list, axis=0)
pred_umap_traj = umap_model.transform(pca_model.transform(pred_all))

# Create timepoint labels
true_cell_tps = np.concatenate([np.repeat(t_idx, traj_data[t_idx].shape[0]) for t_idx in all_tps])
pred_cell_tps = np.concatenate([np.repeat(t_idx, all_recon_obs[t_idx].shape[0]) for t_idx in all_tps])

# Plot
save_path = os.path.join(current_dir, "TimelyGPT_Results.png")
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps_idx, save_path=save_path)

print("="*70)
print("COMPLETE!")

