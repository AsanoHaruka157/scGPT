import sys
import os
import pathlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import geomloss  # 需要 pip install geomloss
import logging
import gc
import matplotlib.pyplot as plt

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
project_root = CURRENT_DIR.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from plotting.PlottingUtils import umapWithPCA

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, memory monitoring disabled")

# ==========================================
# 1. 引用 TimelyGPT 模块
# ==========================================

try:
    from model.TimelyGPT_CTS.layers.configs import RetNetConfig
    from model.TimelyGPT_CTS.layers.Retention_layers import RetNetBlock
except ImportError:
    # 兼容处理
    pass

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage=""):
    """记录内存使用情况"""
    if HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024**3)
            logger.info(f"[{stage}] Memory usage: {mem_gb:.2f} GB (RSS)")
            return mem_gb
        except Exception as e:
            logger.warning(f"[{stage}] Failed to get memory usage: {e}")
            return None
    else:
        logger.info(f"[{stage}] Memory monitoring unavailable (psutil not installed)")
        return None

# ==========================================
# 2. 模型组件
# ==========================================
class SpatialEncoder(nn.Module):
    def __init__(self, input_dim=2, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
    def forward(self, coords):
        return self.net(coords)

class GeneVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
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
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

class GeneDecoder(nn.Module):
    def __init__(self, latent_dim, n_genes, hidden_dims=[256, 512]):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], n_genes))
        self.decoder_net = nn.Sequential(*layers)
    def forward(self, z):
        return self.decoder_net(z)

# ==========================================
# 3. 自回归TimelyGPT
# ==========================================
class SpatiotemporalTimelyGPT(nn.Module):
    def __init__(self, config, n_genes, n_timepoints):
        super().__init__()
        self.latent_dim = config.d_model
        self.n_timepoints = n_timepoints
        
        self.gene_encoder = GeneVAEEncoder(n_genes + 2, self.latent_dim)
        self.spatial_encoder = SpatialEncoder(input_dim=2, d_model=self.latent_dim)
        
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.gene_decoder = GeneDecoder(self.latent_dim, n_genes)

    def forward(self, x_genes, x_coords):
        # 1. 初始 Embedding (t=0 的状态估计)
        encoder_input = torch.cat([x_genes, x_coords], dim=-1)
        mu, log_var = self.gene_encoder(encoder_input)
        z_gene = self.gene_encoder.reparameterize(mu, log_var)
        z_space = self.spatial_encoder(x_coords)
        z_combined = z_gene + z_space 
        
        # 2. 自回归生成：从 t=0 出发逐步扩展序列，而不是简单复制
        token_sequence = z_combined.unsqueeze(1)  # [B, 1, d_model], 仅含 t0
        recon_steps = []
        latent_steps = []

        for t in range(self.n_timepoints):
            hidden_states = token_sequence

            # 让 RetNet 处理当前序列，得到所有时间步的隐状态
            for block in self.blocks:
                block_out = block(hidden_states, sequence_offset=0, forward_impl='parallel')
                hidden_states = block_out[0]

            hidden_states = self.ln_f(hidden_states)

            # 当前时间步的隐状态与基因重建
            current_hidden = hidden_states[:, -1, :]          # 只取最新一步
            current_recon = self.gene_decoder(current_hidden)  # [B, n_genes]

            latent_steps.append(current_hidden)
            recon_steps.append(current_recon)

            # 将当前生成的隐状态作为下一个时间步的输入 token，保证自回归
            if t < self.n_timepoints - 1:
                token_sequence = torch.cat([token_sequence, current_hidden.unsqueeze(1)], dim=1)

        # 堆叠得到完整轨迹
        recon_seq = torch.stack(recon_steps, dim=1)   # [B, T, n_genes]
        latent_seq = torch.stack(latent_steps, dim=1) # [B, T, d_model]

        return recon_seq, mu, log_var, latent_seq

# ==========================================
# 4. 数据加载与挖空（分批加载以防OOM）
# ==========================================
def load_and_split_data(path, hold_out_indices=[3, 6], n_top_genes=2000, 
                       use_backed=False, chunk_size=None):
    """
    hold_out_indices: 需要挖掉的时间点索引 (用于验证插值能力)
    use_backed: 是否使用backed模式（不一次性加载到内存）
    chunk_size: 如果指定，分批处理数据块大小
    """
    log_memory_usage("Before loading")
    logger.info(f"Loading data from {path}...")
    logger.info(f"Using backed mode: {use_backed}, chunk_size: {chunk_size}")
    
    try:
        # 使用backed模式可以避免一次性加载所有数据到内存
        if use_backed:
            logger.info("Using backed mode - data stays on disk")
            adata = sc.read_h5ad(path, backed='r')  # 'r' for read-only
        else:
            logger.info("Loading full dataset into memory")
            adata = sc.read_h5ad(path)
        
        log_memory_usage("After reading h5ad")
        
        logger.info(f"Data shape: {adata.shape}")
        logger.info(f"Data size estimate: {adata.shape[0] * adata.shape[1] * 4 / (1024**3):.2f} GB (if float32)")
        
        logger.info("Preprocessing...")
        # 对于backed模式，某些操作可能需要转换为内存模式
        if use_backed:
            logger.info("Converting to memory mode for preprocessing...")
            adata = adata.to_memory()
            log_memory_usage("After converting to memory")
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        log_memory_usage("After normalize_total")
        
        sc.pp.log1p(adata)
        log_memory_usage("After log1p")
        
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
        log_memory_usage("After highly_variable_genes")
        
        # 转换为numpy数组
        logger.info("Converting to numpy arrays...")
        genes = adata.X
        if hasattr(genes, "toarray"): 
            logger.info("Converting sparse matrix to dense...")
            genes = genes.toarray()
        elif hasattr(genes, "to_numpy"):
            genes = genes.to_numpy()
        else:
            genes = np.array(genes)
        
        log_memory_usage("After converting genes")
        
        coords = adata.obsm['spatial']
        if hasattr(coords, "to_numpy"):
            coords = coords.to_numpy()
        else:
            coords = np.array(coords)
        
        log_memory_usage("After converting coords")
        
        # 映射时间点
        time_labels = sorted(adata.obs['timepoint'].unique())
        time_map = {t: i for i, t in enumerate(time_labels)}
        times = adata.obs['timepoint'].map(time_map).values.astype(int)
        
        logger.info(f"Total Timepoints: {len(time_labels)}")
        logger.info(f"Time Labels: {time_labels}")
        logger.info(f"Hold-out (Validation) Indices: {hold_out_indices} -> {[time_labels[i] for i in hold_out_indices]}")
        logger.info(f"Total cells: {len(times)}")
        
        # 清理adata释放内存
        del adata
        gc.collect()
        log_memory_usage("After deleting adata")
        
        # --- 划分训练集和验证集 ---
        # 训练集：不包含 hold_out_indices 时间点的细胞
        train_mask = ~np.isin(times, hold_out_indices)
        
        logger.info(f"Training cells: {train_mask.sum()}, Validation cells: {(~train_mask).sum()}")
        
        # 如果指定了chunk_size，分批处理
        if chunk_size and train_mask.sum() > chunk_size:
            logger.info(f"Using chunked processing with chunk_size={chunk_size}")
            # 先处理训练数据
            train_indices = np.where(train_mask)[0]
            n_chunks = (len(train_indices) + chunk_size - 1) // chunk_size
            
            train_genes_list = []
            train_coords_list = []
            train_times_list = []
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(train_indices))
                chunk_indices = train_indices[start_idx:end_idx]
                
                train_genes_list.append(genes[chunk_indices])
                train_coords_list.append(coords[chunk_indices])
                train_times_list.append(times[chunk_indices])
                
                log_memory_usage(f"Processed chunk {i+1}/{n_chunks}")
            
            train_data = {
                'genes': np.concatenate(train_genes_list, axis=0),
                'coords': np.concatenate(train_coords_list, axis=0),
                'times': np.concatenate(train_times_list, axis=0)
            }
            del train_genes_list, train_coords_list, train_times_list
            gc.collect()
        else:
            train_data = {
                'genes': genes[train_mask],
                'coords': coords[train_mask],
                'times': times[train_mask]
            }
        
        log_memory_usage("After creating train_data")
        
        # 验证集：只包含 hold_out_indices 时间点的细胞 (Ground Truth)
        val_data = {}
        for t_idx in hold_out_indices:
            mask = times == t_idx
            val_data[t_idx] = {
                'genes': genes[mask].copy(),
                'coords': coords[mask].copy()
            }
        
        log_memory_usage("After creating val_data")
        
        # 获取 t=0 的数据用于验证时的输入 (Seed)
        seed_mask = times == 0
        seed_data = {
            'genes': genes[seed_mask].copy(),
            'coords': coords[seed_mask].copy()
        }
        
        log_memory_usage("After creating seed_data")
        
        # 清理原始数据
        del genes, coords, times
        gc.collect()
        log_memory_usage("After cleanup")
        
        n_genes = train_data['genes'].shape[1]
        logger.info(f"Final data shapes - Train: {train_data['genes'].shape}, n_genes: {n_genes}")
        
        return train_data, val_data, seed_data, n_genes, len(time_labels)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        log_memory_usage("Error state")
        raise


def subset_array_dict(array_dict, max_cells):
    """Optionally down-sample cells to reduce memory footprint."""
    if max_cells is None:
        return array_dict
    total = array_dict['genes'].shape[0]
    if max_cells >= total:
        return array_dict
    subset_idx = np.random.choice(total, max_cells, replace=False)
    for key in array_dict:
        array_dict[key] = array_dict[key][subset_idx]
    return array_dict


def sample_indices(total, desired, allow_replacement=True):
    """Sample indices, falling back to replacement only if requested."""
    if total == 0:
        return np.array([], dtype=int)
    if desired is None:
        desired = total
    if desired <= 0:
        return np.array([], dtype=int)
    if allow_replacement and desired > total:
        return np.random.choice(total, desired, replace=True)
    return np.random.choice(total, min(desired, total), replace=False)


def plot_spatial_interp_umap(true_genes,
                             pred_genes,
                             true_coords,
                             pred_coords,
                             t_label,
                             save_path,
                             n_samples=1000):
    """
    在真实2D空间坐标上，对比插值时间点的真实 vs 预测表达。
    颜色来自基因表达的 UMAP+PCA 嵌入（类似 scNODE 的可视化），坐标使用真实空间位置。
    """
    true_genes = np.asarray(true_genes)
    pred_genes = np.asarray(pred_genes)
    true_coords = np.asarray(true_coords)
    pred_coords = np.asarray(pred_coords)

    if n_samples is None:
        true_sub = true_genes
        pred_sub = pred_genes
        true_xy = true_coords
        pred_xy = pred_coords
    else:
        n_true = min(n_samples, true_genes.shape[0])
        n_pred = min(n_samples, pred_genes.shape[0])
        true_idx = np.random.choice(true_genes.shape[0], n_true, replace=False)
        pred_idx = np.random.choice(pred_genes.shape[0], n_pred, replace=False)

        true_sub = true_genes[true_idx]
        pred_sub = pred_genes[pred_idx]
        true_xy = true_coords[true_idx]
        pred_xy = pred_coords[pred_idx]

    # 合并表达做一次 UMAP+PCA，得到 2D 表达嵌入
    combined_expr = np.concatenate([true_sub, pred_sub], axis=0)
    umap_emb, _, _ = umapWithPCA(combined_expr,
                                 n_neighbors=50,
                                 min_dist=0.1,
                                 pca_pcs=50)
    # 用 UMAP 第一维作为颜色
    umap_dim = umap_emb[:, 0]
    vmin, vmax = umap_dim.min(), umap_dim.max()
    split = true_sub.shape[0]
    true_color = umap_dim[:split]
    pred_color = umap_dim[split:]

    cmap = plt.cm.viridis
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sc1 = axes[0].scatter(true_xy[:, 0], true_xy[:, 1],
                          s=5, c=true_color, cmap=cmap,
                          vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='none')
    axes[0].set_title(f"True cells @ t={t_label}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    sc2 = axes[1].scatter(pred_xy[:, 0], pred_xy[:, 1],
                          s=5, c=pred_color, cmap=cmap,
                          vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='none')
    axes[1].set_title(f"Predicted cells @ t={t_label}")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # 共享 colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("UMAP-1 (expression embedding)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved spatial interpolation figure to {save_path}")
    plt.close(fig)


def pretrain_vae(model,
                 train_genes,
                 train_coords,
                 device,
                 iters=500,
                 batch_size=256,
                 lr=3e-4):
    """预训练 VAE (gene_encoder + gene_decoder)。"""
    encoder = model.gene_encoder
    decoder = model.gene_decoder
    encoder.train()
    decoder.train()

    dataset = TensorDataset(
        torch.from_numpy(train_genes).float(),
        torch.from_numpy(train_coords).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.95, 0.99))

    pbar = tqdm(range(iters), desc="Pre-training VAE")
    best_loss = float("inf")
    best_state = None

    for _ in pbar:
        try:
            batch_g, batch_xy = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_g, batch_xy = next(loader_iter)

        batch_g = batch_g.to(device)
        batch_xy = batch_xy.to(device)
        optimizer.zero_grad()

        enc_input = torch.cat([batch_g, batch_xy], dim=1)
        mu, log_var = encoder(enc_input)
        z = encoder.reparameterize(mu, log_var)
        recon = decoder(z)
        recon_loss = F.mse_loss(recon, batch_g)
        recon_loss.backward()
        optimizer.step()

        pbar.set_postfix({"Recon": f"{recon_loss.item():.6f}"})

        if recon_loss.item() < best_loss:
            best_loss = recon_loss.item()
            best_state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict()
            }

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
    logger.info(f"VAE pre-training finished. Best loss={best_loss:.6f}")


def generate_synthetic_t0(model, seed_data, n_synth, device):
    """使用预训练好的 VAE，从 t0 真实细胞拟合的潜在分布采样，生成更多 t0 细胞。"""
    encoder = model.gene_encoder
    decoder = model.gene_decoder
    encoder.eval()
    decoder.eval()

    genes = torch.from_numpy(seed_data['genes']).float().to(device)
    coords = torch.from_numpy(seed_data['coords']).float().to(device)
    with torch.no_grad():
        enc_input = torch.cat([genes, coords], dim=1)
        mu, log_var = encoder(enc_input)
        mus = mu.cpu().numpy()
        log_vars = log_var.cpu().numpy()

    component_idx = np.random.choice(len(mus), size=n_synth, replace=True)
    selected_mu = mus[component_idx]
    selected_log_var = log_vars[component_idx]
    eps = np.random.randn(*selected_mu.shape)
    latent = selected_mu + np.exp(0.5 * selected_log_var) * eps

    latent_tensor = torch.from_numpy(latent).float().to(device)
    with torch.no_grad():
        synth_genes = decoder(latent_tensor).cpu().numpy()

    coord_idx = np.random.choice(seed_data['coords'].shape[0], size=n_synth, replace=True)
    synth_coords = seed_data['coords'][coord_idx]

    return {
        "genes": synth_genes,
        "coords": synth_coords
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal TimelyGPT.")
    parser.add_argument("--data-path", type=str, default="../data/mouse.h5ad")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128,  # 减小默认batch size
                        help="Training batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hold-out", type=int, nargs="+", default=[3, 6],
                        help="Time indices to hold out for validation.")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--max-train-cells", type=int, default=None,
                        help="Optional limit on number of training cells to load.")
    parser.add_argument("--pretrain-iters", type=int, default=500,
                        help="Iterations for VAE pre-training.")
    parser.add_argument("--pretrain-batch-size", type=int, default=512,
                        help="Batch size for VAE pre-training.")
    parser.add_argument("--pretrain-lr", type=float, default=3e-4,
                        help="Learning rate for VAE pre-training.")
    parser.add_argument("--n-synth-t0", type=int, default=2000,
                        help="Number of synthetic t0 cells to generate for autoregressive training.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; >0 enables prefetch/pin_memory.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Prefetch factor when num_workers > 0.")
    parser.add_argument("--use-backed", action="store_true",
                        help="Use scanpy backed mode to avoid loading full dataset into memory")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Process data in chunks of this size (for memory efficiency)")
    parser.add_argument("--log-file", type=str, default="training.log",
                        help="Log file path")
    return parser.parse_args()

# ==========================================
# 5. 训练与验证
# ==========================================
def train_and_validate(args):
    # 配置
    DATA_PATH = args.data_path
    LATENT_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    HOLD_OUT = args.hold_out
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
    else:
        print("No GPU available, using CPU")
    
    # 1. 准备数据
    try:
        log_memory_usage("Before data loading")
        train_data, val_data_dict, seed_data, n_genes, n_timepoints = \
            load_and_split_data(
                DATA_PATH, 
                hold_out_indices=HOLD_OUT, 
                n_top_genes=args.n_top_genes,
                use_backed=args.use_backed,
                chunk_size=args.chunk_size
            )
        log_memory_usage("After data loading")
        
        if args.max_train_cells:
            logger.info(f"Subsetting training data to {args.max_train_cells} cells")
            train_data = subset_array_dict(train_data, args.max_train_cells)
            log_memory_usage("After subsetting")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        log_memory_usage("Error during data loading")
        raise
    
    # 打印t0和插值时间点的细胞数量
    n_t0_real = seed_data['genes'].shape[0]
    print("\n" + "="*70)
    print(f"t=0 时刻真实细胞数量: {n_t0_real}")
    print("插值时间点 (Hold-out) 的细胞数量:")
    for t_hole in sorted(HOLD_OUT):
        n_cells = val_data_dict[t_hole]['genes'].shape[0]
        print(f"  t={t_hole}: {n_cells} 个细胞")
    print("="*70 + "\n")
    logger.info(f"t=0 real cells: {n_t0_real}")
    logger.info(f"Hold-out timepoints cell counts: {[(t, val_data_dict[t]['genes'].shape[0]) for t in sorted(HOLD_OUT)]}")
    
    # 训练时间点：按照"挖掉的时间点"做 OT（用于插值监督）
    train_tps_idx = sorted([int(t) for t in HOLD_OUT])
    logger.info(f"Train time indices (OT targets / hold-out): {train_tps_idx}")

    train_genes_by_t = {
        t: val_data_dict[t]['genes']
        for t in train_tps_idx
    }
    
    # 2. 模型
    # 必须用真实的 Config
    from model.TimelyGPT_CTS.layers.configs import RetNetConfig
    config = RetNetConfig(d_model=LATENT_DIM, num_layers=3, num_heads=16, qk_dim=128, v_dim=128, ffn_proj_size=4*128,
                          forward_impl='parallel', use_bias_in_msr_out=False)
    
    model = SpatiotemporalTimelyGPT(config, n_genes, n_timepoints).to(DEVICE)

    # 2a. 预训练 VAE（只使用真实训练集细胞）
    pretrain_vae(model,
                 train_data['genes'],
                 train_data['coords'],
                 device=DEVICE,
                 iters=args.pretrain_iters,
                 batch_size=args.pretrain_batch_size,
                 lr=args.pretrain_lr)

    # 2b. 基于 t0 真实细胞生成更多 synthetic t0 细胞（用于评估和画图）
    n_t0_real = seed_data['genes'].shape[0]
    logger.info(f"Real t0 cells: {n_t0_real}")
    
    # 计算所有hold-out时间点中真实细胞的最大数量，确保生成足够的synthetic细胞
    max_true_cells = max([val_data_dict[t]['genes'].shape[0] for t in HOLD_OUT])
    # 生成至少与最大真实细胞数量相等的synthetic细胞（多生成一些以确保充足）
    n_synth_for_eval = max(100000, int(max_true_cells * 1.2))  # 至少10万，或真实细胞数的1.2倍
    logger.info(f"Max true cells at hold-out timepoints: {max_true_cells}")
    logger.info(f"Generating {n_synth_for_eval} synthetic cells for evaluation/plotting")
    synthetic_seed = generate_synthetic_t0(model,
                                           seed_data,
                                           n_synth=n_synth_for_eval,
                                           device=DEVICE)
    logger.info(f"Synthetic t0 cells generated for evaluation/plotting: {synthetic_seed['genes'].shape[0]}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # OT Loss（用于在训练时间点匹配真实分布）
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
    LATENT_COEFF = 1.0
    iters_per_epoch = 100  # 每个 epoch 内的迭代次数，可按需调整

    print("\n=== 自回归训练：从 t0 真实细胞出发，在训练时间点上匹配分布 ===")
    logger.info("Start autoregressive OT training")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        pbar = tqdm(range(iters_per_epoch), desc=f"Train Ep {epoch+1}")

        for _ in pbar:
            # ----- 1. 从 t0 真实细胞里采样一批起点（训练时使用真实t0细胞数量）-----
            n_t0_real = seed_data['genes'].shape[0]
            batch_size = min(BATCH_SIZE, n_t0_real)
            rand_idx = np.random.choice(n_t0_real, size=batch_size, replace=False)
            batch_g = torch.from_numpy(seed_data['genes'][rand_idx]).float().to(DEVICE)
            batch_c = torch.from_numpy(seed_data['coords'][rand_idx]).float().to(DEVICE)
            
            optimizer.zero_grad()
            
            # ----- 2. 自回归生成整条轨迹：[B, T, Genes] -----
            recon_seq, mu, log_var, z_seq = model(batch_g, batch_c)
            
            # ----- 3. 在所有训练时间点上计算 Sinkhorn OT loss -----
            ot_loss = 0.0
            for t in train_tps_idx:
                # 预测群体：这批细胞在时间 t 的表达
                pred_x = recon_seq[:, t, :]  # [B, n_genes]

                # 真实群体：该时间点的所有真实细胞
                true_x_all = train_genes_by_t[t]
                subsample_size = min(200, true_x_all.shape[0])
                subsample_idx = np.random.choice(true_x_all.shape[0], subsample_size, replace=False)
                true_x = torch.from_numpy(true_x_all[subsample_idx]).float().to(DEVICE)

                ot_loss = ot_loss + ot_solver(pred_x, true_x)
            
            # ----- 4. 轨迹平滑正则 -----
            latent_drift_loss = torch.mean((z_seq[:, 1:, :] - z_seq[:, :-1, :]) ** 2)
            
            loss = ot_loss + LATENT_COEFF * latent_drift_loss

            # ----- 5. 反向传播 -----
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "OT": f"{ot_loss.item():.4f}",
                "Smooth": f"{latent_drift_loss.item():.4f}",
            })

        avg_loss = float(np.mean(epoch_losses))
        print(f"Ep {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch+1} finished, Avg Loss={avg_loss:.4f}")

    # ------------------------------------------------------------------
    # 训练结束后：在插值时间点上做一次评估 + 2D 空间可视化
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        # 计算所有hold-out时间点中真实细胞的最大数量
        max_true_cells = max([val_data_dict[t]['genes'].shape[0] for t in HOLD_OUT])
        n_available = synthetic_seed['genes'].shape[0]
        
        fig_dir = os.path.join(project_root, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        for t_hole in HOLD_OUT:
            # 真实插值时间点的全量群体
            true_genes_all = val_data_dict[t_hole]['genes']
            true_coords_all = val_data_dict[t_hole]['coords']
            n_true_cells = true_genes_all.shape[0]

            # 对每个时间点单独评估：以真实该时间点的细胞数为批大小，从t0合成池采样起点
            eval_chunk = 5000  # 可调；根据显存调整
            need = n_true_cells
            seed_idx = np.random.choice(n_available, size=need, replace=(need > n_available))
            logger.info(f"t={t_hole}: eval {need} cells (true={n_true_cells}, available synth={n_available}, chunk={eval_chunk})")

            pred_chunks = []
            for start in range(0, need, eval_chunk):
                end = min(start + eval_chunk, need)
                seed_g_chunk = torch.from_numpy(synthetic_seed['genes'][seed_idx[start:end]]).float().to(DEVICE)
                seed_c_chunk = torch.from_numpy(synthetic_seed['coords'][seed_idx[start:end]]).float().to(DEVICE)
                with torch.no_grad():
                    recon_chunk, _, _, _ = model(seed_g_chunk, seed_c_chunk)
                pred_chunks.append(recon_chunk.cpu())
                del seed_g_chunk, seed_c_chunk, recon_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            recon_seq_vis = torch.cat(pred_chunks, dim=0).numpy()  # [need, T, genes]

            # 使用该时间点对应的真实空间位置和数量作图
            pred_genes_sub = recon_seq_vis[:, t_hole, :]
            pred_coords_sub = true_coords_all  # 坐标用真实t_hole的空间位置

            save_path = os.path.join(fig_dir, f"TimelyGPT_spatial_t{t_hole}.png")
            plot_spatial_interp_umap(true_genes_all,
                                     pred_genes_sub,
                                     true_coords_all,
                                     pred_coords_sub,
                                     t_label=t_hole,
                                     save_path=save_path,
                                     n_samples=None)  # 不下采样

if __name__ == "__main__":
    cli_args = parse_args()
    
    # 更新日志文件路径
    if cli_args.log_file != "training.log":
        # 移除旧的handler，添加新的
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(logging.FileHandler(cli_args.log_file))
    
    logger.info("="*70)
    logger.info("Starting training with arguments:")
    for key, value in vars(cli_args).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*70)
    
    log_memory_usage("Program start")
    
    # 将数据路径
    if cli_args.data_path.startswith('../'):
        rel_path = cli_args.data_path.replace('../', '', 1)
        data_path_abs = os.path.join(project_root, rel_path)
    elif not os.path.isabs(cli_args.data_path):
        data_path_abs = os.path.join(project_root, cli_args.data_path)
    else:
        data_path_abs = cli_args.data_path
    
    cli_args.data_path = data_path_abs
    logger.info(f"Resolved data path: {data_path_abs}")
    
    try:
        if os.path.exists(cli_args.data_path):
            train_and_validate(cli_args)
        else:
            logger.error(f"Dataset not found: {cli_args.data_path}")
            print(f"Dataset missing. Checked path: {cli_args.data_path}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        log_memory_usage("Error state")
        raise