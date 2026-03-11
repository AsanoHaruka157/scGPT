import sys
import os
import pathlib
import argparse
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import logging
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ==========================================
# 环境与路径设置 (保持与pretrain.py一致以加载模块)
# ==========================================
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
project_root = CURRENT_DIR.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

# 导入项目模块
from plotting.PlottingUtils import umapWithPCA

# 字体设置
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'DejaVu Sans' in available_fonts:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
elif 'Arial' in available_fonts:
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
else:
    plt.rcParams['font.family'] = 'sans-serif'

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 模型定义 (必须与pretrain.py完全一致)
# ==========================================
class SpatialEncoder(nn.Module):
    def __init__(self, input_dim=2, d_model=128):
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
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256]):
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
        # Eval模式下通常可以直接返回mu，但为了复现可视化逻辑，保持采样
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class GeneDecoder(nn.Module):
    def __init__(self, latent_dim=128, n_genes=2000, hidden_dims=[256, 512]):
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
# 数据加载函数
# ==========================================
def load_data_by_timepoint(path, n_top_genes=2000):
    logger.info(f"Loading data from {path}...")
    adata = sc.read_h5ad(path)
    
    # 预处理流程 (必须与训练时一致)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
    genes = adata.X
    if hasattr(genes, "toarray"): genes = genes.toarray()
    elif hasattr(genes, "to_numpy"): genes = genes.to_numpy()
    
    coords = adata.obsm['spatial']
    if hasattr(coords, "to_numpy"): coords = coords.to_numpy()
    
    # 时间点排序逻辑
    def sort_timepoints(tp_list):
        def timepoint_key(tp):
            if isinstance(tp, str) and tp.startswith('E'):
                try:
                    num = float(tp[1:])
                    return (0, num) if num == 9.5 else (1, num)
                except: return (2, str(tp))
            return (2, str(tp))
        return sorted(tp_list, key=timepoint_key)
    
    time_labels = sort_timepoints(adata.obs['timepoint'].unique().tolist())
    time_map = {t: i for i, t in enumerate(time_labels)}
    times = adata.obs['timepoint'].map(time_map).values.astype(int)
    
    # Cell type
    cell_type_col = None
    for col in ['cell_type', 'annotation', 'celltype', 'CellType']:
        if col in adata.obs.columns:
            cell_type_col = col
            break
    cell_types = adata.obs[cell_type_col].values if cell_type_col else None
    
    data_by_timepoint = {}
    for t_idx in range(len(time_labels)):
        mask = times == t_idx
        data_by_timepoint[t_idx] = {
            'genes': genes[mask].copy(),
            'coords': coords[mask].copy()
        }
        if cell_types is not None:
            data_by_timepoint[t_idx]['cell_types'] = cell_types[mask].copy()
            
    n_genes = data_by_timepoint[0]['genes'].shape[1]
    logger.info(f"Data loaded. N_genes: {n_genes}, N_timepoints: {len(time_labels)}")
    return data_by_timepoint, n_genes

# ==========================================
# 可视化函数 (核心评估逻辑)
# ==========================================
def visualize_reconstruction(vae_encoder, vae_decoder, mlp, data_by_timepoint, 
                               timepoints_to_plot, device, latent_dim, save_dir):
    
    vae_encoder.eval()
    vae_decoder.eval()
    mlp.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for t_idx in timepoints_to_plot:
        if t_idx not in data_by_timepoint:
            logger.warning(f"Timepoint {t_idx} not found, skipping...")
            continue
        
        logger.info(f"Visualizing timepoint {t_idx}...")
        
        true_genes = data_by_timepoint[t_idx]['genes']
        true_coords = data_by_timepoint[t_idx]['coords']
        n_cells = true_genes.shape[0]
        
        has_cell_types = 'cell_types' in data_by_timepoint[t_idx]
        true_cell_types = data_by_timepoint[t_idx]['cell_types'] if has_cell_types else None
        
        # 重建
        batch_size = 512
        recon_genes_list = []
        z_combined_list = []  # 保存隐变量用于ARI计算
        
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                end_idx = min(i + batch_size, n_cells)
                batch_genes = torch.from_numpy(true_genes[i:end_idx]).float().to(device)
                batch_coords = torch.from_numpy(true_coords[i:end_idx]).float().to(device)
                
                mu, log_var = vae_encoder(batch_genes)
                z_vae = vae_encoder.reparameterize(mu, log_var)
                z_spatial = mlp(batch_coords)
                z_combined = z_vae + z_spatial
                recon_batch = vae_decoder(z_combined)
                
                recon_genes_list.append(recon_batch.cpu().numpy())
                z_combined_list.append(z_combined.cpu().numpy())  # 保存隐变量
        
        recon_genes = np.concatenate(recon_genes_list, axis=0)
        z_combined_all = np.concatenate(z_combined_list, axis=0)  # 所有细胞的隐变量
        
        # 在隐变量空间上进行聚类（用于ARI计算）
        n_clusters = len(np.unique(true_cell_types)) if has_cell_types else 15
        logger.info(f"Clustering latent space z_combined into {n_clusters} clusters for ARI calculation...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        latent_cluster_labels = kmeans.fit_predict(z_combined_all)

        # Computing ARI score on latent space
        ari_score = None
        if has_cell_types:
            ari_score = adjusted_rand_score(true_cell_types, latent_cluster_labels)
            logger.info(f"Timepoint {t_idx} - ARI on latent space: {ari_score:.4f}")
        
        # UMAP & Clustering for visualization
        logger.info(f"Computing UMAP for reconstructed genes at t={t_idx} (for visualization)...")
        recon_umap_emb, _, _ = umapWithPCA(
            recon_genes, n_neighbors=50, min_dist=0.1, pca_pcs=50
        )
        
        # 对UMAP嵌入进行聚类（用于可视化）
        logger.info(f"Clustering UMAP embedding into {n_clusters} clusters (for visualization)...")
        kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        recon_cluster_labels = kmeans_umap.fit_predict(recon_umap_emb)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot settings helper
        def plot_scatter(ax, coords, labels, title, is_cell_type=False):
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)
            
            if n_labels <= 20: cmap = plt.cm.tab20
            elif n_labels <= 40: cmap = plt.cm.tab20b
            else: cmap = plt.cm.gist_ncar
            
            label_to_color = {l: cmap(i / max(n_labels, 1)) for i, l in enumerate(unique_labels)}
            colors = [label_to_color[l] for l in labels]
            
            ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6, c=colors, edgecolors='none')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=label_to_color[l], label=str(l)) for l in unique_labels]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=8, ncol=1, framealpha=0.9, title='Cell Types' if is_cell_type else 'Clusters')

        # Left: True
        if has_cell_types:
            plot_scatter(axes[0], true_coords, true_cell_types, 
                        f'True Cells @ t={t_idx}\n(n={n_cells})', is_cell_type=True)
        else:
            axes[0].scatter(true_coords[:, 0], true_coords[:, 1], s=10, alpha=0.6, c='steelblue')
            axes[0].set_title(f'True Cells @ t={t_idx}')
        
        # Right: Recon
        title_right = f'Reconstructed (ARI={ari_score:.3f})' if ari_score else 'Reconstructed'
        plot_scatter(axes[1], true_coords, recon_cluster_labels, title_right)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{latent_dim}t{t_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close(fig)

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained VAE models.")
    parser.add_argument("--data-path", type=str, default="../data/mouse.h5ad", help="Path to h5ad data")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory with .pt files")
    parser.add_argument("--fig-dir", type=str, default="./figures", help="Output directory for images")
    parser.add_argument("--latent-dim", type=int, default=1024, help="Must match training setting")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Must match training setting")
    parser.add_argument("--viz-timepoints", type=int, nargs="+", default=[0, 2], help="Timepoints to visualize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. 路径检查
    if args.data_path.startswith('../'):
        data_path_abs = os.path.join(project_root, args.data_path.replace('../', '', 1))
    elif not os.path.isabs(args.data_path):
        data_path_abs = os.path.join(project_root, args.data_path)
    else:
        data_path_abs = args.data_path

    vae_path = os.path.join(args.model_dir, f"vae_lat{args.latent_dim}.pt")
    mlp_path = os.path.join(args.model_dir, f"mlp_lat{args.latent_dim}.pt")

    if not os.path.exists(vae_path) or not os.path.exists(mlp_path):
        logger.error(f"Model files not found for latent_dim={args.latent_dim}.")
        logger.error(f"Looked for: {vae_path} AND {mlp_path}")
        return

    # 2. 加载数据
    data_by_timepoint, n_genes = load_data_by_timepoint(data_path_abs, n_top_genes=args.n_top_genes)

    # 3. 初始化模型结构
    logger.info("Initializing models...")
    vae_encoder = GeneVAEEncoder(input_dim=n_genes, latent_dim=args.latent_dim).to(device)
    vae_decoder = GeneDecoder(latent_dim=args.latent_dim, n_genes=n_genes).to(device)
    mlp = SpatialEncoder(input_dim=2, d_model=args.latent_dim).to(device)

    # 4. 加载权重 (关键步骤)
    logger.info("Loading model weights...")
    try:
        # VAE包含 encoder, decoder, n_genes, latent_dim 等信息
        vae_checkpoint = torch.load(vae_path, map_location=device)
        vae_encoder.load_state_dict(vae_checkpoint['encoder'])
        vae_decoder.load_state_dict(vae_checkpoint['decoder'])
        
        # 验证n_genes是否匹配
        if vae_checkpoint['n_genes'] != n_genes:
            logger.warning(f"Warning: Model saved with n_genes={vae_checkpoint['n_genes']}, but data has {n_genes}")

        # MLP包含 model, latent_dim
        mlp_checkpoint = torch.load(mlp_path, map_location=device)
        mlp.load_state_dict(mlp_checkpoint['model'])
        
        logger.info("Weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        return

    # 5. 执行评估与可视化
    visualize_reconstruction(
        vae_encoder,
        vae_decoder,
        mlp,
        data_by_timepoint,
        args.viz_timepoints,
        device,
        args.latent_dim,
        save_dir=args.fig_dir
    )
    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()