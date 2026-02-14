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
import geomloss
import logging
import gc
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免字体问题
import matplotlib.pyplot as plt
# 设置matplotlib字体，避免字体警告
import matplotlib.font_manager as fm
# 获取系统可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
# 优先使用DejaVu Sans，如果不存在则使用其他可用字体
if 'DejaVu Sans' in available_fonts:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
elif 'Arial' in available_fonts:
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
elif 'Liberation Sans' in available_fonts:
    plt.rcParams['font.family'] = 'Liberation Sans'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
else:
    # 使用默认字体，但不设置sans-serif以避免警告
    plt.rcParams['font.family'] = 'sans-serif'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='findfont: Generic family')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 先设置路径，再导入项目模块
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
project_root = CURRENT_DIR.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

# 路径设置后再导入项目模块
from plotting.PlottingUtils import umapWithPCA

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, memory monitoring disabled")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pretrain.log'),
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
# 模型组件
# ==========================================
class SpatialEncoder(nn.Module):
    """MLP：将2D空间坐标映射到128维"""
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
    """VAE编码器：将基因表达编码到隐空间"""
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
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

class GeneDecoder(nn.Module):
    """VAE解码器：从隐空间解码到基因表达"""
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
# 数据加载
# ==========================================
def load_data_by_timepoint(path, n_top_genes=2000, use_backed=False):
    """
    加载数据并按时间点组织
    返回: data_by_timepoint = {time_idx: {'genes': array, 'coords': array}}
    """
    log_memory_usage("Before loading")
    logger.info(f"Loading data from {path}...")
    
    try:
        if use_backed:
            logger.info("Using backed mode - data stays on disk")
            adata = sc.read_h5ad(path, backed='r')
        else:
            logger.info("Loading full dataset into memory")
            adata = sc.read_h5ad(path)
        
        log_memory_usage("After reading h5ad")
        logger.info(f"Data shape: {adata.shape}")
        
        logger.info("Preprocessing...")
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
        
        # 映射时间点 - 自定义排序确保E9.5排在最前面
        def sort_timepoints(tp_list):
            """自定义排序：E9.5排在最前面，其他按数值排序"""
            def timepoint_key(tp):
                if isinstance(tp, str) and tp.startswith('E'):
                    # 提取数字部分
                    try:
                        num = float(tp[1:])
                        # E9.5应该排在最前面，给它一个很小的值
                        if num == 9.5:
                            return (0, num)
                        else:
                            return (1, num)
                    except:
                        return (2, str(tp))
                return (2, str(tp))
            return sorted(tp_list, key=timepoint_key)
        
        time_labels = sort_timepoints(adata.obs['timepoint'].unique().tolist())
        time_map = {t: i for i, t in enumerate(time_labels)}
        times = adata.obs['timepoint'].map(time_map).values.astype(int)
        
        logger.info(f"Total Timepoints: {len(time_labels)}")
        logger.info(f"Time Labels: {time_labels}")
        logger.info(f"Total cells: {len(times)}")
        
        # 获取cell type信息（如果存在）
        cell_type_col = None
        for col in ['cell_type', 'annotation', 'celltype', 'CellType']:
            if col in adata.obs.columns:
                cell_type_col = col
                break
        
        if cell_type_col:
            logger.info(f"Found cell type column: {cell_type_col}")
            cell_types = adata.obs[cell_type_col].values
        else:
            logger.warning("No cell type column found, will use default colors in visualization")
            cell_types = None
        
        # 按时间点组织数据
        data_by_timepoint = {}
        for t_idx in range(len(time_labels)):
            mask = times == t_idx
            data_by_timepoint[t_idx] = {
                'genes': genes[mask].copy(),
                'coords': coords[mask].copy()
            }
            if cell_types is not None:
                data_by_timepoint[t_idx]['cell_types'] = cell_types[mask].copy()
            logger.info(f"Timepoint {t_idx}: {data_by_timepoint[t_idx]['genes'].shape[0]} cells")
        
        # 清理
        del adata, genes, coords, times
        if cell_types is not None:
            del cell_types
        gc.collect()
        log_memory_usage("After cleanup")
        
        n_genes = data_by_timepoint[0]['genes'].shape[1]
        logger.info(f"Number of genes: {n_genes}")
        
        return data_by_timepoint, n_genes, len(time_labels)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        log_memory_usage("Error state")
        raise

# ==========================================
# 训练函数
# ==========================================
def train_joint(vae_encoder, vae_decoder, mlp, data_by_timepoint, device, 
                epochs=10, batch_size=256, lr=1e-3):
    """
    联合训练VAE和MLP：
    1. 将每个时刻的真实细胞群输入VAE编码器，得到隐空间向量 z_vae
    2. MLP将对应时刻的2D空间坐标映射到128维空间编码 z_spatial
    3. 组合：z_combined = z_vae + z_spatial
    4. VAE解码器从z_combined解码得到预测的基因表达
    5. 使用Sinkhorn距离测量预测细胞群体与真实细胞群之间的距离
    6. 同时更新VAE和MLP的参数
    """
    logger.info("="*70)
    logger.info("Joint Training VAE and MLP")
    logger.info("="*70)
    
    vae_encoder.train()
    vae_decoder.train()
    mlp.train()
    
    # 合并所有参数（VAE + MLP）
    all_params = list(vae_encoder.parameters()) + list(vae_decoder.parameters()) + list(mlp.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    
    # Sinkhorn距离计算器
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
    
    timepoints = sorted(data_by_timepoint.keys())
    
    # 计算总迭代次数用于进度条
    total_iterations = epochs * len(timepoints)
    pbar = tqdm(total=total_iterations, desc="Training")
    
    # 记录loss用于画曲线
    all_epoch_losses = []
    all_epoch_ot_losses = []
    all_epoch_kl_losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_ot_losses = []
        epoch_kl_losses = []
        
        for t_idx in timepoints:
            genes_t = data_by_timepoint[t_idx]['genes']
            coords_t = data_by_timepoint[t_idx]['coords']
            
            # 创建数据加载器
            dataset = TensorDataset(
                torch.from_numpy(genes_t).float(),
                torch.from_numpy(coords_t).float()
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            for batch_genes, batch_coords in loader:
                batch_genes = batch_genes.to(device)
                batch_coords = batch_coords.to(device)
                
                optimizer.zero_grad()
                
                # 1. VAE编码器：将真实细胞群的基因表达编码到隐空间
                mu, log_var = vae_encoder(batch_genes)
                z_vae = vae_encoder.reparameterize(mu, log_var)  # [B, latent_dim]
                
                # 2. MLP：将空间坐标映射到128维空间编码
                z_spatial = mlp(batch_coords)  # [B, latent_dim]
                
                # 3. 组合：隐空间向量 + 空间位置编码
                z_combined = z_vae + z_spatial  # [B, latent_dim]
                
                # 4. VAE解码器：从组合后的隐空间向量解码得到预测的基因表达
                recon_genes = vae_decoder(z_combined)  # [B, n_genes]
                
                # 5. 使用Sinkhorn距离计算损失（预测细胞群体 vs 真实细胞群体）
                ot_loss = ot_solver(recon_genes, batch_genes)
                
                # 6. KL散度正则化（可选）
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                
                # 总损失
                loss = ot_loss + 0.001 * kl_loss
                
                # 反向传播，同时更新VAE和MLP的参数
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_ot_losses.append(ot_loss.item())
                epoch_kl_losses.append(kl_loss.item())
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "Epoch": f"{epoch+1}/{epochs}",
                "t": t_idx,
                "Loss": f"{np.mean(epoch_losses):.4f}" if epoch_losses else "N/A"
            })
        
        avg_loss = np.mean(epoch_losses)
        avg_ot = np.mean(epoch_ot_losses)
        avg_kl = np.mean(epoch_kl_losses)
        
        all_epoch_losses.append(avg_loss)
        all_epoch_ot_losses.append(avg_ot)
        all_epoch_kl_losses.append(avg_kl)
    
    pbar.close()
    logger.info("Joint training completed")
    logger.info(f"Final - Loss: {all_epoch_losses[-1]:.6f} | OT: {all_epoch_ot_losses[-1]:.6f} | KL: {all_epoch_kl_losses[-1]:.6f}")
    
    return all_epoch_losses, all_epoch_ot_losses, all_epoch_kl_losses

# ==========================================
# 绘制Loss曲线
# ==========================================
def plot_loss_curves(epoch_losses, epoch_ot_losses, epoch_kl_losses, save_path):
    """绘制训练loss曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(epoch_losses) + 1)
    
    # 总Loss
    axes[0].plot(epochs, epoch_losses, 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # OT Loss
    axes[1].plot(epochs, epoch_ot_losses, 'r-', linewidth=2, label='OT Loss')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('OT Loss', fontsize=10)
    axes[1].set_title('Sinkhorn OT Loss', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # KL Loss
    axes[2].plot(epochs, epoch_kl_losses, 'g-', linewidth=2, label='KL Loss')
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('KL Loss', fontsize=10)
    axes[2].set_title('KL Divergence Loss', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved loss curves to {save_path}")
    plt.close(fig)

# ==========================================
# 可视化函数
# ==========================================
def visualize_reconstruction(vae_encoder, vae_decoder, mlp, data_by_timepoint, 
                               timepoints_to_plot, device, latent_dim, save_dir="./figures"):
    """
    可视化重建结果：对比真实细胞和重建细胞的UMAP图
    timepoints_to_plot: 要可视化的时间点索引列表，例如 [0, 2]
    """
    logger.info("="*70)
    logger.info("Visualizing reconstruction results")
    logger.info("="*70)
    
    vae_encoder.eval()
    vae_decoder.eval()
    mlp.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有时间点的标签（用于显示）
    all_timepoints = sorted(data_by_timepoint.keys())
    
    logger.info(f"Will visualize {len(timepoints_to_plot)} timepoints: {timepoints_to_plot}")
    
    for t_idx in tqdm(timepoints_to_plot, desc="Visualizing timepoints"):
        if t_idx not in data_by_timepoint:
            logger.warning(f"Timepoint {t_idx} not found, skipping...")
            continue
        
        logger.info(f"Visualizing timepoint {t_idx}...")
        
        # 获取该时间点的真实数据
        true_genes = data_by_timepoint[t_idx]['genes']
        true_coords = data_by_timepoint[t_idx]['coords']
        n_cells = true_genes.shape[0]
        
        # 获取cell type信息（如果存在）
        has_cell_types = 'cell_types' in data_by_timepoint[t_idx]
        if has_cell_types:
            true_cell_types = data_by_timepoint[t_idx]['cell_types']
        else:
            true_cell_types = None
        
        # 分批处理以避免内存问题
        batch_size = 512
        recon_genes_list = []
        
        logger.info(f"Reconstructing {n_cells} cells for timepoint {t_idx}...")
        with torch.no_grad():
            pbar = tqdm(range(0, n_cells, batch_size), desc=f"Reconstructing t={t_idx}", leave=False)
            for i in pbar:
                end_idx = min(i + batch_size, n_cells)
                batch_genes = torch.from_numpy(true_genes[i:end_idx]).float().to(device)
                batch_coords = torch.from_numpy(true_coords[i:end_idx]).float().to(device)
                
                # 重建流程
                mu, log_var = vae_encoder(batch_genes)
                z_vae = vae_encoder.reparameterize(mu, log_var)
                z_spatial = mlp(batch_coords)
                z_combined = z_vae + z_spatial
                recon_batch = vae_decoder(z_combined)
                
                recon_genes_list.append(recon_batch.cpu().numpy())
                pbar.set_postfix({"cells": f"{end_idx}/{n_cells}"})
        
        recon_genes = np.concatenate(recon_genes_list, axis=0)
        
        # 对重建的基因表达进行UMAP降维
        logger.info(f"Computing UMAP for reconstructed genes at t={t_idx}...")
        recon_umap_emb, _, _ = umapWithPCA(
            recon_genes,
            n_neighbors=50,
            min_dist=0.1,
            pca_pcs=50
        )
        
        # 对UMAP嵌入进行聚类，类别数量等于细胞类型数量
        if has_cell_types:
            n_clusters = len(np.unique(true_cell_types))  # 等于cell type数量
        else:
            n_clusters = 15  # 默认15个类别
        
        logger.info(f"Clustering UMAP embedding into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        recon_cluster_labels = kmeans.fit_predict(recon_umap_emb)

        # Computing ARI score
        ari_score = None
        if has_cell_types:
            # 将字符串类型的 cell_types 转换为数值标签用于计算
            # 注意：adjusted_rand_score 可以直接处理字符串标签，不需要手动转数值，但要确保维度对齐
            ari_score = adjusted_rand_score(true_cell_types, recon_cluster_labels)
            logger.info(f"Timepoint {t_idx} - Adjusted Rand Index (ARI): {ari_score:.4f}")
        else:
            logger.warning(f"Timepoint {t_idx} - No ground truth cell types found, skipping ARI calculation.")
        
        # 绘制对比图（增加宽度以容纳图例）
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：真实细胞群的2D空间位置，颜色为cell type
        if has_cell_types:
            # 为每个cell type分配颜色
            unique_types = np.unique(true_cell_types)
            n_types = len(unique_types)
            if n_types <= 20:
                cmap = plt.cm.tab20
            elif n_types <= 40:
                cmap = plt.cm.tab20b
            else:
                cmap = plt.cm.gist_ncar
            
            type_to_color = {t: cmap(i / max(n_types, 1)) for i, t in enumerate(unique_types)}
            colors_left = [type_to_color[t] for t in true_cell_types]
            
            scatter_left = axes[0].scatter(true_coords[:, 0], true_coords[:, 1], 
                                          s=10, alpha=0.6, c=colors_left, edgecolors='none')
            axes[0].set_title(f'True Cells @ t={t_idx}\n(n={n_cells}, colored by cell type)', 
                            fontsize=12, fontweight='bold')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=type_to_color[t], label=str(t)) for t in unique_types]
            axes[0].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=8, ncol=1, framealpha=0.9, title='Cell Types')
        else:
            # 如果没有cell type，使用默认颜色
            axes[0].scatter(true_coords[:, 0], true_coords[:, 1], 
                          s=10, alpha=0.6, c='steelblue', edgecolors='none')
            axes[0].set_title(f'True Cells @ t={t_idx}\n(n={n_cells})', 
                            fontsize=12, fontweight='bold')
        
        axes[0].set_xlabel('X Coordinate', fontsize=10)
        axes[0].set_ylabel('Y Coordinate', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        
        # 右图：重建细胞群的2D空间位置（使用真实坐标），颜色为重建基因表达的UMAP聚类标签
        # 使用离散类别着色，类似左图的cell type着色方式
        unique_clusters = np.unique(recon_cluster_labels)
        n_clusters_actual = len(unique_clusters)
        
        # 为每个聚类分配颜色
        if n_clusters_actual <= 20:
            cmap_cluster = plt.cm.tab20
        elif n_clusters_actual <= 40:
            cmap_cluster = plt.cm.tab20b
        else:
            cmap_cluster = plt.cm.gist_ncar
        
        cluster_to_color = {c: cmap_cluster(i / max(n_clusters_actual, 1)) for i, c in enumerate(unique_clusters)}
        colors_right = [cluster_to_color[c] for c in recon_cluster_labels]
        
        scatter_right = axes[1].scatter(true_coords[:, 0], true_coords[:, 1], 
                                      s=10, alpha=0.6, c=colors_right, edgecolors='none')
        # 设置标题，如果计算了ARI则显示
        if ari_score is not None:
            axes[1].set_title(f'Reconstructed (ARI={ari_score:.3f})\n(n={n_cells}, latent_dim={latent_dim}, colored by {n_clusters_actual} clusters)', 
                             fontsize=12, fontweight='bold')
        else:
            axes[1].set_title(f'Reconstructed Cells at t={t_idx}\n(n={n_cells}, latent_dim={latent_dim}, colored by {n_clusters_actual} clusters)', 
                             fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X Coordinate', fontsize=10)
        axes[1].set_ylabel('Y Coordinate', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements_right = [Patch(facecolor=cluster_to_color[c], label=f'Cluster {c}') for c in sorted(unique_clusters)]
        axes[1].legend(handles=legend_elements_right, bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=8, ncol=1, framealpha=0.9, title='Clusters')
        
        plt.tight_layout()
        
        # 保存图片，文件名包含latent_dim与时间点，例如：512t0.png
        save_path = os.path.join(save_dir, f"{latent_dim}t{t_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        plt.close(fig)
    
    logger.info("Visualization completed")

# ==========================================
# 主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain VAE and MLP models.")
    parser.add_argument("--data-path", type=str, default="../data/mouse.h5ad",
                        help="Path to the h5ad data file")
    parser.add_argument("--latent-dim", type=int, default=1024,
                        help="Latent dimension for VAE (default: 128)")
    parser.add_argument("--n-top-genes", type=int, default=3000,
                        help="Number of top highly variable genes to use")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for joint training")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--use-backed", action="store_true",
                        help="Use scanpy backed mode to avoid loading full dataset into memory")
    parser.add_argument("--save-dir", type=str, default="./models",
                        help="Directory to save models")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization after training (visualization is enabled by default)")
    parser.add_argument("--viz-timepoints", type=int, nargs="+", default=[0, 2],
                        help="Timepoint indices to visualize (default: [0, 2])")
    parser.add_argument("--fig-dir", type=str, default="./figures",
                        help="Directory to save figures")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("="*70)
    logger.info("Starting pretraining")
    logger.info("="*70)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*70)
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    log_memory_usage("Program start")
    
    # 解析数据路径
    if args.data_path.startswith('../'):
        rel_path = args.data_path.replace('../', '', 1)
        data_path_abs = os.path.join(project_root, rel_path)
    elif not os.path.isabs(args.data_path):
        data_path_abs = os.path.join(project_root, args.data_path)
    else:
        data_path_abs = args.data_path
    
    logger.info(f"Resolved data path: {data_path_abs}")
    
    if not os.path.exists(data_path_abs):
        logger.error(f"Dataset not found: {data_path_abs}")
        raise FileNotFoundError(f"Dataset not found: {data_path_abs}")
    
    # 1. 加载数据
    try:
        log_memory_usage("Before data loading")
        data_by_timepoint, n_genes, n_timepoints = load_data_by_timepoint(
            data_path_abs,
            n_top_genes=args.n_top_genes,
            use_backed=args.use_backed
        )
        log_memory_usage("After data loading")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        raise
    
    # 2. 创建模型
    logger.info("Creating models...")
    vae_encoder = GeneVAEEncoder(input_dim=n_genes, latent_dim=args.latent_dim).to(device)
    vae_decoder = GeneDecoder(latent_dim=args.latent_dim, n_genes=n_genes).to(device)
    mlp = SpatialEncoder(input_dim=2, d_model=args.latent_dim).to(device)
    
    logger.info(f"VAE Encoder parameters: {sum(p.numel() for p in vae_encoder.parameters()):,}")
    logger.info(f"VAE Decoder parameters: {sum(p.numel() for p in vae_decoder.parameters()):,}")
    logger.info(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # 3. 联合训练VAE和MLP
    try:
        epoch_losses, epoch_ot_losses, epoch_kl_losses = train_joint(
            vae_encoder,
            vae_decoder,
            mlp,
            data_by_timepoint,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        log_memory_usage("After joint training")
        
        # 绘制loss曲线
        fig_dir = args.fig_dir
        os.makedirs(fig_dir, exist_ok=True)
        # loss曲线文件名包含latent_dim
        loss_curve_path = os.path.join(fig_dir, f"loss_curves_lat{args.latent_dim}.png")
        plot_loss_curves(epoch_losses, epoch_ot_losses, epoch_kl_losses, loss_curve_path)
    except Exception as e:
        logger.error(f"Joint training failed: {e}", exc_info=True)
        raise
    
    # 5. 保存模型
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型文件名包含latent_dim，避免覆盖
    vae_path = os.path.join(save_dir, f"vae_lat{args.latent_dim}.pt")
    mlp_path = os.path.join(save_dir, f"mlp_lat{args.latent_dim}.pt")
    
    # 保存VAE（编码器和解码器）
    vae_state = {
        'encoder': vae_encoder.state_dict(),
        'decoder': vae_decoder.state_dict(),
        'n_genes': n_genes,
        'latent_dim': args.latent_dim
    }
    torch.save(vae_state, vae_path)
    logger.info(f"Saved VAE model to {vae_path}")
    
    # 保存MLP
    mlp_state = {
        'model': mlp.state_dict(),
        'latent_dim': args.latent_dim
    }
    torch.save(mlp_state, mlp_path)
    logger.info(f"Saved MLP model to {mlp_path}")
    
    # 6. 可视化（默认启用，除非指定 --no-visualize）
    if not args.no_visualize:
        try:
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
            log_memory_usage("After visualization")
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            # 可视化失败不影响主流程
    
    logger.info("="*70)
    logger.info("Pretraining completed successfully!")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pretraining interrupted by user")
    except Exception as e:
        logger.error(f"Pretraining failed: {e}", exc_info=True)
        log_memory_usage("Error state")
        raise