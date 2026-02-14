<<<<<<< HEAD
'''
Description:
    Run our scNODE on the single-cell dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root) # Change the current working directory to the project root

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
# 设置字体，避免警告
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import torch
import numpy as np
import time

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec
from plotting.__init__ import *
from plotting.visualization import plotPredAllTime, plotPredTestTime, computeDrift, plotStream, plotStreamByCellType
from plotting.PlottingUtils import umapWithPCA, computeLatentEmbedding
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
from optim.evaluation import globalEvaluation

# ======================================================
# Load data and pre-processing
print("=" * 70)
# Specify the dataset: zebrafish, drosophila, wot
# Representing ZB, DR, SC, repectively
data_name= "zebrafish"
print("[ {} ]".format(data_name).center(60))
# Specify the type of prediction tasks: three_interpolation, two_forecasting, three_forecasting, remove_recovery
# The tasks feasible for each dataset:
#   zebrafish (ZB): three_interpolation, two_forecasting, remove_recovery
#   drosophila (DR): three_interpolation, three_forecasting, remove_recovery
#   wot (SC): three_interpolation, three_forecasting, remove_recovery
# They denote easy, medium, and hard tasks respectively.
split_type = "three_interpolation"
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps, all_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

all_tps = list(all_tps)  # Convert to list
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
tps = torch.FloatTensor(all_tps)
train_tps = torch.FloatTensor(train_tps)
test_tps = torch.FloatTensor(test_tps)
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# ======================================================
# Model training - Time-limited competition (300 seconds)
TIME_LIMIT = 300  # 训练时间限制（秒）
pretrain_lr = 1e-3
latent_coeff = 1.0 # regularization coefficient: beta
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000

latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type) # use tuned hyperparameters
latent_ode_model = constructscNODEModel(
    n_genes, latent_dim=latent_dim,
    enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
    latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
    ode_method="euler"
)

# Start timing
print("=" * 70)
print(f"Starting time-limited training ({TIME_LIMIT} seconds)...")
train_start_time = time.time()

# Import necessary modules for custom training loop
import geomloss
import itertools

# Pre-training the VAE component (quick pre-train)
latent_encoder = latent_ode_model.latent_encoder
obs_decoder = latent_ode_model.obs_decoder
all_train_data = torch.cat(train_data, dim=0)
pretrain_iters = 100  # Reduced for time efficiency
if pretrain_iters > 0 and (time.time() - train_start_time) < TIME_LIMIT:
    dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
    dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99))
    latent_encoder.train()
    obs_decoder.train()
    for i in range(pretrain_iters):
        if (time.time() - train_start_time) >= TIME_LIMIT:
            break
        rand_idx = np.random.choice(all_train_data.shape[0], size=batch_size, replace=False)
        batch_data = all_train_data[rand_idx, :]
        dim_reduction_optimizer.zero_grad()
        latent_mu, latent_std = latent_encoder(batch_data)
        latent_sample = latent_mu + torch.randn_like(latent_std) * latent_std
        recon_obs = obs_decoder(latent_sample)
        recon_loss = torch.mean((recon_obs - batch_data) ** 2)
        recon_loss.backward()
        dim_reduction_optimizer.step()

# Dynamic training with time limit
optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
loss_list = []
epoch = 0

# Convert train_tps to list once for model forward (to avoid tensor/numpy issues)
train_tps_list = train_tps.detach().numpy().tolist() if isinstance(train_tps, torch.Tensor) else train_tps

while (time.time() - train_start_time) < TIME_LIMIT:
    epoch += 1
    for iter_idx in range(100):  # iterations per epoch
        if (time.time() - train_start_time) >= TIME_LIMIT:
            break
        
        rand_t_idx = np.random.choice(len(train_tps))
        rand_idx = np.random.choice(train_data[rand_t_idx].shape[0], size=batch_size, replace=False)
        batch_data = train_data[rand_t_idx][rand_idx, :]
        
        optimizer.zero_grad()
        latent_ode_model.train()
        # Model expects: forward(data_list, tps, batch_size=None)
        # data should be a list with data at first timepoint
        recon_obs, first_latent_dist, first_tp_data, latent_seq = latent_ode_model([batch_data], torch.FloatTensor(train_tps_list))
        
        # Compute loss
        ot_loss = 0
        for t_idx, t in enumerate(train_tps):
            pred_x = recon_obs[:, t_idx, :]
            true_x = train_data[t_idx]
            subsample_size = min(200, true_x.shape[0])
            subsample_idx = np.random.choice(true_x.shape[0], subsample_size, replace=False)
            ot_loss += ot_solver(pred_x, true_x[subsample_idx])
        
        latent_drift_loss = torch.mean((latent_seq[:, 1:, :] - latent_seq[:, :-1, :]) ** 2)
        loss = ot_loss + latent_coeff * latent_drift_loss
        loss.backward()
        optimizer.step()
        loss_list.append((loss.item(), ot_loss.item(), latent_drift_loss.item()))
    
    if epoch % 5 == 0:
        elapsed = time.time() - train_start_time
        remaining = TIME_LIMIT - elapsed
        print(f"Epoch {epoch}, Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s, Loss: {loss.item():.4f}")

# Get final predictions
latent_ode_model.eval()
with torch.no_grad():
    first_obs = train_data[0]
    recon_obs, first_latent_dist, first_tp_data, latent_seq = latent_ode_model([first_obs], torch.FloatTensor(train_tps_list))

# End timing
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print("=" * 70)
print(f"Training completed! Total epochs: {epoch}")
print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
print("=" * 70)

all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)  # (# cells, # tps, # genes)

# ======================================================
# Visualization - loss curve
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
plt.show()

# Visualization - 2D UMAP embeddings
print("Compare true and reconstructed data...")
true_data = [each.detach().numpy() for each in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])
reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy(), 
                 save_path="benchmark/scNODE_Results.png")

# Compute evaluation metrics
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
    print(pred_global_metric)

# # ======================================================
# # Save results
# save_dir = "../res/single_cell/experimental/{}".format(data_name)
# res_filename = "{}/{}-{}-scNODE-res.npy".format(save_dir, data_name, split_type)
# state_filename = "{}/{}-{}-scNODE-state_dict.pt".format(save_dir, data_name, split_type)
# print("Saving to {}".format(res_filename))
# np.save(
#     res_filename,
#     {"true": [each.detach().numpy() for each in traj_data],
#      "pred": [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])],
#      "first_latent_dist": first_latent_dist,
#      "latent_seq": latent_seq,
#      "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
#      "loss": loss_list,
#      },
#     allow_pickle=True
# )
# torch.save(latent_ode_model.state_dict(), state_filename)

=======
'''
Description:
    Run our scNODE on the single-cell dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root) # Change the current working directory to the project root

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
# 设置字体，避免警告
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import torch
import numpy as np
import time

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec
from plotting.__init__ import *
from plotting.visualization import plotPredAllTime, plotPredTestTime, computeDrift, plotStream, plotStreamByCellType
from plotting.PlottingUtils import umapWithPCA, computeLatentEmbedding
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
from optim.evaluation import globalEvaluation

# ======================================================
# Load data and pre-processing
print("=" * 70)
# Specify the dataset: zebrafish, drosophila, wot
# Representing ZB, DR, SC, repectively
data_name= "zebrafish"
print("[ {} ]".format(data_name).center(60))
# Specify the type of prediction tasks: three_interpolation, two_forecasting, three_forecasting, remove_recovery
# The tasks feasible for each dataset:
#   zebrafish (ZB): three_interpolation, two_forecasting, remove_recovery
#   drosophila (DR): three_interpolation, three_forecasting, remove_recovery
#   wot (SC): three_interpolation, three_forecasting, remove_recovery
# They denote easy, medium, and hard tasks respectively.
split_type = "three_interpolation"
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps, all_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

all_tps = list(all_tps)  # Convert to list
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
tps = torch.FloatTensor(all_tps)
train_tps = torch.FloatTensor(train_tps)
test_tps = torch.FloatTensor(test_tps)
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# ======================================================
# Model training - Time-limited competition (300 seconds)
TIME_LIMIT = 300  # 训练时间限制（秒）
pretrain_lr = 1e-3
latent_coeff = 1.0 # regularization coefficient: beta
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000

latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type) # use tuned hyperparameters
latent_ode_model = constructscNODEModel(
    n_genes, latent_dim=latent_dim,
    enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
    latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
    ode_method="euler"
)

# Start timing
print("=" * 70)
print(f"Starting time-limited training ({TIME_LIMIT} seconds)...")
train_start_time = time.time()

# Import necessary modules for custom training loop
import geomloss
import itertools

# Pre-training the VAE component (quick pre-train)
latent_encoder = latent_ode_model.latent_encoder
obs_decoder = latent_ode_model.obs_decoder
all_train_data = torch.cat(train_data, dim=0)
pretrain_iters = 100  # Reduced for time efficiency
if pretrain_iters > 0 and (time.time() - train_start_time) < TIME_LIMIT:
    dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
    dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99))
    latent_encoder.train()
    obs_decoder.train()
    for i in range(pretrain_iters):
        if (time.time() - train_start_time) >= TIME_LIMIT:
            break
        rand_idx = np.random.choice(all_train_data.shape[0], size=batch_size, replace=False)
        batch_data = all_train_data[rand_idx, :]
        dim_reduction_optimizer.zero_grad()
        latent_mu, latent_std = latent_encoder(batch_data)
        latent_sample = latent_mu + torch.randn_like(latent_std) * latent_std
        recon_obs = obs_decoder(latent_sample)
        recon_loss = torch.mean((recon_obs - batch_data) ** 2)
        recon_loss.backward()
        dim_reduction_optimizer.step()

# Dynamic training with time limit
optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
loss_list = []
epoch = 0

# Convert train_tps to list once for model forward (to avoid tensor/numpy issues)
train_tps_list = train_tps.detach().numpy().tolist() if isinstance(train_tps, torch.Tensor) else train_tps

while (time.time() - train_start_time) < TIME_LIMIT:
    epoch += 1
    for iter_idx in range(100):  # iterations per epoch
        if (time.time() - train_start_time) >= TIME_LIMIT:
            break
        
        rand_t_idx = np.random.choice(len(train_tps))
        rand_idx = np.random.choice(train_data[rand_t_idx].shape[0], size=batch_size, replace=False)
        batch_data = train_data[rand_t_idx][rand_idx, :]
        
        optimizer.zero_grad()
        latent_ode_model.train()
        # Model expects: forward(data_list, tps, batch_size=None)
        # data should be a list with data at first timepoint
        recon_obs, first_latent_dist, first_tp_data, latent_seq = latent_ode_model([batch_data], torch.FloatTensor(train_tps_list))
        
        # Compute loss
        ot_loss = 0
        for t_idx, t in enumerate(train_tps):
            pred_x = recon_obs[:, t_idx, :]
            true_x = train_data[t_idx]
            subsample_size = min(200, true_x.shape[0])
            subsample_idx = np.random.choice(true_x.shape[0], subsample_size, replace=False)
            ot_loss += ot_solver(pred_x, true_x[subsample_idx])
        
        latent_drift_loss = torch.mean((latent_seq[:, 1:, :] - latent_seq[:, :-1, :]) ** 2)
        loss = ot_loss + latent_coeff * latent_drift_loss
        loss.backward()
        optimizer.step()
        loss_list.append((loss.item(), ot_loss.item(), latent_drift_loss.item()))
    
    if epoch % 5 == 0:
        elapsed = time.time() - train_start_time
        remaining = TIME_LIMIT - elapsed
        print(f"Epoch {epoch}, Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s, Loss: {loss.item():.4f}")

# Get final predictions
latent_ode_model.eval()
with torch.no_grad():
    first_obs = train_data[0]
    recon_obs, first_latent_dist, first_tp_data, latent_seq = latent_ode_model([first_obs], torch.FloatTensor(train_tps_list))

# End timing
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print("=" * 70)
print(f"Training completed! Total epochs: {epoch}")
print(f"Training time: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
print("=" * 70)

all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)  # (# cells, # tps, # genes)

# ======================================================
# Visualization - loss curve
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
plt.show()

# Visualization - 2D UMAP embeddings
print("Compare true and reconstructed data...")
true_data = [each.detach().numpy() for each in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])
reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy(), 
                 save_path="benchmark/scNODE_Results.png")

# Compute evaluation metrics
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
    print(pred_global_metric)

# # ======================================================
# # Save results
# save_dir = "../res/single_cell/experimental/{}".format(data_name)
# res_filename = "{}/{}-{}-scNODE-res.npy".format(save_dir, data_name, split_type)
# state_filename = "{}/{}-{}-scNODE-state_dict.pt".format(save_dir, data_name, split_type)
# print("Saving to {}".format(res_filename))
# np.save(
#     res_filename,
#     {"true": [each.detach().numpy() for each in traj_data],
#      "pred": [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])],
#      "first_latent_dist": first_latent_dist,
#      "latent_seq": latent_seq,
#      "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
#      "loss": loss_list,
#      },
#     allow_pickle=True
# )
# torch.save(latent_ode_model.state_dict(), state_filename)

>>>>>>> 79adf705c1bc4c71af40b3b22de696eaf1c9a4f4
