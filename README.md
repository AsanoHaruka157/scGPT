scGPT: Modeling single-cell transcriptomics (scRNA-seq) time series with GPT (Generative Pre-trained Transformer) models

### Goals
- Interpolation: infer gene expression matrices between observed time points to mitigate sparse temporal coverage due to high sampling costs.
- Extrapolation: predict future expressions and recover cell trajectory evolution.

Main experimental scripts are under `benchmark/`.

### Codes
*This project was initially copied from the scNODE project, then TrajGPT and TimelyGPT were added.*
This repository contains three time-series modeling approaches: TimelyGPT, TrajGPT, and scNODE. Below is a brief overview and key components.

1) TimelyGPT (continuous, multivariate time series)
- Idea: Retention/RetNet-style recurrent attention with convolutional subsampling/upsampling and Reversible Instance Normalization (RevIN), supporting pretraining and forecasting for continuous multivariate time series.
- Library location: `TimelyGPT_CTS/`
- Internal components:
  - `Retention_layers.RetNetBlock`: recurrent attention (Retention) block
  - `Conv_layers.Conv1dSubampling` and `Conv1dUpsampling`: temporal down/up sampling
  - `RevIN.RevIN`: reversible instance normalization to reduce distribution shift
  - `heads` and `snippets`: task heads (pretrain/forecast/clf/reg) and utilities

2) TrajGPT (irregularly-sampled time series / health trajectories)
- Idea: Selective Recurrent Attention (SRA) for irregularly sampled time series; supports pretraining, forecasting, and classification. This repo also includes a variant for continuous inputs (e.g., single-cell expression vectors).
- Library location: `TrajGPT/`
- Internal components:
  - `SRA_layers.SRA_Block`: stacked SRA modules
  - `heads`: task heads for pretrain/forecast/classification
  - `Embed.TokenEmbeddingFixed`: used for discrete code settings

3) scNODE (generative time-series model based on neural ODEs)
- Idea: evolve in a VAE latent space via an ODE solver, then decode back to gene expression space for interpolation and extrapolation of single-cell time series.
- Library location: `model/`

### Code structure (excerpt)
- `benchmark/`: benchmark experiments (e.g., `1_SingleCell_scNODE.py`, `4_SingleCell_TrajGPT.py`, `5_SingleCell_TimelyGPT.py`).
- `optim/`, `plotting/`, `downstream_analysis/`: training, evaluation, and visualization utilities.
- `data/`: datasets and preprocessing scripts.
