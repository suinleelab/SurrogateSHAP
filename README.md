# SurrogateSHAP

Code repository for **SurrogateSHAP: Training-Free Contributor Attribution for Text-to-Image (T2I) Models**.

## Overview

This repository provides a framework for data contributor attribution in generative models across three distinct settings: Class-Conditional Diffusion (DDPM on CIFAR-20), Latent Diffusion (Stable Diffusion on ArtBench), and Rectified Flow Transformers (FLUX.1-dev on Fashion-Product).

Given an observed model behavior (e.g., FID, IS, aesthetic score, mean lpips, and diversity), SurrogateSHAP helps identify which contributors most influence that behavior—efficiently.


## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
cfg_att/
├── configs/
│   ├── ddim_config.py            # DDIM/DDPM configs
│   └── sd_config.py              # Stable Diffusion LoRA configs
|
├── src/
│   ├── models/                   # Model architectures
│   ├── attributions/
|   |   └──methods/               # Gradient computes for IF approximations, e.g. D-TRAK, TRAK, DAS, IF
|   |      └──shapley             # GBT fitting and SurrogateSHAP compute
│   ├── datasets.py               # Dataset loaders
│   └── utils.py                  # Utilities
|
├── cifar20/
│   ├── main.py                   # Train diffusion model
│   ├── compute_model_behavior.py # Model behavior evaluation (e.g., FID & IS)
│   ├── prune.py                  # Pruning steps for sFT
│   ├── unlearn.py                # Fine-tuning proxy (e.g., sFT)
│   └── compute_lds.py            # Compute LDS for methods for all methods, e.g. SurrogateSHAP and other baselines
|
├── text_to_image_sd/             # Stable Diffusion workflows
|   └──baselines/                 # code for baseline computation, e.g. clip similarity & pixel similarity
│   ├── train_text_to_image_lora.py # Train SD with LoRA
│   ├── compute_model_behavior.py # Model behavior evaluation (Aesthetic Score)
│   ├── prune_lora.py             # Pruning steps for sFT
│   ├── baseline_lds.py           # Compute LDS for baseline methods, e.g. LOO, D-TRAK, TRAK, DAS
│   └── compute_lds.py            # Compute LDS for methods based on Shapley value, e.g. SurrogateSHAP
|
├── text_to_image_flux/           # FLUX.1dev
|   ├──preprocess_product_data/   # Preprocessing steps for fashion-product dataset
|   └──baselines/                 # code for baseline computation, e.g. clip similarity & pixel similarity
│   ├── train_lora_flux.py.py     # Train SD with LoRA
│   ├── compute_fashion_metrics.py # Model behavior evaluation (LPIPS, Diversity)
│   ├── prune_lora.py             # Pruning steps for sFT
│   ├── baseline_lds.py           # Compute LDS for baseline methods, e.g. LOO, D-TRAK, TRAK, DAS
│   └── compute_lds.py            # Compute LDS for methods based on Shapley value, e.g. SurrogateSHAP
|
└── shapley_exp/                  # Synthetic experiments for SurrogateSHAP
```


## Usage: CIFAR-20

### 1. Train a Diffusion Model

```bash
python cifar20/main.py \
  --dataset cifar20 \
  --method retrain \
  --opt_seed 42 \
  --outdir ./outputs
```

### 2. Compute Model Behavior

Evaluate subsets to compute model behaviors (FID & Inception Score):

```bash
python cifar20/compute_model_behavior.py \
  --dataset cifar20 \
  --ckpt_path ./outputs/full/models \
  --removal_dist shapley \
  --removal_idx 0
```

### 3. Compute Attribution Scores with Linear Datamodel Score (LDS)

**SurrogateSHAP** (training-free):
```bash
python cifar20/compute_lds.py \
  --fit_db ./results/cifar20/subset_results \
  --dataset cifar20 \
  --method treeshap \
  --model_behavior fid (inception_score) \
  --datamodel_alpha 0.5 \
  --v0 250.0 \
  --v1 20.2
```

Notes:
- The datamodel must be trained using the same pipeline as **retraining** and **model behavior evaluation**.
- `fit_db` should point to the directory containing the evaluated subset results used to fit the surrogate datamodel.
- `model_behavior` specifies which metric to attribute (e.g., `fid`, `inception_score`).

## Usage: ArtBench (Post-Impressionism)

ArtBench (Post-Impressionism) and Fashion-Product follow similar LoRA fine-tune pipeline. Here we include ArtBench as the example.

### 1. Train Text-to-Image LoRA

```bash
cd text_to_image_sd
python train_text_to_image_lora.py \
  --dataset_name="artbench_post_impressionism" \
  --resolution=512 \
  --train_batch_size=4 \
  --learning_rate=1e-4
```

### 2. Compute Model Behavior

Evaluate subsets to compute aesthetic scores:

```bash
python text_to_image_sd/compute_model_behavior.py \
  --lora_dir ./outputs/artbench/lora \
  --removal_dist shapley \
  --removal_seed 0
```

### 3. Compute Attribution Scores with Linear Datamodel Score (LDS)

**SurrogateSHAP** (training-free):
```bash
python text_to_image_sd/compute_lds.py \
  --fit_db ./results/artbench/subset_results \
  --null_db ./results/artbench/null_model \
  --full_db ./results/artbench/full_model \
  --dataset artbench \
  --method treeshap \
  --model_behavior_key aesthetic_score_avg \
  --datamodel_alpha 0.5
```

## Replicating Synthetic Experiment

To validate the approximation fidelity of SurrogateSHAP against ground-truth Shapley values:
```bash
python shapley_exp/surrogate.py --func interaction --sample_sizes 32 64 128 256 512 600 700 800 900
```
