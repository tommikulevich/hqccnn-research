# ⚛️ Hybrid Quantum-Classical CNN for Image Classification

## Overview

This project implements hybrid quantum–classical CNN models and infrastructure to train and evaluate them on various image datasets. The core idea is to combine classical layers with a quantum module which is used as a classifier or feature extractor. The codebase contains two model implementations:

- `hqnn-parallel`: classical convolutional layers with a hybrid classifier (fully connected layers + parallel quantum layers).
- `hqnn-quanv`: an alternative quantum-convolution style architecture.

## Key Components

- Config-driven experiments using YAML files (`configs/default.yaml`).
- Dataset utilities and registries to plug datasets, models, loss functions, optimizers and schedulers via a registry pattern.
- Training loop with checkpointing and optional hyperparameter search hooks.
- Inference tools that save per-layer activations and produce activation plots for analysis.

## Repo Structure

- `configs/` – YAML configuration templates
- `data/` – data directory
- `src/` – source code
  - `config/` – config schema, loader and registries
  - `dataprep/` – dataset loaders and transforms
  - `models/` – model implementations (`hqnn_parallel`, `hqnn_quanv`)
  - `runners/` – `train`, `inference`, and `test` entrypoints
  - `search/` – hyperparameter search utilities (grid search available)
  - `trainer/` – training loop implementation
  - `utils/` – common utilities, logging, and seeding
- `tools/` - plotting tools

## Setup

```cmd
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Experiments are configured via YAML files. The default configuration is `configs/default.yaml`. Major sections and common options:

- `seed` — global seed for reproducibility.
- `logging` — directories for logs, checkpoints and MLflow dashboard; checkpoint frequency.
- `data` — dataset name and parameters:
  - `data_dir` — path to dataset root (default `data/`).
  - `num_classes` — number of target classes.
  - `num_workers`, `batch_size` — DataLoader parameters.
  - `dry_run_batches` — number of batches used in dry runs.
  - `input_shape` — [C, H, W] expected input channels and size.
  - `split` — train/val/test split fractions.
  - `transforms` — lists of torchvision transform entries. Each transform is specified as a mapping with `name` and optional `params`.
- `model` — the model name (registered keys such as `hqnn-parallel`) and parameters with model-specific hyperparameters (both classical and quantum).
- `training` — training runtime options: `epochs`, `device` (e.g. `cpu` or `cuda`).
- `loss` — loss function name and parameters (mapped via registry).
- `optimizer` — optimizer name and constructor parameters (e.g. `lr`, `weight_decay`).
- `scheduler` — scheduler name and params.
- `search` — hyperparameter search method and params.

The code loads the YAML into a `Config` object using `src/config/loader.py` and validates keys via `src/config/schema.py`.

## Registries

This project uses a registry pattern (`src/config/registry.py`) to keep components pluggable. Below are the components currently registered and the keys you can use in `configs/*.yaml`.

- Datasets (use values from `config.enums.DatasetName`):
  - MNIST
  - FASHION_MNIST
  - CIFAR10
  - CIFAR100
  - EUROSAT
  - SVHN
  - CUSTOM (maps to `torchvision.datasets.ImageFolder`)

- Models (use values from `config.enums.ModelName`):
  - HQNN_PARALLEL
  - HQNN_PARALLEL_CLASSIC_CNN
  - HQNN_QUANV
  - HQNN_QUANV_CLASSIC_CNN

- Loss functions (use values from `config.enums.LossName`):
  - CROSS_ENTROPY

- Optimizers (use values from `config.enums.OptimizerName`):
  - ADAM
  - SGD

- Schedulers (use values from `config.enums.SchedulerName`):
  - NONE
  - STEPLR
  - EXPOTENTIALLR

- Search methods (use values from `config.enums.SearchMethod`):
  - NONE
  - GRID

To extend registries, add your component to `src/config/registry.py` and expose a corresponding enum value in `src/config/enums.py`. The trainer and runner code construct components using the registry entries, so new components become immediately available to YAML configs.

## Example Usage

The main entrypoint is `src/main.py`. The script accepts a `--config` path and provides submodes for training, testing and inference.

Basic training (default config):

```cmd
python src/main.py --config configs/default.yaml
```

Resume training from a checkpoint:

```cmd
python src/main.py --config configs/default.yaml --resume outputs/checkpoints/best.pth
```

Dry-run training (sanity check, small number of batches):

```cmd
python src/main.py --config configs/default.yaml --dry-run
```

Run inference on an image file or directory:

```cmd
python src/main.py --config configs/default.yaml --infer --checkpoint outputs/checkpoints/best.pth --input data/custom_images --output outputs/inference
```

Run a dedicated test checkpoint evaluation:

```cmd
python src/main.py --config configs/default.yaml --test outputs/checkpoints/best.pth --output outputs/test
```

## Results

If you are interested in the results of experiments - please do not hesitate to contact me by e-mail (tommikulevich@gmail.com).

## Citation

> Mikulevich, Tomash. Exploring the Potential of Hybrid Quantum-Classical Convolutional Neural Networks in Solving Image Classification Problems. Master’s thesis (unpublished), Gdańsk University of Technology, 2025.
