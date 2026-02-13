# High-Street Occupancy Tracking (MOTIP)

This guide explains how to install and run the project with Docker (default path), then lists local non-Docker steps as optional.

## 1. Prerequisites

- Linux with NVIDIA GPU (recommended for training/inference speed)
- NVIDIA driver + CUDA runtime (`nvidia-smi` should work)
- `git`
- `sudo` access on the VM

## 2. Prepare datasets

Expected dataset root:

```text
./datasets/
  hspot/
    train/
    val/
    test/
    train_seqmap.txt
    val_seqmap.txt
    test_seqmap.txt
```

The default config uses:
- `--config-path ./configs/high_street_property_occupancy_tracking.yaml`
- `--data-root ./datasets/`

## 3. Docker setup (default)

```bash
make bootstrap-gpu
newgrp docker
make build-gpu
```

`make bootstrap-gpu` installs Docker (official repo) and NVIDIA Container Toolkit.

## 4. Two-VM workflow (recommended for limited GPU hours)

### 4.1 CPU VM smoke-test (no GPU usage)

```bash
make bootstrap-cpu
newgrp docker
make build-cpu
make smoke-cpu
```

This validates Docker/project dependencies and command startup.  
Do not expect full training/inference on CPU VM.

### 4.2 GPU VM actual run

```bash
make bootstrap-gpu
newgrp docker
make build-gpu
make train
```

Outputs are written under `./outputs/`.

## 5. Run hyperparameter tuning (Docker, optimize validation HOTA)

```bash
make tune
```

Key outputs:
- Optuna DB: `hspot_hota_optuna.db`
- Best-trial summary: `./outputs/optuna_hspot/best_trial.json`

## 6. Evaluate best checkpoint on test split (Docker)

Replace `<best_checkpoint_path>` with your best trial checkpoint:

```bash
make eval BEST_CKPT=./outputs/<best_checkpoint_path>.pth
```

## 7. Optional: local (non-Docker) setup

This section is optional. Use it only if you explicitly want to run outside Docker.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Local baseline training:

```bash
uv run accelerate launch --num_processes=1 train.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --exp-name hspot_baseline
```

Local Optuna tuning:

```bash
uv run python optuna_tune.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --inference-dataset HSPOT \
  --inference-split val \
  --study-name hspot_hota_optuna \
  --storage sqlite:///hspot_hota_optuna.db \
  --n-trials 30 \
  --epochs 10 \
  --output-root ./outputs/optuna_hspot
```

Local final evaluation:

```bash
uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --inference-mode evaluate \
  --inference-dataset HSPOT \
  --inference-split test \
  --inference-model <best_checkpoint_path> \
  --outputs-dir ./outputs/hspot_final_test
```
