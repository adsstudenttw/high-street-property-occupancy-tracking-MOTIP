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

## 5. Evaluate the pretrained MOTIP baseline on HSPOT val (Docker)

Download the pretrained MOTIP checkpoint first. The DanceTrack MOTIP checkpoint is referenced in [docs/MODEL_ZOO.md](./docs/MODEL_ZOO.md).

Default checkpoint path used by the Makefile:

```text
./pretrains/r50_deformable_detr_motip_dancetrack.pth
```

Run the baseline evaluation on the validation split:

```bash
make baseline-val
```

Or override the checkpoint path explicitly:

```bash
make baseline-val BASELINE_CKPT=./pretrains/<pretrained_motip_checkpoint>.pth
```

This gives you the zero-shot baseline on `HSPOT val` before finetuning or hyperparameter tuning.

## 6. Run hyperparameter tuning (Docker, optimize validation HOTA)

```bash
make tune
```

Key outputs:
- Optuna DB: `hspot_hota_optuna.db`
- Best-trial summary: `./outputs/optuna_hspot/best_trial.json`
- Best tuned checkpoint path (in JSON): `best_checkpoint_path`

Default HSPOT tuning budget:
- `N_TRIALS=40`
- `TUNE_EPOCHS=6`
- `TUNE_TIMEOUT=360000` (100 GPU-hours wall-clock cap)
- `TUNE_PRUNER_STARTUP_TRIALS=8`
- `TUNE_PRUNER_WARMUP_STEPS=3`

This keeps each trial short enough to search broadly on a single GPU.

## 7. Evaluate best tuned checkpoint on test split (Docker)

Evaluate the best checkpoint produced during Optuna tuning:

```bash
make eval-final
```

This uses `./outputs/optuna_hspot/best_trial.json` and the checkpoint metadata written by `make tune`.

If needed, you can still evaluate a specific checkpoint manually:

```bash
make eval BEST_CKPT=./outputs/<final_checkpoint_path>.pth
```

## 8. Optional: local (non-Docker) setup

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

Local pretrained MOTIP baseline evaluation on `HSPOT val`:

```bash
uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --inference-mode evaluate \
  --inference-dataset HSPOT \
  --inference-split val \
  --inference-model ./pretrains/r50_deformable_detr_motip_dancetrack.pth \
  --outputs-dir ./outputs/hspot_pretrained_val
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
  --n-trials 40 \
  --timeout 360000 \
  --pruner-startup-trials 8 \
  --pruner-warmup-steps 3 \
  --epochs 6 \
  --output-root ./outputs/optuna_hspot
```

Local evaluation using the best checkpoint from tuning:

```bash
uv run python eval_best_from_tuning.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --best-trial-json ./outputs/optuna_hspot/best_trial.json \
  --output-root ./outputs/optuna_hspot \
  --inference-dataset HSPOT \
  --inference-split test \
  --outputs-dir ./outputs/hspot_final_test
```
