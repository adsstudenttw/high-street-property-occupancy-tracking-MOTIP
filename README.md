# High-Street Occupancy Tracking (MOTIP)

This guide explains how to install and run the project on a SURF Research Cloud Ubuntu 22.04 CUDA VM with Docker as the default path, then lists local non-Docker steps as optional.

## 1. Prerequisites

- Ubuntu 22.04 VM on SURF Research Cloud
- NVIDIA GPU with CUDA 12-capable driver (`nvidia-smi` should work)
- NVIDIA driver + CUDA runtime (`nvidia-smi` should work)
- `git`
- `sudo` access on the VM
- A mounted SURF volume with enough free space for the repo, datasets, outputs, Docker images, and container runtime state

## 2. Prepare the SURF volume

Create and mount your `motip_storage` volume first, then clone this repository onto that mounted filesystem.

Example layout:

```text
<mounted motip_storage>/
  high-street-property-occupancy-tracking-MOTIP/
    datasets/
    outputs/
    pretrains/
    .surf-storage/
```

The default setup assumes you run all `make` commands from the cloned repo on that mounted volume. In that case:
- the repo itself lives on the SURF volume
- `./datasets`, `./outputs`, and `./pretrains` live on the SURF volume
- Docker `data-root`, containerd `root`, Docker temp files, and container cache dirs live under `./.surf-storage/`
- the Optuna SQLite DB lives at `./.surf-storage/optuna/hspot_hota_optuna.db`

## 3. Prepare datasets

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

## 4. Download required pretrained weights

Create the `./pretrains/` directory in the repo and download the files required for your intended workflow.

Required for training and hyperparameter tuning:
- `r50_deformable_detr_coco_dancetrack.pth`
  Source: [DanceTrack DETR pretrain](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth)

Also required for the zero-shot baseline evaluation flow:
- `r50_deformable_detr_motip_dancetrack.pth`
  Source: [DanceTrack MOTIP checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth)

Example:

```bash
mkdir -p ./pretrains
curl -L -o ./pretrains/r50_deformable_detr_coco_dancetrack.pth \
  https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth
curl -L -o ./pretrains/r50_deformable_detr_motip_dancetrack.pth \
  https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth
```

The training config in this fork expects `./pretrains/r50_deformable_detr_coco_dancetrack.pth` by default.

## 5. MLflow logging (standard)

MLflow logging is the standard workflow for this project. The HSPOT config keeps logging enabled via `USE_WANDB: True` and points `MLFLOW_TRACKING_URI` at the configured SURF endpoint in [configs/high_street_property_occupancy_tracking.yaml](./configs/high_street_property_occupancy_tracking.yaml).

Before running `make train`, confirm that the configured `MLFLOW_TRACKING_URI` is reachable from inside the Docker container. If the URI is unreachable, training can fail when the logger starts.

Disabling MLflow is possible, but it is not the default workflow documented here.

## 6. Docker setup on SURF RC (default)

```bash
make prepare-storage
make bootstrap-gpu
newgrp docker
TORCH_CUDA_ARCH_LIST="8.6" make build-gpu
```

`make bootstrap-gpu` installs Docker from the official Docker apt repo, installs the NVIDIA Container Toolkit, and moves Docker/containerd storage under `./.surf-storage/` on the mounted SURF volume.

For the SURF VM described here, `TORCH_CUDA_ARCH_LIST="8.6"` is recommended because the environment uses an NVIDIA A10 GPU.

If you want a different storage path on the same mounted volume, override it explicitly:

```bash
make bootstrap-gpu STORAGE_ROOT=/path/on/mounted/volume/motip_storage_runtime
```

## 7. Establish the baseline on HSPOT val (Docker)

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

This is the baseline establishment stage. It gives you the zero-shot baseline on `HSPOT val` before finetuning or hyperparameter tuning.

## 8. Run finetuning (Docker)

```bash
make train
```

`make train` runs the finetuning stage for this fork. Internally it calls `train.py` with `--run-stage finetuning`.

Outputs are written under `./outputs/`, which stays on the SURF volume because the repo is cloned there.

## 9. Run hyperparameter tuning (Docker, optimize validation HOTA)

```bash
make tune
```

Key outputs:
- Optuna DB: `./.surf-storage/optuna/hspot_hota_optuna.db`
- Best-trial summary: `./outputs/optuna_hspot/best_trial.json`
- Best tuned checkpoint path (in JSON): `best_checkpoint_path`

Default HSPOT tuning budget:
- `N_TRIALS=40`
- `TUNE_EPOCHS=6`
- `TUNE_TIMEOUT=360000` (100 GPU-hours wall-clock cap)
- `TUNE_PRUNER_STARTUP_TRIALS=8`
- `TUNE_PRUNER_WARMUP_STEPS=3`

This keeps each trial short enough to search broadly on a single GPU.

## 10. Evaluate best tuned checkpoint on test split (Docker)

Evaluate the best checkpoint produced during Optuna tuning:

```bash
make eval-final
```

This uses `./outputs/optuna_hspot/best_trial.json` and the checkpoint metadata written by `make tune`.

If needed, you can still evaluate a specific checkpoint manually:

```bash
make eval BEST_CKPT=./outputs/<final_checkpoint_path>.pth
```

## 11. Storage Notes

The Docker-oriented workflow now keeps the large project-managed files off the VM root disk as long as the repo is cloned onto the mounted SURF volume:
- repo checkout, datasets, checkpoints, and outputs stay in the repo tree on the volume
- Docker image layers and container writable layers live under `./.surf-storage/docker`
- containerd content and snapshot storage live under `./.surf-storage/containerd`
- container temp files and Python/tool caches live under `./.surf-storage/tmp` and `./.surf-storage/cache`

One limitation remains: Ubuntu system packages installed by `apt` still live on the root filesystem. That part cannot be cleanly relocated by this project setup.

## 12. Optional: local (non-Docker) setup

This section is optional. Use it only if you explicitly want to run outside Docker.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --python 3.12
cd models/ops
sh make.sh
cd ../..
```

Optional validation of the CUDA extension:

```bash
cd models/ops
uv run --no-sync python test.py
cd ../..
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
  --storage sqlite:///./.surf-storage/optuna/hspot_hota_optuna.db \
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
