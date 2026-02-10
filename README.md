# High-Street Occupancy Tracking (MOTIP)

This guide explains how to install and run the project end-to-end:
1. Install dependencies (`uv`).
2. Prepare datasets.
3. Train a baseline model.
4. Tune hyperparameters with Optuna (Bayesian optimization + pruning).
5. Evaluate the best model on the test split.

## 1. Prerequisites

- Linux with NVIDIA GPU (recommended for training/inference speed)
- NVIDIA driver + CUDA runtime compatible with PyTorch 2.4.0 / CUDA 12.1
- `git`
- `uv` (https://docs.astral.sh/uv/)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and install

From the project root:

```bash
uv sync
```

This creates `.venv` and installs dependencies from `pyproject.toml`.

## 3. Prepare datasets

Expected root:

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

Your config uses:
- `--config-path ./configs/high_street_property_occupancy_tracking.yaml`
- `--data-root ./datasets/`

## 4. Train baseline model (1 GPU)

```bash
uv run accelerate launch --num_processes=1 train.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --exp-name hspot_baseline
```

Outputs are written under `./outputs/`.

## 5. Hyperparameter tuning with Optuna (optimize validation HOTA)

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

Key outputs:
- Optuna DB: `hspot_hota_optuna.db`
- Best-trial summary: `./outputs/optuna_hspot/best_trial.json`

## 6. Evaluate best checkpoint on test split

Replace `<best_checkpoint_path>` with the best trial checkpoint path.

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

## 7. Optional: run with Docker

Build image:

```bash
docker compose build
```

Run shell in container:

```bash
docker compose run --rm motip bash
```

Run baseline training in container:

```bash
docker compose run --rm motip uv run accelerate launch --num_processes=1 train.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --exp-name hspot_baseline
```
