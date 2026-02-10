# How to run

1. Baseline training (1 GPU):
```bash
accelerate launch --num_processes=1 train.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --exp-name hspot_baseline
```

2. Optuna tuning (validation HOTA + pruning):
```bash
python3 optuna_tune.py \
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

3. Final test evaluation (best trial checkpoint):
```bash
accelerate launch --num_processes=1 submit_and_evaluate.py \
  --config-path ./configs/high_street_property_occupancy_tracking.yaml \
  --data-root ./datasets/ \
  --inference-mode evaluate \
  --inference-dataset HSPOT \
  --inference-split test \
  --inference-model <best_checkpoint_path> \
  --outputs-dir ./outputs/hspot_final_test
```
