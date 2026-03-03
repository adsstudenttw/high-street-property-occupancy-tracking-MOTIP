SHELL := /bin/bash

CONFIG ?= ./configs/high_street_property_occupancy_tracking.yaml
DATA_ROOT ?= ./datasets/
EXP_NAME ?= hspot_train
BASELINE_CKPT ?= ./pretrains/r50_deformable_detr_motip_dancetrack.pth
STUDY_NAME ?= hspot_hota_optuna
STORAGE ?= sqlite:///hspot_hota_optuna.db
N_TRIALS ?= 40
TUNE_EPOCHS ?= 6
OUTPUT_ROOT ?= ./outputs/optuna_hspot
BEST_TRIAL_JSON ?= ./outputs/optuna_hspot/best_trial.json
FINAL_OUTPUT_DIR ?= ./outputs/hspot_final_train
FINAL_EXP_NAME ?= hspot_final_train
FINAL_SUMMARY_JSON ?= ./outputs/hspot_final_train/final_train_summary.json
FINAL_EPOCHS ?= 20
BEST_CKPT ?= ./outputs/REPLACE_WITH_BEST_CHECKPOINT.pth

.PHONY: help bootstrap bootstrap-cpu bootstrap-gpu build build-cpu build-gpu shell smoke-cpu train baseline-val tune train-final eval-final eval

help:
	@echo "Available targets:"
	@echo "  make bootstrap      - Auto bootstrap (detects GPU, installs Docker + optional NVIDIA toolkit)"
	@echo "  make bootstrap-cpu  - CPU VM bootstrap (Docker only)"
	@echo "  make bootstrap-gpu  - GPU VM bootstrap (Docker + NVIDIA toolkit)"
	@echo "  make build          - Build Docker image (CUDA ops enabled)"
	@echo "  make build-cpu      - Build CPU-safe image (skip CUDA ops compile)"
	@echo "  make build-gpu      - Build GPU image (compile CUDA ops)"
	@echo "  make shell          - Open shell in project container"
	@echo "  make smoke-cpu      - Run CPU smoke test command in container"
	@echo "  make train          - Run training in container"
	@echo "  make baseline-val   - Evaluate pretrained MOTIP checkpoint on HSPOT val in container"
	@echo "  make tune           - Run Optuna tuning in container"
	@echo "  make train-final    - Train final HSPOT model from best Optuna hyperparameters"
	@echo "  make eval-final     - Evaluate the best final-training checkpoint on HSPOT test"
	@echo "  make eval           - Evaluate BEST_CKPT on test split in container"

bootstrap:
	bash ./scripts/bootstrap_vm.sh auto

bootstrap-cpu:
	bash ./scripts/bootstrap_vm.sh cpu

bootstrap-gpu:
	bash ./scripts/bootstrap_vm.sh gpu

build:
	BUILD_CUDA_OPS=1 docker compose build

build-cpu:
	BUILD_CUDA_OPS=0 docker compose build

build-gpu:
	BUILD_CUDA_OPS=1 docker compose build

shell:
	docker compose run --rm motip bash

smoke-cpu:
	docker compose run --rm motip uv run python -c "import sys, yaml, accelerate, optuna, mlflow; print('python', sys.version)"

train:
	docker compose run --rm motip uv run accelerate launch --num_processes=1 train.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--exp-name $(EXP_NAME) \
		--run-stage finetuning

baseline-val:
	docker compose run --rm motip uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-mode evaluate \
		--inference-dataset HSPOT \
		--inference-split val \
		--inference-model $(BASELINE_CKPT) \
		--outputs-dir ./outputs/hspot_pretrained_val \
		--run-stage baseline_establishment

tune:
	docker compose run --rm motip uv run python optuna_tune.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-dataset HSPOT \
		--inference-split val \
		--study-name $(STUDY_NAME) \
		--storage $(STORAGE) \
		--n-trials $(N_TRIALS) \
		--epochs $(TUNE_EPOCHS) \
		--output-root $(OUTPUT_ROOT)

train-final:
	docker compose run --rm motip uv run python train_best_from_tuning.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-dataset HSPOT \
		--inference-split val \
		--best-trial-json $(BEST_TRIAL_JSON) \
		--output-dir $(FINAL_OUTPUT_DIR) \
		--exp-name $(FINAL_EXP_NAME) \
		--epochs $(FINAL_EPOCHS)

eval-final:
	docker compose run --rm motip uv run python eval_best_from_final_train.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--summary-json $(FINAL_SUMMARY_JSON) \
		--inference-dataset HSPOT \
		--inference-split test \
		--outputs-dir ./outputs/hspot_final_test

eval:
	docker compose run --rm motip uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-mode evaluate \
		--inference-dataset HSPOT \
		--inference-split test \
		--inference-model $(BEST_CKPT) \
		--outputs-dir ./outputs/hspot_final_test \
		--run-stage final_evaluation
