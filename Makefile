SHELL := /bin/bash

HOST_PROJECT_ROOT := $(CURDIR)
STORAGE_ROOT ?= $(HOST_PROJECT_ROOT)/.surf-storage
CONFIG ?= ./configs/high_street_property_occupancy_tracking.yaml
DATA_ROOT ?= ./datasets/
EXP_NAME ?= hspot_train
BASELINE_EXP_NAME ?= hspot_baseline_val
BASELINE_CKPT ?= ./pretrains/r50_deformable_detr_motip_dancetrack.pth
BASELINE_REL_PE_LENGTH ?= 30
BASELINE_MISS_TOLERANCE ?= 30
STUDY_NAME ?= hspot_hota_optuna
STORAGE ?= sqlite:///./.surf-storage/optuna/hspot_hota_optuna.db
N_TRIALS ?= 40
TUNE_EPOCHS ?= 6
TUNE_TIMEOUT ?= 360000
TUNE_SAMPLER_SEED ?= 42
TUNE_PRUNER_STARTUP_TRIALS ?= 8
TUNE_PRUNER_WARMUP_STEPS ?= 3
TUNE_PRUNER_INTERVAL_STEPS ?= 1
OUTPUT_ROOT ?= ./outputs/optuna_hspot
BEST_TRIAL_JSON ?= ./outputs/optuna_hspot/best_trial.json
BEST_CKPT ?= ./outputs/REPLACE_WITH_BEST_CHECKPOINT.pth

COMPOSE = HOST_PROJECT_ROOT="$(HOST_PROJECT_ROOT)" docker compose

.PHONY: help prepare-storage bootstrap bootstrap-gpu build build-gpu shell train baseline-val tune eval-final eval

help:
	@echo "Available targets:"
	@echo "  make prepare-storage - Create repo-local SURF volume directories under $(STORAGE_ROOT)"
	@echo "  make bootstrap      - Bootstrap the CUDA VM and place Docker storage on the SURF volume"
	@echo "  make bootstrap-gpu  - Alias for make bootstrap"
	@echo "  make build          - Build Docker image (CUDA ops enabled)"
	@echo "  make build-gpu      - Alias for make build"
	@echo "  make shell          - Open shell in project container"
	@echo "  make train          - Run training in container"
	@echo "  make baseline-val   - Evaluate pretrained MOTIP checkpoint on HSPOT val in container"
	@echo "  make tune           - Run Optuna tuning in container"
	@echo "  make eval-final     - Evaluate the best checkpoint from Optuna tuning on HSPOT test"
	@echo "  make eval           - Evaluate BEST_CKPT on test split in container"
	@echo ""
	@echo "Variables:"
	@echo "  STORAGE_ROOT=$(STORAGE_ROOT)"
	@echo "  STORAGE=$(STORAGE)"

prepare-storage:
	mkdir -p datasets
	mkdir -p outputs
	mkdir -p pretrains
	mkdir -p "$(STORAGE_ROOT)/cache/xdg"
	mkdir -p "$(STORAGE_ROOT)/cache/uv"
	mkdir -p "$(STORAGE_ROOT)/cache/pip"
	mkdir -p "$(STORAGE_ROOT)/cache/wandb"
	mkdir -p "$(STORAGE_ROOT)/cache/torch"
	mkdir -p "$(STORAGE_ROOT)/cache/matplotlib"
	mkdir -p "$(STORAGE_ROOT)/tmp"
	mkdir -p "$(STORAGE_ROOT)/optuna"
	mkdir -p "$(STORAGE_ROOT)/mlruns"

bootstrap: prepare-storage
	bash ./scripts/bootstrap_vm.sh "$(STORAGE_ROOT)"

bootstrap-gpu: prepare-storage
	bash ./scripts/bootstrap_vm.sh gpu "$(STORAGE_ROOT)"

build: prepare-storage
	BUILD_CUDA_OPS=1 $(COMPOSE) build

build-gpu: prepare-storage
	BUILD_CUDA_OPS=1 $(COMPOSE) build

shell: prepare-storage
	$(COMPOSE) run --rm motip bash

train: prepare-storage
	$(COMPOSE) run --rm -T motip uv run accelerate launch --num_processes=1 train.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--exp-name $(EXP_NAME) \
		--run-stage finetuning

baseline-val: prepare-storage
	$(COMPOSE) run --rm -T motip uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--exp-name $(BASELINE_EXP_NAME) \
		--rel-pe-length $(BASELINE_REL_PE_LENGTH) \
		--inference-mode evaluate \
		--inference-dataset HSPOT \
		--inference-split val \
		--inference-model $(BASELINE_CKPT) \
		--miss-tolerance $(BASELINE_MISS_TOLERANCE) \
		--outputs-dir ./outputs/hspot_pretrained_val \
		--run-stage baseline_establishment

tune: prepare-storage
	$(COMPOSE) run --rm -T motip uv run python optuna_tune.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-dataset HSPOT \
		--inference-split val \
		--study-name $(STUDY_NAME) \
		--storage $(STORAGE) \
		--n-trials $(N_TRIALS) \
		--timeout $(TUNE_TIMEOUT) \
		--sampler-seed $(TUNE_SAMPLER_SEED) \
		--pruner-startup-trials $(TUNE_PRUNER_STARTUP_TRIALS) \
		--pruner-warmup-steps $(TUNE_PRUNER_WARMUP_STEPS) \
		--pruner-interval-steps $(TUNE_PRUNER_INTERVAL_STEPS) \
		--epochs $(TUNE_EPOCHS) \
		--output-root $(OUTPUT_ROOT)

eval-final: prepare-storage
	$(COMPOSE) run --rm -T motip uv run python eval_best_from_tuning.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--best-trial-json $(BEST_TRIAL_JSON) \
		--output-root $(OUTPUT_ROOT) \
		--inference-dataset HSPOT \
		--inference-split test \
		--outputs-dir ./outputs/hspot_final_test

eval: prepare-storage
	$(COMPOSE) run --rm -T motip uv run accelerate launch --num_processes=1 submit_and_evaluate.py \
		--config-path $(CONFIG) \
		--data-root $(DATA_ROOT) \
		--inference-mode evaluate \
		--inference-dataset HSPOT \
		--inference-split test \
		--inference-model $(BEST_CKPT) \
		--outputs-dir ./outputs/hspot_final_test \
		--run-stage final_evaluation
