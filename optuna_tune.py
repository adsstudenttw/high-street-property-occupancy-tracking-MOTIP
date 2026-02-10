# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import json
import copy
import argparse

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "optuna is required for hyperparameter tuning. Install it with `pip install optuna`."
    ) from e

from train import train_engine
from utils.misc import yaml_to_dict
from configs.util import load_super_config


def build_parser():
    parser = argparse.ArgumentParser("Optuna HOTA tuning entrypoint.", add_help=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--super-config-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--inference-dataset", type=str, default=None)
    parser.add_argument("--inference-split", type=str, default="val")
    parser.add_argument("--study-name", type=str, default="motip_hota_tuning")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruner-startup-trials", type=int, default=3)
    parser.add_argument("--pruner-warmup-steps", type=int, default=2)
    parser.add_argument("--pruner-interval-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="./outputs/optuna")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser


def build_base_config(args):
    cfg = yaml_to_dict(args.config_path)
    if args.super_config_path is not None:
        cfg = load_super_config(cfg, args.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    if args.data_root is not None:
        cfg["DATA_ROOT"] = args.data_root
    if args.inference_dataset is not None:
        cfg["INFERENCE_DATASET"] = args.inference_dataset
    if args.inference_split is not None:
        cfg["INFERENCE_SPLIT"] = args.inference_split
    return cfg


def sample_hparams(trial, cfg):
    cfg["LR"] = trial.suggest_float("LR", 5e-6, 5e-4, log=True)
    cfg["WEIGHT_DECAY"] = trial.suggest_float("WEIGHT_DECAY", 1e-5, 5e-3, log=True)
    cfg["DET_THRESH"] = trial.suggest_float("DET_THRESH", 0.1, 0.7)
    cfg["NEWBORN_THRESH"] = trial.suggest_float("NEWBORN_THRESH", 0.2, 0.9)
    cfg["ID_THRESH"] = trial.suggest_float("ID_THRESH", 0.05, 0.5)
    cfg["MISS_TOLERANCE"] = trial.suggest_int("MISS_TOLERANCE", 10, 40)
    cfg["MAX_CLIP_NORM"] = trial.suggest_float("MAX_CLIP_NORM", 0.05, 0.3)
    cfg["LR_DICTIONARY_SCALE"] = trial.suggest_float("LR_DICTIONARY_SCALE", 0.3, 2.0)
    return cfg


def make_objective(base_cfg, args):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg = sample_hparams(trial, cfg)
        cfg["OUTPUTS_DIR"] = os.path.join(args.output_root, f"trial_{trial.number:04d}")
        cfg["EXP_GROUP"] = args.study_name
        cfg["EXP_NAME"] = f"{args.study_name}_trial_{trial.number:04d}"
        cfg["SAVE_CHECKPOINT_PER_EPOCH"] = 1
        if args.epochs is not None:
            cfg["EPOCHS"] = args.epochs
        if args.disable_mlflow:
            cfg["USE_WANDB"] = False
        cfg["SEED"] = int(base_cfg.get("SEED", 42)) + trial.number
        if cfg.get("INFERENCE_DATASET", None) is None:
            cfg["INFERENCE_DATASET"] = cfg["DATASETS"][0]
        if cfg.get("INFERENCE_SPLIT", None) is None:
            cfg["INFERENCE_SPLIT"] = "val"

        best_hota = None

        def _report(epoch: int, hota: float, global_step: int):
            nonlocal best_hota
            if best_hota is None or hota > best_hota:
                best_hota = hota
            trial.report(hota, step=epoch)
            trial.set_user_attr("latest_global_step", int(global_step))

        def _should_prune(epoch: int, hota: float, global_step: int) -> bool:
            return trial.should_prune()

        summary = train_engine(
            config=cfg,
            report_intermediate=_report,
            should_prune=_should_prune,
        )

        if summary.get("best_eval_hota") is not None:
            best_hota = summary["best_eval_hota"]
        if best_hota is None:
            raise RuntimeError(
                "No HOTA was produced during tuning. Ensure INFERENCE_DATASET and INFERENCE_SPLIT are set for eval."
            )
        if summary.get("pruned", False):
            raise optuna.TrialPruned()
        trial.set_user_attr("best_eval_hota", float(best_hota))
        trial.set_user_attr("outputs_dir", cfg["OUTPUTS_DIR"])
        return float(best_hota)

    return objective


def main():
    parser = build_parser()
    args = parser.parse_args()

    base_cfg = build_base_config(args)
    os.makedirs(args.output_root, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
        interval_steps=args.pruner_interval_steps,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        make_objective(base_cfg=base_cfg, args=args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    best_payload = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value_hota": study.best_value,
        "best_params": study.best_params,
    }
    with open(os.path.join(args.output_root, "best_trial.json"), "w") as f:
        json.dump(best_payload, f, indent=2)
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
