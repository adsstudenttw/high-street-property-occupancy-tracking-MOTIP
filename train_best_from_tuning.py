import os
import json
import argparse

from train import train_engine
from tuning_utils import build_base_config, apply_tuned_hparams


def build_parser():
    parser = argparse.ArgumentParser(
        "Train a final model from the best Optuna hyperparameters.", add_help=True
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--super-config-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--inference-dataset", type=str, default=None)
    parser.add_argument("--inference-split", type=str, default=None)
    parser.add_argument("--best-trial-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/hspot_final_train")
    parser.add_argument("--exp-name", type=str, default="hspot_final_train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    cfg = build_base_config(
        config_path=args.config_path,
        super_config_path=args.super_config_path,
        data_root=args.data_root,
        inference_dataset=args.inference_dataset,
        inference_split=args.inference_split,
    )

    with open(args.best_trial_json, "r") as f:
        best_payload = json.load(f)

    best_params = best_payload.get("best_params")
    if not isinstance(best_params, dict) or len(best_params) == 0:
        raise RuntimeError(
            f"No best_params found in '{args.best_trial_json}'. Run tuning first."
        )

    cfg = apply_tuned_hparams(cfg=cfg, params=best_params)
    cfg["OUTPUTS_DIR"] = args.output_dir
    cfg["EXP_NAME"] = args.exp_name
    cfg["EXP_GROUP"] = best_payload.get("study_name")
    cfg["RUN_STAGE"] = "final_model_training"
    cfg["HPO_STUDY_NAME"] = best_payload.get("study_name")
    cfg["HPO_TRIAL_NUMBER"] = best_payload.get("best_trial_number")
    cfg["HPO_STAGE_ITER"] = best_payload.get("best_trial_number")
    if args.epochs is not None:
        cfg["EPOCHS"] = args.epochs
    if args.disable_mlflow:
        cfg["USE_WANDB"] = False

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "best_trial_used.json"), "w") as f:
        json.dump(best_payload, f, indent=2)

    summary = train_engine(config=cfg)
    summary["study_name"] = best_payload.get("study_name")
    summary["hpo_best_trial_number"] = best_payload.get("best_trial_number")
    with open(os.path.join(args.output_dir, "final_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
