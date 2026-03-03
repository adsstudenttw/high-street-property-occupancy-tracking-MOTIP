import json
import argparse

from submit_and_evaluate import submit_and_evaluate
from tuning_utils import build_base_config


def build_parser():
    parser = argparse.ArgumentParser(
        "Evaluate the best checkpoint produced by the final training stage.",
        add_help=True,
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--super-config-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--inference-dataset", type=str, default="HSPOT")
    parser.add_argument("--inference-split", type=str, default="test")
    parser.add_argument("--outputs-dir", type=str, default="./outputs/hspot_final_test")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    best_checkpoint_path = summary.get("best_checkpoint_path")
    if not best_checkpoint_path:
        raise RuntimeError(
            f"No best_checkpoint_path found in '{args.summary_json}'. "
            "Run the final training stage first."
        )

    cfg = build_base_config(
        config_path=args.config_path,
        super_config_path=args.super_config_path,
        data_root=args.data_root,
        inference_dataset=args.inference_dataset,
        inference_split=args.inference_split,
    )
    cfg["INFERENCE_MODE"] = "evaluate"
    cfg["INFERENCE_MODEL"] = best_checkpoint_path
    cfg["OUTPUTS_DIR"] = args.outputs_dir
    cfg["RUN_STAGE"] = "final_evaluation"
    cfg["HPO_STUDY_NAME"] = summary.get("study_name")
    cfg["HPO_TRIAL_NUMBER"] = summary.get("hpo_best_trial_number")
    cfg["HPO_STAGE_ITER"] = summary.get("hpo_best_trial_number")
    if args.disable_mlflow:
        cfg["USE_WANDB"] = False

    submit_and_evaluate(config=cfg)


if __name__ == "__main__":
    main()
