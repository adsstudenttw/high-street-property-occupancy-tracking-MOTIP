import argparse
import glob
import json
import os
import re

from submit_and_evaluate import submit_and_evaluate
from tuning_utils import build_base_config


def build_parser():
    parser = argparse.ArgumentParser(
        "Evaluate the best checkpoint produced during Optuna tuning.",
        add_help=True,
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--super-config-path", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--best-trial-json", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="./outputs/optuna_hspot")
    parser.add_argument("--inference-dataset", type=str, default="HSPOT")
    parser.add_argument("--inference-split", type=str, default="test")
    parser.add_argument("--outputs-dir", type=str, default="./outputs/hspot_final_test")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser


def _checkpoint_epoch(path: str) -> int:
    matched = re.search(r"checkpoint_(\d+)\.pth$", os.path.basename(path))
    if matched is None:
        return -1
    return int(matched.group(1))


def resolve_best_checkpoint_path(best_payload: dict, output_root: str) -> tuple[str, bool]:
    checkpoint_path = best_payload.get("best_checkpoint_path")
    if checkpoint_path:
        return checkpoint_path, False

    trial_outputs_dir = best_payload.get("best_trial_outputs_dir")
    if trial_outputs_dir is None:
        trial_number = best_payload.get("best_trial_number")
        if trial_number is None:
            raise RuntimeError(
                "Could not resolve checkpoint path: best_trial_number is missing from best-trial JSON."
            )
        trial_outputs_dir = os.path.join(output_root, f"trial_{int(trial_number):04d}")

    checkpoints = sorted(
        glob.glob(os.path.join(trial_outputs_dir, "checkpoint_*.pth")),
        key=_checkpoint_epoch,
    )
    if not checkpoints:
        raise RuntimeError(
            "Could not resolve checkpoint path from tuning outputs. "
            "Rerun `make tune` so best_trial.json includes best_checkpoint_path."
        )
    return checkpoints[-1], True


def main():
    args = build_parser().parse_args()

    with open(args.best_trial_json, "r") as f:
        best_payload = json.load(f)

    best_checkpoint_path, used_fallback = resolve_best_checkpoint_path(
        best_payload=best_payload,
        output_root=args.output_root,
    )
    if used_fallback:
        print(
            "Warning: best_checkpoint_path missing in best-trial JSON. "
            f"Using latest trial checkpoint at '{best_checkpoint_path}'."
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
    cfg["RUN_STAGE"] = "final_evaluation_from_tuning"
    cfg["HPO_STUDY_NAME"] = best_payload.get("study_name")
    cfg["HPO_TRIAL_NUMBER"] = best_payload.get("best_trial_number")
    cfg["HPO_STAGE_ITER"] = best_payload.get("best_trial_number")
    if args.disable_mlflow:
        cfg["USE_WANDB"] = False

    submit_and_evaluate(config=cfg)


if __name__ == "__main__":
    main()
