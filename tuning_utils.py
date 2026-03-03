# Shared helpers for Optuna tuning and final training from best tuned params.

from __future__ import annotations

import copy

from utils.misc import yaml_to_dict
from configs.util import load_super_config


def build_base_config(
    config_path: str,
    super_config_path: str | None = None,
    data_root: str | None = None,
    inference_dataset: str | None = None,
    inference_split: str | None = None,
):
    cfg = yaml_to_dict(config_path)
    if super_config_path is not None:
        cfg = load_super_config(cfg, super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    if data_root is not None:
        cfg["DATA_ROOT"] = data_root
    if inference_dataset is not None:
        cfg["INFERENCE_DATASET"] = inference_dataset
    if inference_split is not None:
        cfg["INFERENCE_SPLIT"] = inference_split
    return cfg


def apply_tuned_hparams(cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key in [
        "LR",
        "WEIGHT_DECAY",
        "LR_BACKBONE_SCALE",
        "LR_DICTIONARY_SCALE",
        "LR_WARMUP_EPOCHS",
        "MAX_CLIP_NORM",
        "ID_LOSS_WEIGHT",
        "ASSIGNMENT_PROTOCOL",
        "DET_THRESH",
        "NEWBORN_THRESH",
        "ID_THRESH",
        "MISS_TOLERANCE",
        "AREA_THRESH",
    ]:
        if key in params:
            cfg[key] = params[key]
    return cfg
