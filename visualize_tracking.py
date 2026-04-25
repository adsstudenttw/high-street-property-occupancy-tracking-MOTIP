import argparse
import json
import os
import re
from collections import defaultdict

from PIL import Image, ImageDraw

from data.joint_dataset import dataset_classes


GT_COLOR = (57, 181, 74)
PRED_COLOR = (255, 140, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


def _checkpoint_epoch(path: str) -> int:
    matched = re.search(r"checkpoint_(\d+)\.pth$", os.path.basename(path))
    if matched is None:
        return -1
    return int(matched.group(1))


def _parse_mot_boxes(file_path: str) -> dict[int, list[dict]]:
    boxes_by_frame = defaultdict(list)
    if not os.path.exists(file_path):
        return boxes_by_frame
    with open(file_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0]) - 1
            obj_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            boxes_by_frame[frame_id].append(
                {"id": obj_id, "bbox": [x, y, w, h]}
            )
    return boxes_by_frame


def _gt_boxes_for_frame(annotation: dict) -> list[dict]:
    gt_boxes = []
    for obj_id, bbox in zip(annotation["id"], annotation["bbox"]):
        gt_boxes.append(
            {
                "id": int(obj_id.item()),
                "bbox": [float(v) for v in bbox.tolist()],
            }
        )
    return gt_boxes


def _draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int]):
    x, y, w, h = box
    draw.rectangle((x, y, x + w, y + h), outline=color, width=3)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    color: tuple[int, int, int],
):
    text_bbox = draw.textbbox((x, y), label)
    draw.rectangle(text_bbox, fill=TEXT_BG)
    draw.text((x, y), label, fill=TEXT_COLOR)
    draw.rectangle(text_bbox, outline=color, width=1)


def render_tracking_visualizations(
    data_root: str,
    dataset: str,
    split: str,
    tracker_dir: str,
    output_dir: str,
    max_frames_per_sequence: int = 0,
):
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    if not os.path.isdir(tracker_dir):
        raise FileNotFoundError(f"Tracker directory does not exist: '{tracker_dir}'.")

    inference_dataset = dataset_classes[dataset](
        data_root=data_root,
        split=split,
        load_annotation=True,
    )
    sequence_names = sorted(inference_dataset.sequence_infos.keys())

    rendered_images = 0
    for sequence_name in sequence_names:
        tracker_path = os.path.join(tracker_dir, f"{sequence_name}.txt")
        pred_boxes_by_frame = _parse_mot_boxes(tracker_path)
        sequence_output_dir = os.path.join(overlay_dir, sequence_name)
        os.makedirs(sequence_output_dir, exist_ok=True)

        image_paths = inference_dataset.image_paths[sequence_name]
        annotations = inference_dataset.annotations[sequence_name]

        frames_to_render = len(image_paths)
        if max_frames_per_sequence > 0:
            frames_to_render = min(frames_to_render, max_frames_per_sequence)

        for frame_idx in range(frames_to_render):
            image = Image.open(image_paths[frame_idx]).convert("RGB")
            draw = ImageDraw.Draw(image)

            for gt in _gt_boxes_for_frame(annotations[frame_idx]):
                _draw_box(draw, gt["bbox"], GT_COLOR)
                _draw_label(
                    draw,
                    gt["bbox"][0],
                    max(gt["bbox"][1] - 14, 0),
                    f"GT {gt['id']}",
                    GT_COLOR,
                )

            for pred in pred_boxes_by_frame.get(frame_idx, []):
                _draw_box(draw, pred["bbox"], PRED_COLOR)
                _draw_label(
                    draw,
                    pred["bbox"][0],
                    pred["bbox"][1],
                    f"PRED {pred['id']}",
                    PRED_COLOR,
                )

            draw.rectangle((8, 8, 210, 52), fill=TEXT_BG)
            draw.text((14, 12), "Green: GT", fill=GT_COLOR)
            draw.text((14, 30), "Red: Prediction", fill=PRED_COLOR)
            image.save(
                os.path.join(sequence_output_dir, f"{frame_idx + 1:06d}.jpg"),
                quality=95,
            )
            rendered_images += 1

    return {
        "overlay_dir": overlay_dir,
        "rendered_images": rendered_images,
        "sequence_count": len(sequence_names),
    }


def resolve_tracker_dir(
    stage: str,
    stage_root: str | None,
    dataset: str,
    split: str,
    inference_group: str,
    model_name: str | None,
    epoch: int | None,
    best_trial_json: str | None,
) -> tuple[str, str]:
    if stage == "best_trial":
        if best_trial_json is None:
            raise ValueError("--best-trial-json is required for stage=best_trial.")
        with open(best_trial_json, "r") as f:
            payload = json.load(f)
        checkpoint_path = payload.get("best_checkpoint_path")
        if checkpoint_path is None:
            raise RuntimeError("best_trial.json does not contain best_checkpoint_path.")
        checkpoint_epoch = _checkpoint_epoch(checkpoint_path)
        if checkpoint_epoch < 0:
            raise RuntimeError(
                f"Could not infer checkpoint epoch from '{checkpoint_path}'."
            )
        trial_root = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        tracker_dir = os.path.join(
            trial_root,
            "train",
            "eval_during_train",
            f"epoch_{checkpoint_epoch}",
            "evaluate",
            inference_group,
            dataset,
            split,
            checkpoint_name,
            "tracker",
        )
        output_root = os.path.join(trial_root, "visualizations", "best_trial")
        return tracker_dir, output_root

    if stage_root is None:
        raise ValueError("--stage-root is required for this stage.")

    if stage in {"baseline", "final"}:
        if model_name is None:
            candidates = _find_model_candidates(
                stage_root=stage_root,
                dataset=dataset,
                split=split,
                inference_group=inference_group,
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not uniquely resolve the model directory under '{stage_root}'. "
                    f"Found {len(candidates)} candidates."
                )
            model_name = candidates[0]
        tracker_dir = _resolve_existing_tracker_dir(
            candidates=[
                os.path.join(
                    stage_root,
                    "evaluate",
                    inference_group,
                    dataset,
                    split,
                    model_name,
                    "tracker",
                ),
                os.path.join(stage_root, "tracker"),
            ],
            stage=stage,
        )
        output_root = os.path.join(stage_root, "visualizations", stage)
        return tracker_dir, output_root

    if stage == "finetuning":
        if epoch is None:
            raise ValueError("--epoch is required for stage=finetuning.")
        if model_name is None:
            model_name = f"checkpoint_{epoch}"
        epoch_root = os.path.join(
            stage_root, "train", "eval_during_train", f"epoch_{epoch}"
        )
        tracker_dir = _resolve_existing_tracker_dir(
            candidates=[
                os.path.join(
                    epoch_root,
                    "evaluate",
                    inference_group,
                    dataset,
                    split,
                    model_name,
                    "tracker",
                ),
                os.path.join(epoch_root, "tracker"),
            ],
            stage=stage,
        )
        output_root = os.path.join(
            epoch_root,
            "visualizations",
        )
        return tracker_dir, output_root

    raise ValueError(f"Unsupported stage '{stage}'.")


def _find_model_candidates(
    stage_root: str,
    dataset: str,
    split: str,
    inference_group: str,
) -> list[str]:
    base_dir = os.path.join(stage_root, "evaluate", inference_group, dataset, split)
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))
    )


def _resolve_existing_tracker_dir(candidates: list[str], stage: str) -> str:
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    formatted = "\n".join(candidates)
    raise FileNotFoundError(
        f"Could not find a tracker directory for stage '{stage}'. Checked:\n{formatted}"
    )


def main():
    parser = argparse.ArgumentParser(
        "Render GT/prediction bounding box overlays for completed stage outputs.",
        add_help=True,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["baseline", "finetuning", "best_trial", "final"],
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="HSPOT")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--stage-root", type=str)
    parser.add_argument("--best-trial-json", type=str)
    parser.add_argument("--inference-group", type=str, default="default")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--tracker-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    args = parser.parse_args()

    tracker_dir = args.tracker_dir
    output_dir = args.output_dir
    if tracker_dir is None or output_dir is None:
        tracker_dir, output_dir = resolve_tracker_dir(
            stage=args.stage,
            stage_root=args.stage_root,
            dataset=args.dataset,
            split=args.split,
            inference_group=args.inference_group,
            model_name=args.model_name,
            epoch=args.epoch,
            best_trial_json=args.best_trial_json,
        )

    summary = render_tracking_visualizations(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        tracker_dir=tracker_dir,
        output_dir=output_dir,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    summary["tracker_dir"] = tracker_dir
    summary["output_dir"] = output_dir
    print(summary)


if __name__ == "__main__":
    main()
