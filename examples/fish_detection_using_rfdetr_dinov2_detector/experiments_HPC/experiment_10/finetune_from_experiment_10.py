#!/usr/bin/env python
"""Fine-tune RF-DETR Nano from the Experiment 10 trained checkpoint.

The new dataset is expected to be in COCO/RoboFlow layout:

    dataset_dir/
      train/_annotations.coco.json
      train/*.jpg|*.png
      val/_annotations.coco.json
      val/*.jpg|*.png

If your validation split is named ``valid`` instead of ``val``, create a
``val`` symlink or pass a dataset directory that already has both splits.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano


DEFAULT_SOURCE_CHECKPOINT = (
    _SCRIPT_DIR / "checkpoints" / "rfdetr_nano_best.weights.h5"
)
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "finetune_runs" / "from_experiment_10"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the best Experiment 10 RF-DETR Nano checkpoint and fine-tune "
            "it on a new COCO-format dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="COCO-format dataset root containing train/_annotations.coco.json.",
    )
    parser.add_argument(
        "--source-checkpoint",
        default=str(DEFAULT_SOURCE_CHECKPOINT),
        help="Experiment 10 checkpoint to initialize from.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for fine-tuning checkpoints/logs.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous fine-tuning run from output-dir/checkpoint.weights.h5 "
            "instead of initializing from --source-checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help=(
            "Load compatible weights when the new dataset class count differs "
            "from the source checkpoint. Classification-head mismatches are skipped."
        ),
    )
    return parser.parse_args()


def read_coco_classes(dataset_dir):
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Missing training annotations: {ann_path}\n"
            "Expected COCO/RoboFlow layout with train/_annotations.coco.json."
        )
    with ann_path.open() as f:
        anns = json.load(f)
    categories = [
        category
        for category in anns.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    if not categories:
        raise ValueError(f"No usable categories found in {ann_path}")
    categories = sorted(categories, key=lambda category: category["id"])
    return [category["name"] for category in categories]


def ensure_val_split(dataset_dir):
    val_dir = dataset_dir / "val"
    valid_dir = dataset_dir / "valid"
    if val_dir.exists():
        return
    if valid_dir.exists():
        try:
            val_dir.symlink_to(valid_dir.resolve(), target_is_directory=True)
            print(f"Created symlink: {val_dir} -> {valid_dir}")
        except OSError:
            shutil.copytree(valid_dir, val_dir)
            print(f"Copied validation split: {valid_dir} -> {val_dir}")
        return
    raise FileNotFoundError(
        f"Missing validation split: expected {val_dir} or {valid_dir}"
    )


def build_detector(num_classes, checkpoint_path, allow_class_mismatch):
    detector = RFDETRNano(num_classes=num_classes)

    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {checkpoint_path}\n"
            "Train experiment_10 first or pass --source-checkpoint."
        )

    print(f"Loading source checkpoint: {checkpoint_path}")
    detector.model.model.load_weights(
        str(checkpoint_path),
        skip_mismatch=allow_class_mismatch,
    )
    return detector


def save_finetune_config(args, output_dir, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_experiment": "experiment_10",
        "source_checkpoint": str(Path(args.source_checkpoint).expanduser().resolve()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "num_workers": args.num_workers,
        "use_ema": not args.no_ema,
        "amp": not args.no_amp,
        "resume": args.resume,
        "allow_class_mismatch": args.allow_class_mismatch,
    }
    config_path = output_dir / "finetune_config.json"
    with config_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved fine-tuning config: {config_path}")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    ensure_val_split(dataset_dir)
    class_names = read_coco_classes(dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")

    detector = build_detector(
        num_classes=len(class_names),
        checkpoint_path=source_checkpoint,
        allow_class_mismatch=args.allow_class_mismatch,
    )
    detector.model.class_names = class_names

    config = TrainConfig(
        dataset_dir=str(dataset_dir),
        dataset_file="coco_json",
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        lr_scheduler="cosine",
        lr_min_factor=0.0,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        clip_max_norm=0.1,
        use_ema=not args.no_ema,
        ema_decay=0.993,
        ema_tau=100,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=False,
        eval_interval=args.eval_interval,
        eval_ema=not args.no_ema,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        run_test=False,
        class_names=class_names,
        resume=args.resume,
    )

    save_finetune_config(args, output_dir, class_names)

    print("Starting fine-tuning ...")
    detector.train_from_config(config)

    final_path = output_dir / "rfdetr_nano_finetuned_final.weights.h5"
    detector.model.model.save_weights(str(final_path))
    print(f"Saved final fine-tuned weights: {final_path}")


if __name__ == "__main__":
    main()
