#!/usr/bin/env python
"""Fine-tune RF-DETR Nano on the pre-split Labelimage_Fish COCO dataset.

No checkpoint trained on DeepFish is loaded. ``RFDETRNano`` initializes from
its configured original weights, ``lwdetr_nano.weights.h5``.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
_EXAMPLE_DIR = _SCRIPT_DIR.parents[1]
_EXP10_DIR = _SCRIPT_DIR.parent / "experiment_10"
for path in (_PAZ_ROOT, _EXAMPLE_DIR, _EXP10_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano


def load_finetune_helpers():
    helper_path = _EXP10_DIR / "finetune_from_experiment_10.py"
    if not helper_path.exists():
        raise FileNotFoundError(
            f"Required fine-tuning helper script not found: {helper_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "experiment_10_finetune_helpers", helper_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fine-tuning helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_finetune_helpers = load_finetune_helpers()
ensure_valid_split = _finetune_helpers.ensure_valid_split
prepare_class_filtered_dataset = _finetune_helpers.prepare_class_filtered_dataset
prepare_oversampled_dataset = _finetune_helpers.prepare_oversampled_dataset
read_coco_classes = _finetune_helpers.read_coco_classes

DEFAULT_DATASET_DIR = Path(
    "/mnt/beegfs/home/jguo/datasets/Labelimage_Fish_coco_split_70_20_10"
)
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "finetune_original_rfdetr_nano"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune original pretrained RF-DETR Nano directly on "
            "datasets/Labelimage_Fish_coco_split_70_20_10 without "
            "DeepFish pretraining."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--grad-accum-steps", type=positive_int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=positive_int, default=10)
    parser.add_argument("--eval-interval", type=nonnegative_int, default=1)
    parser.add_argument("--num-workers", type=nonnegative_int, default=4)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--filtered-dataset-dir", default=None)
    parser.add_argument("--label-remapped-dataset-dir", default=None)
    parser.add_argument("--overwrite-filtered-dataset", action="store_true")
    parser.add_argument("--oversample-classes", default="")
    parser.add_argument("--rare-repeat-factor", type=nonnegative_int, default=0)
    parser.add_argument("--oversampled-dataset-dir", default=None)
    parser.add_argument("--overwrite-oversampled-dataset", action="store_true")
    parser.add_argument(
        "--oversample-copy-mode", choices=("copy", "symlink"), default="symlink"
    )
    parser.add_argument("--rare-hflip-prob", type=float, default=0.5)
    parser.add_argument("--rare-color-jitter", type=float, default=0.35)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoint.weights.h5.",
    )
    return parser.parse_args()




def build_original_detector(num_classes):
    print("Initializing original pretrained RF-DETR Nano ...", flush=True)
    detector = RFDETRNano(num_classes=num_classes)
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.model(dummy, training=True)
    print(
        "Initialization source: "
        f"{detector.model_config.pretrain_weights or 'random weights'}",
        flush=True,
    )
    return detector


def save_experiment_config(args, output_dir, dataset_dir, class_names, detector):
    payload = {
        "experiment": "experiment_8",
        "description": (
            "Original pretrained RF-DETR Nano fine-tuned directly on the "
            "pre-split Labelimage_Fish COCO dataset; no DeepFish checkpoint"
        ),
        "variant": "RFDETRNano",
        "initialization": detector.model_config.pretrain_weights,
        "source_checkpoint": None,
        "dataset_dir_requested": str(args.dataset_dir.expanduser().resolve()),
        "dataset_dir_prepared": str(dataset_dir),
        "output_dir": str(output_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "lr_scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "use_ema": not args.no_ema,
        "amp": not args.no_amp,
        "num_workers": args.num_workers,
        "augmentation": "native RF-DETR square_resize_div_64",
        "include_classes": args.include_classes,
        "oversample_classes": args.oversample_classes,
        "rare_repeat_factor": args.rare_repeat_factor,
        "resume": args.resume,
    }
    config_path = output_dir / "experiment_config.json"
    with config_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved experiment config: {config_path}", flush=True)


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    valid_ann = dataset_dir / "valid" / "_annotations.coco.json"
    if not train_ann.exists() or not valid_ann.exists():
        raise FileNotFoundError(
            "Expected the pre-split COCO dataset to contain:\n"
            f"  {train_ann}\n"
            f"  {valid_ann}\n"
            "Pass the correct dataset root with --dataset-dir if it is stored "
            "elsewhere."
        )

    ensure_valid_split(dataset_dir)
    dataset_dir = prepare_class_filtered_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)
    dataset_dir = prepare_oversampled_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)

    class_names = read_coco_classes(dataset_dir)
    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Classes ({len(class_names)}): {class_names}", flush=True)

    detector = build_original_detector(num_classes=len(class_names))
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

    save_experiment_config(args, output_dir, dataset_dir, class_names, detector)
    print("Starting direct fine-tuning ...", flush=True)
    detector.train_from_config(config)

    final_path = output_dir / "rfdetr_nano_finetuned_final.weights.h5"
    detector.model.model.save_weights(str(final_path))
    print(f"Saved final fine-tuned weights: {final_path}", flush=True)


if __name__ == "__main__":
    main()
