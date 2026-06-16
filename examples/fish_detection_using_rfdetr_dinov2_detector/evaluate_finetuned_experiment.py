#!/usr/bin/env python
"""Evaluate the fine-tuned model on a COCO test split."""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[1]
_SRC_DIR = _SCRIPT_DIR / "src"
_EXP_DIR = _SCRIPT_DIR / "experiments_HPC" / "experiment_10"

for path in (_PAZ_ROOT, _SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)
from train_utils import setup_logging, validate_epoch_full

logger = logging.getLogger(__name__)

DEFAULT_TEST_DIR = (
    _SCRIPT_DIR / "datasets" / "Labelimage_Fish_coco_split_70_20_10" / "test"
)
DEFAULT_FINETUNE_DIR = _EXP_DIR / "finetune_runs" / "from_experiment_10"
DEFAULT_CHECKPOINT = DEFAULT_FINETUNE_DIR / "rfdetr_nano_finetuned_final.weights.h5"
DEFAULT_CONFIG = DEFAULT_FINETUNE_DIR / "finetune_config.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class _TeeWriter:
    def __init__(self, original, log_path):
        self._original = original
        self._log_file = open(log_path, "a")

    def write(self, text):
        self._original.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)


class CocoSplitDataset:
    """Small COCO split adapter for ``validate_epoch_full``."""

    def __init__(self, split_dir, class_names, resolution=384):
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.ann_path = self.split_dir / "_annotations.coco.json"
        self.resolution = resolution
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self._class_to_id = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        if not self.ann_path.exists():
            raise FileNotFoundError(
                f"Missing COCO annotations: {self.ann_path}"
            )
        with self.ann_path.open() as f:
            coco = json.load(f)

        self._category_id_to_label = build_category_mapping(
            coco.get("categories", []), self.class_names
        )
        self._images = list(coco.get("images", []))
        self._annotations_by_image = {}
        for ann in coco.get("annotations", []):
            if ann.get("category_id") not in self._category_id_to_label:
                continue
            self._annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_info = self._images[idx]
        image_path = resolve_image_path(self.split_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if self.resolution is not None:
            image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0

        width = float(image_info.get("width") or orig_w)
        height = float(image_info.get("height") or orig_h)
        boxes = []
        labels = []
        for ann in self._annotations_by_image.get(image_info["id"], []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            clipped = clip_bbox_xywh(*bbox, width, height)
            if clipped is None:
                continue
            x, y, w, h = clipped
            boxes.append([
                (x + w / 2.0) / width,
                (y + h / 2.0) / height,
                w / width,
                h / height,
            ])
            labels.append(self._category_id_to_label[ann["category_id"]])

        if boxes:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
        return image_np, {"boxes": boxes_arr, "labels": labels_arr}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an RF-DETR Nano model fine-tuned by "
            "experiments_HPC/experiment_10/finetune_from_experiment_10.py "
            "using the same validate_epoch_full metrics as experiment_10.py."
        )
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="COCO test split directory containing _annotations.coco.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Fine-tuned .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="finetune_config.json written by the fine-tune script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "evaluation_results",
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-batches",
        type=nonnegative_int,
        default=0,
        help="Debug limit. 0 evaluates all batches.",
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help="Load compatible weights while skipping mismatched layers.",
    )
    parser.add_argument(
        "--class-names",
        default="",
        help=(
            "Comma-separated class names. By default, class names are read "
            "from finetune_config.json, then from test COCO categories."
        ),
    )
    return parser.parse_args()


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


def clip_bbox_xywh(x, y, w, h, image_width, image_height):
    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x) + float(w), float(image_width)))
    y2 = max(0.0, min(float(y) + float(h), float(image_height)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    return [x1, y1, bw, bh]


def resolve_image_path(split_dir, file_name):
    candidates = [
        split_dir / file_name,
        split_dir.parent / file_name,
        split_dir / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image referenced by COCO JSON was not found: {split_dir / file_name}"
    )


def read_coco(split_dir):
    ann_path = Path(split_dir).expanduser().resolve() / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")
    with ann_path.open() as f:
        return json.load(f)


def class_names_from_coco(split_dir):
    coco = read_coco(split_dir)
    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    categories = sorted(categories, key=lambda category: category["id"])
    if not categories:
        raise ValueError(
            f"No usable categories found in {Path(split_dir) / '_annotations.coco.json'}"
        )
    return [category["name"] for category in categories]


def read_class_names(args):
    if args.class_names.strip():
        return [
            name.strip()
            for name in args.class_names.split(",")
            if name.strip()
        ]

    config_path = args.finetune_config.expanduser().resolve()
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        class_names = config.get("class_names")
        if class_names:
            return list(class_names)

    return class_names_from_coco(args.test_dir)


def build_category_mapping(categories, class_names):
    by_name = {
        category["name"]: category["id"]
        for category in categories
        if category.get("supercategory", "") != "none"
    }
    if all(name in by_name for name in class_names):
        return {by_name[name]: idx for idx, name in enumerate(class_names)}

    usable = [
        category
        for category in categories
        if category.get("supercategory", "") != "none"
    ]
    usable = sorted(usable, key=lambda category: category["id"])
    if len(usable) != len(class_names):
        available = [category.get("name") for category in usable]
        raise ValueError(
            "Cannot map COCO category ids to model class indices. "
            f"Model classes: {class_names}; COCO categories: {available}"
        )
    return {
        category["id"]: idx
        for idx, category in enumerate(usable)
    }


def resolve_checkpoint(path):
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.exists():
        return checkpoint

    fallback = checkpoint.parent / "checkpoint_best_total.weights.h5"
    if checkpoint == DEFAULT_CHECKPOINT.resolve() and fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}\n"
        "Run finetune_from_experiment_10.py first or pass --checkpoint."
    )


def build_detector(num_classes, checkpoint_path, allow_class_mismatch):
    detector = RFDETRNano(num_classes=num_classes)
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.model.load_weights(
        str(checkpoint_path),
        skip_mismatch=allow_class_mismatch,
    )
    detector.model.model(dummy, training=True)
    return detector


def make_train_config(dataset_root, output_dir, batch_size, class_names):
    return TrainConfig(
        dataset_dir=str(dataset_root),
        dataset_file="coco_json",
        output_dir=str(output_dir),
        epochs=1,
        batch_size=batch_size,
        grad_accum_steps=1,
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        lr_scheduler="cosine",
        lr_min_factor=0.0,
        warmup_epochs=0.0,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        checkpoint_interval=10,
        early_stopping=False,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=0,
        run_test=False,
        class_names=class_names,
    )


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def write_csv(row, csv_path):
    fieldnames = [
        "checkpoint",
        "checkpoint_path",
        "test_dir",
        "num_test_images",
        "test_loss",
        "test_loss_ce",
        "test_loss_bbox",
        "test_loss_giou",
        "test_mAP_50",
        "test_mAP_50_95",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
        "test_num_gt_boxes",
        "test_num_pred_boxes",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    # output_dir = args.output_dir.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir))
    sys.stdout = _TeeWriter(sys.stdout, output_dir / "output.txt")

    import keras

    test_dir = args.test_dir.expanduser().resolve()
    checkpoint = resolve_checkpoint(args.checkpoint)
    class_names = read_class_names(args)
    dataset = CocoSplitDataset(test_dir, class_names, resolution=384)
    indices = list(range(len(dataset)))
    max_batches = args.max_batches if args.max_batches > 0 else None

    logger.info("=" * 68)
    logger.info("EXPERIMENT 10 FINE-TUNED MODEL EVALUATION")
    logger.info("=" * 68)
    logger.info("Keras backend       : %s", keras.backend.backend())
    logger.info("Checkpoint          : %s", checkpoint)
    logger.info("Fine-tune config    : %s", args.finetune_config)
    logger.info("Test split          : %s", test_dir)
    logger.info("Test images         : %d", len(dataset))
    logger.info("Classes (%d)        : %s", len(class_names), class_names)
    logger.info("Output dir          : %s", output_dir)

    start = time.time()
    detector = build_detector(
        num_classes=len(class_names),
        checkpoint_path=checkpoint,
        allow_class_mismatch=args.allow_class_mismatch,
    )
    detector.model.class_names = class_names
    train_config = make_train_config(
        dataset_root=test_dir.parent,
        output_dir=output_dir,
        batch_size=args.batch_size,
        class_names=class_names,
    )
    criterion, _ = build_criterion_from_config(
        detector.model_config, train_config
    )
    metrics = validate_epoch_full(
        model=detector.model.model,
        criterion=criterion,
        dataset=dataset,
        indices=indices,
        batch_size=args.batch_size,
        num_classes=len(class_names),
        class_names=class_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_batches=max_batches,
        logger=logger,
        prefix="test",
    )
    elapsed = time.time() - start

    logger.info("")
    logger.info("-" * 68)
    logger.info(
        "mAP@50=%.4f mAP@50:95=%.4f precision=%.4f recall=%.4f f1=%.4f loss=%.4f",
        metrics["test_mAP_50"],
        metrics["test_mAP_50_95"],
        metrics["test_precision"],
        metrics["test_recall"],
        metrics["test_f1"],
        metrics["test_loss"],
    )
    logger.info("Evaluation completed in %.1fs", elapsed)

    results = {
        "source_experiment": "experiment_10",
        "fine_tune_script": str(_EXP_DIR / "finetune_from_experiment_10.py"),
        "metric_source": "experiment_10.validate_epoch_full",
        "checkpoint": str(checkpoint),
        "finetune_config": str(args.finetune_config),
        "test_dir": str(test_dir),
        "num_test_images": len(dataset),
        "class_names": class_names,
        "settings": {
            "batch_size": args.batch_size,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "max_batches": max_batches,
            "allow_class_mismatch": args.allow_class_mismatch,
        },
        "elapsed_seconds": elapsed,
        "metrics": jsonable(metrics),
    }
    summary_row = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "test_dir": str(test_dir),
        "num_test_images": len(dataset),
        "elapsed_seconds": elapsed,
    }
    summary_row.update({
        key: jsonable(value)
        for key, value in metrics.items()
        if not key.startswith("per_class_")
    })

    json_path = output_dir / "finetuned_test_results.json"
    csv_path = output_dir / "finetuned_test_summary.csv"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(summary_row, csv_path)

    logger.info("Saved JSON results: %s", json_path)
    logger.info("Saved CSV summary : %s", csv_path)
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
