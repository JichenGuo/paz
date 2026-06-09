#!/usr/bin/env python
"""Experiment 13_2: RF-DETR Nano on DeepFish + FathomNet as ``sea_animal``.

Training uses DeepFish ``train`` images plus FathomNet training COCO data.
Validation uses DeepFish ``valid`` images plus FathomNet validation COCO data,
with every source label remapped to the single ``sea_animal`` class.
"""

import json
import logging
import math
import os
import shutil
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
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
_SRC_DIR = _SCRIPT_DIR.parents[1] / "src"
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)
from metrics_tracker import MetricsTracker
from train_utils import setup_logging, validate_epoch_full

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAME = "sea_animal"


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


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_path(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return Path(default).expanduser().resolve()
    return Path(value).expanduser().resolve()


def _safe_symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _image_size(path):
    with Image.open(path) as image:
        return image.size


def _clip_bbox_xywh(x, y, w, h, image_width, image_height):
    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x) + float(w), float(image_width)))
    y2 = max(0.0, min(float(y) + float(h), float(image_height)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    return [x1, y1, bw, bh]


def _read_yolo_boxes(label_path, image_width, image_height):
    if not label_path.exists():
        return []
    boxes = []
    with label_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 YOLO fields in {label_path}:{line_no}, got {len(parts)}"
                )
            _, cx, cy, bw, bh = [float(value) for value in parts]
            abs_w = bw * image_width
            abs_h = bh * image_height
            abs_x = cx * image_width - abs_w / 2.0
            abs_y = cy * image_height - abs_h / 2.0
            clipped = _clip_bbox_xywh(
                abs_x, abs_y, abs_w, abs_h, image_width, image_height
            )
            if clipped is not None:
                boxes.append(clipped)
    return boxes


def _add_annotations(annotations, boxes, image_id, next_annotation_id):
    for bbox in boxes:
        _, _, width, height = bbox
        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [float(value) for value in bbox],
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        next_annotation_id += 1
    return next_annotation_id


def _collect_deepfish_split(deepfish_dir, split_name, target_dir, images,
                            annotations, next_image_id, next_annotation_id):
    if not deepfish_dir.exists():
        raise FileNotFoundError(f"DeepFish dataset directory not found: {deepfish_dir}")
    image_paths = sorted(
        path
        for path in deepfish_dir.glob(f"*/{split_name}/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No DeepFish {split_name} images found under {deepfish_dir}/*/{split_name}"
        )

    before_annotations = len(annotations)
    for source_path in image_paths:
        subset_name = source_path.parents[1].name
        file_name = f"deepfish_{split_name}_{subset_name}_{source_path.name}"
        width, height = _image_size(source_path)
        _safe_symlink(source_path, target_dir / file_name)
        images.append(
            {
                "id": next_image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
            }
        )
        boxes = _read_yolo_boxes(source_path.with_suffix(".txt"), width, height)
        next_annotation_id = _add_annotations(
            annotations, boxes, next_image_id, next_annotation_id
        )
        next_image_id += 1

    return {
        "images": len(image_paths),
        "annotations": len(annotations) - before_annotations,
        "next_image_id": next_image_id,
        "next_annotation_id": next_annotation_id,
    }


def _find_coco_annotations(dataset_dir, split_name):
    candidates = [
        dataset_dir / split_name / "_annotations.coco.json",
        dataset_dir / f"{split_name}_dataset.json",
    ]
    if split_name == "train":
        candidates.append(dataset_dir / "train_dataset.json")
    if split_name == "valid":
        candidates.extend([
            dataset_dir / "valid_dataset.json",
            dataset_dir / "val_dataset.json",
            dataset_dir / "test_dataset.json",
            dataset_dir / "train_dataset.json",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No FathomNet {split_name} COCO annotations found under {dataset_dir}"
    )


def _resolve_coco_image_path(dataset_dir, ann_path, file_name):
    candidates = [
        dataset_dir / file_name,
        ann_path.parent / file_name,
        dataset_dir / Path(file_name).name,
        ann_path.parent / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _collect_fathomnet_split(fathomnet_dir, split_name, target_dir, images,
                             annotations, next_image_id, next_annotation_id):
    ann_path = _find_coco_annotations(fathomnet_dir, split_name)
    with ann_path.open() as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    before_annotations = len(annotations)
    missing_images = []
    added_images = 0
    for image in coco.get("images", []):
        source_path = _resolve_coco_image_path(fathomnet_dir, ann_path, image["file_name"])
        if source_path is None:
            missing_images.append(str(fathomnet_dir / image["file_name"]))
            continue

        file_name = f"fathomnet_{split_name}_{Path(image['file_name']).name}"
        image_width = int(image.get("width") or 0)
        image_height = int(image.get("height") or 0)
        if image_width <= 0 or image_height <= 0:
            image_width, image_height = _image_size(source_path)
        _safe_symlink(source_path, target_dir / file_name)
        images.append(
            {
                "id": next_image_id,
                "width": image_width,
                "height": image_height,
                "file_name": file_name,
            }
        )
        added_images += 1

        boxes = []
        for ann in anns_by_image.get(image["id"], []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            clipped = _clip_bbox_xywh(*bbox, image_width, image_height)
            if clipped is not None:
                boxes.append(clipped)
        next_annotation_id = _add_annotations(
            annotations, boxes, next_image_id, next_annotation_id
        )
        next_image_id += 1

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} FathomNet {split_name} images referenced by JSON are missing. "
            f"First missing files:\n{preview}"
        )

    return {
        "images": added_images,
        "annotations": len(annotations) - before_annotations,
        "source_annotations": str(ann_path),
        "source_categories": len(coco.get("categories", [])),
        "next_image_id": next_image_id,
        "next_annotation_id": next_annotation_id,
    }


def _write_split_coco(split_dir, images, annotations):
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": CLASS_NAME, "supercategory": "sea_animal"}
        ],
        "info": {
            "description": "Experiment 13_2 DeepFish/FathomNet sea_animal dataset"
        },
        "licenses": [],
    }
    with (split_dir / "_annotations.coco.json").open("w") as f:
        json.dump(coco, f, indent=2)


def _prepare_merged_coco(deepfish_dir, fathomnet_train_dir, fathomnet_valid_dir,
                         output_dir, rebuild=True):
    coco_dir = output_dir / "_coco_sea_animal"
    train_dir = coco_dir / "train"
    valid_dir = coco_dir / "valid"
    train_ann_path = train_dir / "_annotations.coco.json"
    valid_ann_path = valid_dir / "_annotations.coco.json"
    if coco_dir.exists() and rebuild:
        shutil.rmtree(coco_dir)
    if train_ann_path.exists() and valid_ann_path.exists() and not rebuild:
        logger.info("Using existing merged COCO dataset: %s", coco_dir)
        return coco_dir, _MergedCocoDataset(coco_dir, "train"), _MergedCocoDataset(coco_dir, "valid")

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_images = []
    train_annotations = []
    next_train_image_id = 1
    next_train_annotation_id = 1

    logger.info("Collecting DeepFish train images from %s", deepfish_dir)
    deepfish_train_summary = _collect_deepfish_split(
        deepfish_dir, "train", train_dir, train_images, train_annotations,
        next_train_image_id, next_train_annotation_id,
    )
    next_train_image_id = deepfish_train_summary.pop("next_image_id")
    next_train_annotation_id = deepfish_train_summary.pop("next_annotation_id")

    logger.info("Collecting FathomNet train images from %s", fathomnet_train_dir)
    fathomnet_train_summary = _collect_fathomnet_split(
        fathomnet_train_dir, "train", train_dir, train_images, train_annotations,
        next_train_image_id, next_train_annotation_id,
    )
    fathomnet_train_summary.pop("next_image_id")
    fathomnet_train_summary.pop("next_annotation_id")

    valid_images = []
    valid_annotations = []
    next_valid_image_id = 1
    next_valid_annotation_id = 1

    logger.info("Collecting DeepFish valid images from %s", deepfish_dir)
    deepfish_valid_summary = _collect_deepfish_split(
        deepfish_dir, "valid", valid_dir, valid_images, valid_annotations,
        next_valid_image_id, next_valid_annotation_id,
    )
    next_valid_image_id = deepfish_valid_summary.pop("next_image_id")
    next_valid_annotation_id = deepfish_valid_summary.pop("next_annotation_id")

    logger.info("Collecting FathomNet valid images from %s", fathomnet_valid_dir)
    fathomnet_valid_summary = _collect_fathomnet_split(
        fathomnet_valid_dir, "valid", valid_dir, valid_images, valid_annotations,
        next_valid_image_id, next_valid_annotation_id,
    )
    fathomnet_valid_summary.pop("next_image_id")
    fathomnet_valid_summary.pop("next_annotation_id")

    _write_split_coco(train_dir, train_images, train_annotations)
    _write_split_coco(valid_dir, valid_images, valid_annotations)

    train_ds = _MergedCocoDataset(coco_dir, "train")
    valid_ds = _MergedCocoDataset(coco_dir, "valid")
    summary = {
        "class_name": CLASS_NAME,
        "deepfish_dir": str(deepfish_dir),
        "fathomnet_train_dir": str(fathomnet_train_dir),
        "fathomnet_valid_dir": str(fathomnet_valid_dir),
        "train": {
            "deepfish": deepfish_train_summary,
            "fathomnet": fathomnet_train_summary,
            "total_images": len(train_images),
            "total_annotations": len(train_annotations),
        },
        "valid": {
            "deepfish": deepfish_valid_summary,
            "fathomnet": fathomnet_valid_summary,
            "total_images": len(valid_images),
            "total_annotations": len(valid_annotations),
        },
        "output_dir": str(coco_dir),
    }
    with (coco_dir / "merge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Merged COCO dataset: train=%d images/%d ann, valid=%d images/%d ann -> %s",
        len(train_images), len(train_annotations),
        len(valid_images), len(valid_annotations), coco_dir,
    )
    return coco_dir, train_ds, valid_ds


class _MergedCocoDataset:
    def __init__(self, coco_dir, split_name, resolution=None):
        self.coco_dir = Path(coco_dir)
        self.split_name = split_name
        self.split_dir = self.coco_dir / split_name
        self.resolution = resolution
        self.class_names = [CLASS_NAME]
        self.num_classes = 1
        self._class_to_id = {CLASS_NAME: 0}
        with (self.split_dir / "_annotations.coco.json").open() as f:
            coco = json.load(f)
        self._images = list(coco.get("images", []))
        self._annotations_by_image = {}
        for ann in coco.get("annotations", []):
            self._annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_info = self._images[idx]
        image_path = self.split_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if self.resolution is not None:
            image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0

        boxes = []
        labels = []
        width = float(image_info.get("width") or orig_w)
        height = float(image_info.get("height") or orig_h)
        for ann in self._annotations_by_image.get(image_info["id"], []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            clipped = _clip_bbox_xywh(x, y, w, h, width, height)
            if clipped is None:
                continue
            x, y, w, h = clipped
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            bw = w / width
            bh = h / height
            boxes.append([cx, cy, bw, bh])
            labels.append(0)

        if boxes:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
        return image_np, {"boxes": boxes_arr, "labels": labels_arr}


class ExperimentTracker:
    """Per-epoch validation, checkpointing, plotting and summaries."""

    def __init__(
        self,
        model_ref,
        keras_model,
        criterion,
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
        num_classes,
        class_names,
        exp_dir,
        batch_size,
        total_epochs,
        conf_threshold=0.3,
        iou_threshold=0.5,
        val_eval_interval=1,
        train_eval_interval=1,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    ):
        self.model_ref = model_ref
        self.keras_model = keras_model
        self.val_model = keras_model
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.num_classes = num_classes
        self.class_names = class_names
        self.exp_dir = Path(exp_dir)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.val_eval_interval = val_eval_interval
        self.train_eval_interval = train_eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "plots").mkdir(parents=True, exist_ok=True)

        self.tracker = MetricsTracker(
            output_dir=str(self.exp_dir),
            model_name="rfdetr_nano",
            plot_interval=1,
        )
        self.best_val_loss = float("inf")
        self.early_best_val_loss = float("inf")
        self.early_stop_wait = 0

    def on_epoch_end(self, log_stats):
        epoch = log_stats.get("epoch", 0)
        train_loss = float(log_stats.get("train_loss", log_stats.get("loss", 0.0)))
        train_lr = log_stats.get("train_lr", log_stats.get("lr", 0.0))

        if not math.isfinite(train_loss):
            logger.error("NaN/Inf loss detected (%.4f) - requesting stop", train_loss)
            self.model_ref.request_early_stop()
            return

        run_val_eval = (
            self.val_eval_interval > 0
            and (epoch + 1) % self.val_eval_interval == 0
        )
        run_train_eval = (
            self.train_eval_interval > 0
            and (epoch + 1) % self.train_eval_interval == 0
        )

        if run_val_eval:
            logger.info("  Running evaluation on VAL set...")
            val_t0 = time.time()
            val_metrics = validate_epoch_full(
                model=self.val_model,
                criterion=self.criterion,
                dataset=self.val_dataset,
                indices=self.val_indices,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                logger=logger,
                prefix="val",
            )
            logger.info("  Val evaluation completed in %.1fs", time.time() - val_t0)
        else:
            val_metrics = {}
            logger.info("  Skipping VAL evaluation this epoch (interval=%d)", self.val_eval_interval)

        if run_train_eval:
            logger.info("  Running evaluation on TRAIN set...")
            train_eval_t0 = time.time()
            train_eval_metrics = validate_epoch_full(
                model=self.val_model,
                criterion=self.criterion,
                dataset=self.train_dataset,
                indices=self.train_indices,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                logger=None,
                prefix="train",
            )
            logger.info("  Train evaluation completed in %.1fs", time.time() - train_eval_t0)
        else:
            train_eval_metrics = {}
            logger.info("  Skipping TRAIN evaluation this epoch (interval=%d)", self.train_eval_interval)

        val_loss = val_metrics.get("val_loss", 0.0)
        val_mAP_50 = val_metrics.get("val_mAP_50", 0.0)
        val_mAP_50_95 = val_metrics.get("val_mAP_50_95", 0.0)
        val_precision = val_metrics.get("val_precision", 0.0)
        val_recall = val_metrics.get("val_recall", 0.0)
        val_f1 = val_metrics.get("val_f1", 0.0)
        val_accuracy = val_metrics.get("val_accuracy", 0.0)
        val_num_gt = val_metrics.get("val_num_gt_boxes", 0)
        val_num_pred = val_metrics.get("val_num_pred_boxes", 0)
        val_loss_ce = val_metrics.get("val_loss_ce", 0.0)
        val_loss_bbox = val_metrics.get("val_loss_bbox", 0.0)
        val_loss_giou = val_metrics.get("val_loss_giou", 0.0)

        train_mAP_50 = train_eval_metrics.get("train_mAP_50", 0.0)
        train_mAP_50_95 = train_eval_metrics.get("train_mAP_50_95", 0.0)
        train_precision = train_eval_metrics.get("train_precision", 0.0)
        train_recall = train_eval_metrics.get("train_recall", 0.0)
        train_f1 = train_eval_metrics.get("train_f1", 0.0)
        train_accuracy = train_eval_metrics.get("train_accuracy", 0.0)
        train_num_gt = train_eval_metrics.get("train_num_gt_boxes", 0)
        train_num_pred = train_eval_metrics.get("train_num_pred_boxes", 0)
        lr_val = float(train_lr) if isinstance(train_lr, (int, float)) else 0.0

        logger.info("")
        logger.info("-" * 60)
        logger.info("Epoch %d Summary", epoch)
        logger.info("-" * 60)
        logger.info("  LOSSES:")
        logger.info("    Train Loss (total) : %.4f", train_loss)
        logger.info("    Val Loss   (total) : %.4f", val_loss)
        logger.info("    Val   loss_ce      : %.4f", val_loss_ce)
        logger.info("    Val   loss_bbox    : %.4f", val_loss_bbox)
        logger.info("    Val   loss_giou    : %.4f", val_loss_giou)
        logger.info("  OPTIMIZATION:")
        logger.info("    Learning rate      : %.2e", lr_val)
        logger.info("  TRAIN EVALUATION:")
        logger.info("    mAP@50             : %.4f", train_mAP_50)
        logger.info("    mAP@50:95          : %.4f", train_mAP_50_95)
        logger.info("    Precision          : %.4f", train_precision)
        logger.info("    Recall             : %.4f", train_recall)
        logger.info("    F1 Score           : %.4f", train_f1)
        logger.info("    Accuracy           : %.4f", train_accuracy)
        logger.info("    GT Boxes           : %d", train_num_gt)
        logger.info("    Pred Boxes         : %d", train_num_pred)
        logger.info("  VAL EVALUATION:")
        logger.info("    mAP@50             : %.4f", val_mAP_50)
        logger.info("    mAP@50:95          : %.4f", val_mAP_50_95)
        logger.info("    Precision          : %.4f", val_precision)
        logger.info("    Recall             : %.4f", val_recall)
        logger.info("    F1 Score           : %.4f", val_f1)
        logger.info("    Accuracy           : %.4f", val_accuracy)
        logger.info("    GT Boxes           : %d", val_num_gt)
        logger.info("    Pred Boxes         : %d", val_num_pred)
        logger.info("-" * 60)

        self.tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mAP_50=val_mAP_50,
            val_mAP_50_95=val_mAP_50_95,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            val_num_gt_boxes=val_num_gt,
            val_num_pred_boxes=val_num_pred,
            train_mAP_50=train_mAP_50,
            train_mAP_50_95=train_mAP_50_95,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,
            train_accuracy=train_accuracy,
            train_num_gt_boxes=train_num_gt,
            train_num_pred_boxes=train_num_pred,
            learning_rate=lr_val,
            per_class_precision=val_metrics.get("per_class_precision"),
            per_class_recall=val_metrics.get("per_class_recall"),
            per_class_f1=val_metrics.get("per_class_f1"),
            per_class_ap50=val_metrics.get("per_class_ap50"),
            val_loss_ce=val_loss_ce,
            val_loss_bbox=val_loss_bbox,
            val_loss_giou=val_loss_giou,
        )

        ckpt_name = (
            f"rfdetr_nano_epoch_{epoch:04d}"
            f"_val_loss_{val_loss:.4f}"
            f"_mAP_{val_mAP_50:.4f}.weights.h5"
        )
        ckpt_path = self.ckpt_dir / ckpt_name

        if run_val_eval and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.keras_model.save_weights(str(ckpt_path))
            logger.info("  [Checkpoint] NEW BEST (val_loss=%.4f): %s", val_loss, ckpt_path)
            best_path = self.ckpt_dir / "rfdetr_nano_best.weights.h5"
            self.keras_model.save_weights(str(best_path))
            logger.info("  [Checkpoint] Updated best: %s", best_path)
        elif run_val_eval:
            logger.info(
                "  [Checkpoint] No improvement (current=%.4f, best=%.4f) - skipped",
                val_loss, self.best_val_loss,
            )
        else:
            logger.info("  [Checkpoint] Skipped because VAL was not evaluated")

        if run_val_eval:
            if val_loss < self.early_best_val_loss - self.early_stopping_min_delta:
                self.early_best_val_loss = val_loss
                self.early_stop_wait = 0
            else:
                self.early_stop_wait += 1
                logger.info(
                    "  [EarlyStopping] No val_loss improvement for %d/%d epochs",
                    self.early_stop_wait, self.early_stopping_patience,
                )
                if self.early_stop_wait >= self.early_stopping_patience:
                    logger.info("  [EarlyStopping] Requesting stop")
                    self.model_ref.request_early_stop()

        if self.tracker.should_plot(epoch, self.total_epochs):
            self.tracker.generate_plots()
            logger.info("  [Plots] Updated: %s", self.exp_dir / "plots")

    def on_train_end(self):
        self.tracker.generate_plots()
        logger.info("")
        logger.info("=" * 68)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 68)
        logger.info("  Best val loss     : %.4f", self.best_val_loss)
        logger.info("  Experiment dir    : %s", self.exp_dir)
        logger.info("  Checkpoints       : %s", self.ckpt_dir)
        logger.info("  Plots             : %s", self.exp_dir / "plots")
        logger.info("  Metrics log       : %s", self.tracker.log_path)
        if self.tracker.history["epoch"]:
            logger.info("\n%s", self.tracker.format_epoch_summary(-1))
        logger.info("=" * 68)


def main():
    exp_dir = _SCRIPT_DIR
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    setup_logging(str(exp_dir))
    sys.stdout = _TeeWriter(sys.stdout, exp_dir / "output.txt")

    import keras

    print("Keras backend:", keras.backend.backend())
    try:
        import jax
        print("JAX devices:", jax.devices())
    except Exception as exc:
        print("JAX device diagnostics unavailable:", exc)

    logger.info("=" * 68)
    logger.info("EXPERIMENT 13_2: RF-DETR Nano - DeepFish + FathomNet sea_animal")
    logger.info("=" * 68)

    deepfish_dir = _env_path("DEEPFISH_DIR", Path("/mnt/beegfs/home/jguo/datasets/Deepfish"))
    fathomnet_dir = _env_path("FATHOMNET_DIR", Path("/mnt/beegfs/home/jguo/datasets/fathomnet"))
    fathomnet_valid_dir = _env_path(
        "FATHOMNET_VALID_DIR", Path("/mnt/beegfs/home/jguo/datasets/images_fathom_test")
    )
    rebuild_dataset = bool(_env_int("RFDETR_REBUILD_DATASET", 1))

    batch_size = _env_int("RFDETR_BATCH_SIZE", 16)
    grad_accum_steps = _env_int("RFDETR_GRAD_ACCUM_STEPS", 1)
    num_workers = _env_int("RFDETR_NUM_WORKERS", 4)
    epochs = _env_int("RFDETR_EPOCHS", 20)
    train_eval_interval = _env_int("RFDETR_TRAIN_EVAL_INTERVAL", 1)
    val_eval_interval = _env_int("RFDETR_VAL_EVAL_INTERVAL", 1)
    early_stopping_patience = _env_int("RFDETR_EARLY_STOPPING_PATIENCE", 10)
    early_stopping_min_delta = _env_float("RFDETR_EARLY_STOPPING_MIN_DELTA", 0.001)
    base_lr = _env_float("RFDETR_LR", 1e-4)
    lr_encoder = _env_float("RFDETR_LR_ENCODER", 1.5e-4)
    warmup_epochs = _env_float("RFDETR_WARMUP_EPOCHS", 0.0)

    coco_dir, train_ds, val_ds = _prepare_merged_coco(
        deepfish_dir, fathomnet_dir, fathomnet_valid_dir, exp_dir,
        rebuild=rebuild_dataset,
    )
    train_ds.resolution = 384
    val_ds.resolution = 384
    train_indices = list(range(len(train_ds)))
    val_indices = list(range(len(val_ds)))

    logger.info(
        "COCO data: %d train, %d valid -> %s",
        len(train_indices), len(val_indices), coco_dir,
    )

    logger.info("Creating RFDETRNano (num_classes=1) ...")
    model = RFDETRNano(num_classes=1)
    dummy = np.ones((1, 384, 384, 3), dtype="float32") * 0.5
    model.model.model(dummy, training=True)
    model.model.class_names = [CLASS_NAME]
    logger.info(
        "Model ready - resolution=%d, group_detr=%d",
        model.model_config.resolution,
        model.model_config.group_detr,
    )

    config = TrainConfig(
        dataset_dir=str(coco_dir),
        dataset_file="coco_json",
        output_dir=str(exp_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=base_lr,
        lr_encoder=lr_encoder,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        lr_scheduler="cosine",
        lr_min_factor=0.0,
        warmup_epochs=warmup_epochs,
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
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=False,
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=num_workers,
        run_test=False,
        class_names=[CLASS_NAME],
    )

    val_criterion, _ = build_criterion_from_config(model.model_config, config)
    tracker = ExperimentTracker(
        model_ref=model,
        keras_model=model.model.model,
        criterion=val_criterion,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_indices=train_indices,
        val_indices=val_indices,
        num_classes=1,
        class_names=[CLASS_NAME],
        exp_dir=exp_dir,
        batch_size=batch_size,
        total_epochs=epochs,
        conf_threshold=0.3,
        iou_threshold=0.5,
        val_eval_interval=val_eval_interval,
        train_eval_interval=train_eval_interval,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    model.callbacks["on_fit_epoch_end"].append(tracker.on_epoch_end)
    model.callbacks["on_train_end"].append(tracker.on_train_end)

    exp_config = {
        "experiment": "experiment_13_2",
        "description": "RF-DETR Nano trained on DeepFish train + FathomNet train; validates on DeepFish valid + FathomNet validation as sea_animal",
        "variant": "RFDETRNano",
        "api": "high-level (RFDETRNano.train_from_config) with ExperimentTracker validation",
        "class_names": [CLASS_NAME],
        "num_classes": 1,
        "deepfish_dir": str(deepfish_dir),
        "fathomnet_dir": str(fathomnet_dir),
        "fathomnet_valid_dir": str(fathomnet_valid_dir),
        "coco_dir": str(coco_dir),
        "train_images": len(train_indices),
        "val_images": len(val_indices),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": base_lr,
        "lr_encoder": lr_encoder,
        "lr_scheduler": "cosine",
        "lr_min_factor": 0.0,
        "lr_component_decay": 0.7,
        "lr_vit_layer_decay": 0.8,
        "weight_decay": 1e-4,
        "warmup_epochs": warmup_epochs,
        "clip_max_norm": 0.1,
        "ema_decay": 0.993,
        "ema_tau": 100,
        "group_detr": model.model_config.group_detr,
        "resolution": model.model_config.resolution,
        "drop_path": 0.0,
        "multi_scale": False,
        "expanded_scales": False,
        "square_resize_div_64": True,
        "num_workers": num_workers,
        "amp": True,
        "built_in_eval_interval": 0,
        "built_in_eval_ema": False,
        "early_stopping": "ExperimentTracker val_loss patience",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "train_eval_interval": train_eval_interval,
        "val_eval_interval": val_eval_interval,
        "validation": "validate_epoch_full per epoch; all source labels remapped to sea_animal",
    }
    with (exp_dir / "experiment_config.json").open("w") as f:
        json.dump(exp_config, f, indent=2)
    logger.info("Config saved to %s", exp_dir / "experiment_config.json")
    for key, value in sorted(exp_config.items()):
        logger.info("  %-28s: %s", key, value)

    logger.info("Starting training ...")
    model.train_from_config(config)

    final_path = exp_dir / "checkpoints" / "rfdetr_nano_sea_animal_final.weights.h5"
    model.model.model.save_weights(str(final_path))
    logger.info("Final weights saved -> %s", final_path)


if __name__ == "__main__":
    main()
    sys.exit(0)
