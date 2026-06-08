#!/usr/bin/env python
"""Experiment 12: RF-DETR Large on DeepFish + FathomNet as ``sea_animal``.

This active implementation is intentionally placed before the older copied
Experiment 11 body below.  When run as a script, it prepares a merged
train-only COCO dataset and exits after training, so the legacy copy below is
not executed.
"""

import json as _json
import logging as _logging
import os as _os
import shutil as _shutil
import sys as _sys
from pathlib import Path as _Path

_os.environ["KERAS_BACKEND"] = "jax"
_os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
_os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as _np
from PIL import Image as _Image

_EXP12_SCRIPT_DIR = _Path(__file__).resolve().parent
_EXP12_PAZ_ROOT = _EXP12_SCRIPT_DIR.parents[3]
_EXP12_SRC_DIR = _EXP12_SCRIPT_DIR.parents[1] / "src"
if str(_EXP12_PAZ_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_EXP12_PAZ_ROOT))
if str(_EXP12_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_EXP12_SRC_DIR))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig as _TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRLarge as _RFDETRLarge
from train_utils import setup_logging as _setup_logging

_EXP12_LOGGER = _logging.getLogger(__name__)
_EXP12_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_EXP12_CLASS_NAME = "sea_animal"


class _Exp12TeeWriter:
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

    def __getattr__(self, name):
        return getattr(self._original, name)


def _exp12_env_int(name, default):
    value = _os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _exp12_env_float(name, default):
    value = _os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _exp12_env_path(name, default):
    value = _os.environ.get(name)
    if value is None or value == "":
        return _Path(default).expanduser().resolve()
    return _Path(value).expanduser().resolve()


def _exp12_safe_symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _exp12_image_size(path):
    with _Image.open(path) as image:
        return image.size


def _exp12_clip_bbox_xywh(x, y, w, h, image_width, image_height):
    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x) + float(w), float(image_width)))
    y2 = max(0.0, min(float(y) + float(h), float(image_height)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    return [x1, y1, bw, bh]


def _exp12_read_yolo_boxes(label_path, image_width, image_height):
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
            clipped = _exp12_clip_bbox_xywh(
                abs_x, abs_y, abs_w, abs_h, image_width, image_height
            )
            if clipped is not None:
                boxes.append(clipped)
    return boxes


def _exp12_add_annotations(annotations, boxes, image_id, next_annotation_id):
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


def _exp12_collect_deepfish(deepfish_dir, target_dir, images, annotations,
                            next_image_id, next_annotation_id):
    if not deepfish_dir.exists():
        raise FileNotFoundError(f"DeepFish dataset directory not found: {deepfish_dir}")
    image_paths = sorted(
        path
        for path in deepfish_dir.glob("*/train/*")
        if path.is_file() and path.suffix.lower() in _EXP12_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No DeepFish training images found under {deepfish_dir}/*/train"
        )

    before_annotations = len(annotations)
    for source_path in image_paths:
        subset_name = source_path.parents[1].name
        file_name = f"deepfish_{subset_name}_{source_path.name}"
        width, height = _exp12_image_size(source_path)
        _exp12_safe_symlink(source_path, target_dir / file_name)
        images.append(
            {
                "id": next_image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
            }
        )
        boxes = _exp12_read_yolo_boxes(source_path.with_suffix(".txt"), width, height)
        next_annotation_id = _exp12_add_annotations(
            annotations, boxes, next_image_id, next_annotation_id
        )
        next_image_id += 1

    return {
        "images": len(image_paths),
        "annotations": len(annotations) - before_annotations,
        "next_image_id": next_image_id,
        "next_annotation_id": next_annotation_id,
    }


def _exp12_collect_fathomnet(fathomnet_dir, target_dir, images, annotations,
                             next_image_id, next_annotation_id):
    ann_path = fathomnet_dir / "train_dataset.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"FathomNet annotations not found: {ann_path}")
    with ann_path.open() as f:
        coco = _json.load(f)

    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    before_annotations = len(annotations)
    missing_images = []
    for image in coco.get("images", []):
        source_path = fathomnet_dir / image["file_name"]
        if not source_path.exists():
            missing_images.append(str(source_path))
            continue

        file_name = f"fathomnet_{_Path(image['file_name']).name}"
        image_width = int(image.get("width") or 0)
        image_height = int(image.get("height") or 0)
        if image_width <= 0 or image_height <= 0:
            image_width, image_height = _exp12_image_size(source_path)
        _exp12_safe_symlink(source_path, target_dir / file_name)
        images.append(
            {
                "id": next_image_id,
                "width": image_width,
                "height": image_height,
                "file_name": file_name,
            }
        )

        boxes = []
        for ann in anns_by_image.get(image["id"], []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            clipped = _exp12_clip_bbox_xywh(*bbox, image_width, image_height)
            if clipped is not None:
                boxes.append(clipped)
        next_annotation_id = _exp12_add_annotations(
            annotations, boxes, next_image_id, next_annotation_id
        )
        next_image_id += 1

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} FathomNet images referenced by JSON are missing. "
            f"First missing files:\n{preview}"
        )

    return {
        "images": len(coco.get("images", [])),
        "annotations": len(annotations) - before_annotations,
        "source_categories": len(coco.get("categories", [])),
        "next_image_id": next_image_id,
        "next_annotation_id": next_annotation_id,
    }


def _exp12_prepare_merged_coco(deepfish_dir, fathomnet_dir, output_dir, rebuild=True):
    coco_dir = output_dir / "_coco_sea_animal"
    train_dir = coco_dir / "train"
    ann_path = train_dir / "_annotations.coco.json"
    if coco_dir.exists() and rebuild:
        _shutil.rmtree(coco_dir)
    if ann_path.exists() and not rebuild:
        _EXP12_LOGGER.info("Using existing merged COCO dataset: %s", coco_dir)
        return coco_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    next_image_id = 1
    next_annotation_id = 1

    _EXP12_LOGGER.info("Collecting DeepFish train images from %s", deepfish_dir)
    deepfish_summary = _exp12_collect_deepfish(
        deepfish_dir, train_dir, images, annotations, next_image_id, next_annotation_id
    )
    next_image_id = deepfish_summary.pop("next_image_id")
    next_annotation_id = deepfish_summary.pop("next_annotation_id")

    _EXP12_LOGGER.info("Collecting FathomNet images from %s", fathomnet_dir)
    fathomnet_summary = _exp12_collect_fathomnet(
        fathomnet_dir, train_dir, images, annotations, next_image_id, next_annotation_id
    )
    fathomnet_summary.pop("next_image_id")
    fathomnet_summary.pop("next_annotation_id")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 0,
                "name": _EXP12_CLASS_NAME,
                "supercategory": "sea_animal",
            }
        ],
        "info": {
            "description": "Experiment 12 merged DeepFish/FathomNet sea_animal dataset"
        },
        "licenses": [],
    }
    with ann_path.open("w") as f:
        _json.dump(coco, f, indent=2)

    summary = {
        "class_name": _EXP12_CLASS_NAME,
        "deepfish_dir": str(deepfish_dir),
        "fathomnet_dir": str(fathomnet_dir),
        "deepfish": deepfish_summary,
        "fathomnet": fathomnet_summary,
        "total_images": len(images),
        "total_annotations": len(annotations),
        "output_dir": str(coco_dir),
    }
    with (coco_dir / "merge_summary.json").open("w") as f:
        _json.dump(summary, f, indent=2)

    _EXP12_LOGGER.info(
        "Merged COCO dataset: %d images, %d annotations -> %s",
        len(images),
        len(annotations),
        coco_dir,
    )
    return coco_dir


def _exp12_main():
    exp_dir = _EXP12_SCRIPT_DIR
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    _setup_logging(str(exp_dir))
    _sys.stdout = _Exp12TeeWriter(_sys.stdout, exp_dir / "output.txt")

    import keras as _keras

    print("Keras backend:", _keras.backend.backend())
    try:
        import jax as _jax

        print("JAX devices:", _jax.devices())
    except Exception as exc:
        print("JAX device diagnostics unavailable:", exc)

    _EXP12_LOGGER.info("=" * 68)
    _EXP12_LOGGER.info("EXPERIMENT 12: RF-DETR Large — DeepFish + FathomNet sea_animal")
    _EXP12_LOGGER.info("=" * 68)

    deepfish_dir = _exp12_env_path("DEEPFISH_DIR", _Path("/mnt/beegfs/home/jguo/datasets/Deepfish"))
    fathomnet_dir = _exp12_env_path("FATHOMNET_DIR", _Path("/mnt/beegfs/home/jguo/datasets/fathomnet"))
    rebuild_dataset = bool(_exp12_env_int("RFDETR_REBUILD_DATASET", 1))

    batch_size = _exp12_env_int("RFDETR_BATCH_SIZE", 16)
    grad_accum_steps = _exp12_env_int("RFDETR_GRAD_ACCUM_STEPS", 1)
    num_workers = _exp12_env_int("RFDETR_NUM_WORKERS", 4)
    epochs = _exp12_env_int("RFDETR_EPOCHS", 20)
    base_lr = _exp12_env_float("RFDETR_LR", 1e-4)
    lr_encoder = _exp12_env_float("RFDETR_LR_ENCODER", 1.5e-4)
    warmup_epochs = _exp12_env_float("RFDETR_WARMUP_EPOCHS", 0.0)

    coco_dir = _exp12_prepare_merged_coco(
        deepfish_dir, fathomnet_dir, exp_dir, rebuild=rebuild_dataset
    )

    _EXP12_LOGGER.info("Creating RFDETRLarge (num_classes=1) ...")
    model = _RFDETRLarge(num_classes=1)
    dummy = _np.ones((1, 704, 704, 3), dtype="float32") * 0.5
    model.model.model(dummy, training=True)
    model.model.class_names = [_EXP12_CLASS_NAME]

    config = _TrainConfig(
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
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=num_workers,
        run_test=False,
        class_names=[_EXP12_CLASS_NAME],
    )

    exp_config = {
        "experiment": "experiment_12",
        "description": "RF-DETR Large trained on DeepFish train + FathomNet as sea_animal",
        "variant": "RFDETRLarge",
        "class_names": [_EXP12_CLASS_NAME],
        "num_classes": 1,
        "deepfish_dir": str(deepfish_dir),
        "fathomnet_dir": str(fathomnet_dir),
        "coco_dir": str(coco_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": base_lr,
        "lr_encoder": lr_encoder,
        "warmup_epochs": warmup_epochs,
        "resolution": model.model_config.resolution,
        "validation": "disabled; all requested source images are used for training",
    }
    with (exp_dir / "experiment_config.json").open("w") as f:
        _json.dump(exp_config, f, indent=2)

    _EXP12_LOGGER.info("Starting training ...")
    model.train_from_config(config)

    final_path = exp_dir / "checkpoints" / "rfdetr_large_sea_animal_final.weights.h5"
    model.model.model.save_weights(str(final_path))
    _EXP12_LOGGER.info("Final weights saved -> %s", final_path)


if __name__ == "__main__":
    _exp12_main()
    _sys.exit(0)

"""Experiment 12: RF-DETR Large — native RF-DETR augmentation benchmark.
Trained on both FathomNet and DeepFish datasets
Identical to Experiment 5 in every hyperparameter, but replaces the
custom ``pipeline2`` augmentation with the native RF-DETR augmentation
pipeline (``make_coco_transforms_square_div_64``).

Native RF-DETR train augmentation used here
(square_resize_div_64=True, multi_scale=False):
  - RandomHorizontalFlip  (p = 0.5)
  - RandomSelect(
        SquareResize([384]),
        Compose([
            RandomResize([400, 500, 600]),
            RandomSizeCrop(384, 600),
            SquareResize([384]),
        ])
    )
  - ToTensor + Normalize(ImageNet mean/std)

Differences vs Experiment 5:
  - No custom _AugmentedDataLoader / DetectionDataGenerator for training.
    ``train_from_config`` builds the native COCO data loader internally.
  - RandomSizeCrop augmentation adds spatial diversity that pipeline2 lacks.
  - LR-schedule monkey-patch removed: ``detr.py`` now builds the schedule.
  - No engine patch needed: upstream engine.py already uses training=True
    in Phase 1 (bug fixed in PAZ codebase).

Hyperparameters (identical to Experiment 5):
  - Variant:        RFDETRLarge
  - Resolution:     704 × 704
  - Epochs:         20 by default (override with RFDETR_EPOCHS)
  - Batch size:     16 (grad_accum_steps=1, effective batch = 16)

  - LR:             1e-4 cosine → 0   (warmup = 0 epochs)
  - LR encoder:     1.5e-4
  - Weight decay:   1e-4
  - EMA decay:      0.993
  - group_detr:     13
  - drop_path:      0.0
  - Early stopping: patience=10
"""

import os
# Must be set before importing keras/paz. ``setdefault`` is not enough here:
# cluster/container environments may already define KERAS_BACKEND=tensorflow,
# which would build tf.Variable weights that JAX cannot differentiate.
os.environ["KERAS_BACKEND"] = "jax"
import sys
import json
import math
import time
import datetime
import logging

import numpy as np
# from paz.datasets.deepfish import download
# download()


# ---------------------------------------------------------------------------
# Tee stdout → log file so MetricLogger per-step prints appear in output.txt
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Duplicate writes to both the original stream and a file."""

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


# ---------------------------------------------------------------------------
# Path setup (must come before framework imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAZ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "src"))
if _PAZ_ROOT not in sys.path:
    sys.path.insert(0, _PAZ_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


import keras
print("Keras backend:", keras.backend.backend())
try:
    import jax
    print("JAX devices:", jax.devices())
except Exception as exc:
    print("JAX device diagnostics unavailable:", exc)

# ---------------------------------------------------------------------------
# Project imports (after path setup and engine patch)
# ---------------------------------------------------------------------------
from paz.models.detection.dino_v2_object_detection.detr import RFDETRLarge
from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)
from dataset import DeepFishDataset
from train_utils import prepare_coco_dataset, setup_logging, validate_epoch_full
from metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


def _env_int(name, default):
    """Read a positive integer from the environment."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed



# =====================================================================
# Experiment tracker — validation, checkpoints, plots, summaries
# =====================================================================

class ExperimentTracker:
    """Per-epoch validation, checkpointing, plotting and summaries.

    Plugs into ``train_from_config`` via ``on_fit_epoch_end`` /
    ``on_train_end`` callbacks.
    """

    def __init__(
        self,
        model_ref,
        keras_model,
        criterion,
        dataset,
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
        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.num_classes = num_classes
        self.class_names = class_names
        self.exp_dir = exp_dir
        self.ckpt_dir = os.path.join(exp_dir, "checkpoints")
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.val_eval_interval = val_eval_interval
        self.train_eval_interval = train_eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

        self.tracker = MetricsTracker(
            output_dir=exp_dir,
            model_name="rfdetr_large",
            plot_interval=1,
        )
        self.best_val_loss = float("inf")
        self.early_best_val_loss = float("inf")
        self.early_stop_wait = 0

    # ------------------------------------------------------------------

    def on_epoch_end(self, log_stats):
        epoch = log_stats.get("epoch", 0)
        train_loss = float(
            log_stats.get("train_loss", log_stats.get("loss", 0.0))
        )
        train_lr = log_stats.get("train_lr", log_stats.get("lr", 0.0))

        # NaN / Inf guard
        if not math.isfinite(train_loss):
            logger.error(
                "NaN/Inf loss detected (%.4f) — requesting stop", train_loss
            )
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
            # Validation on VAL set
            logger.info("  Running evaluation on VAL set...")
            val_t0 = time.time()
            val_metrics = validate_epoch_full(
                model=self.val_model,
                criterion=self.criterion,
                dataset=self.dataset,
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
            logger.info(
                "  Skipping VAL evaluation this epoch "
                "(interval=%d)", self.val_eval_interval,
            )

        if run_train_eval:
            # Validation on TRAIN set (monitor overfitting)
            logger.info("  Running evaluation on TRAIN set...")
            train_eval_t0 = time.time()
            train_eval_metrics = validate_epoch_full(
                model=self.val_model,
                criterion=self.criterion,
                dataset=self.dataset,
                indices=self.train_indices,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                logger=None,
                prefix="train",
            )
            logger.info(
                "  Train evaluation completed in %.1fs",
                time.time() - train_eval_t0,
            )
        else:
            train_eval_metrics = {}
            logger.info(
                "  Skipping TRAIN evaluation this epoch "
                "(interval=%d)", self.train_eval_interval,
            )

        # Extract metrics
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

        # Epoch summary
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

        # MetricsTracker
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

        # Checkpointing (best_keep strategy)
        ckpt_name = (
            f"rfdetr_large_epoch_{epoch:04d}"
            f"_val_loss_{val_loss:.4f}"
            f"_mAP_{val_mAP_50:.4f}.weights.h5"
        )
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)

        if run_val_eval and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.keras_model.save_weights(ckpt_path)
            logger.info(
                "  [Checkpoint] NEW BEST (val_loss=%.4f): %s",
                val_loss, ckpt_path,
            )
            best_path = os.path.join(self.ckpt_dir, "rfdetr_large_best.weights.h5")
            self.keras_model.save_weights(best_path)
            logger.info("  [Checkpoint] Updated best: %s", best_path)
        elif run_val_eval:
            logger.info(
                "  [Checkpoint] No improvement "
                "(current=%.4f, best=%.4f) — skipped",
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


        # Plots
        if self.tracker.should_plot(epoch, self.total_epochs):
            self.tracker.generate_plots()
            logger.info(
                "  [Plots] Updated: %s",
                os.path.join(self.exp_dir, "plots"),
            )

    # ------------------------------------------------------------------

    def on_train_end(self):
        self.tracker.generate_plots()
        logger.info("")
        logger.info("=" * 68)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 68)
        logger.info("  Best val loss     : %.4f", self.best_val_loss)
        logger.info("  Experiment dir    : %s", self.exp_dir)
        logger.info("  Checkpoints       : %s", self.ckpt_dir)
        logger.info(
            "  Plots             : %s",
            os.path.join(self.exp_dir, "plots"),
        )
        logger.info("  Metrics log       : %s", self.tracker.log_path)
        if self.tracker.history["epoch"]:
            logger.info("\n%s", self.tracker.format_epoch_summary(-1))
        logger.info("=" * 68)


# =====================================================================
# Main
# =====================================================================

def main():
    EXP_DIR = _SCRIPT_DIR
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "plots"), exist_ok=True)
    setup_logging(EXP_DIR)

    sys.stdout = _TeeWriter(sys.stdout, os.path.join(EXP_DIR, "output.txt"))

    logger.info("=" * 68)
    logger.info("EXPERIMENT 12: RF-DETR Large — Native RF-DETR Augmentation")
    logger.info("=" * 68)

    # RF-DETR's loader uses batch_size * grad_accum_steps, then the engine
    # splits that batch into grad_accum_steps micro-batches.  So these defaults
    # keep effective batch = 16 while increasing the actual GPU micro-batch
    # from 4 to 16 versus the previous 4 x 4 setup.
    BATCH_SIZE = _env_int("RFDETR_BATCH_SIZE", 16)
    GRAD_ACCUM_STEPS = _env_int("RFDETR_GRAD_ACCUM_STEPS", 1)
    NUM_WORKERS = _env_int("RFDETR_NUM_WORKERS", 4)
    EPOCHS = _env_int("RFDETR_EPOCHS", 20)
    BASE_LR = 1e-4
    WARMUP_EPOCHS = 0.0
    TRAIN_EVAL_INTERVAL = 1


    # Load dataset: resolution=None for COCO creation (native loader resizes),
    # eval_ds resolution=704 for validate_epoch_full (uses make_batches).
    logger.info("Loading DeepFish dataset ...")
    ds = DeepFishDataset(resolution=None)
    eval_ds = DeepFishDataset(resolution=704)
    logger.info("DeepFish: %d images, %d classes %s",
                len(ds), ds.num_classes, ds.class_names)

    coco_dir, train_indices, val_indices = prepare_coco_dataset(
        ds, EXP_DIR, val_split=0.2, seed=42,
    )
    logger.info("COCO data: %d train, %d val -> %s",
                len(train_indices), len(val_indices), coco_dir)

    # build_roboflow expects {dataset_dir}/valid/ (not val/)
    valid_link = os.path.join(coco_dir, "valid")
    val_dir = os.path.join(coco_dir, "val")
    if os.path.isdir(val_dir) and not os.path.exists(valid_link):
        os.symlink(os.path.abspath(val_dir), valid_link)
        logger.info("Created symlink: valid -> val")

    logger.info("Creating RFDETRLarge (num_classes=1) ...")
    model = RFDETRLarge(num_classes=1)

    # Warm the training-mode JAX trace before the loop starts
    _dummy = np.ones((1, 704, 704, 3), dtype="float32") * 0.5
    model.model.model(_dummy, training=True)
    logger.info("Model ready — resolution=%d, group_detr=%d",
                model.model_config.resolution,
                model.model_config.group_detr)

    # Build criterion for validate_epoch_full
    config = TrainConfig(
        dataset_dir=coco_dir,
        dataset_file="coco_json",
        output_dir=EXP_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps= GRAD_ACCUM_STEPS,
        lr=BASE_LR,
        lr_encoder=1.5e-4,
        lr_component_decay=0.7,
        lr_vit_layer_decay=0.8,
        lr_scheduler="cosine",
        lr_min_factor=0.0,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        drop_path=0.0,
        # Native RF-DETR augmentation — fixed 384 square, no multi-scale resize
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        # Other flags
        checkpoint_interval=10,
        # ExperimentTracker handles val-loss early stopping.  Built-in
        # validation is disabled below to avoid duplicate full COCO evals.
        early_stopping=False,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=NUM_WORKERS,
        run_test=False,
        class_names=eval_ds.class_names,
    )

    val_criterion, _ = build_criterion_from_config(model.model_config, config)

    # Register experiment tracker callbacks
    tracker = ExperimentTracker(
        model_ref=model,
        keras_model=model.model.model,
        criterion=val_criterion,
        dataset=eval_ds,
        train_indices=train_indices,
        val_indices=val_indices,
        num_classes=eval_ds.num_classes,
        class_names=eval_ds.class_names,
        exp_dir=EXP_DIR,
        batch_size=BATCH_SIZE,
        total_epochs=EPOCHS,
        conf_threshold=0.3,
        iou_threshold=0.5,
        val_eval_interval=1,
        train_eval_interval=TRAIN_EVAL_INTERVAL,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
    )
    model.callbacks["on_fit_epoch_end"].append(tracker.on_epoch_end)
    model.callbacks["on_train_end"].append(tracker.on_train_end)

    # Save config for reproducibility
    exp_config = {
        "experiment": "experiment_11",
        "description": (
            "RF-DETR Large with native RF-DETR augmentation "
            "(RandomHorizontalFlip + RandomSizeCrop + SquareResize); "
            "all other hyperparameters identical to Experiment 5"
        ),
        "variant": "RFDETRLarge",
        "api": "high-level (RFDETRLarge.train_from_config)",
        "augmentation": "native RF-DETR (make_coco_transforms_square_div_64)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "lr": BASE_LR,
        "lr_encoder": 1.5e-4,
        "lr_scheduler": "cosine",
        "lr_min_factor": 0.0,
        "lr_component_decay": 0.7,
        "lr_vit_layer_decay": 0.8,
        "weight_decay": 1e-4,
        "warmup_epochs": WARMUP_EPOCHS,
        "clip_max_norm": 0.1,
        "ema_decay": 0.993,
        "ema_tau": 100,
        "group_detr": model.model_config.group_detr,
        "resolution": model.model_config.resolution,
        "drop_path": 0.0,
        "multi_scale": False,
        "expanded_scales": False,
        "square_resize_div_64": True,
        "num_workers": NUM_WORKERS,
        "amp": True,
        "built_in_eval_interval": 0,
        "built_in_eval_ema": False,
        "early_stopping": "ExperimentTracker val_loss patience",
        "early_stopping_patience": 10,
        "train_eval_interval": TRAIN_EVAL_INTERVAL,
        "val_split": 0.2,
        "seed": 42,
        "dataset": "DeepFish",
        "train_images": len(train_indices),
        "val_images": len(val_indices),
        "validation": "validate_epoch_full per epoch (val + train-eval)",
        "monkey_patches": [
            "engine.train_one_epoch: training=True in Phase 1 eager forward",
            "warm training-mode JAX trace after model init",
        ],
        "vs_experiment_5": (
            "Same hyperparameters; augmentation changed from pipeline2 "
            "(hflip + color jitter) to native RF-DETR "
            "(hflip + RandomSizeCrop + SquareResize)"
        ),
    }
    config_path = os.path.join(EXP_DIR, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    logger.info("Config saved to %s", config_path)

    for k, v in sorted(exp_config.items()):
        logger.info("  %-28s: %s", k, v)

    # Train — native COCO loader is built internally by train_from_config
    # (no data_loader_train argument → _build_data_loader is called with
    #  the coco_dir and make_coco_transforms_square_div_64 augmentation).
    logger.info("Starting training ...")
    model.train_from_config(config)

    final_path = os.path.join(EXP_DIR, "checkpoints", "rfdetr_large_final.weights.h5")
    model.model.model.save_weights(final_path)
    logger.info("Final weights saved -> %s", final_path)


if __name__ == "__main__":
    main()
