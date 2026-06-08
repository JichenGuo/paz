#!/usr/bin/env python
"""Experiment 13: RF-DETR Large on DeepFish + FathomNet as ``sea_animal``.
"""

import json
import logging
import os
import shutil
import sys
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
from paz.models.detection.dino_v2_object_detection.detr import RFDETRLarge
from train_utils import setup_logging

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


def _collect_deepfish(deepfish_dir, target_dir, images, annotations,
                            next_image_id, next_annotation_id):
    if not deepfish_dir.exists():
        raise FileNotFoundError(f"DeepFish dataset directory not found: {deepfish_dir}")
    image_paths = sorted(
        path
        for path in deepfish_dir.glob("*/train/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No DeepFish training images found under {deepfish_dir}/*/train"
        )

    before_annotations = len(annotations)
    for source_path in image_paths:
        subset_name = source_path.parents[1].name
        file_name = f"deepfish_{subset_name}_{source_path.name}"
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


def _collect_fathomnet(fathomnet_dir, target_dir, images, annotations,
                             next_image_id, next_annotation_id):
    ann_path = fathomnet_dir / "train_dataset.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"FathomNet annotations not found: {ann_path}")
    with ann_path.open() as f:
        coco = json.load(f)

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

        file_name = f"fathomnet_{Path(image['file_name']).name}"
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


def _prepare_merged_coco(deepfish_dir, fathomnet_dir, output_dir, rebuild=True):
    coco_dir = output_dir / "_coco_sea_animal"
    train_dir = coco_dir / "train"
    ann_path = train_dir / "_annotations.coco.json"
    if coco_dir.exists() and rebuild:
        shutil.rmtree(coco_dir)
    if ann_path.exists() and not rebuild:
        logger.info("Using existing merged COCO dataset: %s", coco_dir)
        return coco_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    next_image_id = 1
    next_annotation_id = 1

    logger.info("Collecting DeepFish train images from %s", deepfish_dir)
    deepfish_summary = _collect_deepfish(
        deepfish_dir, train_dir, images, annotations, next_image_id, next_annotation_id
    )
    next_image_id = deepfish_summary.pop("next_image_id")
    next_annotation_id = deepfish_summary.pop("next_annotation_id")

    logger.info("Collecting FathomNet images from %s", fathomnet_dir)
    fathomnet_summary = _collect_fathomnet(
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
                "name": CLASS_NAME,
                "supercategory": "sea_animal",
            }
        ],
        "info": {
            "description": "Experiment 13 merged DeepFish/FathomNet sea_animal dataset"
        },
        "licenses": [],
    }
    with ann_path.open("w") as f:
        json.dump(coco, f, indent=2)

    summary = {
        "class_name": CLASS_NAME,
        "deepfish_dir": str(deepfish_dir),
        "fathomnet_dir": str(fathomnet_dir),
        "deepfish": deepfish_summary,
        "fathomnet": fathomnet_summary,
        "total_images": len(images),
        "total_annotations": len(annotations),
        "output_dir": str(coco_dir),
    }
    with (coco_dir / "merge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Merged COCO dataset: %d images, %d annotations -> %s",
        len(images),
        len(annotations),
        coco_dir,
    )
    return coco_dir


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
    logger.info("EXPERIMENT 13: RF-DETR Large — DeepFish + FathomNet sea_animal")
    logger.info("=" * 68)

    deepfish_dir = _env_path("DEEPFISH_DIR", Path("/mnt/beegfs/home/jguo/datasets/Deepfish"))
    fathomnet_dir = _env_path("FATHOMNET_DIR", Path("/mnt/beegfs/home/jguo/datasets/fathomnet"))
    rebuild_dataset = bool(_env_int("RFDETR_REBUILD_DATASET", 1))

    batch_size = _env_int("RFDETR_BATCH_SIZE", 16)
    grad_accum_steps = _env_int("RFDETR_GRAD_ACCUM_STEPS", 1)
    num_workers = _env_int("RFDETR_NUM_WORKERS", 4)
    epochs = _env_int("RFDETR_EPOCHS", 20)
    base_lr = _env_float("RFDETR_LR", 1e-4)
    lr_encoder = _env_float("RFDETR_LR_ENCODER", 1.5e-4)
    warmup_epochs = _env_float("RFDETR_WARMUP_EPOCHS", 0.0)

    coco_dir = _prepare_merged_coco(
        deepfish_dir, fathomnet_dir, exp_dir, rebuild=rebuild_dataset
    )

    logger.info("Creating RFDETRLarge (num_classes=1) ...")
    model = RFDETRLarge(num_classes=1)
    dummy = np.ones((1, 704, 704, 3), dtype="float32") * 0.5
    model.model.model(dummy, training=True)
    model.model.class_names = [CLASS_NAME]

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
        eval_interval=0,
        eval_ema=False,
        amp=True,
        num_workers=num_workers,
        run_test=False,
        class_names=[CLASS_NAME],
    )

    exp_config = {
        "experiment": "experiment_13",
        "description": "RF-DETR Large trained on DeepFish train + FathomNet as sea_animal",
        "variant": "RFDETRLarge",
        "class_names": [CLASS_NAME],
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
        json.dump(exp_config, f, indent=2)

    logger.info("Starting training ...")
    model.train_from_config(config)

    final_path = exp_dir / "checkpoints" / "rfdetr_large_sea_animal_final.weights.h5"
    model.model.model.save_weights(str(final_path))
    logger.info("Final weights saved -> %s", final_path)


if __name__ == "__main__":
    main()
    sys.exit(0)
