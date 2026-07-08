#!/usr/bin/env python
"""Experiment 15: RF-DETR Large on DeepFish + OzFish as ``fish``.

This experiment prepares a COCO-style dataset with two splits:

    _coco_fish/
      train/  DeepFish train images + 80% of OzFish frames
      valid/  remaining 20% of OzFish frames

OzFish annotations are read from SageMaker GroundTruth JSON Lines manifests.
"""

import json
import logging
import os
import random
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
CLASS_NAME = "fish"
OZFISH_TRAIN_RATIO = 0.80


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


def _collect_deepfish(
    deepfish_dir,
    target_dir,
    images,
    annotations,
    next_image_id,
    next_annotation_id,
):
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


def _source_key_from_image_name(filename):
    lower_name = filename.lower()
    first_extension_index = None
    first_extension = None
    for extension in IMAGE_EXTENSIONS:
        index = lower_name.find(extension)
        if index == -1:
            continue
        if first_extension_index is None or index < first_extension_index:
            first_extension_index = index
            first_extension = extension
    if first_extension_index is None:
        return filename
    return filename[:first_extension_index + len(first_extension)]


def _build_ozfish_image_index(ozfish_images_dir):
    if not ozfish_images_dir.exists():
        raise FileNotFoundError(
            f"OzFish frames_labelled directory not found: {ozfish_images_dir}"
        )
    image_index = {}
    for image_path in sorted(ozfish_images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source_key = _source_key_from_image_name(image_path.name)
        image_index.setdefault(source_key, image_path)
    if not image_index:
        raise FileNotFoundError(f"No OzFish images found under {ozfish_images_dir}")
    return image_index


def _get_ozfish_payload(record):
    for value in record.values():
        if not isinstance(value, dict):
            continue
        if "annotations" in value and "image_size" in value:
            return value
    return None


def _load_ozfish_records(ozfish_images_dir, ozfish_manifests_dir, manifest_glob):
    if not ozfish_manifests_dir.exists():
        raise FileNotFoundError(
            f"OzFish manifests directory not found: {ozfish_manifests_dir}"
        )
    manifest_paths = sorted(
        path for path in ozfish_manifests_dir.glob(manifest_glob) if path.is_file()
    )
    if not manifest_paths:
        raise FileNotFoundError(
            f"No OzFish manifest files matched {ozfish_manifests_dir / manifest_glob}"
        )

    image_index = _build_ozfish_image_index(ozfish_images_dir)
    records = []
    missing_images = []
    for manifest_path in manifest_paths:
        with manifest_path.open() as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {manifest_path}:{line_number}"
                    ) from error
                source_ref = Path(record.get("source-ref", "")).name
                payload = _get_ozfish_payload(record)
                if payload is None:
                    continue
                image_path = image_index.get(source_ref)
                if image_path is None:
                    missing_images.append(f"{manifest_path}:{line_number} {source_ref}")
                    continue
                records.append(
                    {
                        "source_ref": source_ref,
                        "image_path": image_path,
                        "payload": payload,
                    }
                )

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} OzFish manifest rows reference missing images. "
            f"First missing rows:\n{preview}"
        )
    if not records:
        raise FileNotFoundError(
            f"No usable OzFish manifest rows found in {ozfish_manifests_dir}"
        )
    return records


def _split_records(records, train_ratio, seed):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"OZFISH_TRAIN_RATIO must be between 0 and 1, got {train_ratio}")
    if len(records) < 2:
        raise ValueError("Need at least 2 OzFish records to create train/valid splits")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    train_count = max(1, min(train_count, len(shuffled) - 1))
    return shuffled[:train_count], shuffled[train_count:]


def _ozfish_boxes(payload, image_width, image_height):
    image_size = payload.get("image_size", [{}])[0]
    manifest_width = float(image_size.get("width") or image_width)
    manifest_height = float(image_size.get("height") or image_height)
    x_scale = image_width / manifest_width
    y_scale = image_height / manifest_height

    boxes = []
    for annotation in payload.get("annotations", []):
        clipped = _clip_bbox_xywh(
            float(annotation["left"]) * x_scale,
            float(annotation["top"]) * y_scale,
            float(annotation["width"]) * x_scale,
            float(annotation["height"]) * y_scale,
            image_width,
            image_height,
        )
        if clipped is not None:
            boxes.append(clipped)
    return boxes


def _collect_ozfish(
    records,
    target_dir,
    split_name,
    images,
    annotations,
    next_image_id,
    next_annotation_id,
):
    before_annotations = len(annotations)
    for record in records:
        source_path = record["image_path"]
        batch_name = source_path.parent.name
        file_name = f"ozfish_{split_name}_{batch_name}_{source_path.name}"
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
        boxes = _ozfish_boxes(record["payload"], image_width, image_height)
        next_annotation_id = _add_annotations(
            annotations, boxes, next_image_id, next_annotation_id
        )
        next_image_id += 1

    return {
        "images": len(records),
        "annotations": len(annotations) - before_annotations,
        "next_image_id": next_image_id,
        "next_annotation_id": next_annotation_id,
    }


def _make_coco(images, annotations):
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 0,
                "name": CLASS_NAME,
                "supercategory": "fish",
            }
        ],
        "info": {
            "description": "Experiment 15 DeepFish/OzFish fish dataset"
        },
        "licenses": [],
    }


def _write_coco(split_dir, images, annotations):
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "_annotations.coco.json").open("w") as f:
        json.dump(_make_coco(images, annotations), f, indent=2)


def _copy_best_checkpoints(exp_dir):
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_aliases = [
        (
            exp_dir / "checkpoint_best_total.weights.h5",
            checkpoint_dir / "rfdetr_large_fish_best.weights.h5",
        ),
        (
            exp_dir / "checkpoint_best_regular.weights.h5",
            checkpoint_dir / "rfdetr_large_fish_best_regular.weights.h5",
        ),
        (
            exp_dir / "checkpoint_best_ema.weights.h5",
            checkpoint_dir / "rfdetr_large_fish_best_ema.weights.h5",
        ),
    ]

    copied = []
    for source_path, target_path in checkpoint_aliases:
        if not source_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
        logger.info("Best checkpoint alias saved -> %s", target_path)

    if not copied:
        logger.warning(
            "No built-in best checkpoints were found to alias. "
            "Check that validation ran and produced checkpoint_best_*.weights.h5."
        )
    return copied


def _prepare_merged_coco(
    deepfish_dir,
    ozfish_images_dir,
    ozfish_manifests_dir,
    output_dir,
    rebuild=True,
    ozfish_train_ratio=OZFISH_TRAIN_RATIO,
    split_seed=42,
    manifest_glob="*",
):
    coco_dir = output_dir / "_coco_fish"
    train_dir = coco_dir / "train"
    valid_dir = coco_dir / "valid"
    train_ann_path = train_dir / "_annotations.coco.json"
    valid_ann_path = valid_dir / "_annotations.coco.json"
    if coco_dir.exists() and rebuild:
        shutil.rmtree(coco_dir)
    if train_ann_path.exists() and valid_ann_path.exists() and not rebuild:
        logger.info("Using existing merged COCO dataset: %s", coco_dir)
        return coco_dir

    train_images = []
    train_annotations = []
    valid_images = []
    valid_annotations = []
    next_image_id = 1
    next_annotation_id = 1

    logger.info("Collecting DeepFish train images from %s", deepfish_dir)
    deepfish_summary = _collect_deepfish(
        deepfish_dir,
        train_dir,
        train_images,
        train_annotations,
        next_image_id,
        next_annotation_id,
    )
    next_image_id = deepfish_summary.pop("next_image_id")
    next_annotation_id = deepfish_summary.pop("next_annotation_id")

    logger.info(
        "Loading OzFish manifests from %s and frames from %s",
        ozfish_manifests_dir,
        ozfish_images_dir,
    )
    ozfish_records = _load_ozfish_records(
        ozfish_images_dir, ozfish_manifests_dir, manifest_glob
    )
    ozfish_train_records, ozfish_valid_records = _split_records(
        ozfish_records, ozfish_train_ratio, split_seed
    )

    logger.info(
        "Collecting OzFish split: %d train / %d valid",
        len(ozfish_train_records),
        len(ozfish_valid_records),
    )
    ozfish_train_summary = _collect_ozfish(
        ozfish_train_records,
        train_dir,
        "train",
        train_images,
        train_annotations,
        next_image_id,
        next_annotation_id,
    )
    next_image_id = ozfish_train_summary.pop("next_image_id")
    next_annotation_id = ozfish_train_summary.pop("next_annotation_id")

    ozfish_valid_summary = _collect_ozfish(
        ozfish_valid_records,
        valid_dir,
        "valid",
        valid_images,
        valid_annotations,
        next_image_id,
        next_annotation_id,
    )
    ozfish_valid_summary.pop("next_image_id")
    ozfish_valid_summary.pop("next_annotation_id")

    _write_coco(train_dir, train_images, train_annotations)
    _write_coco(valid_dir, valid_images, valid_annotations)

    summary = {
        "class_name": CLASS_NAME,
        "deepfish_dir": str(deepfish_dir),
        "ozfish_images_dir": str(ozfish_images_dir),
        "ozfish_manifests_dir": str(ozfish_manifests_dir),
        "ozfish_train_ratio": ozfish_train_ratio,
        "split_seed": split_seed,
        "deepfish": deepfish_summary,
        "ozfish_train": ozfish_train_summary,
        "ozfish_valid": ozfish_valid_summary,
        "train_images": len(train_images),
        "train_annotations": len(train_annotations),
        "valid_images": len(valid_images),
        "valid_annotations": len(valid_annotations),
        "output_dir": str(coco_dir),
    }
    with (coco_dir / "merge_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Merged COCO dataset: train=%d images/%d annotations, "
        "valid=%d images/%d annotations -> %s",
        len(train_images),
        len(train_annotations),
        len(valid_images),
        len(valid_annotations),
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
    logger.info("EXPERIMENT 15: RF-DETR Large -- DeepFish + OzFish fish")
    logger.info("=" * 68)

    deepfish_dir = _env_path("DEEPFISH_DIR", Path("/mnt/beegfs/home/jguo/datasets/Deepfish"))
    ozfish_images_dir = _env_path(
        "OZFISH_IMAGES_DIR", Path("/mnt/beegfs/home/jguo/datasets/OzFish/frames_labelled")
    )
    ozfish_manifests_dir = _env_path(
        "OZFISH_MANIFESTS_DIR", Path("/mnt/beegfs/home/jguo/datasets/OzFish/manifests")
    )
    ozfish_train_ratio = _env_float("OZFISH_TRAIN_RATIO", OZFISH_TRAIN_RATIO)
    ozfish_split_seed = _env_int("OZFISH_SPLIT_SEED", 42)
    ozfish_manifest_glob = os.environ.get("OZFISH_MANIFEST_GLOB", "*")
    rebuild_dataset = bool(_env_int("RFDETR_REBUILD_DATASET", 1))

    batch_size = _env_int("RFDETR_BATCH_SIZE", 16)
    grad_accum_steps = _env_int("RFDETR_GRAD_ACCUM_STEPS", 1)
    num_workers = _env_int("RFDETR_NUM_WORKERS", 4)
    epochs = _env_int("RFDETR_EPOCHS", 20)
    base_lr = _env_float("RFDETR_LR", 1e-4)
    lr_encoder = _env_float("RFDETR_LR_ENCODER", 1.5e-4)
    warmup_epochs = _env_float("RFDETR_WARMUP_EPOCHS", 0.0)

    coco_dir = _prepare_merged_coco(
        deepfish_dir,
        ozfish_images_dir,
        ozfish_manifests_dir,
        exp_dir,
        rebuild=rebuild_dataset,
        ozfish_train_ratio=ozfish_train_ratio,
        split_seed=ozfish_split_seed,
        manifest_glob=ozfish_manifest_glob,
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
        eval_interval=1,
        eval_ema=True,
        amp=True,
        num_workers=num_workers,
        run_test=False,
        class_names=[CLASS_NAME],
    )

    exp_config = {
        "experiment": "experiment_15",
        "description": "RF-DETR Large trained on DeepFish train + OzFish 80/20 as fish",
        "variant": "RFDETRLarge",
        "class_names": [CLASS_NAME],
        "num_classes": 1,
        "deepfish_dir": str(deepfish_dir),
        "ozfish_images_dir": str(ozfish_images_dir),
        "ozfish_manifests_dir": str(ozfish_manifests_dir),
        "ozfish_train_ratio": ozfish_train_ratio,
        "ozfish_split_seed": ozfish_split_seed,
        "ozfish_manifest_glob": ozfish_manifest_glob,
        "coco_dir": str(coco_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": base_lr,
        "lr_encoder": lr_encoder,
        "warmup_epochs": warmup_epochs,
        "resolution": model.model_config.resolution,
        "validation": "OzFish 20% split in valid/",
    }
    with (exp_dir / "experiment_config.json").open("w") as f:
        json.dump(exp_config, f, indent=2)

    logger.info("Starting training ...")
    model.train_from_config(config)
    _copy_best_checkpoints(exp_dir)

    final_path = exp_dir / "checkpoints" / "rfdetr_large_fish_final.weights.h5"
    model.model.model.save_weights(str(final_path))
    logger.info("Final weights saved -> %s", final_path)


if __name__ == "__main__":
    main()
    sys.exit(0)
