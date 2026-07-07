#!/usr/bin/env python
"""Train a crop species classifier using the RF-DETR DINO backbone.

The pipeline is:
  COCO annotation bbox -> crop with jitter/padding -> RF-DETR DINO backbone
  -> multi-scale global pooling -> MLP species classifier.

By default the DINO backbone is frozen. Use --unfreeze-backbone to fine-tune it
with a smaller learning rate after you have a stable frozen-backbone baseline.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import keras
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano, RFDETRLarge  # noqa: E402
from species_classifier import (  # noqa: E402
    build_crop_records,
    compute_balanced_class_weights,
    load_coco_annotations,
    split_records,
)


DEFAULT_DATASET_DIR = _PAZ_ROOT / "datasets" / "fathomnet"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "species_classifier_runs" / "fathomnet_dino_backbone"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype="float32")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a species classifier on top of RF-DETR's DINO backbone."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--annotation-file",
        default=None,
        help=(
            "COCO annotation JSON. Defaults to DATASET_DIR/train_dataset.json, "
            "or DATASET_DIR/train/_annotations.coco.json for split COCO layout."
        ),
    )
    parser.add_argument(
        "--valid-annotation-file",
        default=None,
        help=(
            "Optional validation COCO JSON. If omitted, records from "
            "--annotation-file are split by --val-fraction."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--detector", choices=("nano", "large"), default="large")
    parser.add_argument(
        "--detector-checkpoint",
        required=True,
        help="Finetuned RF-DETR .weights.h5 checkpoint used to initialize DINO.",
    )
    parser.add_argument(
        "--detector-num-classes",
        type=int,
        default=1,
        help=(
            "Number of detector classes used when constructing RF-DETR before "
            "loading the checkpoint. Use 1 for binary fish/sea_animal models."
        ),
    )
    parser.add_argument(
        "--allow-detector-class-mismatch",
        action="store_true",
        help="Load detector weights with skip_mismatch=True.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument(
        "--bbox-jitter",
        type=float,
        default=0.12,
        help=(
            "Training-time random bbox center/size jitter as a fraction of bbox "
            "size. This simulates imperfect detector crops."
        ),
    )
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--include-classes",
        default="",
        help="Comma-separated species names to train. Empty keeps all categories.",
    )
    parser.add_argument(
        "--no-balanced-weights",
        action="store_true",
        help="Disable inverse-frequency sample weights for class imbalance.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune the RF-DETR DINO backbone instead of freezing it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Keras PyDataset worker count.",
    )
    return parser.parse_args()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def default_annotation_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    train_json = dataset_dir / "train_dataset.json"
    if train_json.exists():
        return train_json, None

    train_split_json = dataset_dir / "train" / "_annotations.coco.json"
    valid_split_json = dataset_dir / "valid" / "_annotations.coco.json"
    val_split_json = dataset_dir / "val" / "_annotations.coco.json"
    if train_split_json.exists():
        valid_json = valid_split_json if valid_split_json.exists() else None
        if valid_json is None and val_split_json.exists():
            valid_json = val_split_json
        return train_split_json, valid_json
    return train_json, None


def jitter_bbox_xywh(bbox, rng, jitter_strength):
    x, y, w, h = [float(value) for value in bbox]
    if jitter_strength <= 0:
        return [x, y, w, h]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    cx += rng.uniform(-jitter_strength, jitter_strength) * w
    cy += rng.uniform(-jitter_strength, jitter_strength) * h
    scale_w = 1.0 + rng.uniform(-jitter_strength, jitter_strength)
    scale_h = 1.0 + rng.uniform(-jitter_strength, jitter_strength)
    scale_w = max(0.35, scale_w)
    scale_h = max(0.35, scale_h)
    new_w = w * scale_w
    new_h = h * scale_h
    return [cx - 0.5 * new_w, cy - 0.5 * new_h, new_w, new_h]


def crop_with_padding(image, bbox_xywh, padding):
    width, height = image.size
    x, y, w, h = [float(value) for value in bbox_xywh]
    pad_x = w * padding
    pad_y = h * padding
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(width, int(math.ceil(x + w + pad_x)))
    bottom = min(height, int(math.ceil(y + h + pad_y)))
    if right <= left or bottom <= top:
        left = max(0, int(math.floor(x)))
        top = max(0, int(math.floor(y)))
        right = min(width, int(math.ceil(x + max(1.0, w))))
        bottom = min(height, int(math.ceil(y + max(1.0, h))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box after clipping: {bbox_xywh}")
    return image.crop((left, top, right, bottom))


def resize_letterbox(image, image_size, fill=(0, 0, 0)):
    image = ImageOps.contain(image, (image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def augment_crop(image, rng, jitter=0.25, hflip_prob=0.5):
    if rng.random() < hflip_prob:
        image = ImageOps.mirror(image)
    if jitter > 0:
        for enhancer_cls in (
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
        ):
            factor = 1.0 + rng.uniform(-jitter, jitter)
            image = enhancer_cls(image).enhance(max(0.1, factor))
    return image


def normalize_for_dino(image_array):
    image_array = image_array.astype("float32") / 255.0
    return (image_array - IMAGENET_MEAN) / IMAGENET_STD


class DinoCropDataset(keras.utils.PyDataset):
    """Crop dataset that emits ImageNet-normalized detector-backbone inputs."""

    def __init__(
        self,
        records,
        image_size=224,
        batch_size=32,
        shuffle=False,
        augment=False,
        crop_padding=0.12,
        bbox_jitter=0.0,
        seed=42,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.records = list(records)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.crop_padding = float(crop_padding)
        self.bbox_jitter = float(bbox_jitter)
        self.seed = int(seed)
        self.class_weights = class_weights
        self.rng = random.Random(seed)
        self.indices = list(range(len(self.records)))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images = []
        labels = []
        weights = []
        for record_index in batch_indices:
            record = self.records[record_index]
            image = Image.open(record["image"]).convert("RGB")
            bbox = record["bbox_xywh"]
            if self.augment:
                bbox = jitter_bbox_xywh(bbox, self.rng, self.bbox_jitter)
            crop = crop_with_padding(image, bbox, self.crop_padding)
            if self.augment:
                crop = augment_crop(crop, self.rng)
            crop = resize_letterbox(crop, self.image_size)
            images.append(normalize_for_dino(np.asarray(crop, dtype="float32")))
            labels.append(record["label"])
            if self.class_weights is not None:
                weights.append(float(self.class_weights.get(record["label"], 1.0)))

        x = np.stack(images, axis=0)
        y = np.asarray(labels, dtype="int32")
        if self.class_weights is None:
            return x, y
        return x, y, np.asarray(weights, dtype="float32")

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class DinoBackboneSpeciesClassifier(keras.Model):
    """Species classifier that reuses RF-DETR's DINO backbone."""

    def __init__(self, backbone, num_classes, head_dim=512, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.projection = keras.layers.Dense(head_dim, activation="relu", name="projection")
        self.dropout = keras.layers.Dropout(dropout, name="dropout")
        self.classifier = keras.layers.Dense(num_classes, activation="softmax", name="species")

    def call(self, inputs, training=False):
        features, _ = self.backbone(inputs, training=training)
        pooled = []
        for feature in features:
            if isinstance(feature, (list, tuple)):
                feature = feature[0]
            elif hasattr(feature, "decompose"):
                feature = feature.decompose()[0]
            pooled.append(keras.ops.mean(feature, axis=(1, 2)))
        x = keras.ops.concatenate(pooled, axis=-1)
        x = self.projection(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)


def build_detector(detector_name, checkpoint_path, num_classes, skip_mismatch):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")
    detector_cls = RFDETRNano if detector_name == "nano" else RFDETRLarge
    detector = detector_cls(num_classes=num_classes)
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    print(f"Loading detector checkpoint: {checkpoint_path}")
    detector.model.model.load_weights(str(checkpoint_path), skip_mismatch=skip_mismatch)
    return detector


def build_model(args, num_species):
    detector = build_detector(
        args.detector,
        args.detector_checkpoint,
        args.detector_num_classes,
        args.allow_detector_class_mismatch,
    )
    backbone = detector.model.model.backbone
    backbone.trainable = bool(args.unfreeze_backbone)
    model = DinoBackboneSpeciesClassifier(
        backbone=backbone,
        num_classes=num_species,
        head_dim=args.head_dim,
        dropout=args.dropout,
        name="DINO_RFDETR_species_classifier",
    )
    dummy = np.zeros((1, args.image_size, args.image_size, 3), dtype="float32")
    model(dummy, training=False)
    return model


def load_records(args, dataset_dir, annotation_file, valid_annotation_file):
    include_classes = [
        name.strip() for name in args.include_classes.split(",") if name.strip()
    ]
    coco = load_coco_annotations(annotation_file)
    records, class_names, skipped = build_crop_records(
        coco,
        dataset_dir,
        min_box_size=args.min_box_size,
        include_classes=include_classes,
    )
    if valid_annotation_file is None:
        train_records, val_records = split_records(
            records,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        val_skipped = 0
    else:
        valid_coco = load_coco_annotations(valid_annotation_file)
        val_records, val_class_names, val_skipped = build_crop_records(
            valid_coco,
            dataset_dir,
            min_box_size=args.min_box_size,
            include_classes=include_classes,
        )
        if val_class_names != class_names:
            raise ValueError(
                "Train/valid class names differ: "
                f"train={class_names}, valid={val_class_names}"
            )
        train_records = records
    return train_records, val_records, class_names, skipped, val_skipped


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    default_train_json, default_valid_json = default_annotation_paths(dataset_dir)
    annotation_file = (
        Path(args.annotation_file).expanduser().resolve()
        if args.annotation_file
        else default_train_json
    )
    valid_annotation_file = (
        Path(args.valid_annotation_file).expanduser().resolve()
        if args.valid_annotation_file
        else default_valid_json
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, val_records, class_names, skipped, val_skipped = load_records(
        args,
        dataset_dir,
        annotation_file,
        valid_annotation_file,
    )
    num_classes = len(class_names)
    class_weights = None
    if not args.no_balanced_weights:
        class_weights = compute_balanced_class_weights(train_records, num_classes)

    train_dataset = DinoCropDataset(
        train_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        crop_padding=args.crop_padding,
        bbox_jitter=args.bbox_jitter,
        seed=args.seed,
        class_weights=class_weights,
        workers=args.workers,
        use_multiprocessing=args.workers > 1,
    )
    val_dataset = DinoCropDataset(
        val_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        crop_padding=args.crop_padding,
        bbox_jitter=0.0,
        seed=args.seed,
        workers=args.workers,
        use_multiprocessing=args.workers > 1,
    )

    model = build_model(args, num_classes)
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    top_k = min(3, num_classes)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=top_k,
                name=f"top{top_k}_accuracy",
            ),
        ],
    )

    checkpoint_path = output_dir / "reefshield_dino_species_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    train_counts = Counter(record["class_name"] for record in train_records)
    val_counts = Counter(record["class_name"] for record in val_records)
    config = {
        "dataset_dir": str(dataset_dir),
        "annotation_file": str(annotation_file),
        "valid_annotation_file": str(valid_annotation_file) if valid_annotation_file else None,
        "output_dir": str(output_dir),
        "detector": args.detector,
        "detector_checkpoint": str(Path(args.detector_checkpoint).expanduser().resolve()),
        "detector_num_classes": args.detector_num_classes,
        "backbone_trainable": bool(args.unfreeze_backbone),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction if valid_annotation_file is None else None,
        "seed": args.seed,
        "crop_padding": args.crop_padding,
        "bbox_jitter": args.bbox_jitter,
        "min_box_size": args.min_box_size,
        "head_dim": args.head_dim,
        "dropout": args.dropout,
        "class_names": class_names,
        "num_classes": num_classes,
        "include_classes": [name.strip() for name in args.include_classes.split(",") if name.strip()],
        "skipped_annotations": skipped,
        "skipped_valid_annotations": val_skipped,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "balanced_weights": not args.no_balanced_weights,
        "train_class_counts": dict(sorted(train_counts.items())),
        "val_class_counts": dict(sorted(val_counts.items())),
    }
    save_json(output_dir / "classifier_config.json", config)
    save_json(output_dir / "train_records.json", train_records)
    save_json(output_dir / "val_records.json", val_records)

    print(f"Dataset: {dataset_dir}")
    print(f"Train annotations: {annotation_file}")
    print(f"Valid annotations: {valid_annotation_file}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Crop records: train={len(train_records)} val={len(val_records)}")
    print(f"Backbone trainable: {model.backbone.trainable}")
    print(f"Output: {output_dir}")
    model.summary()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = output_dir / "reefshield_dino_species_final.weights.h5"
    model.save_weights(str(final_path))
    print(f"Saved best weights: {checkpoint_path}")
    print(f"Saved final weights: {final_path}")
    print(f"Saved classifier config: {output_dir / 'classifier_config.json'}")


if __name__ == "__main__":
    main()
