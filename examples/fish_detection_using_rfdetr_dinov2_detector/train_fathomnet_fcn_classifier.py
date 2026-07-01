#!/usr/bin/env python
"""Train an FCN crop classifier for FathomNet sea-animal species labels."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import keras


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from species_classifier import (
    FathomNetCropDataset,
    build_crop_records,
    build_fcn_classifier,
    compute_balanced_class_weights,
    load_coco_annotations,
    split_records,
)


DEFAULT_DATASET_DIR = _PAZ_ROOT / "datasets" / "fathomnet"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "species_classifier_runs" / "fathomnet_fcn"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a fully-convolutional crop classifier from FathomNet "
            "COCO annotations. Each annotation bbox becomes one species crop."
        )
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="Defaults to DATASET_DIR/train_dataset.json.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--min-box-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--include-classes",
        default="",
        help="Comma-separated class names to train. Empty keeps all categories.",
    )
    parser.add_argument(
        "--no-balanced-weights",
        action="store_true",
        help="Disable inverse-frequency sample weights for class imbalance.",
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


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    annotation_file = (
        Path(args.annotation_file).expanduser().resolve()
        if args.annotation_file
        else dataset_dir / "train_dataset.json"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    train_records, val_records = split_records(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    num_classes = len(class_names)
    class_weights = None
    if not args.no_balanced_weights:
        class_weights = compute_balanced_class_weights(train_records, num_classes)

    train_dataset = FathomNetCropDataset(
        train_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        crop_padding=args.crop_padding,
        seed=args.seed,
        class_weights=class_weights,
        workers=args.workers,
        use_multiprocessing=args.workers > 1,
    )
    val_dataset = FathomNetCropDataset(
        val_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        crop_padding=args.crop_padding,
        seed=args.seed,
        workers=args.workers,
        use_multiprocessing=args.workers > 1,
    )

    model = build_fcn_classifier(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes,
        width=args.width,
        dropout=args.dropout,
    )
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

    checkpoint_path = output_dir / "fathomnet_fcn_best.weights.h5"
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
            min_lr=1e-6,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
    ]

    train_counts = Counter(record["class_name"] for record in train_records)
    val_counts = Counter(record["class_name"] for record in val_records)
    config = {
        "dataset_dir": str(dataset_dir),
        "annotation_file": str(annotation_file),
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "crop_padding": args.crop_padding,
        "min_box_size": args.min_box_size,
        "width": args.width,
        "dropout": args.dropout,
        "class_names": class_names,
        "num_classes": num_classes,
        "include_classes": include_classes,
        "skipped_annotations": skipped,
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
    print(f"Annotations: {annotation_file}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Crop records: train={len(train_records)} val={len(val_records)}")
    print(f"Skipped annotations: {skipped}")
    print(f"Output: {output_dir}")
    model.summary()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = output_dir / "fathomnet_fcn_final.weights.h5"
    model.save_weights(str(final_path))
    print(f"Saved best weights: {checkpoint_path}")
    print(f"Saved final weights: {final_path}")
    print(f"Saved classifier config: {output_dir / 'classifier_config.json'}")


if __name__ == "__main__":
    main()
