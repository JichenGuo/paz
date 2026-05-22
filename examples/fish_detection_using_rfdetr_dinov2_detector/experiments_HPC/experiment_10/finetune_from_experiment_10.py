#!/usr/bin/env python
"""Fine-tune RF-DETR Nano from the Experiment 10 trained checkpoint.

The new dataset is expected to be in COCO/RoboFlow layout:

    dataset_dir/
      train/_annotations.coco.json
      train/*.jpg|*.png
      valid/_annotations.coco.json
      valid/*.jpg|*.png

If your validation split is named ``val`` instead of ``valid``, the script
creates a ``valid`` symlink or copy automatically.
"""

import argparse
from collections import Counter
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    parser.add_argument("--epochs", type=int, default=30)
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
        "--oversample-classes",
        default="",
        help=(
            "Comma-separated rare class names to oversample, e.g. "
            "'crab,lobster'. Disabled by default."
        ),
    )
    parser.add_argument(
        "--rare-repeat-factor",
        type=int,
        default=0,
        help=(
            "Number of augmented copies to create for every training image "
            "containing one of --oversample-classes. 0 disables oversampling."
        ),
    )
    parser.add_argument(
        "--oversampled-dataset-dir",
        default=None,
        help=(
            "Where to write the generated oversampled dataset. Defaults to "
            "OUTPUT_DIR/oversampled_dataset."
        ),
    )
    parser.add_argument(
        "--overwrite-oversampled-dataset",
        action="store_true",
        help="Remove the generated oversampled dataset first if it exists.",
    )
    parser.add_argument(
        "--oversample-copy-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to place original images in the oversampled dataset.",
    )
    parser.add_argument(
        "--rare-hflip-prob",
        type=float,
        default=0.5,
        help="Horizontal-flip probability for generated rare-class copies.",
    )
    parser.add_argument(
        "--rare-color-jitter",
        type=float,
        default=0.35,
        help="Color/contrast/brightness jitter strength for rare-class copies.",
    )
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


def load_coco(ann_path):
    with ann_path.open() as f:
        return json.load(f)


def copy_or_link(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def category_ids_for_names(coco, class_names):
    wanted = {name.strip() for name in class_names if name.strip()}
    mapping = {
        category["name"]: category["id"]
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    }
    missing = sorted(wanted - set(mapping))
    if missing:
        raise ValueError(
            f"Oversample classes not found in COCO categories: {missing}. "
            f"Available: {sorted(mapping)}"
        )
    return {mapping[name] for name in wanted}


def count_objects_by_class(coco):
    id_to_name = {
        category["id"]: category["name"]
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    }
    counts = Counter(
        annotation["category_id"]
        for annotation in coco.get("annotations", [])
        if annotation.get("category_id") in id_to_name
    )
    return {id_to_name[cat_id]: counts[cat_id] for cat_id in sorted(id_to_name)}


def augment_rare_image(image, rng, hflip_prob, jitter_strength):
    flipped = rng.random() < hflip_prob
    if flipped:
        image = ImageOps.mirror(image)

    if jitter_strength > 0:
        for enhancer_cls in (
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
        ):
            factor = 1.0 + rng.uniform(-jitter_strength, jitter_strength)
            image = enhancer_cls(image).enhance(max(0.1, factor))
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    return image, flipped


def flip_bbox_xywh(bbox, image_width):
    x, y, w, h = [float(value) for value in bbox]
    return [image_width - x - w, y, w, h]


def normalize_bbox_xywh(bbox):
    return [float(value) for value in bbox]


def copy_existing_split(source_dir, target_dir, split_name, copy_mode):
    source_split_dir = source_dir / split_name
    if not source_split_dir.exists():
        return

    target_split_dir = target_dir / split_name
    target_split_dir.mkdir(parents=True, exist_ok=True)

    for path in source_split_dir.iterdir():
        target_path = target_split_dir / path.name
        if path.name == "_annotations.coco.json":
            shutil.copy2(path, target_path)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            copy_or_link(path, target_path, copy_mode)


def prepare_oversampled_dataset(args, source_dataset_dir, output_dir):
    rare_names = [
        name.strip()
        for name in args.oversample_classes.split(",")
        if name.strip()
    ]
    if args.rare_repeat_factor <= 0 or not rare_names:
        print(
            "Rare-class oversampling disabled. To enable it, pass "
            "--oversample-classes and --rare-repeat-factor > 0.",
            flush=True,
        )
        return source_dataset_dir

    prepared_dir = (
        Path(args.oversampled_dataset_dir).expanduser().resolve()
        if args.oversampled_dataset_dir
        else output_dir / "oversampled_dataset"
    )
    print("Preparing oversampled dataset ...", flush=True)
    print(f"  Source      : {source_dataset_dir}", flush=True)
    print(f"  Destination : {prepared_dir}", flush=True)
    print(f"  Rare classes: {rare_names}", flush=True)
    print(f"  Repeat      : {args.rare_repeat_factor}", flush=True)
    if prepared_dir.exists():
        if not args.overwrite_oversampled_dataset:
            raise FileExistsError(
                f"Oversampled dataset already exists: {prepared_dir}\n"
                "Use --overwrite-oversampled-dataset or pass a different "
                "--oversampled-dataset-dir."
            )
        shutil.rmtree(prepared_dir)

    train_dir = source_dataset_dir / "train"
    train_ann_path = train_dir / "_annotations.coco.json"
    coco = load_coco(train_ann_path)
    rare_category_ids = category_ids_for_names(coco, rare_names)

    anns_by_image = {}
    for annotation in coco.get("annotations", []):
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    rare_images = [
        image for image in coco["images"]
        if any(
            ann.get("category_id") in rare_category_ids
            for ann in anns_by_image.get(image["id"], [])
        )
    ]
    if not rare_images:
        raise ValueError(f"No training images contain rare classes: {rare_names}")

    prepared_train_dir = prepared_dir / "train"
    prepared_train_dir.mkdir(parents=True, exist_ok=True)

    new_coco = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    new_images = [dict(image) for image in coco["images"]]
    new_annotations = [dict(annotation) for annotation in coco["annotations"]]

    print(
        f"Linking/copying {len(coco['images'])} original train images "
        f"using mode={args.oversample_copy_mode} ...",
        flush=True,
    )
    for image in coco["images"]:
        src = train_dir / image["file_name"]
        if not src.exists():
            raise FileNotFoundError(f"Image referenced by COCO JSON is missing: {src}")
        copy_or_link(src, prepared_train_dir / image["file_name"], args.oversample_copy_mode)

    rng = random.Random(42)
    next_image_id = max(image["id"] for image in new_images) + 1
    next_annotation_id = max(annotation["id"] for annotation in new_annotations) + 1

    total_augmented = len(rare_images) * args.rare_repeat_factor
    completed_augmented = 0
    print(
        f"Creating {total_augmented} augmented rare-class images "
        f"from {len(rare_images)} source images ...",
        flush=True,
    )
    for image_index, image in enumerate(rare_images, start=1):
        source_path = train_dir / image["file_name"]
        image_annotations = anns_by_image.get(image["id"], [])
        stem = Path(image["file_name"]).stem
        suffix = Path(image["file_name"]).suffix or ".png"

        for repeat_index in range(args.rare_repeat_factor):
            pil_image = Image.open(source_path).convert("RGB")
            aug_image, flipped = augment_rare_image(
                pil_image,
                rng,
                args.rare_hflip_prob,
                args.rare_color_jitter,
            )
            new_file_name = f"{stem}__rare_aug_{repeat_index + 1:02d}{suffix}"
            aug_image.save(prepared_train_dir / new_file_name)
            completed_augmented += 1

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = new_file_name
            new_images.append(new_image)

            for annotation in image_annotations:
                new_annotation = dict(annotation)
                new_annotation["id"] = next_annotation_id
                new_annotation["image_id"] = next_image_id
                if flipped:
                    new_annotation["bbox"] = flip_bbox_xywh(
                        list(new_annotation["bbox"]),
                        image["width"],
                    )
                else:
                    new_annotation["bbox"] = normalize_bbox_xywh(
                        list(new_annotation["bbox"])
                    )
                new_annotations.append(new_annotation)
                next_annotation_id += 1

            next_image_id += 1

            if completed_augmented == 1 or completed_augmented % 25 == 0:
                print(
                    f"  Augmented {completed_augmented}/{total_augmented} "
                    f"(source image {image_index}/{len(rare_images)})",
                    flush=True,
                )

    new_coco["images"] = new_images
    new_coco["annotations"] = new_annotations
    print("Writing oversampled train annotations ...", flush=True)
    with (prepared_train_dir / "_annotations.coco.json").open("w") as f:
        json.dump(new_coco, f, indent=2)

    print("Linking/copying valid/test splits ...", flush=True)
    copy_existing_split(source_dataset_dir, prepared_dir, "valid", args.oversample_copy_mode)
    copy_existing_split(source_dataset_dir, prepared_dir, "test", args.oversample_copy_mode)

    before_counts = count_objects_by_class(coco)
    after_counts = count_objects_by_class(new_coco)
    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "prepared_dataset_dir": str(prepared_dir),
        "oversample_classes": rare_names,
        "rare_repeat_factor": args.rare_repeat_factor,
        "rare_images": len(rare_images),
        "original_train_images": len(coco["images"]),
        "prepared_train_images": len(new_images),
        "object_counts_before": before_counts,
        "object_counts_after": after_counts,
    }
    with (prepared_dir / "oversampling_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Prepared oversampled dataset:", flush=True)
    print(f"  Source             : {source_dataset_dir}", flush=True)
    print(f"  Prepared           : {prepared_dir}", flush=True)
    print(f"  Rare classes       : {rare_names}", flush=True)
    print(f"  Rare images        : {len(rare_images)}", flush=True)
    print(f"  Train images       : {len(coco['images'])} -> {len(new_images)}", flush=True)
    print(f"  Object counts      : {before_counts} -> {after_counts}", flush=True)
    return prepared_dir


def ensure_valid_split(dataset_dir):
    val_dir = dataset_dir / "val"
    valid_dir = dataset_dir / "valid"
    if valid_dir.exists():
        return
    if val_dir.exists():
        try:
            valid_dir.symlink_to(val_dir.resolve(), target_is_directory=True)
            print(f"Created symlink: {valid_dir} -> {val_dir}")
        except OSError:
            shutil.copytree(val_dir, valid_dir)
            print(f"Copied validation split: {val_dir} -> {valid_dir}")
        return
    raise FileNotFoundError(
        f"Missing validation split: expected {valid_dir} or {val_dir}"
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
        "oversample_classes": args.oversample_classes,
        "rare_repeat_factor": args.rare_repeat_factor,
        "oversampled_dataset_dir": args.oversampled_dataset_dir,
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

    ensure_valid_split(dataset_dir)
    dataset_dir = prepare_oversampled_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)
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
