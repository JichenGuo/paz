#!/usr/bin/env python
"""Fine-tune only the RF-DETR Large detection head.

The DINOv2 backbone, transformer, and learned query embeddings are frozen.
Only the classification and bounding-box detection heads are updated.

The new dataset is expected to be in COCO/RoboFlow layout:

    dataset_dir/
      train/_annotations.coco.json
      train/*.jpg|*.png
      valid/_annotations.coco.json
      valid/*.jpg|*.png

If your validation split is named ``val`` instead of ``valid``, the script
creates a ``valid`` symlink or copy automatically.

It also accepts the local FathomNet layout:

    datasets/fathomnet/
      train_dataset.json
      *.jpg|*.png

In that case it creates an 80/20 train/valid split and remaps category IDs to
contiguous zero-based IDs without adding a background category.
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
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))

from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.detr import RFDETRLarge
from src.training_helpers import (
    apply_train_mode,
    count_component_parameters,
    count_parameters,
)


DEFAULT_SOURCE_CHECKPOINT = (
    _SCRIPT_DIR / "checkpoints" / "rfdetr_large_best.weights.h5"
)
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "finetune_runs" / "from_experiment_12"
DEFAULT_FATHOMNET_DIR = _PAZ_ROOT / "datasets" / "fathomnet"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load an RF-DETR Large checkpoint and fine-tune only its detection "
            "head while keeping DINOv2 and the transformer frozen."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_FATHOMNET_DIR),
        help="COCO-format dataset root containing train/_annotations.coco.json.",
    )
    parser.add_argument(
        "--single-coco-json",
        default="train_dataset.json",
        help=(
            "Single COCO annotation JSON to split when dataset-dir does not "
            "already contain train/valid splits."
        ),
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of images used for training when splitting a single COCO JSON.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the generated train/valid split.",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        default=None,
        help=(
            "Where to write a generated train/valid dataset from a single COCO "
            "JSON. Defaults to OUTPUT_DIR/prepared_fathomnet_dataset."
        ),
    )
    parser.add_argument(
        "--overwrite-prepared-dataset",
        action="store_true",
        help="Remove the generated single-JSON split dataset first if it exists.",
    )
    parser.add_argument(
        "--source-checkpoint",
        default=str(DEFAULT_SOURCE_CHECKPOINT),
        help="Experiment 11 checkpoint to initialize from.",
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
        "--include-classes",
        default="",
        help=(
            "Comma-separated class names to keep for training/validation, "
            "e.g. 'fish'. Other class annotations are removed and treated as "
            "background. Disabled by default."
        ),
    )
    parser.add_argument(
        "--filtered-dataset-dir",
        default=None,
        help=(
            "Where to write the generated class-filtered dataset. Defaults to "
            "OUTPUT_DIR/filtered_dataset."
        ),
    )
    parser.add_argument(
        "--label-remapped-dataset-dir",
        default=None,
        help=(
            "Where to write the generated zero-based-label dataset when "
            "--include-classes is not used. Defaults to OUTPUT_DIR/"
            "label_remapped_dataset."
        ),
    )
    parser.add_argument(
        "--overwrite-filtered-dataset",
        action="store_true",
        help="Remove the generated class-filtered dataset first if it exists.",
    )
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


def has_train_valid_layout(dataset_dir):
    return (
        (dataset_dir / "train" / "_annotations.coco.json").exists()
        and (
            (dataset_dir / "valid" / "_annotations.coco.json").exists()
            or (dataset_dir / "val" / "_annotations.coco.json").exists()
        )
    )


def prepare_single_coco_split_dataset(args, source_dataset_dir, output_dir):
    args._single_coco_prepared = False
    if has_train_valid_layout(source_dataset_dir):
        return source_dataset_dir

    ann_path = source_dataset_dir / args.single_coco_json
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Missing COCO annotations: {ann_path}\n"
            "Expected either train/valid COCO folders or a single annotation "
            "JSON such as datasets/fathomnet/train_dataset.json."
        )
    if not 0.0 < args.train_split < 1.0:
        raise ValueError(f"--train-split must be in (0, 1), got {args.train_split}")

    prepared_dir = (
        Path(args.prepared_dataset_dir).expanduser().resolve()
        if args.prepared_dataset_dir
        else output_dir / "prepared_fathomnet_dataset"
    )
    print("Preparing single-JSON COCO dataset split ...", flush=True)
    print(f"  Source JSON  : {ann_path}", flush=True)
    print(f"  Destination  : {prepared_dir}", flush=True)
    print(f"  Train split  : {args.train_split:.2f}", flush=True)
    print(f"  Split seed   : {args.split_seed}", flush=True)
    if prepared_dir.exists():
        if not args.overwrite_prepared_dataset:
            raise FileExistsError(
                f"Prepared dataset already exists: {prepared_dir}\n"
                "Use --overwrite-prepared-dataset or pass a different "
                "--prepared-dataset-dir."
            )
        shutil.rmtree(prepared_dir)

    coco = load_coco(ann_path)
    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    categories = sorted(categories, key=lambda category: category["id"])
    if not categories:
        raise ValueError(f"No usable categories found in {ann_path}")

    old_to_new_category_id = {
        category["id"]: new_id
        for new_id, category in enumerate(categories)
    }
    new_categories = []
    for new_id, category in enumerate(categories):
        new_category = dict(category)
        new_category["id"] = new_id
        new_categories.append(new_category)

    images = [dict(image) for image in coco.get("images", [])]
    if not images:
        raise ValueError(f"No images found in {ann_path}")

    rng = random.Random(args.split_seed)
    shuffled_images = images[:]
    rng.shuffle(shuffled_images)
    n_train = int(round(len(shuffled_images) * args.train_split))
    n_train = min(max(1, n_train), len(shuffled_images) - 1)
    split_images = {
        "train": shuffled_images[:n_train],
        "valid": shuffled_images[n_train:],
    }

    anns_by_image = {}
    for annotation in coco.get("annotations", []):
        category_id = annotation.get("category_id")
        if category_id not in old_to_new_category_id:
            continue
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    split_summaries = {}
    for split_name, split_image_list in split_images.items():
        split_dir = prepared_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_image_ids = {image["id"] for image in split_image_list}
        split_annotations = []
        next_annotation_id = 1
        for image in split_image_list:
            src = source_dataset_dir / image["file_name"]
            if not src.exists():
                raise FileNotFoundError(
                    f"Image referenced by COCO JSON is missing: {src}"
                )
            copy_or_link(src, split_dir / image["file_name"], args.oversample_copy_mode)

            for annotation in anns_by_image.get(image["id"], []):
                if annotation["image_id"] not in split_image_ids:
                    continue
                new_annotation = dict(annotation)
                new_annotation["id"] = next_annotation_id
                new_annotation["category_id"] = old_to_new_category_id[
                    annotation["category_id"]
                ]
                split_annotations.append(new_annotation)
                next_annotation_id += 1

        split_coco = {
            key: value
            for key, value in coco.items()
            if key not in {"images", "annotations", "categories"}
        }
        split_coco["images"] = split_image_list
        split_coco["annotations"] = split_annotations
        split_coco["categories"] = new_categories
        with (split_dir / "_annotations.coco.json").open("w") as f:
            json.dump(split_coco, f, indent=2)

        split_summaries[split_name] = {
            "images": len(split_image_list),
            "annotations": len(split_annotations),
        }

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "source_annotations": str(ann_path),
        "prepared_dataset_dir": str(prepared_dir),
        "train_split": args.train_split,
        "split_seed": args.split_seed,
        "category_id_mapping": {
            str(old_id): new_id for old_id, new_id in old_to_new_category_id.items()
        },
        "class_names": [category["name"] for category in new_categories],
        "splits": split_summaries,
    }
    with (prepared_dir / "single_coco_split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Prepared single-JSON COCO dataset:", flush=True)
    for split_name, split_summary in split_summaries.items():
        print(
            f"  {split_name}: {split_summary['images']} images, "
            f"{split_summary['annotations']} annotations",
            flush=True,
        )
    print(
        "  Categories remapped to zero-based IDs: "
        f"{summary['category_id_mapping']}",
        flush=True,
    )
    args._single_coco_prepared = True
    return prepared_dir


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


def prepare_class_filtered_dataset(args, source_dataset_dir, output_dir):
    included_names = [
        name.strip()
        for name in args.include_classes.split(",")
        if name.strip()
    ]
    if not included_names:
        return prepare_label_remapped_dataset(args, source_dataset_dir, output_dir)

    prepared_dir = (
        Path(args.filtered_dataset_dir).expanduser().resolve()
        if args.filtered_dataset_dir
        else output_dir / "filtered_dataset"
    )
    print("Preparing class-filtered dataset ...", flush=True)
    print(f"  Source       : {source_dataset_dir}", flush=True)
    print(f"  Destination  : {prepared_dir}", flush=True)
    print(f"  Include only : {included_names}", flush=True)
    if prepared_dir.exists():
        if not args.overwrite_filtered_dataset:
            raise FileExistsError(
                f"Filtered dataset already exists: {prepared_dir}\n"
                "Use --overwrite-filtered-dataset or pass a different "
                "--filtered-dataset-dir."
            )
        shutil.rmtree(prepared_dir)

    summaries = {}
    for split_name in ("train", "valid", "test"):
        source_split_dir = source_dataset_dir / split_name
        if not source_split_dir.exists():
            if split_name == "test":
                continue
            raise FileNotFoundError(f"Missing split directory: {source_split_dir}")
        summaries[split_name] = filter_coco_split(
            source_split_dir,
            prepared_dir / split_name,
            included_names,
            args.oversample_copy_mode,
        )

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "prepared_dataset_dir": str(prepared_dir),
        "include_classes": included_names,
        "splits": summaries,
    }
    with (prepared_dir / "class_filter_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Prepared class-filtered dataset:", flush=True)
    for split_name, split_summary in summaries.items():
        print(
            f"  {split_name}: {split_summary['images']} images, "
            f"{split_summary['annotations_before']} -> "
            f"{split_summary['annotations_after']} annotations",
            flush=True,
        )
    return prepared_dir


def prepare_label_remapped_dataset(args, source_dataset_dir, output_dir):
    prepared_dir = (
        Path(args.label_remapped_dataset_dir).expanduser().resolve()
        if args.label_remapped_dataset_dir
        else output_dir / "label_remapped_dataset"
    )
    print("Preparing zero-based-label dataset ...", flush=True)
    print(f"  Source      : {source_dataset_dir}", flush=True)
    print(f"  Destination : {prepared_dir}", flush=True)
    if prepared_dir.exists():
        if not args.overwrite_filtered_dataset:
            raise FileExistsError(
                f"Label-remapped dataset already exists: {prepared_dir}\n"
                "Use --overwrite-filtered-dataset or pass a different "
                "--label-remapped-dataset-dir."
            )
        shutil.rmtree(prepared_dir)

    summaries = {}
    for split_name in ("train", "valid", "test"):
        source_split_dir = source_dataset_dir / split_name
        if not source_split_dir.exists():
            if split_name == "test":
                continue
            raise FileNotFoundError(f"Missing split directory: {source_split_dir}")
        summaries[split_name] = remap_coco_split_labels(
            source_split_dir,
            prepared_dir / split_name,
            args.oversample_copy_mode,
        )

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "prepared_dataset_dir": str(prepared_dir),
        "splits": summaries,
    }
    with (prepared_dir / "label_remap_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Prepared zero-based-label dataset:", flush=True)
    for split_name, split_summary in summaries.items():
        print(
            f"  {split_name}: classes={split_summary['classes']}, "
            f"{split_summary['annotations']} annotations",
            flush=True,
        )
    return prepared_dir


def remap_coco_split_labels(source_split_dir, target_split_dir, copy_mode):
    source_ann_path = source_split_dir / "_annotations.coco.json"
    coco = load_coco(source_ann_path)
    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    categories = sorted(categories, key=lambda category: category["id"])
    old_to_new_category_id = {
        category["id"]: new_id
        for new_id, category in enumerate(categories)
    }
    new_categories = []
    for new_id, category in enumerate(categories):
        new_category = dict(category)
        new_category["id"] = new_id
        new_categories.append(new_category)

    target_split_dir.mkdir(parents=True, exist_ok=True)
    for image in coco.get("images", []):
        source_path = source_split_dir / image["file_name"]
        if not source_path.exists():
            raise FileNotFoundError(
                f"Image referenced by COCO JSON is missing: {source_path}"
            )
        copy_or_link(source_path, target_split_dir / image["file_name"], copy_mode)

    new_annotations = []
    for annotation in coco.get("annotations", []):
        old_category_id = annotation.get("category_id")
        if old_category_id not in old_to_new_category_id:
            continue
        new_annotation = dict(annotation)
        new_annotation["category_id"] = old_to_new_category_id[old_category_id]
        new_annotations.append(new_annotation)

    new_coco = {
        key: value
        for key, value in coco.items()
        if key not in {"categories", "annotations"}
    }
    new_coco["categories"] = new_categories
    new_coco["annotations"] = new_annotations
    with (target_split_dir / "_annotations.coco.json").open("w") as f:
        json.dump(new_coco, f, indent=2)

    return {
        "classes": [category["name"] for category in new_categories],
        "annotations": len(new_annotations),
    }


def filter_coco_split(source_split_dir, target_split_dir, included_names, copy_mode):
    source_ann_path = source_split_dir / "_annotations.coco.json"
    coco = load_coco(source_ann_path)

    categories = [
        category
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    ]
    category_by_name = {category["name"]: category for category in categories}
    missing = sorted(set(included_names) - set(category_by_name))
    if missing:
        raise ValueError(
            f"Included classes not found in {source_ann_path}: {missing}. "
            f"Available: {sorted(category_by_name)}"
        )

    old_to_new_category_id = {}
    new_categories = []
    for new_id, name in enumerate(included_names):
        old_category = dict(category_by_name[name])
        old_to_new_category_id[old_category["id"]] = new_id
        old_category["id"] = new_id
        new_categories.append(old_category)

    target_split_dir.mkdir(parents=True, exist_ok=True)
    for image in coco.get("images", []):
        source_path = source_split_dir / image["file_name"]
        if not source_path.exists():
            raise FileNotFoundError(
                f"Image referenced by COCO JSON is missing: {source_path}"
            )
        copy_or_link(source_path, target_split_dir / image["file_name"], copy_mode)

    new_annotations = []
    next_annotation_id = 1
    for annotation in coco.get("annotations", []):
        old_category_id = annotation.get("category_id")
        if old_category_id not in old_to_new_category_id:
            continue
        new_annotation = dict(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["category_id"] = old_to_new_category_id[old_category_id]
        new_annotations.append(new_annotation)
        next_annotation_id += 1

    new_coco = {
        key: value
        for key, value in coco.items()
        if key not in {"categories", "annotations"}
    }
    new_coco["categories"] = new_categories
    new_coco["annotations"] = new_annotations
    with (target_split_dir / "_annotations.coco.json").open("w") as f:
        json.dump(new_coco, f, indent=2)

    return {
        "images": len(new_coco.get("images", [])),
        "annotations_before": len(coco.get("annotations", [])),
        "annotations_after": len(new_annotations),
        "classes": included_names,
    }


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
    detector = RFDETRLarge(num_classes=num_classes)

    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {checkpoint_path}\n"
            "Train experiment_11 first or pass --source-checkpoint."
        )

    print(f"Loading source checkpoint: {checkpoint_path}")
    detector.model.model.load_weights(
        str(checkpoint_path),
        skip_mismatch=allow_class_mismatch,
    )

    detector.model.model(dummy, training=True)
    return detector


def freeze_for_head_only_training(detector):
    """Make the classification and bounding-box heads the only trainable part."""
    model = detector.model.model
    apply_train_mode(model, "head_only")

    # The repository's generic head_only mode keeps learned queries trainable.
    # Freeze them as well so this script trains the detection heads strictly.
    model.refpoint_embed._trainable = False
    model.query_feat._trainable = False

    component_params = count_component_parameters(model)
    expected_frozen = {"backbone", "transformer", "query_embeddings"}
    for component in expected_frozen:
        trainable = component_params.get(component, {}).get("trainable", 0)
        if trainable:
            raise RuntimeError(
                f"Head-only freezing failed: {component} still has "
                f"{trainable:,} trainable parameters"
            )

    unexpected_components = {
        component
        for component, counts in component_params.items()
        if component != "detection_head" and counts["trainable"] > 0
    }
    if unexpected_components:
        raise RuntimeError(
            "Head-only freezing failed; unexpected trainable components: "
            f"{sorted(unexpected_components)}"
        )
    if component_params.get("detection_head", {}).get("trainable", 0) == 0:
        raise RuntimeError("Detection head has no trainable parameters")

    params = count_parameters(model)
    print("Training mode: detection head only", flush=True)
    print(
        "  Frozen: DINOv2 backbone, feature projector, transformer, queries",
        flush=True,
    )
    print("  Trainable: classification and bounding-box heads", flush=True)
    print(
        f"  Parameters: {params['trainable']:,} trainable / "
        f"{params['total']:,} total "
        f"({params['pct_trainable']:.2f}% trainable)",
        flush=True,
    )
    return {
        "frozen_components": sorted(expected_frozen),
        "trainable_components": ["detection_head"],
        "params": params,
        "component_params": component_params,
    }


def save_finetune_config(args, output_dir, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_experiment": "experiment_11",
        "source_checkpoint": str(Path(args.source_checkpoint).expanduser().resolve()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "single_coco_json": args.single_coco_json,
        "train_split": args.train_split,
        "split_seed": args.split_seed,
        "prepared_dataset_dir": args.prepared_dataset_dir,
        "output_dir": str(output_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "train_mode": "head_only",
        "frozen_components": [
            "dinov2_encoder",
            "backbone_projector",
            "transformer",
            "query_embeddings",
        ],
        "trainable_components": ["detection_head"],
        "augmentation": "native RF-DETR (make_coco_transforms_square_div_64)",
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "num_workers": args.num_workers,
        "use_ema": not args.no_ema,
        "amp": not args.no_amp,
        "resume": args.resume,
        "allow_class_mismatch": args.allow_class_mismatch,
        "include_classes": args.include_classes,
        "filtered_dataset_dir": args.filtered_dataset_dir,
        "label_remapped_dataset_dir": args.label_remapped_dataset_dir,
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

    dataset_dir = prepare_single_coco_split_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)
    if args.include_classes or not getattr(args, "_single_coco_prepared", False):
        dataset_dir = prepare_class_filtered_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)
    dataset_dir = prepare_oversampled_dataset(args, dataset_dir, output_dir)
    ensure_valid_split(dataset_dir)
    class_names = read_coco_classes(dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    if len(class_names) != 1 and not args.allow_class_mismatch:
        args.allow_class_mismatch = True
        print(
            "Detected a multi-class fine-tuning dataset. Loading compatible "
            "source weights with classification-head mismatches skipped.",
            flush=True,
        )

    detector = build_detector(
        num_classes=len(class_names),
        checkpoint_path=source_checkpoint,
        allow_class_mismatch=args.allow_class_mismatch,
    )
    detector.model.class_names = class_names
    freeze_for_head_only_training(detector)

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
        # Native RF-DETR augmentation, matching experiment_10.py:
        # RandomHorizontalFlip + RandomSelect(SquareResize,
        # RandomResize/RandomSizeCrop/SquareResize) + ImageNet normalize.
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

    final_path = output_dir / "rfdetr_large_finetuned_final.weights.h5"
    detector.model.model.save_weights(str(final_path))
    print(f"Saved final fine-tuned weights: {final_path}")


if __name__ == "__main__":
    main()
