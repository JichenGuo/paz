#!/usr/bin/env python
"""Split one COCO/RoboFlow image folder into train/val/test datasets.

Input example:

    datasets/Labelimage_Fish_coco/train/
      _annotations.coco.json
      *.png

Output example:

    datasets/Labelimage_Fish_coco_split_70_20_10/
      train/_annotations.coco.json
      train/*.png
      val/_annotations.coco.json
      val/*.png
      test/_annotations.coco.json
      test/*.png

The output root can be passed directly to ``finetune_from_experiment_10.py``
via ``--dataset-dir``.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


DEFAULT_SOURCE_DIR = Path(
    "/mnt/beegfs/home/jguo/datasets/Labelimage_Fish_coco/train"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/beegfs/home/jguo/datasets/Labelimage_Fish_coco_split_70_20_10"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split a COCO/RoboFlow dataset into train/val/test "
            "folders suitable for RF-DETR fine-tuning."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Folder containing images and a COCO annotation JSON.",
    )
    parser.add_argument(
        "--annotation-file",
        default="_annotations.coco.json",
        help="COCO annotation filename inside --source-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output dataset root to create.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy image files or create symlinks to save disk space.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory first if it already exists.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0; got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive")


def load_coco(source_dir, annotation_file):
    ann_path = source_dir / annotation_file
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    with ann_path.open() as f:
        coco = json.load(f)
    if "images" not in coco or "annotations" not in coco:
        raise ValueError(f"Invalid COCO JSON: {ann_path}")
    return coco, ann_path


def split_images(images, train_ratio, val_ratio, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)
    test_count = total - train_count - val_count

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Dataset is too small for the requested split: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    train_images = shuffled[:train_count]
    val_images = shuffled[train_count:train_count + val_count]
    test_images = shuffled[train_count + val_count:]
    return {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }


def copy_or_link_image(source_path, target_path, copy_mode):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        target_path.symlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, target_path)


def make_split_coco(coco, split_images):
    image_ids = {image["id"] for image in split_images}
    annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if annotation.get("image_id") in image_ids
    ]
    split_coco = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    split_coco["images"] = split_images
    split_coco["annotations"] = annotations
    return split_coco


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def write_split(source_dir, output_dir, split_name, split_images, coco, copy_mode):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    missing_images = []
    for image in split_images:
        file_name = image["file_name"]
        source_path = source_dir / file_name
        if not source_path.exists():
            missing_images.append(str(source_path))
            continue
        copy_or_link_image(source_path, split_dir / file_name, copy_mode)

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} images referenced by annotations are missing. "
            f"First missing files:\n{preview}"
        )

    split_coco = make_split_coco(coco, split_images)
    ann_path = split_dir / "_annotations.coco.json"
    with ann_path.open("w") as f:
        json.dump(split_coco, f, indent=2)

    return {
        "images": len(split_coco["images"]),
        "annotations": len(split_coco["annotations"]),
        "annotation_file": str(ann_path),
    }


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    coco, ann_path = load_coco(source_dir, args.annotation_file)
    prepare_output_dir(output_dir, args.overwrite)

    splits = split_images(
        coco["images"],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    summary = {
        "source_dir": str(source_dir),
        "source_annotation": str(ann_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "copy_mode": args.copy_mode,
        "splits": {},
    }

    for split_name, split_image_records in splits.items():
        split_summary = write_split(
            source_dir,
            output_dir,
            split_name,
            split_image_records,
            coco,
            args.copy_mode,
        )
        summary["splits"][split_name] = split_summary

    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Source annotation: {ann_path}")
    print(f"Output dataset: {output_dir}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name}: {split_summary['images']} images, "
            f"{split_summary['annotations']} annotations"
        )
    print(f"Summary: {summary_path}")
    print("")
    print("Fine-tune with:")
    print(
        "python examples/fish_detection_using_rfdetr_dinov2_detector/"
        "experiments_HPC/experiment_10/finetune_from_experiment_10.py "
        f"--dataset-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
