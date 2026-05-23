#!/usr/bin/env python
"""Tile a COCO/RoboFlow dataset into overlapping square crops.

Input layout:

    dataset_dir/
      train/_annotations.coco.json
      train/*.jpg|*.png
      valid/_annotations.coco.json   # or val/_annotations.coco.json
      valid/*.jpg|*.png
      test/_annotations.coco.json    # optional

Output layout:

    output_dir/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

The output can be passed directly to finetune_from_experiment_10.py.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create overlapping COCO crops for RF-DETR Nano finetuning."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=960)
    parser.add_argument(
        "--stride",
        type=int,
        default=640,
        help="Crop stride in pixels. 640 gives 320 px overlap for 960 crops.",
    )
    parser.add_argument(
        "--min-visible-frac",
        type=float,
        default=0.25,
        help="Keep a clipped box only if this fraction of its area remains.",
    )
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help=(
            "For train only, keep this many empty crops relative to positive "
            "crops. Empty crops help reduce false positives."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_coco(path):
    with path.open() as f:
        return json.load(f)


def write_coco(path, coco):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(coco, f, indent=2)


def find_split_dir(dataset_dir, split):
    candidates = [dataset_dir / split]
    if split == "valid":
        candidates.append(dataset_dir / "val")
    for split_dir in candidates:
        if (split_dir / "_annotations.coco.json").exists():
            return split_dir
    return None


def crop_positions(length, crop_size, stride):
    if length <= crop_size:
        return [0]
    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def clip_bbox_xywh(bbox, crop_x, crop_y, crop_size):
    x, y, w, h = [float(v) for v in bbox]
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    cx1, cy1 = crop_x, crop_y
    cx2, cy2 = crop_x + crop_size, crop_y + crop_size

    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    clipped_w = ix2 - ix1
    clipped_h = iy2 - iy1
    return [ix1 - crop_x, iy1 - crop_y, clipped_w, clipped_h]


def should_keep_box(original_bbox, clipped_bbox, min_visible_frac, min_box_size):
    _, _, original_w, original_h = [float(v) for v in original_bbox]
    _, _, clipped_w, clipped_h = [float(v) for v in clipped_bbox]
    if clipped_w < min_box_size or clipped_h < min_box_size:
        return False
    original_area = max(0.0, original_w) * max(0.0, original_h)
    clipped_area = clipped_w * clipped_h
    if original_area <= 0:
        return False
    return clipped_area / original_area >= min_visible_frac


def image_path_for(split_dir, file_name):
    path = split_dir / file_name
    if path.exists():
        return path
    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image referenced by COCO JSON is missing: {path}")


def tile_split(
    source_split_dir,
    output_split_dir,
    split_name,
    crop_size,
    stride,
    min_visible_frac,
    min_box_size,
    negative_ratio,
    rng,
):
    ann_path = source_split_dir / "_annotations.coco.json"
    coco = load_coco(ann_path)
    output_split_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    new_coco = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    new_images = []
    new_annotations = []
    next_image_id = 1
    next_ann_id = 1
    empty_candidates = []

    for image_info in coco.get("images", []):
        src_path = image_path_for(source_split_dir, image_info["file_name"])
        image = Image.open(src_path).convert("RGB")
        width, height = image.size
        source_anns = annotations_by_image.get(image_info["id"], [])

        x_positions = crop_positions(width, crop_size, stride)
        y_positions = crop_positions(height, crop_size, stride)
        for crop_y in y_positions:
            for crop_x in x_positions:
                crop_anns = []
                for ann in source_anns:
                    clipped = clip_bbox_xywh(
                        ann["bbox"], crop_x, crop_y, crop_size
                    )
                    if clipped is None:
                        continue
                    if not should_keep_box(
                        ann["bbox"],
                        clipped,
                        min_visible_frac,
                        min_box_size,
                    ):
                        continue
                    new_ann = dict(ann)
                    new_ann["id"] = next_ann_id
                    new_ann["image_id"] = next_image_id
                    new_ann["bbox"] = [float(v) for v in clipped]
                    new_ann["area"] = float(clipped[2] * clipped[3])
                    new_ann.pop("segmentation", None)
                    crop_anns.append(new_ann)
                    next_ann_id += 1

                crop_record = (
                    image_info,
                    image,
                    crop_x,
                    crop_y,
                    crop_anns,
                    next_image_id,
                )
                if crop_anns:
                    save_crop(output_split_dir, crop_record, crop_size)
                    new_images.append(make_image_record(crop_record, crop_size))
                    new_annotations.extend(crop_anns)
                    next_image_id += 1
                else:
                    empty_candidates.append(crop_record)

    if split_name == "train" and negative_ratio > 0 and new_images:
        max_empty = int(round(len(new_images) * negative_ratio))
        rng.shuffle(empty_candidates)
        for crop_record in empty_candidates[:max_empty]:
            crop_record = (*crop_record[:-1], next_image_id)
            save_crop(output_split_dir, crop_record, crop_size)
            new_images.append(make_image_record(crop_record, crop_size))
            next_image_id += 1

    new_coco["images"] = new_images
    new_coco["annotations"] = new_annotations
    write_coco(output_split_dir / "_annotations.coco.json", new_coco)

    print(
        f"{split_name}: {len(coco.get('images', []))} images -> "
        f"{len(new_images)} crops, {len(new_annotations)} annotations"
    )


def make_crop_name(image_info, crop_x, crop_y):
    source_name = Path(image_info["file_name"])
    return f"{source_name.stem}__x{crop_x:04d}_y{crop_y:04d}{source_name.suffix}"


def make_image_record(crop_record, crop_size):
    image_info, _, crop_x, crop_y, _, image_id = crop_record
    new_image = dict(image_info)
    new_image["id"] = image_id
    new_image["file_name"] = make_crop_name(image_info, crop_x, crop_y)
    new_image["width"] = crop_size
    new_image["height"] = crop_size
    return new_image


def save_crop(output_split_dir, crop_record, crop_size):
    image_info, image, crop_x, crop_y, _, _ = crop_record
    crop = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
    crop.save(output_split_dir / make_crop_name(image_info, crop_x, crop_y))


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    for split_name in ("train", "valid", "test"):
        source_split_dir = find_split_dir(dataset_dir, split_name)
        if source_split_dir is None:
            if split_name != "test":
                raise FileNotFoundError(
                    f"Missing required split: {dataset_dir / split_name}"
                )
            continue
        tile_split(
            source_split_dir=source_split_dir,
            output_split_dir=output_dir / split_name,
            split_name=split_name,
            crop_size=args.crop_size,
            stride=args.stride,
            min_visible_frac=args.min_visible_frac,
            min_box_size=args.min_box_size,
            negative_ratio=args.negative_ratio,
            rng=rng,
        )

    print(f"Tiled dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
