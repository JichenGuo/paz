#!/usr/bin/env python
"""Visualize OzFish GroundTruth bounding boxes on labelled frames.

The OzFish manifests are JSON Lines files produced by SageMaker GroundTruth.
Each line contains a ``source-ref`` such as ``E000501_R.MP4.31568.png`` and
the corresponding image stored under ``frames_labelled`` uses that source name
with an extra suffix, for example ``E000501_R.MP4.31568.png-176-1.png``.

Example:

    python examples/fish_detection_using_rfdetr_dinov2_detector/visualize_ozfish_ground_truth.py
"""

import argparse
import json
from pathlib import Path


DEFAULT_IMAGES_DIR = Path("datasets/OzFish/frames_labelled")
DEFAULT_MANIFESTS_DIR = Path("datasets/OzFish/manifests")
DEFAULT_OUTPUT_DIR = Path("datasets/OzFish/frames_labelled_gt_boxes")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw OzFish ground-truth boxes and save annotated images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Root folder containing OzFish labelled frames.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=DEFAULT_MANIFESTS_DIR,
        help="Folder containing OzFish JSONL manifest files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination folder for annotated images.",
    )
    parser.add_argument(
        "--manifest-glob",
        default="*",
        help="Glob pattern for manifest files inside --manifests-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of manifest rows to visualize.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness in pixels.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Draw boxes only, without class/confidence labels.",
    )
    return parser.parse_args()


def require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "OpenCV is required to visualize annotations. Install the project "
            "dependencies, including opencv-python."
        ) from error
    return cv2


def source_key_from_image_name(filename):
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


def build_image_index(images_dir):
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_index = {}
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source_key = source_key_from_image_name(image_path.name)
        image_index.setdefault(source_key, image_path)
    return image_index


def get_annotation_payload(record):
    for key, value in record.items():
        if not isinstance(value, dict):
            continue
        if "annotations" in value and "image_size" in value:
            return key, value
    return None, None


def load_manifest_rows(manifests_dir, manifest_glob):
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

    manifest_paths = sorted(
        path for path in manifests_dir.glob(manifest_glob) if path.is_file()
    )
    if not manifest_paths:
        raise FileNotFoundError(
            f"No manifest files matched {manifests_dir / manifest_glob}"
        )

    for manifest_path in manifest_paths:
        with manifest_path.open("r") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield manifest_path, line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {manifest_path}:{line_number}"
                    ) from error


def class_name_for(record, class_id):
    for key, value in record.items():
        if not key.endswith("-metadata") or not isinstance(value, dict):
            continue
        class_map = value.get("class-map", {})
        return class_map.get(str(class_id), str(class_id))
    return str(class_id)


def object_confidence(record, index):
    for key, value in record.items():
        if not key.endswith("-metadata") or not isinstance(value, dict):
            continue
        objects = value.get("objects", [])
        if index < len(objects):
            confidence = objects[index].get("confidence")
            if confidence is not None:
                return float(confidence)
    return None


def scale_box(annotation, image_shape, manifest_image_size):
    image_height, image_width = image_shape[:2]
    manifest_width = manifest_image_size.get("width", image_width)
    manifest_height = manifest_image_size.get("height", image_height)
    x_scale = image_width / manifest_width
    y_scale = image_height / manifest_height

    left = annotation["left"] * x_scale
    top = annotation["top"] * y_scale
    right = (annotation["left"] + annotation["width"]) * x_scale
    bottom = (annotation["top"] + annotation["height"]) * y_scale

    x_min = max(0, min(image_width - 1, round(left)))
    y_min = max(0, min(image_height - 1, round(top)))
    x_max = max(0, min(image_width - 1, round(right)))
    y_max = max(0, min(image_height - 1, round(bottom)))
    return x_min, y_min, x_max, y_max


def draw_label(cv2, image, label, x_min, y_min):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    y_text_top = max(0, y_min - text_height - baseline - 4)
    y_text_bottom = y_text_top + text_height + baseline + 4
    cv2.rectangle(
        image,
        (x_min, y_text_top),
        (x_min + text_width + 6, y_text_bottom),
        BOX_COLOR,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (x_min + 3, y_text_bottom - baseline - 2),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_annotations(cv2, image, record, payload, draw_labels, line_thickness):
    image_size = payload.get("image_size", [{}])[0]
    for index, annotation in enumerate(payload.get("annotations", [])):
        x_min, y_min, x_max, y_max = scale_box(
            annotation, image.shape, image_size
        )
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            BOX_COLOR,
            line_thickness,
        )
        if not draw_labels:
            continue
        class_id = annotation.get("class_id", 0)
        label = class_name_for(record, class_id)
        confidence = object_confidence(record, index)
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        draw_label(cv2, image, label, x_min, y_min)


def output_path_for(image_path, images_dir, output_dir):
    relative_path = image_path.relative_to(images_dir)
    return output_dir / relative_path


def main():
    args = parse_args()
    cv2 = require_cv2()

    image_index = build_image_index(args.images_dir)
    if not image_index:
        raise FileNotFoundError(f"No images found under {args.images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing_images = 0
    skipped_without_annotations = 0
    for manifest_path, line_number, record in load_manifest_rows(
        args.manifests_dir, args.manifest_glob
    ):
        if args.limit is not None and processed >= args.limit:
            break

        source_ref = Path(record.get("source-ref", "")).name
        _, payload = get_annotation_payload(record)
        if payload is None:
            skipped_without_annotations += 1
            continue

        image_path = image_index.get(source_ref)
        if image_path is None:
            missing_images += 1
            print(
                f"[missing] {manifest_path}:{line_number} "
                f"source-ref={source_ref}"
            )
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[unreadable] {image_path}")
            continue

        draw_annotations(
            cv2,
            image,
            record,
            payload,
            draw_labels=not args.no_labels,
            line_thickness=args.line_thickness,
        )

        destination_path = output_path_for(
            image_path, args.images_dir, args.output_dir
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination_path), image):
            raise IOError(f"Failed to write annotated image: {destination_path}")

        processed += 1
        if processed % 100 == 0:
            print(f"[progress] wrote {processed} annotated images")

    print(f"Wrote {processed} annotated images to {args.output_dir}")
    print(f"Missing images: {missing_images}")
    print(f"Rows without annotations: {skipped_without_annotations}")


if __name__ == "__main__":
    main()
