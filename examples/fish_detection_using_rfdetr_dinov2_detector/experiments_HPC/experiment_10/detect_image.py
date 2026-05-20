#!/usr/bin/env python
"""Run inference with the best RF-DETR Nano checkpoint from Experiment 10."""

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = _SCRIPT_DIR.parents[3]
_SRC_DIR = _SCRIPT_DIR.parents[1] / "src"
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano


DEFAULT_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "rfdetr_nano_best.weights.h5"
DEFAULT_CLASS_NAMES = ["fish"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 0, 0)
TEXT_BG_COLOR = (0, 120, 170)
TEXT_FONT_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the Experiment 10 best RF-DETR Nano checkpoint and run "
            "detection on one image or every image in a folder."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an input image or a folder of images.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the annotated output image, or output directory when "
            "--image is a folder. Defaults next to the input."
        ),
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path for detection JSON. Folder mode writes one combined JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names. Experiment 10 defaults to fish.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --image is a folder, scan images recursively.",
    )
    return parser.parse_args()


def load_rgb_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image, np.asarray(image, dtype=np.uint8)


def collect_image_paths(input_path, recursive=False):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input image/folder not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_path}. Supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return image_paths


def make_output_paths(input_path, image_paths, args):
    folder_mode = input_path.is_dir()
    if folder_mode:
        if args.output is None:
            output_dir = input_path / "experiment_10_detections"
        else:
            output_dir = Path(args.output).expanduser().resolve()
        json_path = (
            output_dir / "detections.json"
            if args.json_output is None
            else Path(args.json_output).expanduser().resolve()
        )
        output_paths = {
            image_path: output_dir / f"{image_path.stem}_detected.jpg"
            for image_path in image_paths
        }
    else:
        image_path = image_paths[0]
        output_path = (
            image_path.with_name(f"{image_path.stem}_experiment_10_detected.jpg")
            if args.output is None
            else Path(args.output).expanduser().resolve()
        )
        json_path = (
            output_path.with_suffix(".json")
            if args.json_output is None
            else Path(args.json_output).expanduser().resolve()
        )
        output_paths = {image_path: output_path}

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return output_paths, json_path


def build_model(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train experiment_10 first or pass --checkpoint /path/to/model.weights.h5"
        )

    detector = RFDETRNano(num_classes=1)

    # Build all layers before loading the H5 weights.
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.model.load_weights(str(checkpoint_path))
    detector.model.class_names = DEFAULT_CLASS_NAMES
    return detector


def label_name(label, class_names):
    label = int(label)
    if 0 <= label < len(class_names):
        return class_names[label]
    if 1 <= label <= len(class_names):
        return class_names[label - 1]
    return str(label)


def draw_detections(image, detections, class_names):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", TEXT_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    width, height = annotated.size

    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        text = f"{label_name(label, class_names)} {float(score):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=3)

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        text_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            (x1, text_y, x1 + text_w + 6, text_y + text_h + 4),
            fill=TEXT_BG_COLOR,
        )
        draw.text((x1 + 3, text_y + 2), text, fill=TEXT_COLOR, font=font)

    return annotated


def detections_to_records(detections, class_names):
    records = []
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        records.append(
            {
                "box_xyxy": [float(v) for v in box],
                "score": float(score),
                "label": int(label),
                "class_name": label_name(label, class_names),
            }
        )
    return records


def main():
    args = parse_args()
    input_path = Path(args.image).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]

    if not class_names:
        raise ValueError("--class-names must contain at least one class name")

    image_paths = collect_image_paths(input_path, recursive=args.recursive)
    output_paths, json_path = make_output_paths(input_path, image_paths, args)

    print(f"Loading checkpoint: {checkpoint_path}")
    detector = build_model(checkpoint_path)

    payload = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "threshold": args.threshold,
        "num_images": len(image_paths),
        "images": [],
    }

    total_detections = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_paths[image_path]
        print(f"[{index}/{len(image_paths)}] Detecting: {image_path}")

        image, image_array = load_rgb_image(image_path)
        result = detector.predict(image_array, threshold=args.threshold)[0]
        records = detections_to_records(result, class_names)
        total_detections += len(records)

        annotated = draw_detections(image, result, class_names)
        annotated.save(output_path)

        payload["images"].append(
            {
                "image": str(image_path),
                "annotated_image": str(output_path),
                "num_detections": len(records),
                "detections": records,
            }
        )

        print(f"  Detections: {len(records)}")
        for record in records:
            box = ", ".join(f"{v:.1f}" for v in record["box_xyxy"])
            print(
                f"    {record['class_name']} score={record['score']:.3f} "
                f"box=[{box}]"
            )

    payload["total_detections"] = total_detections
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Processed images: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Detection JSON: {json_path}")


if __name__ == "__main__":
    main()
