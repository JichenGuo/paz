#!/usr/bin/env python
"""Run inference with the best RF-DETR Nano checkpoint from Experiment 10."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))

from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano, RFDETRLarge


DEFAULT_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "rfdetr_nano_best.weights.h5"
DEFAULT_CLASS_NAMES = ["fish"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Publication-friendly annotation style.
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 0, 0)
MIN_TEXT_FONT_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the best RF-DETR Nano or Large model checkpoint and run "
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
        default=None,
        help=(
            "Comma-separated class names. When omitted, this is read from "
            "finetune_config.json next to the checkpoint when available, "
            "otherwise defaults to fish."
        ),
    )
    parser.add_argument(
        "--detector",
        default="NANO",
        help="Choose nano or large detector.",
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


def read_finetune_class_names(checkpoint_path):
    config_path = checkpoint_path.parent / "finetune_config.json"
    if not config_path.exists():
        return None

    with config_path.open() as f:
        config = json.load(f)
    class_names = config.get("class_names")
    if not class_names:
        return None
    return [str(name) for name in class_names]


def resolve_class_names(args, checkpoint_path):
    if args.class_names is not None:
        class_names = [
            name.strip() for name in args.class_names.split(",") if name.strip()
        ]
    else:
        class_names = read_finetune_class_names(checkpoint_path) or DEFAULT_CLASS_NAMES

    if not class_names:
        raise ValueError("--class-names must contain at least one class name")
    return class_names


def build_model_nano(checkpoint_path, class_names):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train experiment_10 first or pass --checkpoint /path/to/model.weights.h5"
        )

    detector = RFDETRNano(num_classes=len(class_names))

    # Build all layers before loading the H5 weights.
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    detector.model.class_names = class_names
    return detector


def build_model_large(checkpoint_path, class_names):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train experiment_10 first or pass --checkpoint /path/to/model.weights.h5"
        )

    detector = RFDETRLarge(num_classes=len(class_names))

    # Build all layers before loading the H5 weights.
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    detector.model.class_names = class_names
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
    width, height = annotated.size

    # Scale annotations with the image so they remain readable when a
    # high-resolution result is reduced to a single- or double-column figure.
    short_side = min(width, height)
    font_size = max(MIN_TEXT_FONT_SIZE, round(short_side * 0.04))
    box_width = max(6, round(short_side * 0.008))
    label_gap = max(3, round(box_width * 0.5))
    text_stroke_width = max(1, round(font_size * 0.04))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

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

        #text = label_name(label, class_names)
        text = f"{label_name(label, class_names)} {float(score):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=box_width)

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        text_x = min(max(0, x1), max(0, width - text_w))
        if y1 >= text_h + label_gap:
            text_y = y1 - text_h - label_gap
        else:
            text_y = min(y1 + label_gap, max(0, height - text_h))

        draw.text(
            (text_x - left, text_y - top),
            text,
            fill=TEXT_COLOR,
            font=font,
            stroke_width=text_stroke_width,
            stroke_fill=(0, 0, 0),
        )

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
    class_names = resolve_class_names(args, checkpoint_path)

    image_paths = collect_image_paths(input_path, recursive=args.recursive)
    output_paths, json_path = make_output_paths(input_path, image_paths, args)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Classes ({len(class_names)}): {class_names}")

    if args.detector.lower() == "nano":
        print("the detector is nano")
        detector = build_model_nano(checkpoint_path, class_names)
    
    elif args.detector.lower() == "large":
        print("the detector is large")
        detector = build_model_large(checkpoint_path, class_names)

    payload = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "threshold": args.threshold,
        "num_images": len(image_paths),
        "images": [],
    }

    total_detections = 0
    inference_times_ms = []
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_paths[image_path]
        print(f"[{index}/{len(image_paths)}] Detecting: {image_path}")

        image, image_array = load_rgb_image(image_path)
        inference_start = time.perf_counter()
        result = detector.predict(image_array, threshold=args.threshold)[0]
        # Converting outputs to Python values also synchronizes asynchronous
        # accelerator work before stopping the timer.
        records = detections_to_records(result, class_names)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
        inference_times_ms.append(inference_time_ms)
        total_detections += len(records)

        annotated = draw_detections(image, result, class_names)
        annotated.save(output_path)

        payload["images"].append(
            {
                "image": str(image_path),
                "annotated_image": str(output_path),
                "inference_time_ms": inference_time_ms,
                "num_detections": len(records),
                "detections": records,
            }
        )

        print(
            f"  Detections: {len(records)}; "
            f"inference: {inference_time_ms:.2f} ms"
        )
        for record in records:
            box = ", ".join(f"{v:.1f}" for v in record["box_xyxy"])
            print(
                f"    {record['class_name']} score={record['score']:.3f} "
                f"box=[{box}]"
            )

    payload["total_detections"] = total_detections
    total_inference_seconds = sum(inference_times_ms) / 1000.0
    payload["inference_timing"] = {
        "scope": (
            "detector prediction and output materialization; excludes image "
            "loading, annotation drawing, and file writing"
        ),
        "total_seconds": total_inference_seconds,
        "mean_ms_per_image": float(np.mean(inference_times_ms)),
        "median_ms_per_image": float(np.median(inference_times_ms)),
        "p95_ms_per_image": float(np.percentile(inference_times_ms, 95)),
        "inference_fps": len(inference_times_ms) / total_inference_seconds,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Processed images: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    timing = payload["inference_timing"]
    print(
        "Inference timing: "
        f"total={timing['total_seconds']:.3f} s, "
        f"mean={timing['mean_ms_per_image']:.2f} ms/image, "
        f"median={timing['median_ms_per_image']:.2f} ms/image, "
        f"p95={timing['p95_ms_per_image']:.2f} ms/image, "
        f"FPS={timing['inference_fps']:.2f}"
    )
    print(f"Detection JSON: {json_path}")


if __name__ == "__main__":
    main()
