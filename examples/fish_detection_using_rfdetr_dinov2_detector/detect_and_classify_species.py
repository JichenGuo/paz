#!/usr/bin/env python
"""Two-stage fish detection and sea-animal species classification."""

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
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from detect_image import (  # noqa: E402
    collect_image_paths,
    label_name,
    load_rgb_image,
    resolve_class_names,
)
from species_classifier import (  # noqa: E402
    build_fcn_classifier,
    crop_with_padding,
    resize_letterbox,
)
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano, RFDETRLarge  # noqa: E402


DEFAULT_DETECTOR_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "rfdetr_nano_best.weights.h5"
DEFAULT_CLASSIFIER_DIR = _SCRIPT_DIR / "species_classifier_runs" / "fathomnet_fcn"
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 96, 120)
TEXT_FONT_SIZE = 26


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run RF-DETR detection first, crop each detection, then classify "
            "the crop with the FathomNet FCN species classifier."
        )
    )
    parser.add_argument("--image", required=True, help="Input image or image folder.")
    parser.add_argument(
        "--detector-checkpoint",
        default=str(DEFAULT_DETECTOR_CHECKPOINT),
        help="Fine-tuned RF-DETR .weights.h5 file.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=None,
        help=(
            "FCN classifier .weights.h5 file. Defaults to "
            "CLASSIFIER_DIR/fathomnet_fcn_best.weights.h5."
        ),
    )
    parser.add_argument(
        "--classifier-dir",
        default=str(DEFAULT_CLASSIFIER_DIR),
        help="Directory containing classifier_config.json.",
    )
    parser.add_argument(
        "--detector",
        default="NANO",
        help="Choose nano or large detector.",
    )    
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-padding", type=float, default=None)
    parser.add_argument("--detector-class-names", default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def build_detector_nano(checkpoint_path, class_names):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    detector = RFDETRNano(num_classes=len(class_names))
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    detector.model.class_names = class_names
    return detector


 def build_detector_large(checkpoint_path, class_names):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    detector = RFDETRLarge(num_classes=len(class_names))
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    detector.model.class_names = class_names
    return detector   


def load_classifier(classifier_dir, classifier_checkpoint):
    classifier_dir = Path(classifier_dir).expanduser().resolve()
    config_path = classifier_dir / "classifier_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Classifier config not found: {config_path}\n"
            "Train train_fathomnet_fcn_classifier.py first or pass --classifier-dir."
        )
    with config_path.open() as f:
        config = json.load(f)

    checkpoint_path = (
        Path(classifier_checkpoint).expanduser().resolve()
        if classifier_checkpoint
        else classifier_dir / "fathomnet_fcn_best.weights.h5"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    image_size = int(config["image_size"])
    class_names = [str(name) for name in config["class_names"]]
    model = build_fcn_classifier(
        input_shape=(image_size, image_size, 3),
        num_classes=len(class_names),
        width=int(config.get("width", 32)),
        dropout=float(config.get("dropout", 0.25)),
    )
    model(np.zeros((1, image_size, image_size, 3), dtype="float32"), training=False)
    model.load_weights(str(checkpoint_path))
    return model, class_names, config, checkpoint_path


def make_output_paths(input_path, image_paths, output_arg, json_output_arg):
    folder_mode = input_path.is_dir()
    if folder_mode:
        output_dir = (
            input_path / "experiment_10_species"
            if output_arg is None
            else Path(output_arg).expanduser().resolve()
        )
        json_path = (
            output_dir / "species_detections.json"
            if json_output_arg is None
            else Path(json_output_arg).expanduser().resolve()
        )
        output_paths = {
            image_path: output_dir / f"{image_path.stem}_species.jpg"
            for image_path in image_paths
        }
    else:
        image_path = image_paths[0]
        output_path = (
            image_path.with_name(f"{image_path.stem}_species.jpg")
            if output_arg is None
            else Path(output_arg).expanduser().resolve()
        )
        json_path = (
            output_path.with_suffix(".json")
            if json_output_arg is None
            else Path(json_output_arg).expanduser().resolve()
        )
        output_paths = {image_path: output_path}

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return output_paths, json_path


def classify_crop(classifier, class_names, image, box_xyxy, image_size, crop_padding, top_k):
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    crop = crop_with_padding(image, [x1, y1, x2 - x1, y2 - y1], crop_padding)
    crop = resize_letterbox(crop, image_size)
    batch = np.asarray(crop, dtype="float32")[None, ...]
    probs = np.asarray(classifier.predict(batch, verbose=0)[0])
    top_k = max(1, min(int(top_k), len(class_names)))
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [
        {
            "class_name": class_names[int(index)],
            "label": int(index),
            "score": float(probs[int(index)]),
        }
        for index in top_indices
    ]
    return predictions


def detector_label_name(label, class_names):
    return label_name(label, class_names)


def draw_records(image, records):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", TEXT_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    width, height = annotated.size

    for record in records:
        x1, y1, x2, y2 = record["box_xyxy"]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        species = record["species_top1"]
        text = (
            f"{species['class_name']} {species['score']:.2f} "
            f"det {record['detector_score']:.2f}"
        )
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=3)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        text_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            (x1, text_y, x1 + text_w + 8, text_y + text_h + 4),
            fill=TEXT_BG_COLOR,
        )
        draw.text((x1 + 4, text_y + 2), text, fill=TEXT_COLOR, font=font)

    return annotated


def main():
    args = parse_args()
    input_path = Path(args.image).expanduser().resolve()
    detector_checkpoint = Path(args.detector_checkpoint).expanduser().resolve()

    detector_args = argparse.Namespace(
        class_names=args.detector_class_names,
    )
    detector_class_names = resolve_class_names(detector_args, detector_checkpoint)
    classifier, species_names, classifier_config, classifier_checkpoint = load_classifier(
        args.classifier_dir,
        args.classifier_checkpoint,
    )
    image_size = int(classifier_config["image_size"])
    crop_padding = (
        float(classifier_config.get("crop_padding", 0.12))
        if args.crop_padding is None
        else args.crop_padding
    )

    image_paths = collect_image_paths(input_path, recursive=args.recursive)
    output_paths, json_path = make_output_paths(
        input_path,
        image_paths,
        args.output,
        args.json_output,
    )

    print(f"Detector checkpoint: {detector_checkpoint}")
    print(f"Detector classes: {detector_class_names}")
    print(f"Classifier checkpoint: {classifier_checkpoint}")
    print(f"Species classes ({len(species_names)}): {species_names}")
    
    if args.detector.lower() == "nano":
        detector = build_detector_nano(detector_checkpoint, detector_class_names)
    elif args.detector.lower() == "large":
        detector = build_detector_large(detector_checkpoint, detector_class_names)

    payload = {
        "input": str(input_path),
        "detector_checkpoint": str(detector_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "threshold": args.threshold,
        "top_k": args.top_k,
        "crop_padding": crop_padding,
        "images": [],
    }

    total_detections = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_paths[image_path]
        print(f"[{index}/{len(image_paths)}] Detecting and classifying: {image_path}")
        image, image_array = load_rgb_image(image_path)
        detections = detector.predict(image_array, threshold=args.threshold)[0]

        records = []
        for box, det_score, det_label in zip(
            detections["boxes"],
            detections["scores"],
            detections["labels"],
        ):
            species_predictions = classify_crop(
                classifier,
                species_names,
                image,
                box,
                image_size,
                crop_padding,
                args.top_k,
            )
            records.append(
                {
                    "box_xyxy": [float(value) for value in box],
                    "detector_score": float(det_score),
                    "detector_label": int(det_label),
                    "detector_class_name": detector_label_name(
                        det_label,
                        detector_class_names,
                    ),
                    "species_top1": species_predictions[0],
                    "species_topk": species_predictions,
                }
            )

        annotated = draw_records(image, records)
        annotated.save(output_path)
        total_detections += len(records)
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
            top1 = record["species_top1"]
            print(
                f"    {top1['class_name']} species={top1['score']:.3f} "
                f"det={record['detector_score']:.3f}"
            )

    payload["total_detections"] = total_detections
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Processed images: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Species JSON: {json_path}")


if __name__ == "__main__":
    main()
