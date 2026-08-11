#!/usr/bin/env python
"""Run RF-DETR video detection and report per-species MaxN."""

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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))

from detect_image import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    detections_to_records,
    build_model_nano,
    build_model_large,
    resolve_class_names,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
COUNT_CLASSES = ("fish", "crab", "lobster")
CLASS_COLORS = {
    "fish": (0, 90, 255),
    "crab": (145, 40, 200),
    "lobster": (255, 110, 0),
}
DEFAULT_CLASS_COLORS = (
    (0, 160, 180), (30, 170, 70), (220, 80, 40),
    (190, 140, 0), (170, 50, 120),
)
DETECTION_TEXT_COLOR = (255, 255, 255)
DETECTION_TEXT_STROKE_COLOR = (0, 0, 0)
LABEL_BORDER_COLOR = (255, 255, 255)
MIN_DETECTION_FONT_SIZE = 48
COUNT_TEXT_COLOR = (255, 255, 255)
COUNT_TEXT_STROKE_COLOR = (0, 0, 0)
COUNT_BG_COLOR = (0, 35, 90)
COUNT_BORDER_COLOR = (255, 255, 255)
MIN_COUNT_FONT_SIZE = 48


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load an Experiment 10 or fine-tuned RF-DETR Nano checkpoint, "
            "detect objects in a video, and save the annotated video."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
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
            "Path for the annotated output video. Defaults next to the input "
            "as INPUT_STEM_detected.mp4."
        ),
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help=(
            "Optional path for frame-level detection JSON. Omit to skip JSON "
            "writing."
        ),
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
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to process.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process after --start-frame.",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV fourcc code for the output video.",
    )
    parser.add_argument(
        "--detector",
        default="NANO",
        help="Choose nano or large detector.",
    )
    parser.add_argument(
        "--count-classes",
        default=",".join(COUNT_CLASSES),
        help=(
            "Comma-separated class names for per-species MaxN in the overlay. "
            "Defaults to fish,crab,lobster."
        ),
    )
    return parser.parse_args()


def make_output_path(video_path, output):
    if output is None:
        return video_path.with_name(f"{video_path.stem}_detected.mp4")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def open_video(video_path):
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"Warning: unlisted video extension: {video_path.suffix}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
    return capture


def video_metadata(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read input video frame size.")
    return fps, width, height, frame_count


def make_writer(output_path, fps, width, height, fourcc):
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")
    return writer


def parse_count_classes(value):
    count_classes = [name.strip() for name in value.split(",") if name.strip()]
    if not count_classes:
        raise ValueError("--count-classes must contain at least one class name")
    return count_classes


def count_records(records, count_classes):
    counts = {name: 0 for name in count_classes}
    name_lookup = {name.lower(): name for name in count_classes}
    for record in records:
        class_name = str(record["class_name"]).lower()
        if class_name in name_lookup:
            counts[name_lookup[class_name]] += 1
    return counts


def make_class_colors(class_names):
    colors = {}
    for index, name in enumerate(class_names):
        colors[name.lower()] = CLASS_COLORS.get(
            name.lower(), DEFAULT_CLASS_COLORS[index % len(DEFAULT_CLASS_COLORS)]
        )
    return colors


def draw_detections_by_class(image, records, class_colors):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    short_side = min(image_width, image_height)
    font_size = max(MIN_DETECTION_FONT_SIZE, round(short_side * 0.065))
    box_width = max(7, round(short_side * 0.009))
    text_stroke_width = max(2, round(font_size * 0.06))
    label_gap = max(4, round(box_width * 0.5))
    padding_x = max(9, round(font_size * 0.3))
    padding_y = max(6, round(font_size * 0.18))
    label_border_width = max(2, round(box_width * 0.35))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for record in records:
        class_name = str(record["class_name"])
        color = class_colors.get(class_name.lower(), DEFAULT_CLASS_COLORS[0])
        x1, y1, x2, y2 = [float(value) for value in record["box_xyxy"]]
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        label = f'{class_name} {float(record["score"]):.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 2 * padding_x
        label_height = bottom - top + 2 * padding_y
        label_x = min(max(0, x1), max(0, image_width - label_width))
        if y1 >= label_height + label_gap:
            label_y = y1 - label_height - label_gap
        else:
            label_y = min(y1 + label_gap, max(0, image_height - label_height))
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color, outline=LABEL_BORDER_COLOR, width=label_border_width,
        )
        draw.text(
            (label_x + padding_x - left, label_y + padding_y - top),
            label, fill=DETECTION_TEXT_COLOR, font=font,
            stroke_width=text_stroke_width,
            stroke_fill=DETECTION_TEXT_STROKE_COLOR,
        )
    return annotated


def draw_count_overlay(image, maxn_counts):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    image_width, image_height = annotated.size
    short_side = min(image_width, image_height)
    font_size = max(MIN_COUNT_FONT_SIZE, round(short_side * 0.06))
    padding = max(12, round(font_size * 0.4))
    margin = max(12, round(short_side * 0.018))
    line_gap = max(8, round(font_size * 0.25))
    border_width = max(3, round(short_side * 0.004))
    text_stroke_width = max(1, round(font_size * 0.04))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    lines = [f"MaxN {name}: {count}" for name, count in maxn_counts.items()]
    if not lines:
        return annotated
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [right - left for left, top, right, bottom in text_boxes]
    line_heights = [bottom - top for left, top, right, bottom in text_boxes]
    box_width = max(line_widths) + 2 * padding
    box_height = (
        sum(line_heights)
        + line_gap * max(0, len(lines) - 1)
        + 2 * padding
    )

    x1 = max(0, image_width - box_width - margin)
    y1 = max(0, image_height - box_height - margin)
    x2 = x1 + box_width
    y2 = y1 + box_height
    draw.rectangle(
        (x1, y1, x2, y2), fill=(*COUNT_BG_COLOR, 230),
        outline=(*COUNT_BORDER_COLOR, 255), width=border_width,
    )

    y = y1 + padding
    for line, line_height, text_box in zip(lines, line_heights, text_boxes):
        left, top, right, _ = text_box
        text_width = right - left
        x = x2 - padding - text_width
        draw.text(
            (x - left, y - top), line, fill=COUNT_TEXT_COLOR, font=font,
            stroke_width=text_stroke_width,
            stroke_fill=COUNT_TEXT_STROKE_COLOR,
        )
        y += line_height + line_gap

    return annotated


def annotate_frame(
    frame_bgr, detector, class_names, count_classes, class_colors, maxn_counts,
    threshold,
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inference_start = time.perf_counter()
    result = detector.predict(frame_rgb, threshold=threshold)[0]
    inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
    records = detections_to_records(result, class_names)
    counts = count_records(records, count_classes)
    for name, count in counts.items():
        maxn_counts[name] = max(maxn_counts[name], count)
    annotated = draw_detections_by_class(
        Image.fromarray(frame_rgb), records, class_colors
    )
    annotated = draw_count_overlay(annotated, maxn_counts)
    annotated_bgr = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
    return annotated_bgr, records, counts, inference_time_ms


def write_json(json_path, payload):
    if json_path is None:
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = make_output_path(video_path, args.output)
    json_path = (
        None
        if args.json_output is None
        else Path(args.json_output).expanduser().resolve()
    )
    class_names = resolve_class_names(args, checkpoint_path)
    count_classes = parse_count_classes(args.count_classes)
    class_colors = make_class_colors(class_names)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"MaxN classes: {count_classes}")
    
    if args.detector.lower() == "nano":
        detector = build_model_nano(checkpoint_path, class_names)
    if args.detector.lower() == "large":
        detector = build_model_large(checkpoint_path, class_names)
    
    capture = open_video(video_path)
    fps, width, height, frame_count = video_metadata(capture)
    writer = make_writer(output_path, fps, width, height, args.fourcc)

    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    payload = {
        "input": str(video_path),
        "checkpoint": str(checkpoint_path),
        "annotated_video": str(output_path),
        "class_names": class_names,
        "count_classes": count_classes,
        "count_metric": "MaxN",
        "threshold": args.threshold,
        "fps": fps,
        "width": width,
        "height": height,
        "input_frame_count": frame_count,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "frames": [],
    }

    processed = 0
    total_detections = 0
    maxn_counts = {name: 0 for name in count_classes}
    inference_times_ms = []
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame_index = args.start_frame + processed
            annotated, records, counts, inference_time_ms = annotate_frame(
                frame,
                detector,
                class_names,
                count_classes,
                class_colors,
                maxn_counts,
                threshold=args.threshold,
            )
            inference_times_ms.append(inference_time_ms)
            writer.write(annotated)

            total_detections += len(records)
            if json_path is not None:
                payload["frames"].append(
                    {
                        "frame_index": frame_index,
                        "num_detections": len(records),
                        "inference_time_ms": inference_time_ms,
                        "inference_fps": (
                            1000.0 / inference_time_ms
                            if inference_time_ms > 0.0
                            else None
                        ),
                        "counts": counts,
                        "maxn": dict(maxn_counts),
                        "detections": records,
                    }
                )

            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"Processed {processed} frames "
                    f"(last frame detections: {len(records)}, "
                    f"MaxN: {dict(maxn_counts)}, "
                    f"inference: {inference_time_ms:.1f} ms)"
                )
    finally:
        capture.release()
        writer.release()

    payload["processed_frames"] = processed
    payload["total_detections"] = total_detections
    payload["final_maxn"] = dict(maxn_counts)
    if inference_times_ms:
        total_inference_seconds = sum(inference_times_ms) / 1000.0
        inference_summary = {
            "scope": "detector.predict only; excludes video I/O and rendering",
            "total_seconds": total_inference_seconds,
            "mean_ms_per_frame": float(np.mean(inference_times_ms)),
            "median_ms_per_frame": float(np.median(inference_times_ms)),
            "p95_ms_per_frame": float(np.percentile(inference_times_ms, 95)),
            "inference_fps": processed / total_inference_seconds,
        }
    else:
        inference_summary = {
            "scope": "detector.predict only; excludes video I/O and rendering",
            "total_seconds": 0.0,
            "mean_ms_per_frame": None,
            "median_ms_per_frame": None,
            "p95_ms_per_frame": None,
            "inference_fps": None,
        }
    payload["inference_timing"] = inference_summary
    write_json(json_path, payload)

    print(f"Processed frames: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Final MaxN: {dict(maxn_counts)}")
    if inference_times_ms:
        print(
            "Inference timing: "
            f"total={inference_summary['total_seconds']:.3f} s, "
            f"mean={inference_summary['mean_ms_per_frame']:.2f} ms/frame, "
            f"median={inference_summary['median_ms_per_frame']:.2f} ms/frame, "
            f"p95={inference_summary['p95_ms_per_frame']:.2f} ms/frame, "
            f"FPS={inference_summary['inference_fps']:.2f}"
        )
    print(f"Annotated video: {output_path}")
    if json_path is not None:
        print(f"Detection JSON: {json_path}")


if __name__ == "__main__":
    main()
