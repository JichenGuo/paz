#!/usr/bin/env python
"""Run RF-DETR Nano detection on a video and save an annotated video."""

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
    draw_detections,
    build_model,
    resolve_class_names,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
COUNT_CLASSES = ("fish", "crab", "lobster")
COUNT_TEXT_COLOR = (255, 255, 255)
COUNT_BG_COLOR = (0, 0, 0)
COUNT_FONT_SIZE = 28
COUNT_MARGIN = 14
COUNT_PADDING = 8


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
        "--count-classes",
        default=",".join(COUNT_CLASSES),
        help=(
            "Comma-separated class names to count in the top-right overlay. "
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


def draw_count_overlay(image, counts):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", COUNT_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    lines = [f"{name}: {count}" for name, count in counts.items()]
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [right - left for left, top, right, bottom in text_boxes]
    line_heights = [bottom - top for left, top, right, bottom in text_boxes]
    line_gap = max(4, COUNT_FONT_SIZE // 5)
    box_width = max(line_widths) + 2 * COUNT_PADDING
    box_height = (
        sum(line_heights)
        + line_gap * max(0, len(lines) - 1)
        + 2 * COUNT_PADDING
    )

    image_width, _ = annotated.size
    x1 = max(0, image_width - box_width - COUNT_MARGIN)
    y1 = COUNT_MARGIN
    x2 = x1 + box_width
    y2 = y1 + box_height
    draw.rectangle((x1, y1, x2, y2), fill=(*COUNT_BG_COLOR, 170))

    y = y1 + COUNT_PADDING
    for line, line_height, text_box in zip(lines, line_heights, text_boxes):
        left, _, right, _ = text_box
        text_width = right - left
        x = x2 - COUNT_PADDING - text_width
        draw.text((x, y), line, fill=COUNT_TEXT_COLOR, font=font)
        y += line_height + line_gap

    return annotated


def annotate_frame(frame_bgr, detector, class_names, count_classes, threshold):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.predict(frame_rgb, threshold=threshold)[0]
    records = detections_to_records(result, class_names)
    annotated = draw_detections(Image.fromarray(frame_rgb), result, class_names)
    counts = count_records(records, count_classes)
    annotated = draw_count_overlay(annotated, counts)
    annotated_bgr = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
    return annotated_bgr, records, counts


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

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Count overlay classes: {count_classes}")
    detector = build_model(checkpoint_path, class_names)

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
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame_index = args.start_frame + processed
            annotated, records, counts = annotate_frame(
                frame,
                detector,
                class_names,
                count_classes,
                threshold=args.threshold,
            )
            writer.write(annotated)

            total_detections += len(records)
            if json_path is not None:
                payload["frames"].append(
                    {
                        "frame_index": frame_index,
                        "num_detections": len(records),
                        "counts": counts,
                        "detections": records,
                    }
                )

            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"Processed {processed} frames "
                    f"(last frame detections: {len(records)})"
                )
    finally:
        capture.release()
        writer.release()

    payload["processed_frames"] = processed
    payload["total_detections"] = total_detections
    write_json(json_path, payload)

    print(f"Processed frames: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Annotated video: {output_path}")
    if json_path is not None:
        print(f"Detection JSON: {json_path}")


if __name__ == "__main__":
    main()
