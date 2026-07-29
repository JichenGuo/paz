#!/usr/bin/env python
"""Track RF-DETR video detections with DeepSORT and count valid tracks."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Must be set before importing keras/paz through detect_video.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detect_video import (  # noqa: E402
    COUNT_CLASSES,
    DEFAULT_CHECKPOINT,
    build_model_nano,
    build_model_large,
    detections_to_records,
    make_writer,
    open_video,
    parse_count_classes,
    resolve_class_names,
    video_metadata,
    write_json,
)


TRACK_CLASS_COLORS = {
    "fish": (0, 90, 255),
    "crab": (145, 40, 200),
    "lobster": (255, 110, 0),
}
DEFAULT_TRACK_COLOR = (0, 160, 180)
TRACK_TEXT_COLOR = (255, 255, 255)
TRACK_TEXT_STROKE_COLOR = (0, 0, 0)
TRACK_LABEL_BORDER_COLOR = (255, 255, 255)
MIN_TRACK_FONT_SIZE = 34
COUNT_TEXT_COLOR = (255, 255, 255)
COUNT_TEXT_STROKE_COLOR = (0, 0, 0)
COUNT_BG_COLOR = (0, 35, 90)
COUNT_BORDER_COLOR = (255, 255, 255)
MIN_COUNT_FONT_SIZE = 34


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect fish/crab/lobster in a video with RF-DETR, track them "
            "with DeepSORT, and count each track once after it reaches a "
            "minimum track length."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the tracked output video. Defaults next to the input as "
            "INPUT_STEM_deepsort_counted.mp4."
        ),
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for frame-level tracking/counting JSON.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--class-names",
        default=None,
        help=(
            "Comma-separated detector class names. When omitted, this is read "
            "from finetune_config.json next to the checkpoint when available, "
            "otherwise defaults to fish."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument(
        "--count-classes",
        default=",".join(COUNT_CLASSES),
        help="Comma-separated class names to count. Defaults to fish,crab,lobster.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Count a track only after it has been observed for at least N frames.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="DeepSORT max_age: frames to keep an unmatched track alive.",
    )
    parser.add_argument(
        "--max-draw-missed-frames",
        type=int,
        default=5,
        help=(
            "Draw predicted tracks for at most this many missed frames. "
            "The track still stays alive internally until --max-age."
        ),
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="DeepSORT n_init: detections required before a track is confirmed.",
    )
    parser.add_argument(
        "--max-iou-distance",
        type=float,
        default=0.7,
        help="DeepSORT IoU matching threshold.",
    )
    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=0.2,
        help="DeepSORT appearance matching threshold.",
    )
    parser.add_argument(
        "--detector",
        default="NANO",
        help="Choose nano or large detector.",
    )
    parser.add_argument(
        "--embedder",
        default="mobilenet",
        help=(
            "deep-sort-realtime embedder name. Use 'none' to disable "
            "appearance embeddings and track from boxes only."
        ),
    )
    return parser.parse_args()


def make_output_path(video_path, output):
    if output is None:
        return video_path.with_name(f"{video_path.stem}_deepsort_counted.mp4")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def build_tracker(args):
    embedder = None if args.embedder.lower() == "none" else args.embedder
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as exc:
        raise ImportError(
            "DeepSORT tracking requires deep-sort-realtime. Install it with:\n"
            "  pip install deep-sort-realtime\n"
            "Then rerun this script."
        ) from exc

    return DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.max_iou_distance,
        max_cosine_distance=args.max_cosine_distance,
        embedder=embedder,
        half=False,
        bgr=True,
    )


def record_to_deepsort_detection(record, count_classes):
    class_name = str(record["class_name"])
    if class_name.lower() not in count_classes:
        return None

    x1, y1, x2, y2 = [float(value) for value in record["box_xyxy"]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None
    return ([x1, y1, width, height], float(record["score"]), class_name)


def detections_to_deepsort(records, count_classes):
    count_class_lookup = {name.lower() for name in count_classes}
    detections = []
    for record in records:
        detection = record_to_deepsort_detection(record, count_class_lookup)
        if detection is not None:
            detections.append(detection)
    return detections


def track_class_name(track):
    if hasattr(track, "get_det_class"):
        class_name = track.get_det_class()
        if class_name is not None:
            return str(class_name)
    if hasattr(track, "det_class") and track.det_class is not None:
        return str(track.det_class)
    return "unknown"


def remembered_track_class_name(track, track_class_names):
    track_id = str(track.track_id)
    class_name = track_class_name(track)
    if class_name != "unknown":
        return class_name
    return track_class_names.get(track_id, class_name)


def track_bbox_xyxy(track):
    left, top, right, bottom = track.to_ltrb()
    return [float(left), float(top), float(right), float(bottom)]


def draw_track_overlay(
    image,
    tracks,
    track_lengths,
    counted_track_ids,
    counted_display_ids,
    track_class_names,
    max_draw_missed_frames,
):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    short_side = min(image_width, image_height)
    font_size = max(MIN_TRACK_FONT_SIZE, round(short_side * 0.045))
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
    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > max_draw_missed_frames:
            continue

        track_id = str(track.track_id)
        x1, y1, x2, y2 = track_bbox_xyxy(track)
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = remembered_track_class_name(track, track_class_names)
        if track_id in counted_track_ids:
            label = f"{class_name} ID {counted_display_ids[track_id]}"
        else:
            label = class_name
        track_color = TRACK_CLASS_COLORS.get(
            class_name.lower(), DEFAULT_TRACK_COLOR
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=track_color,
            width=box_width,
        )
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        label_width = text_width + 2 * padding_x
        label_height = text_height + 2 * padding_y
        label_x = min(max(0, x1), max(0, image_width - label_width))
        if y1 >= label_height + label_gap:
            label_y = y1 - label_height - label_gap
        else:
            label_y = min(y1 + label_gap, max(0, image_height - label_height))
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=track_color,
            outline=TRACK_LABEL_BORDER_COLOR,
            width=label_border_width,
        )
        draw.text(
            (label_x + padding_x - left, label_y + padding_y - top),
            label,
            fill=TRACK_TEXT_COLOR,
            font=font,
            stroke_width=text_stroke_width,
            stroke_fill=TRACK_TEXT_STROKE_COLOR,
        )
    return annotated


def draw_count_overlay(image, counts):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    image_width, image_height = annotated.size
    short_side = min(image_width, image_height)
    font_size = max(MIN_COUNT_FONT_SIZE, round(short_side * 0.04))
    padding = max(12, round(font_size * 0.4))
    margin = max(12, round(short_side * 0.018))
    line_gap = max(8, round(font_size * 0.25))
    border_width = max(3, round(short_side * 0.004))
    text_stroke_width = max(1, round(font_size * 0.04))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    lines = [f"{name}: {count}" for name, count in counts.items()]
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
    y1 = margin
    x2 = x1 + box_width
    y2 = y1 + box_height
    draw.rectangle(
        (x1, y1, x2, y2),
        fill=(*COUNT_BG_COLOR, 230),
        outline=(*COUNT_BORDER_COLOR, 255),
        width=border_width,
    )

    y = y1 + padding
    for line, line_height, text_box in zip(lines, line_heights, text_boxes):
        left, top, right, bottom = text_box
        text_width = right - left
        x = x2 - padding - text_width
        draw.text(
            (x - left, y - top),
            line,
            fill=COUNT_TEXT_COLOR,
            font=font,
            stroke_width=text_stroke_width,
            stroke_fill=COUNT_TEXT_STROKE_COLOR,
        )
        y += line_height + line_gap

    return annotated


def count_tracks_once(
    tracks,
    track_lengths,
    counted_track_ids,
    counted_display_ids,
    track_class_names,
    counts,
    min_frames,
):
    newly_counted = []
    count_lookup = {name.lower(): name for name in counts}
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = str(track.track_id)
        class_name = remembered_track_class_name(track, track_class_names)
        normalized_class = class_name.lower()
        if normalized_class not in count_lookup:
            continue
        if track_id in counted_track_ids:
            continue
        if track_lengths[track_id] < min_frames:
            continue

        counted_name = count_lookup[normalized_class]
        counts[counted_name] += 1
        counted_display_ids[track_id] = counts[counted_name]
        counted_track_ids.add(track_id)
        newly_counted.append(track_id)
    return newly_counted


def tracks_to_records(
    tracks, track_lengths, counted_track_ids, counted_display_ids, track_class_names
):
    records = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = str(track.track_id)
        records.append(
            {
                "track_id": track_id,
                "display_id": counted_display_ids.get(track_id),
                "class_name": remembered_track_class_name(track, track_class_names),
                "box_xyxy": track_bbox_xyxy(track),
                "length_frames": int(track_lengths.get(track_id, 0)),
                "time_since_update": int(track.time_since_update),
                "counted": track_id in counted_track_ids,
            }
        )
    return records


def annotate_frame(
    frame_bgr,
    detector,
    tracker,
    class_names,
    count_classes,
    threshold,
    track_lengths,
    counted_track_ids,
    counted_display_ids,
    track_class_names,
    counts,
    min_track_frames,
    max_draw_missed_frames,
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.predict(frame_rgb, threshold=threshold)[0]
    detection_records = detections_to_records(result, class_names)
    deepsort_detections = detections_to_deepsort(detection_records, count_classes)
    tracks = tracker.update_tracks(deepsort_detections, frame=frame_bgr)

    for track in tracks:
        if track.time_since_update == 0:
            track_id = str(track.track_id)
            track_lengths[track_id] += 1
            class_name = track_class_name(track)
            if class_name != "unknown":
                track_class_names[track_id] = class_name

    newly_counted = count_tracks_once(
        tracks,
        track_lengths,
        counted_track_ids,
        counted_display_ids,
        track_class_names,
        counts,
        min_track_frames,
    )

    # Draw only confirmed tracks; raw detector boxes and labels stay hidden.
    annotated = Image.fromarray(frame_rgb)
    annotated = draw_track_overlay(
        annotated,
        tracks,
        track_lengths,
        counted_track_ids,
        counted_display_ids,
        track_class_names,
        max_draw_missed_frames,
    )
    annotated = draw_count_overlay(annotated, counts)
    annotated_bgr = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
    return annotated_bgr, detection_records, tracks_to_records(
        tracks,
        track_lengths,
        counted_track_ids,
        counted_display_ids,
        track_class_names,
    ), newly_counted


def main():
    args = parse_args()
    if args.min_track_frames < 1:
        raise ValueError("--min-track-frames must be >= 1")
    if args.max_draw_missed_frames < 0:
        raise ValueError("--max-draw-missed-frames must be >= 0")

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
    print(f"Count classes: {count_classes}")
    print(f"Minimum track length for counting: {args.min_track_frames} frames")
    
    if args.detector.lower() == "nano":
        print("the detector is nano")
        detector = build_model_nano(checkpoint_path, class_names)
    elif args.detector.lower() == "large":
        print("the detector is large")
        detector = build_model_large(checkpoint_path, class_names)
            
    tracker = build_tracker(args)

    capture = open_video(video_path)
    fps, width, height, frame_count = video_metadata(capture)
    writer = make_writer(output_path, fps, width, height, args.fourcc)

    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    counts = {name: 0 for name in count_classes}
    track_lengths = defaultdict(int)
    counted_track_ids = set()
    counted_display_ids = {}
    track_class_names = {}
    payload = {
        "input": str(video_path),
        "checkpoint": str(checkpoint_path),
        "tracked_video": str(output_path),
        "class_names": class_names,
        "count_classes": count_classes,
        "threshold": args.threshold,
        "min_track_frames": args.min_track_frames,
        "deepsort": {
            "max_age": args.max_age,
            "max_draw_missed_frames": args.max_draw_missed_frames,
            "n_init": args.n_init,
            "max_iou_distance": args.max_iou_distance,
            "max_cosine_distance": args.max_cosine_distance,
            "embedder": args.embedder,
        },
        "fps": fps,
        "width": width,
        "height": height,
        "input_frame_count": frame_count,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "frames": [],
    }

    processed = 0
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame_index = args.start_frame + processed
            annotated, detections, tracks, newly_counted = annotate_frame(
                frame,
                detector,
                tracker,
                class_names,
                count_classes,
                args.threshold,
                track_lengths,
                counted_track_ids,
                counted_display_ids,
                track_class_names,
                counts,
                args.min_track_frames,
                args.max_draw_missed_frames,
            )
            writer.write(annotated)

            if json_path is not None:
                payload["frames"].append(
                    {
                        "frame_index": frame_index,
                        "num_detections": len(detections),
                        "num_tracks": len(tracks),
                        "counts": dict(counts),
                        "newly_counted_track_ids": newly_counted,
                        "detections": detections,
                        "tracks": tracks,
                    }
                )

            processed += 1
            if processed == 1 or processed % 25 == 0:
                print(
                    f"Processed {processed} frames "
                    f"(active tracks: {len(tracks)}, counts: {dict(counts)})"
                )
    finally:
        capture.release()
        writer.release()

    payload["processed_frames"] = processed
    payload["final_counts"] = dict(counts)
    payload["counted_track_ids"] = sorted(counted_track_ids)
    payload["counted_display_ids"] = counted_display_ids
    payload["track_class_names"] = track_class_names
    write_json(json_path, payload)

    print(f"Processed frames: {processed}")
    print(f"Final counts: {dict(counts)}")
    print(f"Tracked video: {output_path}")
    if json_path is not None:
        print(f"Tracking JSON: {json_path}")


if __name__ == "__main__":
    main()
