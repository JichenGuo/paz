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


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
# _EXPERIMENT_10_DIR = _SCRIPT_DIR / "experiments_HPC" / "experiment_10"
# for path in (_PAZ_ROOT, _EXPERIMENT_10_DIR, _SCRIPT_DIR):
#     if path.is_dir() and str(path) not in sys.path:
#         sys.path.insert(0, str(path))

from detect_video import (  # noqa: E402
    COUNT_BG_COLOR,
    COUNT_CLASSES,
    COUNT_FONT_SIZE,
    COUNT_MARGIN,
    COUNT_PADDING,
    COUNT_TEXT_COLOR,
    DEFAULT_CHECKPOINT,
    build_model,
    detections_to_records,
    draw_detections,
    make_writer,
    open_video,
    parse_count_classes,
    resolve_class_names,
    video_metadata,
    write_json,
)


TRACK_BOX_COLOR = (0, 255, 255)
TRACK_TEXT_COLOR = (255, 255, 255)
TRACK_TEXT_BG_COLOR = (0, 105, 130)
TRACK_FONT_SIZE = 24


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


def track_bbox_xyxy(track):
    left, top, right, bottom = track.to_ltrb()
    return [float(left), float(top), float(right), float(bottom)]


def draw_track_overlay(
    image, tracks, track_lengths, counted_track_ids, counted_display_ids
):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", TRACK_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    image_width, image_height = annotated.size
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = str(track.track_id)
        x1, y1, x2, y2 = track_bbox_xyxy(track)
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = track_class_name(track)
        counted = track_id in counted_track_ids
        if counted:
            display_id = counted_display_ids[track_id]
            label = f"{class_name} id:{display_id} counted"
        else:
            label = f"{class_name} not counted"

        draw.rectangle((x1, y1, x2, y2), outline=TRACK_BOX_COLOR, width=3)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            (x1, text_y, x1 + text_width + 6, text_y + text_height + 4),
            fill=TRACK_TEXT_BG_COLOR,
        )
        draw.text((x1 + 3, text_y + 2), label, fill=TRACK_TEXT_COLOR, font=font)
    return annotated


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


def count_tracks_once(
    tracks,
    track_lengths,
    counted_track_ids,
    counted_display_ids,
    counts,
    min_frames,
):
    newly_counted = []
    count_lookup = {name.lower(): name for name in counts}
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = str(track.track_id)
        class_name = track_class_name(track)
        normalized_class = class_name.lower()
        if normalized_class not in count_lookup:
            continue
        if track_id in counted_track_ids:
            continue
        if track_lengths[track_id] < min_frames:
            continue

        counted_name = count_lookup[normalized_class]
        counted_display_ids[track_id] = counts[counted_name]
        counts[counted_name] += 1
        counted_track_ids.add(track_id)
        newly_counted.append(track_id)
    return newly_counted


def tracks_to_records(tracks, track_lengths, counted_track_ids, counted_display_ids):
    records = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = str(track.track_id)
        records.append(
            {
                "track_id": track_id,
                "display_id": counted_display_ids.get(track_id),
                "class_name": track_class_name(track),
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
    counts,
    min_track_frames,
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

    newly_counted = count_tracks_once(
        tracks,
        track_lengths,
        counted_track_ids,
        counted_display_ids,
        counts,
        min_track_frames,
    )

    annotated = draw_detections(Image.fromarray(frame_rgb), result, class_names)
    annotated = draw_track_overlay(
        annotated, tracks, track_lengths, counted_track_ids, counted_display_ids
    )
    annotated = draw_count_overlay(annotated, counts)
    annotated_bgr = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
    return annotated_bgr, detection_records, tracks_to_records(
        tracks, track_lengths, counted_track_ids, counted_display_ids
    ), newly_counted


def main():
    args = parse_args()
    if args.min_track_frames < 1:
        raise ValueError("--min-track-frames must be >= 1")

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
    detector = build_model(checkpoint_path, class_names)
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
                counts,
                args.min_track_frames,
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
    write_json(json_path, payload)

    print(f"Processed frames: {processed}")
    print(f"Final counts: {dict(counts)}")
    print(f"Tracked video: {output_path}")
    if json_path is not None:
        print(f"Tracking JSON: {json_path}")


if __name__ == "__main__":
    main()
