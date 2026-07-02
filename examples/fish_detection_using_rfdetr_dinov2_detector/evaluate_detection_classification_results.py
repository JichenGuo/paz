#!/usr/bin/env python
"""Evaluate two-stage detection and species-classification JSON results.

The expected prediction format is the JSON written by detect_and_classify_species.py:

    {
      "images": [
        {
          "image": "/path/to/image.png",
          "detections": [
            {
              "box_xyxy": [x1, y1, x2, y2],
              "detector_score": 0.9,
              "species_top1": {"class_name": "bony fish", ...},
              "species_topk": [...]
            }
          ]
        }
      ]
    }

Ground truth is expected to be COCO format with images, annotations, categories.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PREDICTIONS = "datasets/large_sea_animal_ep29_test_fathomnet_03.json"
DEFAULT_GROUND_TRUTH = "datasets/fathom_test_small/test_dataset.json"
IOU_THRESHOLDS_COCO = [round(0.50 + 0.05 * index, 2) for index in range(10)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RF-DETR detection + FCN species classification results "
            "against a COCO-format FathomNet ground-truth JSON."
        )
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to save evaluation summary JSON. Defaults next to predictions.",
    )
    parser.add_argument(
        "--matches-csv",
        default=None,
        help="Optional CSV of matched detections and classification outcomes.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--class-aware-detection",
        action="store_true",
        help=(
            "Require predicted species top-1 to match the GT category during "
            "detection TP/FP matching. By default detection is class-agnostic "
            "and classification is evaluated separately on matched boxes."
        ),
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Debug limit on prediction images. 0 evaluates all predictions.",
    )
    return parser.parse_args()


def load_json(path):
    path = Path(path).expanduser().resolve()
    with path.open() as f:
        return json.load(f), path


def category_maps(coco):
    id_to_name = {
        category["id"]: category["name"]
        for category in coco.get("categories", [])
        if category.get("supercategory", "") != "none"
    }
    name_to_id = {name: category_id for category_id, name in id_to_name.items()}
    return id_to_name, name_to_id


def image_keys(path_or_name):
    path = Path(str(path_or_name))
    return {str(path_or_name), path.name, path.stem}


def build_gt(coco):
    id_to_name, name_to_id = category_maps(coco)
    image_lookup = {}
    image_id_to_key = {}
    for image in coco.get("images", []):
        keys = image_keys(image.get("file_name", ""))
        for key in keys:
            image_lookup[key] = image
        image_id_to_key[image["id"]] = Path(image.get("file_name", str(image["id"]))).name

    annotations_by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        image_key = image_id_to_key.get(image_id)
        if image_key is None:
            continue
        category_id = annotation.get("category_id")
        if category_id not in id_to_name:
            continue
        x, y, w, h = [float(value) for value in annotation.get("bbox", [])[:4]]
        if w <= 0 or h <= 0:
            continue
        annotations_by_image[image_key].append(
            {
                "annotation_id": annotation.get("id"),
                "image_id": image_id,
                "category_id": category_id,
                "class_name": id_to_name[category_id],
                "box_xyxy": [x, y, x + w, y + h],
                "area": float(annotation.get("area", w * h)),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    return image_lookup, annotations_by_image, id_to_name, name_to_id


def prediction_image_key(prediction_image):
    image_path = prediction_image.get("image", "")
    return Path(image_path).name


def normalize_predictions(predictions, limit_images=0):
    images = predictions.get("images", []) if isinstance(predictions, dict) else []
    if limit_images > 0:
        images = images[:limit_images]

    prediction_image_keys = []
    predictions_by_image = defaultdict(list)
    for image_record in images:
        image_key = prediction_image_key(image_record)
        if image_key:
            prediction_image_keys.append(image_key)
        for index, detection in enumerate(image_record.get("detections", [])):
            box = detection.get("box_xyxy") or detection.get("bbox_xyxy")
            if box is None and "bbox" in detection:
                x, y, w, h = [float(value) for value in detection["bbox"][:4]]
                box = [x, y, x + w, y + h]
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue

            species_top1 = detection.get("species_top1") or {}
            topk = detection.get("species_topk") or []
            if species_top1 and not topk:
                topk = [species_top1]
            predictions_by_image[image_key].append(
                {
                    "prediction_index": index,
                    "box_xyxy": [x1, y1, x2, y2],
                    "detector_score": float(detection.get("detector_score", detection.get("score", 0.0))),
                    "detector_class_name": detection.get("detector_class_name"),
                    "species_top1": species_top1.get("class_name"),
                    "species_top1_score": float(species_top1.get("score", 0.0)) if species_top1 else 0.0,
                    "species_topk": [item.get("class_name") for item in topk if item.get("class_name")],
                    "raw": detection,
                }
            )
    for image_predictions in predictions_by_image.values():
        image_predictions.sort(key=lambda item: item["detector_score"], reverse=True)
    return predictions_by_image, sorted(set(prediction_image_keys)), len(images)


def filter_gt_to_prediction_images(gt_by_image, prediction_image_keys):
    return {
        image_key: gt_by_image.get(image_key, [])
        for image_key in prediction_image_keys
    }


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def species_matches(prediction, gt):
    return prediction.get("species_top1") == gt.get("class_name")


def match_at_threshold(predictions_by_image, gt_by_image, iou_threshold, class_aware=False):
    matched_rows = []
    total_gt = sum(len(items) for items in gt_by_image.values())
    total_predictions = sum(len(items) for items in predictions_by_image.values())
    tp = fp = 0
    matched_gt_count = 0
    per_class_counts = defaultdict(lambda: Counter())

    all_image_keys = sorted(set(gt_by_image) | set(predictions_by_image))
    for image_key in all_image_keys:
        gt_items = gt_by_image.get(image_key, [])
        pred_items = predictions_by_image.get(image_key, [])
        used_gt = set()

        for prediction in pred_items:
            best_gt_index = None
            best_iou = 0.0
            for gt_index, gt in enumerate(gt_items):
                if gt_index in used_gt:
                    continue
                if class_aware and not species_matches(prediction, gt):
                    continue
                iou = box_iou(prediction["box_xyxy"], gt["box_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            is_tp = best_gt_index is not None and best_iou >= iou_threshold
            if is_tp:
                used_gt.add(best_gt_index)
                gt = gt_items[best_gt_index]
                tp += 1
                matched_gt_count += 1
                top1_correct = prediction.get("species_top1") == gt["class_name"]
                topk_correct = gt["class_name"] in prediction.get("species_topk", [])
                per_class_counts[gt["class_name"]]["matched"] += 1
                per_class_counts[gt["class_name"]]["top1_correct"] += int(top1_correct)
                per_class_counts[gt["class_name"]]["topk_correct"] += int(topk_correct)
                matched_rows.append(
                    {
                        "image": image_key,
                        "prediction_index": prediction["prediction_index"],
                        "annotation_id": gt.get("annotation_id"),
                        "iou": best_iou,
                        "detector_score": prediction["detector_score"],
                        "gt_class": gt["class_name"],
                        "pred_species_top1": prediction.get("species_top1"),
                        "pred_species_top1_score": prediction.get("species_top1_score"),
                        "pred_species_topk": "|".join(prediction.get("species_topk", [])),
                        "top1_correct": top1_correct,
                        "topk_correct": topk_correct,
                    }
                )
            else:
                fp += 1
                matched_rows.append(
                    {
                        "image": image_key,
                        "prediction_index": prediction["prediction_index"],
                        "annotation_id": None,
                        "iou": best_iou,
                        "detector_score": prediction["detector_score"],
                        "gt_class": None,
                        "pred_species_top1": prediction.get("species_top1"),
                        "pred_species_top1_score": prediction.get("species_top1_score"),
                        "pred_species_topk": "|".join(prediction.get("species_topk", [])),
                        "top1_correct": False,
                        "topk_correct": False,
                    }
                )

        for gt_index, gt in enumerate(gt_items):
            if gt_index not in used_gt:
                per_class_counts[gt["class_name"]]["missed"] += 1

    fn = max(0, total_gt - matched_gt_count)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, total_gt)
    f1 = safe_div(2 * precision * recall, precision + recall)
    top1_accuracy = safe_div(
        sum(row["top1_correct"] for row in matched_rows if row["annotation_id"] is not None),
        tp,
    )
    topk_accuracy = safe_div(
        sum(row["topk_correct"] for row in matched_rows if row["annotation_id"] is not None),
        tp,
    )
    return {
        "iou_threshold": iou_threshold,
        "class_aware_detection": class_aware,
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_top1_accuracy_on_matched": top1_accuracy,
        "classification_topk_accuracy_on_matched": topk_accuracy,
        "matched_rows": matched_rows,
        "per_class_counts": per_class_counts,
    }


def compute_ap(recalls, precisions):
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def average_precision(predictions_by_image, gt_by_image, iou_threshold, class_aware=False):
    total_gt = sum(len(items) for items in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    flat_predictions = []
    for image_key, predictions in predictions_by_image.items():
        for prediction in predictions:
            flat_predictions.append((image_key, prediction))
    flat_predictions.sort(key=lambda item: item[1]["detector_score"], reverse=True)

    used_by_image = defaultdict(set)
    tp_flags = []
    fp_flags = []
    for image_key, prediction in flat_predictions:
        gt_items = gt_by_image.get(image_key, [])
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_items):
            if gt_index in used_by_image[image_key]:
                continue
            if class_aware and not species_matches(prediction, gt):
                continue
            iou = box_iou(prediction["box_xyxy"], gt["box_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            used_by_image[image_key].add(best_gt_index)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    cumulative_tp = 0.0
    cumulative_fp = 0.0
    recalls = []
    precisions = []
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        cumulative_tp += tp_flag
        cumulative_fp += fp_flag
        recalls.append(cumulative_tp / total_gt)
        precisions.append(safe_div(cumulative_tp, cumulative_tp + cumulative_fp))
    return compute_ap(recalls, precisions)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_per_class(per_class_counts):
    rows = []
    for class_name in sorted(per_class_counts):
        counts = per_class_counts[class_name]
        matched = counts["matched"]
        total_gt = matched + counts["missed"]
        rows.append(
            {
                "class_name": class_name,
                "gt_objects": total_gt,
                "matched": matched,
                "missed": counts["missed"],
                "recall": safe_div(matched, total_gt),
                "top1_accuracy_on_matched": safe_div(counts["top1_correct"], matched),
                "topk_accuracy_on_matched": safe_div(counts["topk_correct"], matched),
            }
        )
    return rows


def prediction_summary(predictions_by_image, prediction_image_keys, prediction_image_count):
    species_counts = Counter()
    detection_counts = []
    scores = []
    for predictions in predictions_by_image.values():
        detection_counts.append(len(predictions))
        for prediction in predictions:
            species_counts[prediction.get("species_top1") or "<missing>"] += 1
            scores.append(prediction["detector_score"])
    return {
        "prediction_image_records": prediction_image_count,
        "evaluated_prediction_images": len(prediction_image_keys),
        "images_with_predictions": len(predictions_by_image),
        "total_predictions": sum(detection_counts),
        "mean_predictions_per_image": safe_div(sum(detection_counts), prediction_image_count),
        "mean_detector_score": safe_div(sum(scores), len(scores)),
        "species_top1_counts": dict(species_counts.most_common()),
    }


def write_matches_csv(path, rows):
    if not rows:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    predictions, predictions_path = load_json(args.predictions)
    ground_truth, ground_truth_path = load_json(args.ground_truth)

    _, gt_by_image, id_to_name, _ = build_gt(ground_truth)
    predictions_by_image, prediction_image_keys, prediction_image_count = normalize_predictions(
        predictions,
        limit_images=args.limit_images,
    )
    gt_by_image = filter_gt_to_prediction_images(gt_by_image, prediction_image_keys)

    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else predictions_path.with_name(f"{predictions_path.stem}_evaluation.json")
    )

    summary = {
        "predictions": str(predictions_path),
        "ground_truth": str(ground_truth_path),
        "iou_threshold": args.iou_threshold,
        "class_aware_detection": args.class_aware_detection,
        "num_categories": len(id_to_name),
        "num_gt_images_total": len(ground_truth.get("images", [])),
        "num_gt_annotations_total": len(ground_truth.get("annotations", [])),
        "evaluation_scope": "prediction_images_only",
        "num_evaluated_prediction_images": len(prediction_image_keys),
        "num_evaluated_gt_annotations": sum(
            len(items) for items in gt_by_image.values()
        ),
        "prediction_summary": prediction_summary(
            predictions_by_image,
            prediction_image_keys,
            prediction_image_count,
        ),
    }

    if not ground_truth.get("annotations"):
        summary["status"] = "no_ground_truth_annotations"
        summary["message"] = (
            "Ground-truth JSON contains zero annotations. Detection precision, "
            "recall, mAP, and classification accuracy cannot be computed."
        )
        print(summary["message"])
    else:
        match_summary = match_at_threshold(
            predictions_by_image,
            gt_by_image,
            args.iou_threshold,
            class_aware=args.class_aware_detection,
        )
        ap50 = average_precision(
            predictions_by_image,
            gt_by_image,
            0.50,
            class_aware=args.class_aware_detection,
        )
        aps = [
            average_precision(
                predictions_by_image,
                gt_by_image,
                threshold,
                class_aware=args.class_aware_detection,
            )
            for threshold in IOU_THRESHOLDS_COCO
        ]
        matched_rows = match_summary.pop("matched_rows")
        per_class_counts = match_summary.pop("per_class_counts")
        summary["status"] = "ok"
        summary["detection"] = match_summary
        summary["detection"]["AP50"] = ap50
        summary["detection"]["mAP50_95"] = sum(aps) / len(aps)
        summary["detection"]["AP_by_iou"] = {
            str(threshold): ap for threshold, ap in zip(IOU_THRESHOLDS_COCO, aps)
        }
        summary["classification"] = {
            "top1_accuracy_on_matched_detections": match_summary[
                "classification_top1_accuracy_on_matched"
            ],
            "topk_accuracy_on_matched_detections": match_summary[
                "classification_topk_accuracy_on_matched"
            ],
        }
        summary["per_class"] = summarize_per_class(per_class_counts)
        if args.matches_csv:
            write_matches_csv(args.matches_csv, matched_rows)

        print(
            "Detection: "
            f"TP={match_summary['true_positives']} "
            f"FP={match_summary['false_positives']} "
            f"FN={match_summary['false_negatives']} "
            f"precision={match_summary['precision']:.4f} "
            f"recall={match_summary['recall']:.4f} "
            f"F1={match_summary['f1']:.4f} "
            f"AP50={ap50:.4f} "
            f"mAP50:95={summary['detection']['mAP50_95']:.4f}"
        )
        print(
            "Classification on matched detections: "
            f"top1={summary['classification']['top1_accuracy_on_matched_detections']:.4f} "
            f"topK={summary['classification']['topk_accuracy_on_matched_detections']:.4f}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Prediction image records: {summary['prediction_summary']['prediction_image_records']}")
    print(f"Evaluated prediction images: {summary['prediction_summary']['evaluated_prediction_images']}")
    print(f"Total predictions: {summary['prediction_summary']['total_predictions']}")
    print(f"Saved evaluation JSON: {output_json}")


if __name__ == "__main__":
    main()
