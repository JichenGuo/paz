#!/usr/bin/env python
"""Extract finetuned RF-DETR backbone features and visualize them with UMAP.

Examples
--------
COCO ground-truth boxes, one point per annotated object::

    python examples/fish_detection_using_rfdetr_dinov2_detector/extract_detector_features_umap.py \
        --image-dir datasets/fathomnet \
        --annotation-file datasets/fathomnet/train_dataset.json \
        --checkpoint /path/to/checkpoint.weights.h5 \
        --detector large \
        --point-source gt_boxes

In-house Labelimage_Fish COCO dataset, ignoring category id 0/background::

    python examples/fish_detection_using_rfdetr_dinov2_detector/extract_detector_features_umap.py \
        --image-dir datasets/Labelimage_Fish_coco/train \
        --annotation-file datasets/Labelimage_Fish_coco/train/_annotations.coco.json \
        --checkpoint /path/to/checkpoint.weights.h5 \
        --detector large \
        --class-names crab,fish,lobster \
        --point-source gt_boxes \
        --background-category-ids 0

Only images containing rare crab/lobster, then plot crab/fish/lobster boxes::

    python examples/fish_detection_using_rfdetr_dinov2_detector/extract_detector_features_umap.py \
        --image-dir datasets/Labelimage_Fish_coco/train \
        --annotation-file datasets/Labelimage_Fish_coco/train/_annotations.coco.json \
        --checkpoint /path/to/checkpoint.weights.h5 \
        --detector large \
        --class-names crab,fish,lobster \
        --point-source gt_boxes \
        --include-classes crab,fish,lobster \
        --require-image-classes crab,lobster \
        --background-category-ids 0 \
        --max-points 0

Two-stage prediction JSON, one point per classified detection::

    python examples/fish_detection_using_rfdetr_dinov2_detector/extract_detector_features_umap.py \
        --predictions-json datasets/large_sea_animal_ep29_test_fathomnet_03.json \
        --checkpoint /path/to/checkpoint.weights.h5 \
        --detector large \
        --point-source predicted_boxes
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Must be set before importing keras/paz.
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from PIL import Image


_SCRIPT_DIR = Path(__file__).resolve().parent
_PAZ_ROOT = next(
    parent for parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
    if (parent / "paz" / "models").is_dir()
)
if str(_PAZ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAZ_ROOT))

from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano, RFDETRLarge  # noqa: E402


DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "feature_umap_results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype="float32")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RF-DETR backbone features from test images/boxes and run "
            "UMAP to inspect class clustering."
        )
    )
    parser.add_argument("--image-dir", default=None, help="Directory containing images.")
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="COCO JSON used by --point-source gt_boxes.",
    )
    parser.add_argument(
        "--predictions-json",
        default=None,
        help="Two-stage detection/classification JSON used by predicted_boxes.",
    )
    parser.add_argument("--checkpoint", required=True, help="Finetuned RF-DETR weights.")
    parser.add_argument("--detector", choices=("nano", "large"), default="nano")
    parser.add_argument(
        "--point-source",
        choices=("gt_boxes", "predicted_boxes", "images"),
        default="gt_boxes",
        help="What each UMAP point represents.",
    )
    parser.add_argument(
        "--class-names",
        default="fish",
        help="Comma-separated detector class names for model construction.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP local neighborhood size.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance between embedded points.",
    )
    parser.add_argument(
        "--umap-metric",
        default="euclidean",
        help="Distance metric used by UMAP.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument(
        "--include-classes",
        default="",
        help=(
            "Comma-separated object classes to include as UMAP points. "
            "For the in-house dataset use crab,fish,lobster."
        ),
    )
    parser.add_argument(
        "--require-image-classes",
        default="",
        help=(
            "Comma-separated classes that an image must contain before any "
            "records from that image are used. Matching is image-level OR. "
            "Example: crab,lobster selects all images containing crab or lobster."
        ),
    )
    parser.add_argument(
        "--background-category-ids",
        default="0",
        help=(
            "Comma-separated COCO category ids to ignore as background/root "
            "classes. Defaults to 0 for Roboflow-style datasets."
        ),
    )
    parser.add_argument(
        "--background-class-names",
        default="background,none,__background__,Labelimage_Fish",
        help="Comma-separated category names to ignore as background/root classes.",
    )
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_json(path):
    with Path(path).expanduser().resolve().open() as f:
        return json.load(f)


def collect_image_paths(image_dir, recursive=False):
    image_dir = Path(image_dir).expanduser().resolve()
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_class_names(value):
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--class-names must contain at least one detector class")
    return names


def build_detector(checkpoint_path, detector_name, class_names):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    detector_cls = RFDETRNano if detector_name == "nano" else RFDETRLarge
    detector = detector_cls(num_classes=len(class_names))
    resolution = detector.model_config.resolution
    dummy = np.ones((1, resolution, resolution, 3), dtype="float32") * 0.5
    detector.model.model(dummy, training=False)
    detector.model.load_pretrained_weights(str(checkpoint_path))
    detector.model.class_names = class_names
    return detector


def preprocess_image(image_path, resolution):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype="float32") / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array, original_size


def feature_to_array(feature):
    if isinstance(feature, (list, tuple)):
        feature = feature[0]
    elif hasattr(feature, "decompose"):
        feature = feature.decompose()[0]
    return np.asarray(feature)


def backbone_features(detector, batch):
    features, _ = detector.model.model.backbone(batch, training=False)
    return [feature_to_array(feature) for feature in features]


def pool_feature_map(feature_map, box_xyxy, original_size, resolution):
    # feature_map shape is expected to be (B, H, W, C) for this Keras backbone.
    _, feature_h, feature_w, _ = feature_map.shape
    image_w, image_h = original_size
    x1, y1, x2, y2 = box_xyxy

    x1 = x1 * resolution / image_w
    x2 = x2 * resolution / image_w
    y1 = y1 * resolution / image_h
    y2 = y2 * resolution / image_h

    fx1 = int(np.floor(x1 * feature_w / resolution))
    fx2 = int(np.ceil(x2 * feature_w / resolution))
    fy1 = int(np.floor(y1 * feature_h / resolution))
    fy2 = int(np.ceil(y2 * feature_h / resolution))
    fx1 = max(0, min(feature_w - 1, fx1))
    fy1 = max(0, min(feature_h - 1, fy1))
    fx2 = max(fx1 + 1, min(feature_w, fx2))
    fy2 = max(fy1 + 1, min(feature_h, fy2))

    roi = feature_map[0, fy1:fy2, fx1:fx2, :]
    return roi.mean(axis=(0, 1))


def pool_image_feature(feature_map):
    return feature_map[0].mean(axis=(0, 1))


def parse_csv_set(value, cast=str):
    values = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.add(cast(item))
    return values


def build_category_maps(coco, background_category_ids=None, background_class_names=None):
    background_category_ids = background_category_ids or set()
    background_class_names = background_class_names or set()
    categories = []
    for category in coco.get("categories", []):
        category_id = category.get("id")
        category_name = str(category.get("name", ""))
        if category_id in background_category_ids:
            continue
        if category_name in background_class_names:
            continue
        if category.get("supercategory", "") == "none":
            continue
        categories.append(category)
    categories = sorted(categories, key=lambda category: category["id"])
    id_to_name = {category["id"]: category["name"] for category in categories}
    return id_to_name


def find_image_path(image_root, file_name):
    image_root = Path(image_root).expanduser().resolve()
    path = image_root / file_name
    if path.exists():
        return path
    stem = Path(file_name).stem
    matches = [
        candidate for suffix in IMAGE_EXTENSIONS
        for candidate in image_root.glob(f"{stem}{suffix}")
    ]
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Image not found for COCO file_name={file_name}")


def records_from_gt_boxes(
    annotation_file,
    image_dir,
    min_box_size,
    include_classes,
    require_image_classes,
    background_category_ids,
    background_class_names,
):
    coco = load_json(annotation_file)
    id_to_name = build_category_maps(
        coco,
        background_category_ids=background_category_ids,
        background_class_names=background_class_names,
    )
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    include = {name.strip() for name in include_classes.split(",") if name.strip()}
    required = {
        name.strip() for name in require_image_classes.split(",") if name.strip()
    }

    allowed_image_ids = None
    if required:
        allowed_image_ids = set()
        for annotation in coco.get("annotations", []):
            category_id = annotation.get("category_id")
            class_name = id_to_name.get(category_id)
            if class_name in required:
                allowed_image_ids.add(annotation.get("image_id"))
        if not allowed_image_ids:
            raise ValueError(
                f"No images contain required classes: {sorted(required)}"
            )

    records = []
    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            continue
        category_id = annotation.get("category_id")
        if category_id not in id_to_name:
            continue
        class_name = id_to_name[category_id]
        if include and class_name not in include:
            continue
        image = image_by_id.get(image_id)
        if image is None:
            continue
        x, y, w, h = [float(value) for value in annotation.get("bbox", [])[:4]]
        if w < min_box_size or h < min_box_size:
            continue
        image_path = find_image_path(image_dir, image["file_name"])
        records.append(
            {
                "image_path": str(image_path),
                "image_key": Path(image["file_name"]).name,
                "box_xyxy": [x, y, x + w, y + h],
                "class_name": class_name,
                "source": "gt_box",
                "annotation_id": annotation.get("id"),
            }
        )
    return records


def records_from_predictions(predictions_json, include_classes):
    payload = load_json(predictions_json)
    include = {name.strip() for name in include_classes.split(",") if name.strip()}
    records = []
    for image_record in payload.get("images", []):
        image_path = Path(image_record.get("image", "")).expanduser()
        if not image_path.exists():
            continue
        for index, detection in enumerate(image_record.get("detections", [])):
            top1 = detection.get("species_top1") or {}
            class_name = top1.get("class_name") or detection.get("class_name")
            if not class_name:
                continue
            if include and class_name not in include:
                continue
            box = detection.get("box_xyxy") or detection.get("bbox_xyxy")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            records.append(
                {
                    "image_path": str(image_path),
                    "image_key": image_path.name,
                    "box_xyxy": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "source": "predicted_box",
                    "prediction_index": index,
                    "detector_score": detection.get("detector_score"),
                    "species_score": top1.get("score"),
                }
            )
    return records


def records_from_images(image_dir, recursive):
    records = []
    for image_path in collect_image_paths(image_dir, recursive=recursive):
        records.append(
            {
                "image_path": str(image_path),
                "image_key": image_path.name,
                "box_xyxy": None,
                "class_name": "image",
                "source": "image",
            }
        )
    return records


def limit_records(records, max_points, seed):
    if max_points <= 0 or len(records) <= max_points:
        return records
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=max_points, replace=False)
    return [records[int(index)] for index in sorted(indices)]


def extract_features(detector, records, batch_size):
    resolution = detector.model_config.resolution
    features = []
    metadata = []
    grouped = defaultdict(list)
    for record in records:
        grouped[record["image_path"]].append(record)

    image_paths = sorted(grouped)
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        # Backbone ROI pooling is per image because feature maps have one batch axis.
        for image_index, image_path in enumerate(batch_paths, start=start + 1):
            batch, original_size = preprocess_image(image_path, resolution)
            feature_maps = backbone_features(detector, batch[None, ...])
            for record in grouped[image_path]:
                pooled = []
                if record["box_xyxy"] is None:
                    pooled = [pool_image_feature(feature_map) for feature_map in feature_maps]
                else:
                    pooled = [
                        pool_feature_map(
                            feature_map,
                            record["box_xyxy"],
                            original_size,
                            resolution,
                        )
                        for feature_map in feature_maps
                    ]
                features.append(np.concatenate(pooled, axis=0).astype("float32"))
                metadata.append(dict(record))
            if image_index == 1 or image_index % 25 == 0 or image_index == len(image_paths):
                print(f"Processed images: {image_index}/{len(image_paths)}", flush=True)
    return np.stack(features, axis=0), metadata


def run_umap(features, n_neighbors, min_dist, metric, random_state):
    try:
        from sklearn.preprocessing import StandardScaler
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires scikit-learn and umap-learn. "
            "Install them in your ML environment, e.g. "
            "`pip install scikit-learn umap-learn`."
        ) from exc

    if len(features) < 3:
        raise ValueError("Need at least 3 feature points for UMAP")
    n_neighbors = min(max(2, int(n_neighbors)), len(features) - 1)
    scaled = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(scaled), n_neighbors


def save_metadata_csv(path, metadata, embeddings):
    fieldnames = sorted({key for row in metadata for key in row})
    fieldnames = ["umap_x", "umap_y"] + fieldnames
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, embedding in zip(metadata, embeddings):
            payload = dict(row)
            payload["umap_x"] = float(embedding[0])
            payload["umap_y"] = float(embedding[1])
            writer.writerow(payload)


def compute_clustering_metrics(features, embeddings, metadata):
    labels = [row["class_name"] for row in metadata]
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2 or len(features) <= len(unique_labels):
        return {
            "num_classes": len(unique_labels),
            "message": "Need at least two classes and more points than classes.",
        }
    try:
        from sklearn.metrics import calinski_harabasz_score
        from sklearn.metrics import davies_bouldin_score
        from sklearn.metrics import silhouette_score
    except ImportError:
        return {
            "num_classes": len(unique_labels),
            "message": "scikit-learn metrics unavailable.",
        }

    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    y = np.asarray([label_to_index[label] for label in labels])
    metrics = {
        "num_classes": len(unique_labels),
        "labels": unique_labels,
    }
    for name, values in (("feature", features), ("umap", embeddings)):
        metrics[name] = {
            "silhouette": float(silhouette_score(values, y)),
            "davies_bouldin": float(davies_bouldin_score(values, y)),
            "calinski_harabasz": float(calinski_harabasz_score(values, y)),
        }
    return metrics


def plot_umap(path, embeddings, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["class_name"] for row in metadata]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    label_to_color = {label: cmap(index) for index, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 9))
    for label in unique_labels:
        indices = [index for index, value in enumerate(labels) if value == label]
        points = embeddings[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            alpha=0.78,
            color=label_to_color[label],
            label=f"{label} ({len(indices)})",
            edgecolors="none",
        )
    ax.set_title("RF-DETR backbone features UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)
    if len(unique_labels) <= 35:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.point_source == "gt_boxes":
        if not args.annotation_file or not args.image_dir:
            raise ValueError("gt_boxes requires --annotation-file and --image-dir")
        records = records_from_gt_boxes(
            args.annotation_file,
            args.image_dir,
            args.min_box_size,
            args.include_classes,
            args.require_image_classes,
            parse_csv_set(args.background_category_ids, int),
            parse_csv_set(args.background_class_names, str),
        )
    elif args.point_source == "predicted_boxes":
        if not args.predictions_json:
            raise ValueError("predicted_boxes requires --predictions-json")
        records = records_from_predictions(args.predictions_json, args.include_classes)
    else:
        if not args.image_dir:
            raise ValueError("images requires --image-dir")
        records = records_from_images(args.image_dir, args.recursive)

    if not records:
        raise ValueError("No feature records found for the requested point source")
    records = limit_records(records, args.max_points, args.random_state)
    print(f"Feature points: {len(records)}", flush=True)
    print(f"Class counts: {dict(Counter(row['class_name'] for row in records))}", flush=True)

    detector = build_detector(
        args.checkpoint,
        args.detector,
        read_class_names(args.class_names),
    )
    features, metadata = extract_features(detector, records, args.batch_size)
    embeddings, used_n_neighbors = run_umap(
        features,
        args.n_neighbors,
        args.min_dist,
        args.umap_metric,
        args.random_state,
    )
    clustering_metrics = compute_clustering_metrics(features, embeddings, metadata)

    features_path = output_dir / "detector_features.npz"
    csv_path = output_dir / "detector_features_umap.csv"
    plot_path = output_dir / "detector_features_umap.png"
    summary_path = output_dir / "detector_features_umap_summary.json"

    np.savez_compressed(
        features_path,
        features=features,
        umap=embeddings,
        labels=np.asarray([row["class_name"] for row in metadata]),
        image_keys=np.asarray([row["image_key"] for row in metadata]),
    )
    save_metadata_csv(csv_path, metadata, embeddings)
    plot_umap(plot_path, embeddings, metadata)

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "detector": args.detector,
        "point_source": args.point_source,
        "num_points": len(metadata),
        "feature_dim": int(features.shape[1]),
        "n_neighbors": used_n_neighbors,
        "min_dist": args.min_dist,
        "umap_metric": args.umap_metric,
        "class_counts": dict(Counter(row["class_name"] for row in metadata)),
        "selected_image_count": len({row["image_key"] for row in metadata}),
        "required_image_classes": args.require_image_classes,
        "included_point_classes": args.include_classes,
        "clustering_metrics": clustering_metrics,
        "features_npz": str(features_path),
        "metadata_csv": str(csv_path),
        "plot_png": str(plot_path),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Clustering metrics: {clustering_metrics}")
    print(f"Saved features: {features_path}")
    print(f"Saved UMAP CSV: {csv_path}")
    print(f"Saved UMAP plot: {plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
