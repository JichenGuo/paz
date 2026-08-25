"""Evaluate and visualize held-out RGB-D mesh point-cloud predictions.

Example:
    KERAS_BACKEND=jax python -m \
        paz.graphics.synthetic_data.evaluate_synthetic_mesh_predictions \
        --experiment experiments/resnet18_rgbd_val_mesh_HPC
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import json
from pathlib import Path

import cv2
import keras
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import registers the serialized loss and provides target preprocessing.
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh import (
    MESH_OUTPUT_NAME,
    load_camera_mesh_points,
    sample_mesh_surface,
    world_to_camera_matrix,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import load_records


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def load_inputs(record, max_depth):
    """Loads one named RGB/depth model input pair."""
    root = Path(record["_root"])
    rgb_path = root / record["rgb"]
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(root / record["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    depth = np.clip(depth / max_depth, 0.0, 1.0)[..., None]
    return rgb, depth


def denormalize_points(points, statistics):
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(
        statistics["standard_deviation"], dtype=np.float32
    )
    return np.asarray(points, dtype=np.float32) * std + mean


def camera_to_world_points(points, camera):
    """Maps predicted camera-frame points back to physical world metres."""
    camera_to_world = np.linalg.inv(world_to_camera_matrix(camera))
    homogeneous = np.concatenate([
        points, np.ones((len(points), 1), dtype=np.float32)
    ], axis=1)
    return (homogeneous @ camera_to_world.T)[:, :3].astype(np.float32)


def nearest_distances(first, second):
    """Returns Euclidean distance from every first point to its nearest second."""
    differences = first[:, None, :] - second[None, :, :]
    squared = np.sum(np.square(differences), axis=-1)
    return np.sqrt(np.maximum(np.min(squared, axis=1), 0.0))


def point_cloud_metrics(prediction, target, threshold):
    """Computes symmetric surface metrics in physical metres."""
    prediction_to_target = nearest_distances(prediction, target)
    target_to_prediction = nearest_distances(target, prediction)
    precision = np.mean(prediction_to_target <= threshold)
    recall = np.mean(target_to_prediction <= threshold)
    denominator = precision + recall
    fscore = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    chamfer_m2 = 0.5 * (
        np.mean(np.square(prediction_to_target))
        + np.mean(np.square(target_to_prediction))
    )
    return {
        "chamfer_m2": float(chamfer_m2),
        "chamfer_rmse_m": float(np.sqrt(chamfer_m2)),
        "mean_surface_distance_m": float(0.5 * (
            np.mean(prediction_to_target) + np.mean(target_to_prediction)
        )),
        "hausdorff_distance_m": float(max(
            np.max(prediction_to_target), np.max(target_to_prediction)
        )),
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
    }


def set_equal_3d_limits(axis, point_sets):
    points = np.concatenate(point_sets, axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * np.max(upper - lower), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")


def scatter_points(axis, points, color, label=None, alpha=0.8):
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=7, c=color,
        label=label, alpha=alpha, depthshade=False,
    )


def save_visualization(path, rgb, prediction, target, sample_id, metrics):
    """Saves RGB, separate point clouds, and an overlay in one PNG."""
    figure = plt.figure(figsize=(18, 5))
    image_axis = figure.add_subplot(1, 4, 1)
    image_axis.imshow(rgb)
    image_axis.set_title(f"RGB input: {sample_id}")
    image_axis.axis("off")

    target_axis = figure.add_subplot(1, 4, 2, projection="3d")
    scatter_points(target_axis, target, "tab:blue")
    target_axis.set_title("Ground-truth surface")
    set_equal_3d_limits(target_axis, [prediction, target])

    prediction_axis = figure.add_subplot(1, 4, 3, projection="3d")
    scatter_points(prediction_axis, prediction, "tab:orange")
    prediction_axis.set_title("Predicted point cloud")
    set_equal_3d_limits(prediction_axis, [prediction, target])

    overlay_axis = figure.add_subplot(1, 4, 4, projection="3d")
    scatter_points(overlay_axis, target, "tab:blue", "Ground truth", 0.55)
    scatter_points(
        overlay_axis, prediction, "tab:orange", "Prediction", 0.75
    )
    overlay_axis.set_title(
        f"Overlay\nChamfer RMSE: {metrics['chamfer_rmse_m']:.3f} m, "
        f"F-score: {metrics['fscore']:.3f}"
    )
    overlay_axis.legend(fontsize=8)
    set_equal_3d_limits(overlay_axis, [prediction, target])
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarize(rows):
    summary = {"num_samples": len(rows)}
    for name in rows[0]:
        if name in ("sample_id", "shape"):
            continue
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values)),
        }
    for shape in sorted({row["shape"] for row in rows}):
        shape_rows = [row for row in rows if row["shape"] == shape]
        summary[f"{shape}_num_samples"] = len(shape_rows)
        summary[f"{shape}_chamfer_rmse_m_mean"] = float(np.mean([
            row["chamfer_rmse_m"] for row in shape_rows
        ]))
        summary[f"{shape}_fscore_mean"] = float(np.mean([
            row["fscore"] for row in shape_rows
        ]))
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None,
                        help="Full .keras model; defaults to best.keras.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--distance-threshold", type=float, default=0.05,
                        help="Metre threshold used by precision/recall/F-score.")
    parser.add_argument("--num-visualizations", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional evaluation limit for smoke testing.")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.distance_threshold <= 0.0 or args.num_visualizations < 0:
        raise ValueError("threshold must be positive and count nonnegative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    test_split = args.test_split or args.experiment / "test_split"
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "mesh_evaluation"
    predicted_output = output / "predicted_pointclouds"
    target_output = output / "target_pointclouds"
    visualization_output = output / "visualizations"
    for directory in (predicted_output, target_output, visualization_output):
        directory.mkdir(parents=True, exist_ok=True)

    preprocessing = load_json(args.experiment / "input_preprocessing.json")
    normalization = load_json(args.experiment / "normalization.json")
    mesh_statistics = normalization["targets"][MESH_OUTPUT_NAME]
    num_points = int(preprocessing["num_mesh_points"])
    records = load_records(test_split)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if not records:
        raise ValueError(f"No test records found in {test_split}")
    model = keras.models.load_model(model_path, compile=False)
    if MESH_OUTPUT_NAME not in model.output_names:
        raise ValueError(f"{model_path} has no {MESH_OUTPUT_NAME} output")

    visualization_count = min(args.num_visualizations, len(records))
    visualization_indices = set(np.linspace(
        0, len(records) - 1, visualization_count, dtype=int
    )) if visualization_count else set()
    rows = []
    print(f"Evaluating {len(records)} held-out meshes from {model_path}")
    for index, record in enumerate(records):
        sample_id = Path(record["_metadata_path"]).stem
        rgb, depth = load_inputs(record, preprocessing["max_depth"])
        raw = model.predict(
            {"rgb": rgb[None], "depth": depth[None]}, verbose=0
        )
        predicted_camera = denormalize_points(
            raw[MESH_OUTPUT_NAME][0], mesh_statistics
        )
        predicted_world = camera_to_world_points(
            predicted_camera, record["camera"]
        )
        mesh_path = test_split / "meshes" / f"{sample_id}.ply"
        target_world = sample_mesh_surface(
            mesh_path, num_points, int(sample_id)
        )
        metrics = point_cloud_metrics(
            predicted_world, target_world, args.distance_threshold
        )
        row = {
            "sample_id": sample_id,
            "shape": record["shape"]["type"],
            **metrics,
        }
        rows.append(row)
        trimesh.PointCloud(predicted_world).export(
            predicted_output / f"{sample_id}.ply"
        )
        trimesh.PointCloud(target_world).export(
            target_output / f"{sample_id}.ply"
        )
        if index in visualization_indices:
            save_visualization(
                visualization_output / f"{sample_id}.png", rgb,
                predicted_world, target_world, sample_id, metrics,
            )
        print(
            f"Evaluated {index + 1}/{len(records)}: {sample_id}, "
            f"Chamfer RMSE={metrics['chamfer_rmse_m']:.4f} m, "
            f"F-score={metrics['fscore']:.3f}"
        )

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["distance_threshold_m"] = args.distance_threshold
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved mesh evaluation to {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
