"""Evaluate predicted canonical SDF meshes against original dataset meshes.

The SDF preview meshes are canonical. Ground-truth meshes are stored in world
coordinates, so both meshes are transformed to the camera frame using the
ground-truth pose before physical surface metrics are computed.

Example:
    python -m paz.graphics.synthetic_data.evaluate_sdf_preview_meshes \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --experiment experiments/resnet18_rgbd_sdf
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Not a triangle mesh: {path}")
    return mesh


def world_to_camera_matrix(camera):
    position = np.asarray(camera["position_world_xyz"], dtype=np.float64)
    target = np.asarray(camera["target_world_xyz"], dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    up = np.cross(left, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([left, up, -forward])
    matrix[:3, 3] = -matrix[:3, :3] @ position
    return matrix


def rotation_6d_to_matrix(first, second, epsilon=1e-8):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    axis_x = first / max(np.linalg.norm(first), epsilon)
    second = second - np.dot(axis_x, second) * axis_x
    axis_y = second / max(np.linalg.norm(second), epsilon)
    axis_z = np.cross(axis_x, axis_y)
    return np.stack([axis_x, axis_y, axis_z], axis=-1)


def canonical_to_camera_matrix(metadata):
    obj = metadata["object"]
    view_rotation = world_to_camera_matrix(metadata["camera"])[:3, :3]
    orientation = obj["orientation_camera_6d"]
    # The stored axes include PAZ's view transform, whose left/up rows are not
    # renormalized. Recover a true object-to-world rotation first, then reapply
    # that same view transform for exact alignment with the rendered mesh.
    camera_axes = np.stack([
        orientation["vector_a"], orientation["vector_b"]
    ])
    world_axes = camera_axes @ np.linalg.inv(view_rotation).T
    object_to_world = rotation_6d_to_matrix(world_axes[0], world_axes[1])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        view_rotation @ object_to_world
        * float(obj["scale"])
    )
    transform[:3, 3] = obj["translation_camera_xyz"]
    return transform


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :3]


def sample_surface(mesh, count, seed):
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0.0:
        raise ValueError("Mesh has no finite surface area")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(triangles), count, p=areas / areas.sum())
    chosen = triangles[selected]
    first = np.sqrt(rng.random((count, 1)))
    second = rng.random((count, 1))
    return ((1.0 - first) * chosen[:, 0]
            + first * (1.0 - second) * chosen[:, 1]
            + first * second * chosen[:, 2])


def nearest_distances(source, target, chunk_size=512):
    distances = []
    for begin in range(0, len(source), chunk_size):
        source_chunk = source[begin:begin + chunk_size]
        squared = np.sum(
            np.square(source_chunk[:, None, :] - target[None, :, :]), axis=-1
        )
        distances.append(np.sqrt(np.maximum(squared.min(axis=1), 0.0)))
    return np.concatenate(distances)


def surface_metrics(prediction, target, threshold):
    prediction_to_target = nearest_distances(prediction, target)
    target_to_prediction = nearest_distances(target, prediction)
    precision = np.mean(prediction_to_target <= threshold)
    recall = np.mean(target_to_prediction <= threshold)
    fscore = (0.0 if precision + recall == 0.0 else
              2.0 * precision * recall / (precision + recall))
    chamfer_m2 = 0.5 * (
        np.mean(np.square(prediction_to_target))
        + np.mean(np.square(target_to_prediction))
    )
    return {
        "chamfer_m2": float(chamfer_m2),
        "chamfer_rmse_m": float(np.sqrt(chamfer_m2)),
        "mean_surface_distance_m": float(0.5 * (
            prediction_to_target.mean() + target_to_prediction.mean()
        )),
        "hausdorff_distance_m": float(max(
            prediction_to_target.max(), target_to_prediction.max()
        )),
        "precision": float(precision),
        "recall": float(recall),
        "fscore": float(fscore),
    }


def relative_error(prediction, target):
    return float(abs(prediction - target) / max(abs(target), 1e-12))


def equal_limits(axis, point_sets):
    points = np.concatenate(point_sets)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * np.max(upper - lower), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X camera [m]")
    axis.set_ylabel("Y camera [m]")
    axis.set_zlabel("Z camera [m]")


def scatter(axis, points, color, label=None, alpha=0.7):
    step = max(1, len(points) // 2000)
    shown = points[::step]
    axis.scatter(
        shown[:, 0], shown[:, 1], shown[:, 2], s=3, color=color,
        label=label, alpha=alpha, depthshade=False,
    )


def save_visualization(path, rgb, predicted, target, sample_id, metrics):
    figure = plt.figure(figsize=(18, 5))
    image_axis = figure.add_subplot(1, 4, 1)
    image_axis.imshow(rgb)
    image_axis.set_title(f"Original RGB: {sample_id}")
    image_axis.axis("off")
    target_axis = figure.add_subplot(1, 4, 2, projection="3d")
    scatter(target_axis, target, "tab:blue")
    target_axis.set_title("Original mesh")
    prediction_axis = figure.add_subplot(1, 4, 3, projection="3d")
    scatter(prediction_axis, predicted, "tab:orange")
    prediction_axis.set_title("Predicted SDF mesh")
    overlay_axis = figure.add_subplot(1, 4, 4, projection="3d")
    scatter(overlay_axis, target, "tab:blue", "Original", 0.45)
    scatter(overlay_axis, predicted, "tab:orange", "Prediction", 0.65)
    overlay_axis.set_title(
        f"Overlay\nChamfer RMSE={metrics['chamfer_rmse_m']:.4f} m, "
        f"F-score={metrics['fscore']:.3f}"
    )
    overlay_axis.legend(fontsize=8)
    for axis in (target_axis, prediction_axis, overlay_axis):
        equal_limits(axis, [predicted, target])
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarize(rows):
    summary = {"num_samples": len(rows)}
    excluded = {"sample_id", "shape", "predicted_watertight",
                "target_watertight"}
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "standard_deviation": float(values.std()),
        }
    summary["predicted_watertight_fraction"] = float(np.mean([
        row["predicted_watertight"] for row in rows
    ]))
    for shape in sorted({row["shape"] for row in rows}):
        selected = [row for row in rows if row["shape"] == shape]
        summary[f"{shape}_num_samples"] = len(selected)
        summary[f"{shape}_chamfer_rmse_m_mean"] = float(np.mean([
            row["chamfer_rmse_m"] for row in selected
        ]))
        summary[f"{shape}_fscore_mean"] = float(np.mean([
            row["fscore"] for row in selected
        ]))
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/synthetic_rgbd_1000_v4"))
    parser.add_argument("--experiment", type=Path,
                        default=Path("experiments/resnet18_rgbd_sdf"))
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-surface-points", type=int, default=5000)
    parser.add_argument("--distance-threshold", type=float, default=0.02)
    parser.add_argument("--num-visualizations", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.num_surface_points < 1 or args.distance_threshold <= 0.0:
        raise ValueError("surface point count and threshold must be positive")
    if args.num_visualizations < 0:
        raise ValueError("visualization count cannot be negative")
    predictions = args.predictions or args.experiment / "sdf_preview_meshes"
    output = args.output or args.experiment / "sdf_mesh_evaluation"
    visualization_output = output / "visualizations"
    aligned_output = output / "predicted_camera_meshes"
    visualization_output.mkdir(parents=True, exist_ok=True)
    aligned_output.mkdir(parents=True, exist_ok=True)

    predicted_paths = sorted(predictions.glob("*.ply"))
    if args.max_samples is not None:
        predicted_paths = predicted_paths[:args.max_samples]
    if not predicted_paths:
        raise ValueError(f"No predicted PLY meshes found in {predictions}")
    visualization_count = min(args.num_visualizations, len(predicted_paths))
    visualization_indices = set(np.linspace(
        0, len(predicted_paths) - 1, visualization_count, dtype=int
    )) if visualization_count else set()

    rows = []
    for index, predicted_path in enumerate(predicted_paths):
        sample_id = predicted_path.stem
        metadata_path = args.dataset / "metadata" / f"{sample_id}.json"
        target_path = args.dataset / "meshes" / f"{sample_id}.ply"
        with metadata_path.open() as file:
            metadata = json.load(file)
        predicted_mesh = load_mesh(predicted_path)
        target_mesh = load_mesh(target_path)
        canonical_to_camera = canonical_to_camera_matrix(metadata)
        world_to_camera = world_to_camera_matrix(metadata["camera"])
        predicted_camera_mesh = predicted_mesh.copy()
        predicted_camera_mesh.apply_transform(canonical_to_camera)
        target_camera_mesh = target_mesh.copy()
        target_camera_mesh.apply_transform(world_to_camera)
        predicted_points = sample_surface(
            predicted_camera_mesh, args.num_surface_points, int(sample_id)
        )
        target_points = sample_surface(
            target_camera_mesh, args.num_surface_points, int(sample_id) + 1
        )
        metrics = surface_metrics(
            predicted_points, target_points, args.distance_threshold
        )
        predicted_area = float(predicted_camera_mesh.area)
        target_area = float(target_camera_mesh.area)
        row = {
            "sample_id": sample_id,
            "shape": metadata["shape"]["type"],
            **metrics,
            "surface_area_relative_error": relative_error(
                predicted_area, target_area
            ),
            "predicted_watertight": int(predicted_camera_mesh.is_watertight),
            "target_watertight": int(target_camera_mesh.is_watertight),
        }
        rows.append(row)
        predicted_camera_mesh.export(aligned_output / predicted_path.name)
        if index in visualization_indices:
            bgr = cv2.imread(
                str(args.dataset / metadata["rgb"]), cv2.IMREAD_COLOR
            )
            if bgr is None:
                raise FileNotFoundError(args.dataset / metadata["rgb"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            save_visualization(
                visualization_output / f"{sample_id}.png", rgb,
                predicted_points, target_points, sample_id, metrics,
            )
        print(
            f"Evaluated {index + 1}/{len(predicted_paths)}: {sample_id}, "
            f"Chamfer RMSE={metrics['chamfer_rmse_m']:.4f} m, "
            f"F-score={metrics['fscore']:.3f}"
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["distance_threshold_m"] = args.distance_threshold
    summary["num_surface_points_per_mesh"] = args.num_surface_points
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved evaluation to {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
