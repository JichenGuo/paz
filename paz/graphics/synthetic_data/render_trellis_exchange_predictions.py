"""Render and evaluate two-stage PAZ + official TRELLIS predictions.

The input is the result directory written by ``run_trellis_exchange.py``.
Final world-frame meshes are rendered with the original camera, material, and
lighting so that the comparison isolates geometry and metric alignment.  No
neural network is loaded or executed by this script.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
      paz.graphics.synthetic_data.render_trellis_exchange_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --results experiments/trellis_exchange_results
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jp
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paz
from paz.graphics.synthetic_data.evaluate_sdf_preview_meshes import (
    relative_error,
    sample_surface,
    surface_metrics,
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh found in {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def unique_edges(faces):
    edges = np.concatenate([
        faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]
    ])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def camera_transform(camera):
    return paz.SE3.view_transform(
        jp.asarray(camera["position_world_xyz"]),
        jp.asarray(camera["target_world_xyz"]),
        jp.asarray([0.0, 1.0, 0.0]),
    )


def mesh_node(mesh, material_values):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    color = np.asarray(material_values["color_rgb"], dtype=np.float32)
    colors = np.repeat(color[None], len(vertices), axis=0)
    material = paz.graphics.Material(
        jp.asarray(color), float(material_values["ambient"]),
        float(material_values["diffuse"]),
        float(material_values["specular"]),
        float(material_values["shininess"]),
    )
    return paz.graphics.Mesh(
        jp.asarray(vertices), jp.asarray(colors), jp.eye(4), material,
        jp.asarray(faces), jp.asarray(unique_edges(faces)),
    )


def make_light(metadata, world_to_camera):
    light_camera = np.append(metadata["light"]["position_camera_xyz"], 1.0)
    light_world = np.linalg.inv(np.asarray(world_to_camera)) @ light_camera
    intensity = max(float(metadata["light"]["intensity"]), 0.0)
    return paz.graphics.PointLight(
        jp.full(3, intensity), jp.asarray(light_world[:3])
    )


def render_mesh(mesh, metadata, image_shape, y_fov, tiles, chunk_size,
                face_chunk_size, shadows, include_floor):
    world_to_camera = camera_transform(metadata["camera"])
    nodes = [mesh_node(mesh, metadata["material"])]
    if include_floor:
        floor_material = paz.graphics.Material(
            jp.asarray([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
        )
        nodes.insert(0, paz.graphics.Plane(material=floor_material))
    rgb, depth = paz.graphics.render(
        image_shape, np.deg2rad(y_fov), world_to_camera,
        paz.graphics.Scene(nodes), None,
        make_light(metadata, world_to_camera), tiles=(tiles, tiles),
        chunk_size=chunk_size, face_chunk_size=face_chunk_size,
        shadows=shadows,
    )
    return np.asarray(rgb), np.asarray(depth)


def load_rgbd(metadata, dataset, max_depth):
    path = dataset / metadata["rgb"]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(dataset / metadata["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {path}")
    depth = np.where(
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth),
        depth, 0.0,
    )
    return rgb, depth


def image_metrics(rendered_rgb, target_rgb, rendered_depth, target_depth):
    rgb_difference = rendered_rgb.astype(np.float64) - target_rgb
    rgb_mse = float(np.mean(np.square(rgb_difference)))
    valid = (rendered_depth > 0.0) & (target_depth > 0.0)
    if np.any(valid):
        difference = rendered_depth[valid] - target_depth[valid]
        depth_mae = float(np.mean(np.abs(difference)))
        depth_rmse = float(np.sqrt(np.mean(np.square(difference))))
    else:
        depth_mae = depth_rmse = float("nan")
    return {
        "rgb_mae_0_1": float(np.mean(np.abs(rgb_difference))),
        "rgb_psnr_db": (100.0 if rgb_mse < 1e-10 else
                        float(-10.0 * np.log10(rgb_mse))),
        "depth_mae_m": depth_mae,
        "depth_rmse_m": depth_rmse,
    }


def binary_iou(prediction, target):
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    return float(intersection / union) if union else 1.0


def rotation_6d_to_matrix(values, epsilon=1e-8):
    values = np.asarray(values, dtype=np.float64)
    first, second = values[:3], values[3:]
    axis_x = first / max(float(np.linalg.norm(first)), epsilon)
    second -= np.dot(axis_x, second) * axis_x
    axis_y = second / max(float(np.linalg.norm(second)), epsilon)
    return np.stack([axis_x, axis_y, np.cross(axis_x, axis_y)], axis=-1)


def pose_metrics(result):
    exchange = result.get("exchange_alignment", {})
    prediction = exchange.get("metric_alignment")
    target = exchange.get("ground_truth", {}).get("object")
    if not prediction or not target:
        return {}
    predicted_translation = np.asarray(
        prediction["translation_camera_xyz"], dtype=np.float64
    )
    target_translation = np.asarray(
        target["translation_camera_xyz"], dtype=np.float64
    )
    orientation = target["orientation_camera_6d"]
    target_rotation = rotation_6d_to_matrix(
        orientation["vector_a"] + orientation["vector_b"]
    )
    predicted_rotation = np.asarray(
        prediction["rotation_camera_3x3"], dtype=np.float64
    )
    relative = target_rotation.T @ predicted_rotation
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "translation_error_m": float(np.linalg.norm(
            predicted_translation - target_translation
        )),
        "orientation_error_degrees": float(np.degrees(np.arccos(cosine))),
        "scale_absolute_error": abs(
            float(prediction["scale_metres_per_canonical_unit"])
            - float(target["scale"])
        ),
    }


def add_fscore_thresholds(metrics, predicted, target, thresholds):
    for threshold in thresholds:
        value = surface_metrics(predicted, target, threshold)["fscore"]
        metrics[f"fscore_{int(round(threshold * 1000))}mm"] = value


def save_comparison(path, target_rgb, predicted_rgb, target_depth,
                    predicted_depth, target_mask, predicted_mask, metrics):
    rgb_error = np.mean(np.abs(predicted_rgb - target_rgb), axis=-1)
    valid = (target_depth > 0.0) & (predicted_depth > 0.0)
    depth_error = np.zeros_like(target_depth)
    depth_error[valid] = np.abs(
        predicted_depth[valid] - target_depth[valid]
    )
    positive = np.concatenate([
        target_depth[target_depth > 0.0],
        predicted_depth[predicted_depth > 0.0],
    ])
    depth_max = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    figure, axes = plt.subplots(3, 3, figsize=(15, 13))
    panels = [
        (target_rgb, "Original synthetic RGB", None),
        (np.clip(predicted_rgb, 0.0, 1.0), "PAZ + TRELLIS render", None),
        (rgb_error, f"RGB error; MAE={metrics['rgb_mae_0_1']:.4f}", "magma"),
        (target_depth, "Original depth [m]", "viridis"),
        (predicted_depth, "Predicted depth [m]", "viridis"),
        (depth_error, f"Depth error; MAE={metrics['depth_mae_m']:.4f} m", "magma"),
        (target_mask, "Ground-truth object mask", "gray"),
        (predicted_mask, "Predicted object mask", "gray"),
        (target_mask.astype(np.int8) - predicted_mask.astype(np.int8),
         f"Silhouette difference; IoU={metrics['silhouette_iou']:.3f}", "coolwarm"),
    ]
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        kwargs = {"cmap": cmap} if cmap else {}
        if "depth [m]" in title:
            kwargs.update(vmin=0.0, vmax=depth_max)
        axis.imshow(image, **kwargs)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_geometry_overlay(path, predicted, target, metrics):
    figure = plt.figure(figsize=(15, 5))
    views = ((20, -60), (90, -90), (0, 0))
    labels = ("Perspective", "Top", "Side")
    combined = np.concatenate([predicted, target], axis=0)
    center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))
    radius = max(float(np.max(np.ptp(combined, axis=0))) / 2.0, 1e-3)
    for index, ((elevation, azimuth), label) in enumerate(zip(views, labels)):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        axis.scatter(*target.T, s=1, alpha=0.28, c="tab:blue", label="GT")
        axis.scatter(*predicted.T, s=1, alpha=0.28, c="tab:orange",
                     label="Prediction")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(label)
        if index == 0:
            axis.legend()
    figure.suptitle(
        f"World-frame surfaces: Chamfer RMSE="
        f"{metrics['chamfer_rmse_m']:.4f} m"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def summarize(rows):
    result = {"num_samples": len(rows)}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        if key in {"sample_id", "shape"}:
            continue
        values = np.asarray([
            row.get(key, np.nan) for row in rows
        ], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "standard_deviation": float(values.std()),
            }
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-surface-points", type=int, default=5000)
    parser.add_argument("--fscore-thresholds", type=float, nargs="+",
                        default=(0.01, 0.02, 0.05))
    parser.add_argument("--num-geometry-plots", type=int, default=20)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--clear-caches-every", type=int, default=1)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (args.num_surface_points, args.max_depth, args.y_fov,
                args.tiles, args.chunk_size, args.face_chunk_size)
    if any(value <= 0 for value in positive):
        raise ValueError("Point count, render sizes, and ranges must be positive")
    if any(value <= 0 for value in args.fscore_thresholds):
        raise ValueError("F-score thresholds must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.num_geometry_plots < 0 or args.clear_caches_every < 0:
        raise ValueError("Plot count and cache interval cannot be negative")

    dataset, results = args.dataset.resolve(), args.results.resolve()
    output = (args.output or results / "render_evaluation").resolve()
    directories = {
        name: output / name for name in (
            "rendered_rgb", "rendered_depth", "silhouettes",
            "comparisons", "geometry_overlays", "metadata",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    mesh_paths = sorted((results / "world_meshes").glob("*.ply"))
    if args.max_samples is not None:
        mesh_paths = mesh_paths[:args.max_samples]
    if not mesh_paths:
        raise ValueError(f"No world meshes found in {results / 'world_meshes'}")
    plot_count = min(args.num_geometry_plots, len(mesh_paths))
    plot_indices = set(np.linspace(
        0, len(mesh_paths) - 1, plot_count, dtype=int
    )) if plot_count else set()

    rows, failures = [], []
    for index, predicted_path in enumerate(mesh_paths):
        sample_id = predicted_path.stem
        try:
            metadata = load_json(dataset / "metadata" / f"{sample_id}.json")
            result_path = results / "metadata" / f"{sample_id}.json"
            result = load_json(result_path) if result_path.is_file() else {}
            target_path = dataset / "meshes" / f"{sample_id}.ply"
            predicted_mesh = load_mesh(predicted_path)
            target_mesh = load_mesh(target_path)
            target_rgb, target_depth = load_rgbd(
                metadata, dataset, args.max_depth
            )

            start = time.perf_counter()
            predicted_rgb, predicted_depth = render_mesh(
                predicted_mesh, metadata, target_rgb.shape[:2], args.y_fov,
                args.tiles, args.chunk_size, args.face_chunk_size,
                not args.no_shadows, include_floor=True,
            )
            _, predicted_object_depth = render_mesh(
                predicted_mesh, metadata, target_rgb.shape[:2], args.y_fov,
                args.tiles, args.chunk_size, args.face_chunk_size,
                False, include_floor=False,
            )
            _, target_object_depth = render_mesh(
                target_mesh, metadata, target_rgb.shape[:2], args.y_fov,
                args.tiles, args.chunk_size, args.face_chunk_size,
                False, include_floor=False,
            )
            jax.block_until_ready(predicted_depth)
            render_ms = (time.perf_counter() - start) * 1000.0
            predicted_rgb = np.clip(predicted_rgb, 0.0, 1.0)
            predicted_depth = np.where(
                (predicted_depth > 0.0) &
                (predicted_depth <= args.max_depth), predicted_depth, 0.0,
            ).astype(np.float32)
            predicted_mask = predicted_object_depth > 0.0
            target_mask = target_object_depth > 0.0

            predicted_points = sample_surface(
                predicted_mesh, args.num_surface_points, int(sample_id)
            )
            target_points = sample_surface(
                target_mesh, args.num_surface_points, int(sample_id) + 1
            )
            metrics = image_metrics(
                predicted_rgb, target_rgb, predicted_depth, target_depth
            )
            metrics.update(surface_metrics(
                predicted_points, target_points, args.fscore_thresholds[1]
                if len(args.fscore_thresholds) > 1
                else args.fscore_thresholds[0]
            ))
            add_fscore_thresholds(
                metrics, predicted_points, target_points,
                args.fscore_thresholds,
            )
            metrics.update(pose_metrics(result))
            metrics.update({
                "silhouette_iou": binary_iou(predicted_mask, target_mask),
                "surface_area_relative_error": relative_error(
                    float(predicted_mesh.area), float(target_mesh.area)
                ),
                "predicted_watertight": int(predicted_mesh.is_watertight),
                "render_time_ms": render_ms,
            })
            row = {
                "sample_id": sample_id,
                "shape": metadata["shape"]["type"],
                **metrics,
            }
            rows.append(row)

            bgr = cv2.cvtColor(
                (predicted_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(
                directories["rendered_rgb"] / f"{sample_id}.png"
            ), bgr)
            np.save(
                directories["rendered_depth"] / f"{sample_id}.npy",
                predicted_depth,
            )
            cv2.imwrite(str(
                directories["silhouettes"] / f"{sample_id}.png"
            ), predicted_mask.astype(np.uint8) * 255)
            save_comparison(
                directories["comparisons"] / f"{sample_id}.png",
                target_rgb, predicted_rgb, target_depth, predicted_depth,
                target_mask, predicted_mask, metrics,
            )
            if index in plot_indices:
                save_geometry_overlay(
                    directories["geometry_overlays"] / f"{sample_id}.png",
                    predicted_points, target_points, metrics,
                )
            with (directories["metadata"] / f"{sample_id}.json").open(
                    "w") as file:
                json.dump({
                    "sample_id": sample_id,
                    "predicted_world_mesh": str(predicted_path),
                    "target_world_mesh": str(target_path),
                    "appearance_source": "ground-truth material and lighting",
                    "metrics": metrics,
                }, file, indent=2)
            print(
                f"Evaluated {index + 1}/{len(mesh_paths)}: {sample_id}; "
                f"Chamfer={metrics['chamfer_rmse_m']:.4f} m, "
                f"silhouette IoU={metrics['silhouette_iou']:.3f}, "
                f"depth MAE={metrics['depth_mae_m']:.4f} m"
            )
        except Exception as error:
            failures.append({
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
            print(f"Skipped {sample_id}: {type(error).__name__}: {error}")
        if (args.clear_caches_every > 0 and
                (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    if rows:
        keys = list(rows[0])
        for row in rows[1:]:
            keys.extend(key for key in row if key not in keys)
        with (output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows) if rows else {"num_samples": 0}
    summary.update({
        "num_requested": len(mesh_paths),
        "num_failed": len(failures),
        "mesh_coordinate_frame": "world",
        "render_appearance": "ground-truth material and lighting",
        "fscore_thresholds_metres": list(args.fscore_thresholds),
        "failures": failures,
    })
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved {len(rows)} comparisons to {output}; "
          f"{len(failures)} failures")


if __name__ == "__main__":
    main()
