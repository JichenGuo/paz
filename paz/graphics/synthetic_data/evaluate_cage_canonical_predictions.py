"""Evaluate cage predictions against ground-truth canonical surfaces.

The report separates surface accuracy, normal agreement, size, area, volume,
and common-shape collapse. Distances are measured in the canonical units used
by the training meshes.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import itertools
import json
from pathlib import Path

import keras
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .render_inverse_graphics_cage_predictions import (
    decode_prediction,
    inference_model,
    load_inputs,
    load_json,
    load_mesh_topology,
)
from .train_synthetic_rgbd_inverse_graphics_cage import (
    MESH_OUTPUT_NAME,
    transform_points,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh import (
    world_to_camera_matrix,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    canonical_to_camera_matrix,
)


def load_canonical_mesh(record):
    sample_id = Path(record["_metadata_path"]).stem
    mesh_path = Path(record["_root"]) / "meshes" / f"{sample_id}.ply"
    mesh = trimesh.load(mesh_path, process=False)
    matrix = (np.linalg.inv(canonical_to_camera_matrix(record))
              @ world_to_camera_matrix(record["camera"]))
    vertices = transform_points(np.asarray(mesh.vertices), matrix)
    faces = np.asarray(mesh.faces, np.int32)
    return trimesh.Trimesh(vertices, faces, process=False)


def sample_surface(vertices, faces, count, seed):
    triangles = np.asarray(vertices)[np.asarray(faces, np.int32)]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-12)
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        len(faces), count, p=twice_area / np.sum(twice_area)
    )
    first = np.sqrt(rng.random((count, 1)))
    second = rng.random((count, 1))
    chosen = triangles[selected]
    points = ((1.0 - first) * chosen[:, 0]
              + first * (1.0 - second) * chosen[:, 1]
              + first * second * chosen[:, 2])
    return points.astype(np.float32), normals[selected].astype(np.float32)


def nearest_surface_metrics(predicted, predicted_normals, target,
                            target_normals, thresholds):
    target_tree = cKDTree(target)
    predicted_tree = cKDTree(predicted)
    accuracy, target_indices = target_tree.query(predicted)
    completeness, predicted_indices = predicted_tree.query(target)
    first_normals = target_normals[target_indices]
    second_normals = predicted_normals[predicted_indices]
    first_alignment = np.abs(np.sum(predicted_normals * first_normals, axis=1))
    second_alignment = np.abs(np.sum(target_normals * second_normals, axis=1))
    metrics = {
        "surface_accuracy_mean": float(np.mean(accuracy)),
        "surface_completeness_mean": float(np.mean(completeness)),
        "chamfer_l1": float(0.5 * (
            np.mean(accuracy) + np.mean(completeness)
        )),
        "chamfer_l2": float(0.5 * (
            np.mean(np.square(accuracy))
            + np.mean(np.square(completeness))
        )),
        "normal_consistency": float(0.5 * (
            np.mean(first_alignment) + np.mean(second_alignment)
        )),
    }
    for threshold in thresholds:
        precision = np.mean(accuracy <= threshold)
        recall = np.mean(completeness <= threshold)
        denominator = max(precision + recall, 1e-12)
        name = f"fscore_{threshold:g}"
        metrics[name] = float(2.0 * precision * recall / denominator)
    return metrics


def geometry_metrics(predicted, faces, target_mesh, count, seed, thresholds):
    predicted_points, predicted_normals = sample_surface(
        predicted, faces, count, seed
    )
    target_points, target_normals = sample_surface(
        target_mesh.vertices, target_mesh.faces, count, seed + 1
    )
    metrics = nearest_surface_metrics(
        predicted_points, predicted_normals, target_points, target_normals,
        thresholds,
    )
    predicted_mesh = trimesh.Trimesh(predicted, faces, process=False)
    predicted_extent = np.ptp(predicted, axis=0)
    target_extent = np.ptp(target_mesh.vertices, axis=0)
    metrics.update({
        "extent_mae": float(np.mean(np.abs(
            predicted_extent - target_extent
        ))),
        "surface_area_relative_error": float(abs(
            predicted_mesh.area - target_mesh.area
        ) / max(target_mesh.area, 1e-12)),
        "volume_relative_error": float(abs(
            abs(predicted_mesh.volume) - abs(target_mesh.volume)
        ) / max(abs(target_mesh.volume), 1e-12)),
    })
    return metrics


def summarize_rows(rows):
    names = [name for name in rows[0] if name not in ("sample_id", "shape")]
    summary = {"num_samples": len(rows)}
    for name in names:
        values = np.asarray([row[name] for row in rows], np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values)),
        }
    return summary


def collapse_metrics(meshes):
    grouped = {}
    for shape, vertices in meshes:
        grouped.setdefault(shape, []).append(vertices)
    means = {shape: np.mean(values, axis=0)
             for shape, values in grouped.items()}
    within = {}
    for shape, values in grouped.items():
        pairs = itertools.combinations(values, 2)
        distances = [np.sqrt(np.mean(np.square(a - b))) for a, b in pairs]
        within[shape] = float(np.mean(distances)) if distances else 0.0
    between = {}
    for first, second in itertools.combinations(sorted(means), 2):
        distance = np.sqrt(np.mean(np.square(means[first] - means[second])))
        between[f"{first}_vs_{second}"] = float(distance)
    positive_within = [value for value in within.values() if value > 0.0]
    within_mean = float(np.mean(positive_within)) if positive_within else 0.0
    between_mean = float(np.mean(list(between.values()))) if between else 0.0
    ratio = None if within_mean == 0.0 else between_mean / within_mean
    return {
        "within_shape_vertex_rms": within,
        "between_shape_mean_vertex_rms": between,
        "mean_between_to_within_ratio": ratio,
    }


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=2048)
    parser.add_argument("--fscore-thresholds", type=float, nargs="+",
                        default=(0.02, 0.05, 0.1))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.surface_points < 1 or min(args.fscore_thresholds) <= 0.0:
        raise ValueError("surface sampling settings must be positive")
    dataset = args.dataset.resolve()
    experiment = args.experiment.resolve()
    model_path = args.model or experiment / "best.keras"
    output = args.output or experiment / "canonical_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    split = load_json(experiment / "split.json")
    normalization = load_json(experiment / "normalization.json")
    preprocessing = load_json(experiment / "input_preprocessing.json")
    _, topology, representation = load_mesh_topology(
        experiment, preprocessing
    )
    faces = np.asarray(topology["faces"], np.int32)
    sample_ids = list(split[args.split])
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]
    complete = keras.models.load_model(model_path, compile=False)
    model = inference_model(complete, representation)
    statistics = normalization["targets"]
    mesh_statistics = statistics[MESH_OUTPUT_NAME]
    shape_names = tuple(normalization["shape_names"])
    material_definition = normalization.get("material_definition", {})
    max_depth = float(preprocessing.get("max_depth", 10.0))
    y_fov = float(preprocessing.get("y_fov_degrees", 45.0))
    floor = float(preprocessing.get("render_floor_tolerance_metres", 0.01))
    rows, predicted_meshes = [], []
    for index, sample_id in enumerate(sample_ids):
        metadata_path = dataset / "metadata" / f"{sample_id}.json"
        record = load_json(metadata_path)
        record.update({"_root": str(dataset),
                       "_metadata_path": str(metadata_path)})
        rgb, _, depth, mask = load_inputs(
            record, dataset, max_depth, y_fov, floor
        )
        raw = model.predict({
            "rgb": rgb[None], "depth": depth[None],
            "object_mask": mask[None],
        }, verbose=0)
        vertices, _ = decode_prediction(
            raw, statistics, mesh_statistics, shape_names,
            material_definition,
        )
        target_mesh = load_canonical_mesh(record)
        metrics = geometry_metrics(
            vertices, faces, target_mesh, args.surface_points,
            int(sample_id), tuple(args.fscore_thresholds),
        )
        shape = record["shape"]["type"]
        rows.append({"sample_id": sample_id, "shape": shape, **metrics})
        predicted_meshes.append((shape, np.asarray(vertices)))
        print(f"Evaluated {index + 1}/{len(sample_ids)}: {sample_id}")
    if not rows:
        raise ValueError(f"No samples found in {args.split!r} split")
    by_shape = {}
    for shape in sorted({row["shape"] for row in rows}):
        by_shape[shape] = summarize_rows([
            row for row in rows if row["shape"] == shape
        ])
    report = {
        "model": str(model_path),
        "split": args.split,
        "surface_points": args.surface_points,
        "fscore_thresholds_canonical": args.fscore_thresholds,
        "overall": summarize_rows(rows),
        "by_shape": by_shape,
        "collapse": collapse_metrics(predicted_meshes),
    }
    save_csv(output / "metrics.csv", rows)
    with (output / "summary.json").open("w") as file:
        json.dump(report, file, indent=2)
    print(f"Saved canonical evaluation to {output}")


if __name__ == "__main__":
    main()
