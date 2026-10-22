"""Visualize balanced canonical cage-mesh predictions by shape.

This diagnostic is intended for geometry-only cage experiments. It selects an
equal number of held-out examples per shape, shows each source RGB beside its
canonical prediction using a common metric axis, and reports how much the
predicted meshes vary within and between shape classes.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
      paz.graphics.synthetic_data.visualize_cage_geometry_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/cage_geometry_full
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import itertools
import json
from pathlib import Path

import cv2
import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from paz.graphics.synthetic_data.render_inverse_graphics_atlas_predictions import (
    decode_prediction,
    inference_model,
    load_inputs,
    load_json,
    load_mesh_topology,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_inverse_graphics_atlas import (
    ATLAS_OUTPUT_NAME,
)


SHAPE_COLORS = {
    "cube": (0.25, 0.55, 0.90),
    "cylinder": (0.95, 0.55, 0.20),
    "sphere": (0.35, 0.75, 0.45),
}


def select_balanced_samples(dataset, sample_ids, count, seed):
    """Selects up to ``count`` held-out metadata records for every shape."""
    grouped = {}
    for sample_id in sample_ids:
        metadata = load_json(dataset / "metadata" / f"{sample_id}.json")
        shape = metadata["shape"]["type"]
        grouped.setdefault(shape, []).append((sample_id, metadata))
    rng = np.random.default_rng(seed)
    selected = {}
    for shape in sorted(grouped):
        candidates = grouped[shape]
        order = rng.permutation(len(candidates))
        selected[shape] = [candidates[index] for index in order[:count]]
    insufficient = {
        shape: len(records) for shape, records in selected.items()
        if len(records) < count
    }
    if insufficient:
        print(f"Warning: fewer than {count} test samples available: "
              f"{insufficient}")
    return selected


def draw_mesh(axis, vertices, faces, color, axis_limit):
    """Draws a canonical triangle mesh with fixed axes and shaded faces."""
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    light = np.asarray([0.35, -0.45, 0.82], np.float32)
    light /= np.linalg.norm(light)
    brightness = 0.35 + 0.65 * np.abs(normals @ light)
    base = np.asarray(color, np.float32)
    face_colors = np.clip(brightness[:, None] * base[None], 0.0, 1.0)
    collection = Poly3DCollection(
        triangles, facecolors=face_colors, edgecolors=(0.1, 0.1, 0.1, 0.35),
        linewidths=0.25,
    )
    axis.add_collection3d(collection)
    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_zlim(-axis_limit, axis_limit)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-55)
    axis.set_axis_off()


def pairwise_rms(meshes):
    values = []
    for first, second in itertools.combinations(meshes, 2):
        values.append(float(np.sqrt(np.mean(np.square(first - second)))))
    return values


def summarize_predictions(predictions):
    """Computes extents and fixed-topology within/between-class distances."""
    summary = {"shapes": {}}
    class_means = {}
    all_meshes = []
    for shape, records in predictions.items():
        meshes = [record["vertices"] for record in records]
        stack = np.stack(meshes)
        all_meshes.extend(meshes)
        extents = stack.max(axis=1) - stack.min(axis=1)
        within = pairwise_rms(meshes)
        class_means[shape] = stack.mean(axis=0)
        summary["shapes"][shape] = {
            "num_samples": len(meshes),
            "sample_ids": [record["sample_id"] for record in records],
            "mean_extent_xyz": extents.mean(axis=0).tolist(),
            "standard_deviation_extent_xyz": extents.std(axis=0).tolist(),
            "mean_within_shape_vertex_rms": (
                float(np.mean(within)) if within else 0.0
            ),
        }
    between = {}
    for first, second in itertools.combinations(sorted(class_means), 2):
        distance = np.sqrt(np.mean(np.square(
            class_means[first] - class_means[second]
        )))
        between[f"{first}_vs_{second}"] = float(distance)
    stack = np.stack(all_meshes)
    summary["between_shape_mean_vertex_rms"] = between
    summary["mean_vertex_standard_deviation_across_samples"] = float(
        stack.std(axis=0).mean()
    )
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Defaults to geometry_only.keras, then best.keras.",
    )
    parser.add_argument("--samples-per-shape", type=int, default=3)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--axis-limit", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.samples_per_shape < 1 or args.axis_limit <= 0.0:
        raise ValueError("samples per shape and axis limit must be positive")
    dataset = args.dataset.resolve()
    experiment = args.experiment.resolve()
    output = args.output or experiment / "canonical_geometry_verification.png"
    model_path = args.model
    if model_path is None:
        candidates = (
            experiment / "geometry_only.keras",
            experiment / "best.keras",
        )
        model_path = next((path for path in candidates if path.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(
                "Expected geometry_only.keras or best.keras in experiment"
            )
    split = load_json(experiment / "split.json")
    normalization = load_json(experiment / "normalization.json")
    preprocessing = load_json(experiment / "input_preprocessing.json")
    _, topology, representation = load_mesh_topology(
        experiment, preprocessing
    )
    faces = np.asarray(topology["faces"], np.int32)
    selected = select_balanced_samples(
        dataset, split[args.split], args.samples_per_shape, args.seed
    )
    if not selected:
        raise ValueError(f"No samples found in {args.split!r} split")
    complete_model = keras.models.load_model(model_path, compile=False)
    model = inference_model(complete_model, representation)
    statistics = normalization["targets"]
    mesh_statistics = statistics[ATLAS_OUTPUT_NAME]
    shape_names = tuple(normalization["shape_names"])
    material_definition = normalization.get("material_definition", {})
    max_depth = float(preprocessing.get("max_depth", 10.0))
    predictions = {}
    for shape, records in selected.items():
        predictions[shape] = []
        for sample_id, metadata in records:
            rgb, _, normalized_depth = load_inputs(
                metadata, dataset, max_depth
            )
            raw = model.predict({
                "rgb": rgb[None], "depth": normalized_depth[None]
            }, verbose=0)
            vertices, _ = decode_prediction(
                raw, statistics, mesh_statistics, shape_names,
                material_definition,
            )
            predictions[shape].append({
                "sample_id": sample_id,
                "rgb": rgb,
                "vertices": np.asarray(vertices, np.float32),
            })

    shapes = sorted(predictions)
    columns = max(len(records) for records in predictions.values()) * 2
    figure = plt.figure(
        figsize=(3.4 * columns, 3.7 * len(shapes)), constrained_layout=True
    )
    grid = figure.add_gridspec(len(shapes), columns)
    for row, shape in enumerate(shapes):
        color = SHAPE_COLORS.get(shape, (0.55, 0.65, 0.85))
        for index, record in enumerate(predictions[shape]):
            image_axis = figure.add_subplot(grid[row, 2 * index])
            image_axis.imshow(record["rgb"])
            image_axis.set_title(f"{shape}: {record['sample_id']}\ninput RGB")
            image_axis.set_axis_off()
            mesh_axis = figure.add_subplot(
                grid[row, 2 * index + 1], projection="3d"
            )
            draw_mesh(
                mesh_axis, record["vertices"], faces, color, args.axis_limit
            )
            extent = np.ptp(record["vertices"], axis=0)
            mesh_axis.set_title(
                "canonical prediction\n"
                + "extent=" + np.array2string(extent, precision=2)
            )
    figure.suptitle(
        "Canonical cage geometry by ground-truth shape "
        f"({args.split} split; common axis ±{args.axis_limit:g})",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    summary = summarize_predictions(predictions)
    summary.update({
        "model": str(model_path),
        "split": args.split,
        "axis_limit": args.axis_limit,
        "interpretation": (
            "Very small between-shape RMS distances and similar extents "
            "indicate common-shape collapse."
        ),
    })
    metrics_path = output.with_name(f"{output.stem}_metrics.json")
    with metrics_path.open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved canonical comparison to {output}")
    print(f"Saved collapse diagnostics to {metrics_path}")
    print("Between-shape mean-mesh RMS distances:")
    for pair, distance in summary["between_shape_mean_vertex_rms"].items():
        print(f"  {pair}: {distance:.6f}")


if __name__ == "__main__":
    main()
