"""Train an RGB-D physical model with conditional implicit SDF geometry.

The dual ResNet-18 encoder and existing physical/material/lighting heads are
retained. A geometry latent conditions a point-wise SDF decoder on canonical
XYZ queries. The zero level set is converted to triangles with a dependency-
free marching-tetrahedra implementation.

Example:
    KERAS_BACKEND=jax python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/resnet18_rgbd_sdf
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import shutil
from pathlib import Path

import cv2
import jax
import keras
import numpy as np
import trimesh

from paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 import (
    OUTPUT_NAMES,
    PeriodicWeightsCheckpoint,
    RGBDDataset,
    TargetNormalizer,
    TrainingPlot,
    build_model,
    compile_model,
    save_split_manifest,
    split_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    export_test_split,
    load_records,
)


SDF_OUTPUT_NAME = "sdf_values"


def canonical_sdf(points, shape):
    """Returns exact canonical SDF for PAZ cube, cylinder, or sphere."""
    points = np.asarray(points, dtype=np.float32)
    if shape == "sphere":
        return np.linalg.norm(points, axis=-1) - 1.0
    if shape == "cube":
        offset = np.abs(points) - 1.0
        outside = np.linalg.norm(np.maximum(offset, 0.0), axis=-1)
        inside = np.minimum(np.max(offset, axis=-1), 0.0)
        return outside + inside
    if shape == "cylinder":
        offset = np.stack([
            np.linalg.norm(points[:, (0, 2)], axis=-1) - 1.0,
            np.abs(points[:, 1]) - 1.0,
        ], axis=-1)
        outside = np.linalg.norm(np.maximum(offset, 0.0), axis=-1)
        inside = np.minimum(np.max(offset, axis=-1), 0.0)
        return outside + inside
    raise ValueError(f"Unsupported shape: {shape}")


def sample_surface_points(shape, count, rng):
    """Samples exact canonical primitive surfaces for near-surface queries."""
    if shape == "sphere":
        points = rng.normal(size=(count, 3))
        return points / np.linalg.norm(points, axis=-1, keepdims=True)
    if shape == "cube":
        points = rng.uniform(-1.0, 1.0, size=(count, 3))
        axes = rng.integers(0, 3, size=count)
        signs = rng.choice((-1.0, 1.0), size=count)
        points[np.arange(count), axes] = signs
        return points
    if shape == "cylinder":
        points = np.empty((count, 3), dtype=np.float64)
        side = rng.random(count) < (2.0 / 3.0)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        points[side, 0] = np.cos(angles[side])
        points[side, 2] = np.sin(angles[side])
        points[side, 1] = rng.uniform(-1.0, 1.0, size=np.count_nonzero(side))
        caps = ~side
        radii = np.sqrt(rng.random(np.count_nonzero(caps)))
        points[caps, 0] = radii * np.cos(angles[caps])
        points[caps, 2] = radii * np.sin(angles[caps])
        points[caps, 1] = rng.choice((-1.0, 1.0), size=np.count_nonzero(caps))
        return points
    raise ValueError(f"Unsupported shape: {shape}")


def sample_sdf_queries(record, num_queries, extent, truncation):
    """Samples repeatable uniform and near-surface SDF supervision."""
    sample_id = int(Path(record["_metadata_path"]).stem)
    rng = np.random.default_rng(sample_id)
    uniform_count = num_queries // 2
    surface_count = num_queries - uniform_count
    uniform = rng.uniform(-extent, extent, size=(uniform_count, 3))
    surface = sample_surface_points(
        record["shape"]["type"], surface_count, rng
    )
    surface += rng.normal(0.0, 0.08, size=surface.shape)
    queries = np.concatenate([uniform, surface]).astype(np.float32)
    rng.shuffle(queries)
    distances = canonical_sdf(queries, record["shape"]["type"])
    normalized = np.clip(distances, -truncation, truncation) / truncation
    return queries, normalized[:, None].astype(np.float32)


class RGBDSDFDataset(RGBDDataset):
    """Adds canonical XYZ queries and truncated SDF targets to RGB-D data."""

    def __init__(self, *args, num_sdf_queries, sdf_extent, sdf_truncation,
                 **kwargs):
        self.num_sdf_queries = num_sdf_queries
        self.sdf_extent = sdf_extent
        self.sdf_truncation = sdf_truncation
        super().__init__(*args, **kwargs)
        samples = [
            sample_sdf_queries(
                record, num_sdf_queries, sdf_extent, sdf_truncation
            )
            for record in self.records
        ]
        self.sdf_queries = np.stack([sample[0] for sample in samples])
        self.sdf_targets = np.stack([sample[1] for sample in samples])

    def __getitem__(self, batch_index):
        inputs, targets = super().__getitem__(batch_index)
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        inputs["sdf_query_points"] = self.sdf_queries[indices]
        targets[SDF_OUTPUT_NAME] = self.sdf_targets[indices]
        return inputs, targets


@keras.saving.register_keras_serializable(package="paz")
class CanonicalSDFMAE(keras.metrics.Metric):
    """Mean absolute SDF error in canonical object units."""

    def __init__(self, truncation, name="canonical_sdf_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.truncation = float(truncation)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, target, prediction, sample_weight=None):
        errors = keras.ops.abs(prediction - target) * self.truncation
        values = keras.ops.mean(errors, axis=(-2, -1))
        if sample_weight is not None:
            values = values * keras.ops.cast(sample_weight, values.dtype)
        self.total.assign_add(keras.ops.sum(values))
        self.count.assign_add(keras.ops.cast(keras.ops.size(values), "float32"))

    def result(self):
        return self.total / keras.ops.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config["truncation"] = self.truncation
        return config


@keras.saving.register_keras_serializable(package="paz")
class RepeatGeometryLatent(keras.layers.Layer):
    """Broadcasts one image latent to every SDF query point."""

    def call(self, inputs):
        latent, query_points = inputs
        query_ones = keras.ops.ones_like(query_points[..., :1])
        return query_ones * keras.ops.expand_dims(latent, axis=1)

    def compute_output_shape(self, input_shape):
        latent_shape, query_shape = input_shape
        return query_shape[:-1] + (latent_shape[-1],)


def build_sdf_model(image_shape=(256, 256), l2_regularization=1e-4):
    """Adds a query-conditioned implicit SDF decoder to the physical model."""
    base = build_model(image_shape, l2_regularization)
    regularizer = keras.regularizers.L2(l2_regularization)
    fused = base.get_layer("fused_512d_feature").output
    latent = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=regularizer,
        name="geometry_sdf_latent",
    )(fused)
    query_points = keras.Input((None, 3), name="sdf_query_points")
    repeated = RepeatGeometryLatent(name="repeat_geometry_latent")(
        [latent, query_points]
    )
    decoder = keras.layers.Concatenate(name="sdf_conditioning")(
        [repeated, query_points]
    )
    for index, width in enumerate((256, 128, 64), start=1):
        decoder = keras.layers.Dense(
            width, activation="relu", kernel_regularizer=regularizer,
            name=f"sdf_decoder_dense{index}",
        )(decoder)
    sdf_values = keras.layers.Dense(
        1, activation="tanh", kernel_regularizer=regularizer,
        name=SDF_OUTPUT_NAME,
    )(decoder)
    outputs = dict(base.output)
    outputs[SDF_OUTPUT_NAME] = sdf_values
    inputs = dict(base.input)
    inputs["sdf_query_points"] = query_points
    return keras.Model(inputs, outputs, name="physical_parameter_rgbd_sdf")


CUBE_CORNERS = np.asarray([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int32)
TETRAHEDRA = np.asarray([
    [0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
    [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
], dtype=np.int32)
TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def order_polygon(points):
    """Orders three or four coplanar intersection points around a centroid."""
    points = np.asarray(points)
    center = points.mean(axis=0)
    centered = points - center
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    first, second = right_vectors[0], right_vectors[1]
    angles = np.arctan2(centered @ second, centered @ first)
    return points[np.argsort(angles)]


def marching_tetrahedra(sdf_grid, extent):
    """Extracts the SDF zero level set as a triangle mesh in canonical space."""
    resolution = sdf_grid.shape[0]
    if sdf_grid.shape != (resolution, resolution, resolution):
        raise ValueError("SDF grid must be cubic")
    coordinates = np.linspace(-extent, extent, resolution)
    vertices, faces = [], []
    for x in range(resolution - 1):
        for y in range(resolution - 1):
            for z in range(resolution - 1):
                base = np.asarray([x, y, z])
                indices = base + CUBE_CORNERS
                cube_points = coordinates[indices]
                cube_values = sdf_grid[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ]
                for tetrahedron in TETRAHEDRA:
                    points = cube_points[tetrahedron]
                    values = cube_values[tetrahedron]
                    intersections = []
                    for first, second in TETRA_EDGES:
                        first_value, second_value = values[first], values[second]
                        if (first_value < 0.0) == (second_value < 0.0):
                            continue
                        fraction = first_value / (first_value - second_value)
                        intersections.append(
                            points[first] + fraction
                            * (points[second] - points[first])
                        )
                    if len(intersections) not in (3, 4):
                        continue
                    polygon = order_polygon(intersections)
                    start = len(vertices)
                    vertices.extend(polygon)
                    for corner in range(1, len(polygon) - 1):
                        faces.append((start, start + corner, start + corner + 1))
    if not faces:
        raise ValueError("No SDF zero crossing found in extraction grid")
    mesh = trimesh.Trimesh(
        np.asarray(vertices), np.asarray(faces), process=True
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def predict_sdf_mesh(model, rgb, depth, resolution, extent, chunk_size):
    """Evaluates a conditional SDF grid and extracts its canonical mesh."""
    coordinates = np.linspace(-extent, extent, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    ), axis=-1)
    flat_queries = grid.reshape(-1, 3)
    predictions = []
    for begin in range(0, len(flat_queries), chunk_size):
        queries = flat_queries[begin:begin + chunk_size]
        raw = model.predict({
            "rgb": rgb[None], "depth": depth[None],
            "sdf_query_points": queries[None],
        }, verbose=0)
        predictions.append(np.asarray(raw[SDF_OUTPUT_NAME][0, :, 0]))
    sdf_grid = np.concatenate(predictions).reshape(grid.shape[:3])
    return marching_tetrahedra(sdf_grid, extent)


def load_rgb_depth(record, max_depth):
    root = Path(record["_root"])
    bgr = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(root / record["rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(root / record["depth"]).astype(np.float32)
    depth = np.clip(depth / max_depth, 0.0, 1.0)[..., None]
    return rgb, depth


def export_test_meshes(records, destination):
    mesh_output = destination / "meshes"
    mesh_output.mkdir(parents=True, exist_ok=True)
    for record in records:
        stem = Path(record["_metadata_path"]).stem
        source = Path(record["_root"]) / "meshes" / f"{stem}.ply"
        shutil.copy2(source, mesh_output / source.name)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/resnet18_rgbd_sdf"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--num-sdf-queries", type=int, default=2048)
    parser.add_argument("--sdf-extent", type=float, default=1.25)
    parser.add_argument("--sdf-truncation", type=float, default=0.2)
    parser.add_argument("--sdf-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5)
    parser.add_argument("--lr-reduction-patience", type=int, default=7)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=1e-3)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--preview-meshes", type=int, default=3)
    parser.add_argument("--mesh-resolution", type=int, default=32)
    parser.add_argument("--mesh-query-chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.num_sdf_queries, args.sdf_extent, args.sdf_truncation,
        args.sdf_loss_weight, args.batch_size, args.epochs, args.max_depth,
        args.checkpoint_every, args.mesh_resolution, args.mesh_query_chunk,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes, ranges, and weights must be positive")
    if args.mesh_resolution < 4:
        raise ValueError("mesh resolution must be at least four")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)

    records = load_records(args.dataset)
    train_records, validation_records, test_records = split_records(
        records, args.seed, args.validation_split
    )
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    export_test_meshes(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", train_records, validation_records,
        test_records, args.seed, args.validation_split,
    )
    normalizer = TargetNormalizer.fit(train_records)
    normalizer.save(args.output / "normalization.json")
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "depth_unit": "metres", "max_depth": args.max_depth,
            "geometry_representation": "conditional canonical SDF",
            "num_sdf_queries": args.num_sdf_queries,
            "sdf_extent": args.sdf_extent,
            "sdf_truncation": args.sdf_truncation,
            "mesh_extraction": "marching tetrahedra at SDF zero level",
        }, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    sdf_options = {
        "num_sdf_queries": args.num_sdf_queries,
        "sdf_extent": args.sdf_extent,
        "sdf_truncation": args.sdf_truncation,
    }
    training = RGBDSDFDataset(
        train_records, *common, shuffle=True, seed=args.seed, **sdf_options
    )
    validation = RGBDSDFDataset(
        validation_records, *common, shuffle=False, seed=args.seed,
        **sdf_options,
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_sdf_model(image_shape, args.l2_regularization)
    sdf_loss = keras.losses.Huber(delta=0.1, name="truncated_sdf_huber")
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={SDF_OUTPUT_NAME: sdf_loss},
        extra_metrics={SDF_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(sdf_loss, name="loss"),
            CanonicalSDFMAE(args.sdf_truncation),
        ]},
        extra_loss_weights={SDF_OUTPUT_NAME: args.sdf_loss_weight},
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png", OUTPUT_NAMES + (SDF_OUTPUT_NAME,)
        ),
        keras.callbacks.TerminateOnNaN(),
        PeriodicWeightsCheckpoint(
            args.output / "checkpoints", args.checkpoint_every
        ),
        keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="val_loss", mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=args.lr_reduction_factor,
            patience=args.lr_reduction_patience,
            min_lr=args.min_learning_rate, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            start_from_epoch=args.early_stopping_start_epoch,
            restore_best_weights=True, verbose=1,
        ),
    ]
    print(
        f"Training conditional SDF model on {gpu}: {len(train_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")

    preview_output = args.output / "sdf_preview_meshes"
    preview_output.mkdir(parents=True, exist_ok=True)
    for record in test_records[:args.preview_meshes]:
        sample_id = Path(record["_metadata_path"]).stem
        rgb, depth = load_rgb_depth(record, args.max_depth)
        mesh = predict_sdf_mesh(
            model, rgb, depth, args.mesh_resolution, args.sdf_extent,
            args.mesh_query_chunk,
        )
        mesh.export(preview_output / f"{sample_id}.ply")
        print(f"Saved canonical SDF preview mesh {sample_id}.ply")


if __name__ == "__main__":
    main()
