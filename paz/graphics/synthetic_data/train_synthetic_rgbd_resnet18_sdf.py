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
    MATERIAL_NAMES,
    OUTPUT_NAMES,
    PeriodicWeightsCheckpoint,
    REGRESSION_NAMES,
    RGBDDataset,
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


def world_to_camera_matrix(camera):
    """Reproduces PAZ's saved look-at transform exactly."""
    position = np.asarray(camera["position_world_xyz"], dtype=np.float64)
    target = np.asarray(camera["target_world_xyz"], dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    up = np.cross(left, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([left, up, -forward])
    matrix[:3, 3] = -matrix[:3, :3] @ position
    return matrix


def rotation_from_axes(first, second, epsilon=1e-8):
    first = first / max(np.linalg.norm(first), epsilon)
    second = second - np.dot(first, second) * first
    second = second / max(np.linalg.norm(second), epsilon)
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def canonical_to_camera_matrix(record):
    """Recovers the exact canonical-object to PAZ-camera transformation."""
    view = world_to_camera_matrix(record["camera"])
    orientation = record["object"]["orientation_camera_6d"]
    camera_axes = np.asarray([
        orientation["vector_a"], orientation["vector_b"]
    ], dtype=np.float64)
    world_axes = camera_axes @ np.linalg.inv(view[:3, :3]).T
    object_to_world = rotation_from_axes(world_axes[0], world_axes[1])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        view[:3, :3] @ object_to_world * record["object"]["scale"]
    )
    transform[:3, 3] = record["object"]["translation_camera_xyz"]
    return transform


def transform_vertices(vertices, matrix):
    homogeneous = np.concatenate([
        vertices, np.ones((len(vertices), 1), dtype=np.float64)
    ], axis=-1)
    return (homogeneous @ matrix.T)[:, :3]


def load_canonical_mesh(record, mesh_frame):
    """Loads an arbitrary triangle mesh and maps it to object coordinates."""
    sample_id = Path(record["_metadata_path"]).stem
    path = Path(record["_root"]) / "meshes" / f"{sample_id}.ply"
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh found in {path}")
    if mesh_frame == "world":
        world_to_camera = world_to_camera_matrix(record["camera"])
        world_to_canonical = (
            np.linalg.inv(canonical_to_camera_matrix(record))
            @ world_to_camera
        )
        mesh.vertices = transform_vertices(mesh.vertices, world_to_canonical)
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_mesh_surface(mesh, count, rng):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ), axis=-1)
    if not np.isfinite(areas).all() or areas.sum() <= 0.0:
        raise ValueError("Mesh has no finite surface area")
    chosen = triangles[rng.choice(
        len(triangles), count, p=areas / areas.sum()
    )]
    first = np.sqrt(rng.random((count, 1)))
    second = rng.random((count, 1))
    return ((1.0 - first) * chosen[:, 0]
            + first * (1.0 - second) * chosen[:, 1]
            + first * second * chosen[:, 2])


def winding_inside(points, triangles, chunk_size):
    """Classifies points using generalized winding numbers."""
    inside = []
    for begin in range(0, len(points), chunk_size):
        vectors = triangles[None] - points[
            begin:begin + chunk_size, None, None, :
        ]
        first, second, third = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        numerator = np.einsum(
            "...i,...i->...", first, np.cross(second, third)
        )
        norms = np.linalg.norm(vectors, axis=-1)
        denominator = (
            np.prod(norms, axis=-1)
            + np.einsum("...i,...i->...", first, second) * norms[:, :, 2]
            + np.einsum("...i,...i->...", second, third) * norms[:, :, 0]
            + np.einsum("...i,...i->...", third, first) * norms[:, :, 1]
        )
        solid_angle = 2.0 * np.arctan2(numerator, denominator)
        winding = np.abs(np.sum(solid_angle, axis=1)) / (4.0 * np.pi)
        inside.append(winding > 0.5)
    return np.concatenate(inside)


def mesh_signed_distance(mesh, points, chunk_size):
    """Computes dependency-free signed distances to a watertight mesh."""
    distances = []
    for begin in range(0, len(points), chunk_size):
        _, distance, _ = trimesh.proximity.closest_point_naive(
            mesh, points[begin:begin + chunk_size]
        )
        distances.append(distance)
    distances = np.concatenate(distances)
    inside = winding_inside(points, np.asarray(mesh.triangles), chunk_size)
    distances[inside] *= -1.0
    return distances


def sample_sdf_queries(record, num_queries, extent, truncation, mesh_frame,
                       distance_chunk_size):
    """Samples repeatable SDF supervision from any watertight mesh."""
    sample_id = int(Path(record["_metadata_path"]).stem)
    rng = np.random.default_rng(sample_id)
    mesh = load_canonical_mesh(record, mesh_frame)
    if not mesh.is_watertight:
        raise ValueError(
            f"SDF requires a watertight mesh: {record['_metadata_path']}"
        )
    largest_coordinate = float(np.max(np.abs(mesh.bounds)))
    if largest_coordinate >= extent:
        raise ValueError(
            f"Canonical mesh {record['_metadata_path']} reaches "
            f"{largest_coordinate:.3f}, outside --sdf-extent {extent:.3f}; "
            "increase --sdf-extent or normalize the source mesh"
        )
    uniform_count = num_queries // 2
    surface_count = num_queries - uniform_count
    uniform = rng.uniform(-extent, extent, size=(uniform_count, 3))
    surface = sample_mesh_surface(mesh, surface_count, rng)
    surface += rng.normal(0.0, 0.08, size=surface.shape)
    queries = np.concatenate([uniform, surface]).astype(np.float32)
    rng.shuffle(queries)
    distances = mesh_signed_distance(mesh, queries, distance_chunk_size)
    normalized = np.clip(distances, -truncation, truncation) / truncation
    return queries, normalized[:, None].astype(np.float32)


def extract_generic_targets(record, shape_names):
    shape_index = shape_names.index(record["shape"]["type"])
    shape = np.eye(len(shape_names), dtype=np.float32)[shape_index]
    orientation = record["object"]["orientation_camera_6d"]
    orientation_6d = np.asarray(
        orientation["vector_a"] + orientation["vector_b"], np.float32
    )
    source_material = record["material"]
    material = np.concatenate([
        np.asarray(source_material["color_rgb"], dtype=np.float32),
        np.asarray([
            source_material["diffuse"], source_material["specular"],
            source_material["ambient"], source_material["shininess"],
        ], dtype=np.float32),
    ])
    return {
        "object_translation": np.asarray(
            record["object"]["translation_camera_xyz"], np.float32
        ),
        "object_orientation_6d": np.concatenate([orientation_6d, shape]),
        "object_scale": np.asarray([record["object"]["scale"]], np.float32),
        "shape": shape,
        "material": material,
        "light_position": np.asarray(
            record["light"]["position_camera_xyz"], np.float32
        ),
        "light_intensity": np.asarray(
            [record["light"]["intensity"]], np.float32
        ),
    }


class GenericTargetNormalizer:
    """Normalizes physical targets while supporting arbitrary shape labels."""

    def __init__(self, statistics, shape_names):
        self.statistics = statistics
        self.shape_names = tuple(shape_names)

    @classmethod
    def fit(cls, records, shape_names):
        targets = [
            extract_generic_targets(record, shape_names) for record in records
        ]
        statistics = {}
        for name in REGRESSION_NAMES:
            values = np.stack([target[name] for target in targets])
            mean, deviation = values.mean(axis=0), values.std(axis=0)
            deviation = np.where(deviation < 1e-6, 1.0, deviation)
            statistics[name] = {
                "mean": mean.tolist(),
                "standard_deviation": deviation.tolist(),
            }
        return cls(statistics, shape_names)

    def normalize(self, targets):
        normalized = {
            name: np.asarray(targets[name], np.float32)
            for name in OUTPUT_NAMES
        }
        for name in REGRESSION_NAMES:
            statistics = self.statistics[name]
            mean = np.asarray(statistics["mean"], np.float32)
            deviation = np.asarray(
                statistics["standard_deviation"], np.float32
            )
            normalized[name] = (normalized[name] - mean) / deviation
        return normalized

    def save(self, path):
        payload = {
            "shape_names": self.shape_names,
            "material_definition": {
                "values": MATERIAL_NAMES,
                "source": "generator metadata material fields",
            },
            "targets": self.statistics,
        }
        with path.open("w") as file:
            json.dump(payload, file, indent=2)


class RGBDSDFDataset(RGBDDataset):
    """Adds canonical XYZ queries and truncated SDF targets to RGB-D data."""

    def __init__(self, *args, num_sdf_queries, sdf_extent, sdf_truncation,
                 shape_names, mesh_frame, distance_chunk_size, cache_dir,
                 **kwargs):
        self.num_sdf_queries = num_sdf_queries
        self.sdf_extent = sdf_extent
        self.sdf_truncation = sdf_truncation
        self.shape_names = tuple(shape_names)
        self.mesh_frame = mesh_frame
        self.distance_chunk_size = distance_chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)
        samples = [self.load_or_create_sdf(record) for record in self.records]
        self.sdf_queries = np.stack([sample[0] for sample in samples])
        self.sdf_targets = np.stack([sample[1] for sample in samples])

    def load_or_create_sdf(self, record):
        sample_id = Path(record["_metadata_path"]).stem
        key = (
            f"{sample_id}_q{self.num_sdf_queries}_e{self.sdf_extent:g}"
            f"_t{self.sdf_truncation:g}_{self.mesh_frame}.npz"
        )
        path = self.cache_dir / key
        if path.is_file():
            cached = np.load(path)
            return cached["queries"], cached["targets"]
        sample = sample_sdf_queries(
            record, self.num_sdf_queries, self.sdf_extent,
            self.sdf_truncation, self.mesh_frame, self.distance_chunk_size,
        )
        np.savez_compressed(path, queries=sample[0], targets=sample[1])
        return sample

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        rgb_images, depth_images, targets = [], [], []
        for index in indices:
            record = self.records[index]
            root = Path(record["_root"])
            image_path = root / record["rgb"]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            rgb_images.append(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            depth = np.load(root / record["depth"]).astype(np.float32)
            if depth.shape != rgb_images[-1].shape[:2]:
                raise ValueError(f"RGB/depth mismatch for {image_path}")
            depth_images.append(
                np.clip(depth / self.max_depth, 0.0, 1.0)[..., None]
            )
            targets.append(self.normalizer.normalize(
                extract_generic_targets(record, self.shape_names)
            ))
        inputs = {
            "rgb": np.stack(rgb_images), "depth": np.stack(depth_images),
            "sdf_query_points": self.sdf_queries[indices],
        }
        stacked_targets = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        stacked_targets[SDF_OUTPUT_NAME] = self.sdf_targets[indices]
        return inputs, stacked_targets


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


def rotation_6d_to_matrix_ops(values, epsilon=1e-8):
    first, second = values[..., :3], values[..., 3:6]
    first_norm = keras.ops.linalg.norm(first, axis=-1, keepdims=True)
    axis_x = first / keras.ops.maximum(first_norm, epsilon)
    projection = keras.ops.sum(axis_x * second, axis=-1, keepdims=True)
    second = second - projection * axis_x
    second_norm = keras.ops.linalg.norm(second, axis=-1, keepdims=True)
    axis_y = second / keras.ops.maximum(second_norm, epsilon)
    axis_z = keras.ops.cross(axis_x, axis_y)
    return keras.ops.stack([axis_x, axis_y, axis_z], axis=-1)


@keras.saving.register_keras_serializable(package="paz")
def generic_rotation_loss(target_6d_and_shape, predicted_6d):
    """Chordal SO(3) loss without assumptions about object symmetries."""
    target = rotation_6d_to_matrix_ops(target_6d_and_shape[..., :6])
    prediction = rotation_6d_to_matrix_ops(predicted_6d)
    return 0.5 * keras.ops.sum(
        keras.ops.square(prediction - target), axis=(-2, -1)
    )


@keras.saving.register_keras_serializable(package="paz")
def generic_geodesic_angle_degrees(target_6d_and_shape, predicted_6d):
    target = rotation_6d_to_matrix_ops(target_6d_and_shape[..., :6])
    prediction = rotation_6d_to_matrix_ops(predicted_6d)
    relative = keras.ops.matmul(keras.ops.transpose(target, (0, 2, 1)),
                                prediction)
    cosine = keras.ops.clip(
        (keras.ops.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0, 1.0,
    )
    return keras.ops.arccos(cosine) * (180.0 / np.pi)


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


def build_sdf_model(image_shape=(256, 256), l2_regularization=1e-4,
                    num_shapes=3):
    """Adds a query-conditioned implicit SDF decoder to the physical model."""
    base = build_model(image_shape, l2_regularization, num_shapes)
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
    parser.add_argument(
        "--mesh-frame", choices=("world", "canonical"), default="world",
        help="Coordinate frame of dataset PLY files.",
    )
    parser.add_argument(
        "--distance-chunk-size", type=int, default=32,
        help="Queries per brute-force mesh-distance chunk.",
    )
    parser.add_argument(
        "--sdf-cache", type=Path, default=None,
        help="Defaults to <output>/sdf_cache.",
    )
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
        args.distance_chunk_size,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes, ranges, and weights must be positive")
    if args.mesh_resolution < 4:
        raise ValueError("mesh resolution must be at least four")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)

    records = load_records(args.dataset)
    shape_names = tuple(sorted({
        record["shape"]["type"] for record in records
    }))
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
    normalizer = GenericTargetNormalizer.fit(train_records, shape_names)
    normalizer.save(args.output / "normalization.json")
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "depth_unit": "metres", "max_depth": args.max_depth,
            "geometry_representation": "conditional canonical SDF",
            "num_sdf_queries": args.num_sdf_queries,
            "sdf_extent": args.sdf_extent,
            "sdf_truncation": args.sdf_truncation,
            "sdf_target_source": "ground-truth triangle meshes",
            "mesh_frame": args.mesh_frame,
            "shape_names": shape_names,
            "mesh_extraction": "marching tetrahedra at SDF zero level",
        }, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    sdf_options = {
        "num_sdf_queries": args.num_sdf_queries,
        "sdf_extent": args.sdf_extent,
        "sdf_truncation": args.sdf_truncation,
        "shape_names": shape_names,
        "mesh_frame": args.mesh_frame,
        "distance_chunk_size": args.distance_chunk_size,
        "cache_dir": args.sdf_cache or args.output / "sdf_cache",
    }
    training = RGBDSDFDataset(
        train_records, *common, shuffle=True, seed=args.seed, **sdf_options
    )
    validation = RGBDSDFDataset(
        validation_records, *common, shuffle=False, seed=args.seed,
        **sdf_options,
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_sdf_model(
        image_shape, args.l2_regularization, len(shape_names)
    )
    sdf_loss = keras.losses.Huber(delta=0.1, name="truncated_sdf_huber")
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={
            "object_orientation_6d": generic_rotation_loss,
            SDF_OUTPUT_NAME: sdf_loss,
        },
        extra_metrics={
            "object_orientation_6d": [
                keras.metrics.MeanMetricWrapper(
                    generic_rotation_loss, name="loss"
                ),
                generic_geodesic_angle_degrees,
            ],
            SDF_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(sdf_loss, name="loss"),
                CanonicalSDFMAE(args.sdf_truncation),
            ],
        },
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
        f"Training mesh-supervised SDF for {shape_names} on {gpu}: "
        f"{len(train_records)} train, "
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
