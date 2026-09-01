"""Train RGB ResNet-18 + depth-voxel local features for camera-frame SDF.

Pipeline:
    RGB -> ResNet-18 -----------+
    depth -> point cloud -> voxel grid -> 3D CNN
                                  |
                      camera-frame XYZ query
                                  |
                    local voxel feature + global RGB-D feature
                                  |
                              SDF decoder
                                  |
                       dense evaluation + Marching Cubes

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_voxel_sdf \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/resnet18_depth_voxel_sdf
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import importlib.util
import json
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
    TrainingPlot,
    build_encoder,
    compile_model,
    mlp_branch,
    save_split_manifest,
    split_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    export_test_split,
    load_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    GenericTargetNormalizer,
    RepeatGeometryLatent,
    SDF_OUTPUT_NAME,
    export_test_meshes,
    extract_generic_targets,
    generic_geodesic_angle_degrees,
    generic_rotation_loss,
    mesh_signed_distance,
    sample_mesh_surface,
    world_to_camera_matrix,
)


def camera_ray_directions(height, width, y_fov_degrees):
    """Returns normalized PAZ camera-ray directions for every pixel."""
    y_fov = np.deg2rad(y_fov_degrees)
    aspect = width / height
    height_world = 2.0 * np.tan(y_fov / 2.0)
    width_world = height_world * aspect
    pixel_size = width_world / width
    x_offset = (np.arange(width) + 0.5) * pixel_size
    y_offset = (np.arange(height) + 0.5) * pixel_size
    x_grid, y_grid = np.meshgrid(
        (width_world / 2.0) - x_offset,
        (height_world / 2.0) - y_offset,
    )
    directions = np.stack([-x_grid, y_grid, -np.ones_like(x_grid)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def depth_to_voxel(depth, bounds, resolution, y_fov_degrees):
    """Back-projects metric ray depth into a camera-space occupancy grid."""
    directions = camera_ray_directions(*depth.shape, y_fov_degrees)
    valid = np.isfinite(depth) & (depth > 0.0)
    points = directions[valid] * depth[valid, None]
    lower, upper = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    normalized = (points - lower) / (upper - lower)
    inside = np.all((normalized >= 0.0) & (normalized <= 1.0), axis=-1)
    indices = np.floor(
        normalized[inside] * (resolution - 1)
    ).astype(np.int32)
    voxel = np.zeros((resolution,) * 3, dtype=np.float32)
    if len(indices):
        voxel[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return voxel[..., None]


def load_camera_mesh(record):
    sample_id = Path(record["_metadata_path"]).stem
    path = Path(record["_root"]) / "meshes" / f"{sample_id}.ply"
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh found in {path}")
    mesh.apply_transform(world_to_camera_matrix(record["camera"]))
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_camera_sdf(record, num_queries, truncation, distance_chunk_size):
    """Samples signed-distance supervision in camera coordinates."""
    sample_id = int(Path(record["_metadata_path"]).stem)
    rng = np.random.default_rng(sample_id)
    mesh = load_camera_mesh(record)
    if not mesh.is_watertight:
        raise ValueError(f"SDF requires a watertight mesh: {record['_metadata_path']}")
    uniform_count = num_queries // 2
    surface_count = num_queries - uniform_count
    surface = sample_mesh_surface(mesh, surface_count, rng)
    physical_truncation = truncation * float(record["object"]["scale"])
    surface += rng.normal(0.0, physical_truncation * 0.4, surface.shape)
    lower, upper = mesh.bounds
    padding = max(physical_truncation * 2.0, 1e-3)
    uniform = rng.uniform(lower - padding, upper + padding,
                          size=(uniform_count, 3))
    queries = np.concatenate([uniform, surface]).astype(np.float32)
    rng.shuffle(queries)
    distances = mesh_signed_distance(mesh, queries, distance_chunk_size)
    targets = np.clip(
        distances, -physical_truncation, physical_truncation
    ) / physical_truncation
    return queries, targets[:, None].astype(np.float32)


class RGBVoxelSDFDataset(keras.utils.PyDataset):
    """Loads RGB, builds camera voxels, and supplies camera-frame SDF queries."""

    def __init__(self, records, normalizer, shape_names, batch_size, max_depth,
                 y_fov, voxel_bounds, voxel_resolution, num_sdf_queries,
                 sdf_truncation, distance_chunk_size, cache_dir,
                 shuffle=False, seed=0):
        super().__init__()
        self.records = list(records)
        self.normalizer = normalizer
        self.shape_names = tuple(shape_names)
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.y_fov = y_fov
        self.voxel_bounds = tuple(voxel_bounds)
        self.voxel_resolution = voxel_resolution
        self.num_sdf_queries = num_sdf_queries
        self.sdf_truncation = sdf_truncation
        self.distance_chunk_size = distance_chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        samples = [self.load_or_create_geometry(record) for record in records]
        self.voxels = np.stack([sample[0] for sample in samples])
        self.queries = np.stack([sample[1] for sample in samples])
        self.sdf_targets = np.stack([sample[2] for sample in samples])
        self.on_epoch_end()

    def load_or_create_geometry(self, record):
        sample_id = Path(record["_metadata_path"]).stem
        bounds_key = "_".join(f"{value:g}" for value in self.voxel_bounds)
        path = self.cache_dir / (
            f"{sample_id}_v{self.voxel_resolution}_b{bounds_key}"
            f"_q{self.num_sdf_queries}_t{self.sdf_truncation:g}.npz"
        )
        if path.is_file():
            cached = np.load(path)
            return cached["voxel"], cached["queries"], cached["targets"]
        root = Path(record["_root"])
        depth = np.load(root / record["depth"]).astype(np.float32)
        depth = np.where(
            np.isfinite(depth) & (depth <= self.max_depth), depth, 0.0
        )
        voxel = depth_to_voxel(
            depth, self.voxel_bounds, self.voxel_resolution, self.y_fov
        )
        queries, targets = sample_camera_sdf(
            record, self.num_sdf_queries, self.sdf_truncation,
            self.distance_chunk_size,
        )
        np.savez_compressed(
            path, voxel=voxel, queries=queries, targets=targets
        )
        return voxel, queries, targets

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        images, targets = [], []
        for index in indices:
            record = self.records[index]
            path = Path(record["_root"]) / record["rgb"]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            images.append(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            targets.append(self.normalizer.normalize(
                extract_generic_targets(record, self.shape_names)
            ))
        inputs = {
            "rgb": np.stack(images),
            "depth_voxel": self.voxels[indices],
            "sdf_query_points": self.queries[indices],
        }
        output = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        output[SDF_OUTPUT_NAME] = self.sdf_targets[indices]
        return inputs, output

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


@keras.saving.register_keras_serializable(package="paz")
class CameraVoxelSampler(keras.layers.Layer):
    """Trilinearly interpolates camera-space 3D CNN features."""

    def __init__(self, bounds, resolution, **kwargs):
        super().__init__(**kwargs)
        self.bounds = tuple(float(value) for value in bounds)
        self.resolution = int(resolution)

    def call(self, inputs):
        grid, queries = inputs
        lower = keras.ops.convert_to_tensor(self.bounds[:3], dtype=queries.dtype)
        upper = keras.ops.convert_to_tensor(self.bounds[3:], dtype=queries.dtype)
        coordinates = (queries - lower) / (upper - lower)
        coordinates = keras.ops.clip(coordinates, 0.0, 1.0)
        coordinates = coordinates * (self.resolution - 1)
        base_float = keras.ops.floor(coordinates)
        base = keras.ops.cast(base_float, "int32")
        ceiling = keras.ops.minimum(base + 1, self.resolution - 1)
        fraction = coordinates - base_float
        flat = keras.ops.reshape(
            grid, (keras.ops.shape(grid)[0], self.resolution ** 3,
                   keras.ops.shape(grid)[-1])
        )
        result = keras.ops.zeros_like(queries[..., :1]) * flat[:, :1, :]
        for x_high in (0, 1):
            for y_high in (0, 1):
                for z_high in (0, 1):
                    x = ceiling[..., 0] if x_high else base[..., 0]
                    y = ceiling[..., 1] if y_high else base[..., 1]
                    z = ceiling[..., 2] if z_high else base[..., 2]
                    index = x * self.resolution ** 2 + y * self.resolution + z
                    index = keras.ops.repeat(
                        index[..., None], keras.ops.shape(flat)[-1], axis=-1
                    )
                    corner = keras.ops.take_along_axis(flat, index, axis=1)
                    wx = fraction[..., 0] if x_high else 1.0 - fraction[..., 0]
                    wy = fraction[..., 1] if y_high else 1.0 - fraction[..., 1]
                    wz = fraction[..., 2] if z_high else 1.0 - fraction[..., 2]
                    result = result + corner * (wx * wy * wz)[..., None]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (input_shape[0][-1],)

    def get_config(self):
        config = super().get_config()
        config.update({"bounds": self.bounds, "resolution": self.resolution})
        return config


def build_model(image_shape, num_shapes, voxel_resolution, voxel_bounds,
                l2_regularization):
    """Builds ResNet-18 RGB and camera-space 3D voxel feature branches."""
    regularizer = keras.regularizers.L2(l2_regularization)
    rgb = keras.Input((*image_shape, 3), name="rgb")
    voxel = keras.Input(
        (voxel_resolution,) * 3 + (1,), name="depth_voxel"
    )
    queries = keras.Input((None, 3), name="sdf_query_points")
    rgb_features = build_encoder(rgb, regularizer, "rgb")
    spatial = voxel
    for index, width in enumerate((16, 32, 64), start=1):
        spatial = keras.layers.Conv3D(
            width, 3, padding="same", activation="relu",
            kernel_regularizer=regularizer, name=f"voxel_conv3d_{index}",
        )(spatial)
    voxel_global = keras.layers.GlobalAveragePooling3D(
        name="voxel_global_feature"
    )(spatial)
    global_features = keras.layers.Concatenate(name="rgb_voxel_fusion")([
        rgb_features, voxel_global
    ])
    global_features = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=regularizer,
        name="fused_512d_feature",
    )(global_features)
    geometry = mlp_branch(global_features, "geometry", regularizer)
    material_branch = mlp_branch(global_features, "material", regularizer)
    lighting = mlp_branch(global_features, "lighting", regularizer)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation", kernel_regularizer=regularizer
        )(geometry),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d", kernel_regularizer=regularizer
        )(geometry),
        "object_scale": keras.layers.Dense(
            1, name="object_scale", kernel_regularizer=regularizer
        )(geometry),
        "shape": keras.layers.Dense(
            num_shapes, activation="softmax", name="shape",
            kernel_regularizer=regularizer,
        )(geometry),
        "material": keras.layers.Dense(
            len(MATERIAL_NAMES), name="material", kernel_regularizer=regularizer
        )(material_branch),
        "light_position": keras.layers.Dense(
            3, name="light_position", kernel_regularizer=regularizer
        )(lighting),
        "light_intensity": keras.layers.Dense(
            1, name="light_intensity", kernel_regularizer=regularizer
        )(lighting),
    }
    local = CameraVoxelSampler(
        voxel_bounds, voxel_resolution, name="camera_voxel_sampler"
    )([spatial, queries])
    global_condition = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=regularizer,
        name="global_rgbd_condition",
    )(global_features)
    global_condition = RepeatGeometryLatent(
        name="repeat_global_condition"
    )([global_condition, queries])
    decoded = keras.layers.Concatenate(name="local_global_query_features")([
        local, global_condition, queries
    ])
    for index, width in enumerate((256, 128, 64), start=1):
        decoded = keras.layers.Dense(
            width, activation="relu", kernel_regularizer=regularizer,
            name=f"sdf_decoder_{index}",
        )(decoded)
    outputs[SDF_OUTPUT_NAME] = keras.layers.Dense(
        1, activation="tanh", name=SDF_OUTPUT_NAME,
        kernel_regularizer=regularizer,
    )(decoded)
    return keras.Model(
        {"rgb": rgb, "depth_voxel": voxel, "sdf_query_points": queries},
        outputs, name="resnet18_depth_voxel_camera_sdf",
    )


def decode_center_scale(raw, statistics):
    translation_stats = statistics["object_translation"]
    scale_stats = statistics["object_scale"]
    center = (np.asarray(raw["object_translation"][0])
              * translation_stats["standard_deviation"]
              + translation_stats["mean"])
    scale = (np.asarray(raw["object_scale"][0])
             * scale_stats["standard_deviation"] + scale_stats["mean"])
    return center.astype(np.float32), max(float(scale[0]), 1e-3)


class MeshExtractionError(ValueError):
    """Reports an invalid predicted SDF field with extraction diagnostics."""

    def __init__(self, message, sdf_min=None, sdf_max=None):
        super().__init__(message)
        self.sdf_min = sdf_min
        self.sdf_max = sdf_max


def marching_cubes_mesh(model, rgb, voxel, statistics, extent, resolution,
                        chunk_size):
    """Evaluates camera SDF around predicted pose and applies Marching Cubes."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise ImportError(
            "Marching Cubes requires scikit-image: pip install scikit-image"
        ) from error
    dummy = np.zeros((1, 1, 3), dtype=np.float32)
    raw = model.predict({
        "rgb": rgb[None], "depth_voxel": voxel[None],
        "sdf_query_points": dummy,
    }, verbose=0)
    center, scale = decode_center_scale(raw, statistics)
    radius = extent * scale
    lower, upper = center - radius, center + radius
    axes = [np.linspace(lower[i], upper[i], resolution, dtype=np.float32)
            for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3)
    values = []
    for begin in range(0, len(flat), chunk_size):
        prediction = model.predict({
            "rgb": rgb[None], "depth_voxel": voxel[None],
            "sdf_query_points": flat[begin:begin + chunk_size][None],
        }, verbose=0)
        values.append(np.asarray(prediction[SDF_OUTPUT_NAME][0, :, 0]))
    field = np.concatenate(values).reshape((resolution,) * 3)
    sdf_min, sdf_max = float(field.min()), float(field.max())
    if not sdf_min <= 0.0 <= sdf_max:
        raise MeshExtractionError(
            "Predicted SDF has no zero crossing inside the extraction cube; "
            "check pose/scale predictions or increase --sdf-extent",
            sdf_min=sdf_min, sdf_max=sdf_max,
        )
    spacing = tuple((upper - lower) / (resolution - 1))
    vertices, faces, normals, _ = marching_cubes(
        field, level=0.0, spacing=spacing, gradient_direction="ascent"
    )
    vertices += lower
    return trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/resnet18_depth_voxel_sdf"))
    parser.add_argument("--voxel-resolution", type=int, default=24)
    parser.add_argument(
        "--voxel-bounds", type=float, nargs=6,
        default=(-4.0, -4.0, -10.0, 4.0, 4.0, 0.0),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--num-sdf-queries", type=int, default=2048)
    parser.add_argument("--sdf-extent", type=float, default=1.25)
    parser.add_argument("--sdf-truncation", type=float, default=0.2)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--preview-meshes", type=int, default=3)
    parser.add_argument("--mesh-resolution", type=int, default=48)
    parser.add_argument("--mesh-query-chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    bounds = np.asarray(args.voxel_bounds, dtype=np.float32)
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("voxel maximum bounds must exceed minimum bounds")
    positive = (
        args.voxel_resolution, args.y_fov, args.num_sdf_queries,
        args.sdf_extent, args.sdf_truncation, args.distance_chunk_size,
        args.batch_size, args.epochs, args.max_depth, args.checkpoint_every,
        args.mesh_resolution, args.mesh_query_chunk,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes and ranges must be positive")
    if args.preview_meshes and importlib.util.find_spec("skimage") is None:
        raise ImportError(
            "Marching Cubes preview extraction requires scikit-image. "
            "Install it with `pip install scikit-image`, or train with "
            "`--preview-meshes 0`."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)
    records = load_records(args.dataset)
    shape_names = tuple(sorted({r["shape"]["type"] for r in records}))
    train_records, validation_records, test_records = split_records(
        records, args.seed, args.validation_split
    )
    test_output = args.output / "test_split"
    export_test_split(test_records, test_output)
    export_test_meshes(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", train_records, validation_records,
        test_records, args.seed, args.validation_split,
    )
    normalizer = GenericTargetNormalizer.fit(train_records, shape_names)
    normalizer.save(args.output / "normalization.json")
    preprocessing = {
        "rgb_encoder": "ResNet-18", "depth_representation": "camera voxel",
        "voxel_resolution": args.voxel_resolution,
        "voxel_bounds_camera_xyz": args.voxel_bounds,
        "y_fov_degrees": args.y_fov, "max_depth": args.max_depth,
        "sdf_frame": "camera", "num_sdf_queries": args.num_sdf_queries,
        "sdf_truncation_relative_to_object_scale": args.sdf_truncation,
        "mesh_extraction": "skimage.measure.marching_cubes",
    }
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump(preprocessing, file, indent=2)
    common = {
        "normalizer": normalizer, "shape_names": shape_names,
        "batch_size": args.batch_size, "max_depth": args.max_depth,
        "y_fov": args.y_fov, "voxel_bounds": args.voxel_bounds,
        "voxel_resolution": args.voxel_resolution,
        "num_sdf_queries": args.num_sdf_queries,
        "sdf_truncation": args.sdf_truncation,
        "distance_chunk_size": args.distance_chunk_size,
        "cache_dir": args.output / "geometry_cache",
    }
    training = RGBVoxelSDFDataset(
        train_records, shuffle=True, seed=args.seed, **common
    )
    validation = RGBVoxelSDFDataset(
        validation_records, shuffle=False, seed=args.seed, **common
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_model(
        image_shape, len(shape_names), args.voxel_resolution,
        args.voxel_bounds, args.l2_regularization,
    )
    sdf_loss = keras.losses.Huber(delta=0.1, name="camera_sdf_huber")
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={
            "object_orientation_6d": generic_rotation_loss,
            SDF_OUTPUT_NAME: sdf_loss,
        },
        extra_metrics={
            "object_orientation_6d": [generic_geodesic_angle_degrees],
            SDF_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(sdf_loss, name="loss"),
                keras.metrics.MeanAbsoluteError(name="normalized_sdf_mae"),
            ],
        },
        extra_loss_weights={SDF_OUTPUT_NAME: 1.0},
    )
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(args.output / "loss.png", OUTPUT_NAMES + (SDF_OUTPUT_NAME,)),
        keras.callbacks.TerminateOnNaN(),
        PeriodicWeightsCheckpoint(args.output / "checkpoints",
                                  args.checkpoint_every),
        keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="val_loss",
            save_best_only=True, mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6,
            mode="min", verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, start_from_epoch=20,
            restore_best_weights=True, mode="min", verbose=1,
        ),
    ]
    model.summary()
    print(
        f"Training camera-voxel SDF on {gpu}: {len(train_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test"
    )
    model.fit(training, validation_data=validation, epochs=args.epochs,
              callbacks=callbacks)
    model.save(args.output / "final.keras")
    preview = args.output / "marching_cubes_meshes"
    preview.mkdir(parents=True, exist_ok=True)
    target_meshes = min(args.preview_meshes, len(test_records))
    extraction_failures = []
    successful_meshes = []
    for record in test_records:
        if len(successful_meshes) >= target_meshes:
            break
        sample_id = Path(record["_metadata_path"]).stem
        try:
            root = Path(record["_root"])
            bgr = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read RGB image: {root / record['rgb']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            depth = np.load(root / record["depth"]).astype(np.float32)
            voxel = depth_to_voxel(
                depth, args.voxel_bounds, args.voxel_resolution, args.y_fov
            )
            mesh = marching_cubes_mesh(
                model, rgb, voxel, normalizer.statistics, args.sdf_extent,
                args.mesh_resolution, args.mesh_query_chunk,
            )
            mesh.export(preview / f"{sample_id}.ply")
            successful_meshes.append(sample_id)
            print(f"Saved Marching Cubes mesh {sample_id}.ply "
                  f"({len(successful_meshes)}/{target_meshes})")
        except Exception as error:
            failure = {
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
                "sdf_min": getattr(error, "sdf_min", None),
                "sdf_max": getattr(error, "sdf_max", None),
            }
            extraction_failures.append(failure)
            print(f"Skipped mesh {sample_id}: {failure['message']} "
                  f"(SDF range: {failure['sdf_min']}, "
                  f"{failure['sdf_max']})")

    extraction_report = {
        "requested_successful_meshes": args.preview_meshes,
        "target_successful_meshes": target_meshes,
        "num_attempted": len(successful_meshes) + len(extraction_failures),
        "num_successful": len(successful_meshes),
        "num_failed": len(extraction_failures),
        "successful_sample_ids": successful_meshes,
        "failures": extraction_failures,
    }
    report_path = preview / "extraction_report.json"
    with report_path.open("w") as file:
        json.dump(extraction_report, file, indent=2)
    print(f"Mesh preview extraction completed: {len(successful_meshes)}/"
          f"{target_meshes} successful; report saved to {report_path}")


if __name__ == "__main__":
    main()
