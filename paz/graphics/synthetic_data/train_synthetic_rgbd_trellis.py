"""Train a TRELLIS-inspired structured-latent RGB-D mesh reconstructor.

This is a compact PAZ adaptation, not Microsoft's pretrained TRELLIS model.
It predicts a structured 3D latent from RGB-D and decodes dense scalar,
grid-vertex deformation, and interpolation-weight parameters for direct triangle
mesh extraction. It has no query-point input or pointwise SDF decoder.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_trellis \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/rgbd_trellis_simple_shapes
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import cv2
import json
from pathlib import Path

import jax
import keras
import numpy as np
import trimesh

from paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 import (
    OUTPUT_NAMES,
    PeriodicWeightsCheckpoint,
    TrainingPlot,
    RGBDDataset,
    build_model,
    compile_model,
    save_split_manifest,
    split_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    export_test_split,
    load_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    GenericTargetNormalizer,
    export_test_meshes,
    generic_geodesic_angle_degrees,
    generic_rotation_loss,
    extract_generic_targets,
    winding_inside,
    load_canonical_mesh,
    load_rgb_depth,
)


STRUCTURE_OUTPUT_NAME = "structure_occupancy"
SCALAR_OUTPUT_NAME = "flexicubes_scalar"
DEFORMATION_OUTPUT_NAME = "flexicubes_deformation"
WEIGHTS_OUTPUT_NAME = "flexicubes_weights"


def grid_coordinates(resolution, extent):
    coordinates = np.linspace(
        -extent, extent, resolution, dtype=np.float32
    )
    return np.stack(np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    ), axis=-1)


def closest_mesh_distances(mesh, points, chunk_size):
    distances = []
    for begin in range(0, len(points), chunk_size):
        _, distance, _ = trimesh.proximity.closest_point_naive(
            mesh, points[begin:begin + chunk_size]
        )
        distances.append(distance)
    return np.concatenate(distances)


def build_structure_target(record, resolution, extent, surface_band,
                           mesh_frame, distance_chunk_size):
    """Marks sparse grid cells close to the canonical mesh surface."""
    mesh = load_canonical_mesh(record, mesh_frame)
    coordinates = grid_coordinates(resolution, extent)
    distance = closest_mesh_distances(
        mesh, coordinates.reshape(-1, 3), distance_chunk_size
    )
    occupancy = distance.reshape((resolution,) * 3) <= surface_band
    return occupancy[..., None].astype(np.float32)


class RGBDFlexiCubesDataset(RGBDDataset):
    """Loads RGB-D and cached dense mesh-decoder supervision grids."""

    def __init__(self, *args, structure_resolution, structure_surface_band, mesh_frame, mesh_cache, distance_chunk_size, mesh_extent, **kwargs):
        self.structure_resolution = structure_resolution
        self.structure_surface_band = structure_surface_band
        self.mesh_frame = mesh_frame
        self.mesh_cache = Path(mesh_cache)
        self.mesh_cache.mkdir(parents=True, exist_ok=True)
        self.distance_chunk_size = distance_chunk_size
        self.mesh_extent = mesh_extent
        super().__init__(*args, **kwargs)
        samples = [self.load_or_create_mesh_grids(record) for record in self.records]
        self.structure_targets = np.stack([x[0] for x in samples])
        self.scalar_targets = np.stack([x[1] for x in samples])
        self.deformation_targets = np.stack([x[2] for x in samples])
        self.weight_targets = np.stack([x[3] for x in samples])

    def load_or_create_mesh_grids(self, record):
        sample_id = Path(record["_metadata_path"]).stem
        path = self.mesh_cache / (f"{sample_id}_spatialcanonical_v2_flexicubes_r{self.structure_resolution}_e{self.mesh_extent:g}_{self.mesh_frame}.npz")
        if path.is_file():
            data = np.load(path)
            return tuple(data[name] for name in ("structure", "scalar", "deformation", "weights"))
        # Dataset meshes are stored in world coordinates. The inherited
        # loader uses "world" to request world-to-canonical conversion.
        source_frame = "world" if self.mesh_frame == "canonical" \
            else "canonical"
        mesh = load_canonical_mesh(record, source_frame)
        grid = grid_coordinates(self.structure_resolution, self.mesh_extent)
        points = grid.reshape(-1, 3)
        closest, distance, _ = trimesh.proximity.closest_point_naive(mesh, points)
        inside = winding_inside(points, np.asarray(mesh.triangles), self.distance_chunk_size)
        scalar = inside.reshape(grid.shape[:-1])[..., None].astype(np.float32)
        mask = distance <= self.structure_surface_band
        structure = mask.reshape(grid.shape[:-1])[..., None].astype(np.float32)
        voxel_size = 2.0 * self.mesh_extent / (self.structure_resolution - 1)
        deformation = np.clip((closest - points) / voxel_size, -0.5, 0.5) * mask[:, None]
        deformation = deformation.reshape(grid.shape).astype(np.float32)
        weights = np.ones(grid.shape[:-1] + (1,), dtype=np.float32)
        np.savez_compressed(path, structure=structure, scalar=scalar, deformation=deformation, weights=weights)
        return structure, scalar, deformation, weights

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        rgbs, depths, targets = [], [], []
        for index in indices:
            record = self.records[index]
            root = Path(record["_root"])
            bgr = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(root / record["rgb"])
            rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            depth = np.load(root / record["depth"]).astype(np.float32)
            depths.append(np.clip(depth / self.max_depth, 0.0, 1.0)[..., None])
            targets.append(self.normalizer.normalize(extract_generic_targets(record, self.normalizer.shape_names)))
        output = {name: np.stack([target[name] for target in targets]) for name in OUTPUT_NAMES}
        output.update({STRUCTURE_OUTPUT_NAME: self.structure_targets[indices], SCALAR_OUTPUT_NAME: self.scalar_targets[indices], DEFORMATION_OUTPUT_NAME: self.deformation_targets[indices], WEIGHTS_OUTPUT_NAME: self.weight_targets[indices]})
        return {"rgb": np.stack(rgbs), "depth": np.stack(depths)}, output


@keras.saving.register_keras_serializable(package="paz")
class TrilinearStructuredSampler(keras.layers.Layer):
    """Interpolates local features from a dense implementation of a SLAT."""

    def __init__(self, resolution, extent, **kwargs):
        super().__init__(**kwargs)
        self.resolution = int(resolution)
        self.extent = float(extent)

    def call(self, inputs):
        feature_grid, query_points = inputs
        resolution = self.resolution
        coordinates = (
            (query_points + self.extent) / (2.0 * self.extent)
            * (resolution - 1)
        )
        coordinates = keras.ops.clip(coordinates, 0.0, resolution - 1.0)
        lower_float = keras.ops.floor(coordinates)
        lower = keras.ops.cast(lower_float, "int32")
        upper = keras.ops.minimum(lower + 1, resolution - 1)
        fraction = coordinates - lower_float
        flat = keras.ops.reshape(
            feature_grid,
            (keras.ops.shape(feature_grid)[0], resolution ** 3,
             keras.ops.shape(feature_grid)[-1]),
        )
        result = keras.ops.zeros_like(query_points[..., :1]) * flat[:, :1, :]
        for x_upper in (0, 1):
            for y_upper in (0, 1):
                for z_upper in (0, 1):
                    x = upper[..., 0] if x_upper else lower[..., 0]
                    y = upper[..., 1] if y_upper else lower[..., 1]
                    z = upper[..., 2] if z_upper else lower[..., 2]
                    index = x * resolution * resolution + y * resolution + z
                    index = keras.ops.repeat(
                        index[..., None], keras.ops.shape(flat)[-1], axis=-1
                    )
                    corner = keras.ops.take_along_axis(flat, index, axis=1)
                    x_weight = (fraction[..., 0] if x_upper
                                else 1.0 - fraction[..., 0])
                    y_weight = (fraction[..., 1] if y_upper
                                else 1.0 - fraction[..., 1])
                    z_weight = (fraction[..., 2] if z_upper
                                else 1.0 - fraction[..., 2])
                    weight = x_weight * y_weight * z_weight
                    result = result + corner * weight[..., None]
        return result

    def compute_output_shape(self, input_shape):
        grid_shape, query_shape = input_shape
        return query_shape[:-1] + (grid_shape[-1],)

    def get_config(self):
        config = super().get_config()
        config.update({
            "resolution": self.resolution, "extent": self.extent,
        })
        return config



@keras.saving.register_keras_serializable(package="paz")
class LiftSpatialFeatures3D(keras.layers.Layer):
    """Lifts an XY feature map through Z and appends a Z coordinate."""

    def __init__(self, resolution, **kwargs):
        super().__init__(**kwargs)
        self.resolution = int(resolution)

    def call(self, features):
        lifted = keras.ops.expand_dims(features, axis=3)
        lifted = keras.ops.repeat(lifted, self.resolution, axis=3)
        batch = keras.ops.shape(features)[0]
        z = keras.ops.linspace(-1.0, 1.0, self.resolution)
        z = keras.ops.reshape(z, (1, 1, 1, self.resolution, 1))
        z = keras.ops.broadcast_to(
            z, (batch, self.resolution, self.resolution,
                self.resolution, 1),
        )
        return keras.ops.concatenate([lifted, z], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.resolution, self.resolution,
                self.resolution, input_shape[-1] + 1)

    def get_config(self):
        config = super().get_config()
        config.update({"resolution": self.resolution})
        return config


@keras.saving.register_keras_serializable(package="paz")
class RepeatGlobalGrid(keras.layers.Layer):
    """Broadcasts compact global context over a structured 3D grid."""

    def __init__(self, resolution, **kwargs):
        super().__init__(**kwargs)
        self.resolution = int(resolution)

    def call(self, features):
        features = keras.ops.reshape(
            features, (keras.ops.shape(features)[0], 1, 1, 1,
                       keras.ops.shape(features)[-1]),
        )
        return keras.ops.tile(
            features, (1, self.resolution, self.resolution,
                       self.resolution, 1),
        )

    def compute_output_shape(self, input_shape):
        return ((input_shape[0],) + (self.resolution,) * 3
                + (input_shape[-1],))

    def get_config(self):
        config = super().get_config()
        config.update({"resolution": self.resolution})
        return config


def build_trellis_model(image_shape, num_shapes, structure_resolution,
                        structure_channels, mesh_extent, l2_regularization):
    """Builds a spatial RGB-D structured mesh-parameter decoder."""
    base = build_model(image_shape, l2_regularization, num_shapes)
    regularizer = keras.regularizers.L2(l2_regularization)
    fused = base.get_layer("fused_512d_feature").output
    rgb_spatial = base.get_layer("rgb_stage4_block2_output").output
    depth_spatial = base.get_layer("depth_stage4_block2_output").output
    spatial = keras.layers.Concatenate(name="rgb_depth_spatial_fusion")([
        rgb_spatial, depth_spatial
    ])
    spatial = keras.layers.Conv2D(
        structure_channels, 1, activation="relu",
        kernel_regularizer=regularizer, name="spatial_channel_projection",
    )(spatial)
    spatial = keras.layers.Resizing(
        structure_resolution, structure_resolution,
        interpolation="bilinear", name="spatial_grid_resize",
    )(spatial)
    spatial = LiftSpatialFeatures3D(
        structure_resolution, name="spatial_feature_lift_3d"
    )(spatial)
    global_context = keras.layers.Dense(
        structure_channels, activation="relu",
        kernel_regularizer=regularizer, name="mesh_global_context",
    )(fused)
    global_context = RepeatGlobalGrid(
        structure_resolution, name="repeat_mesh_global_context"
    )(global_context)
    structured = keras.layers.Concatenate(name="spatial_global_3d_fusion")([
        spatial, global_context
    ])
    structured = keras.layers.Conv3D(
        structure_channels, 3, padding="same", activation="relu",
        kernel_regularizer=regularizer, name="structured_latent_grid",
    )(structured)
    for index in range(2):
        residual = structured
        structured = keras.layers.Conv3D(structure_channels, 3, padding="same", activation="relu", kernel_regularizer=regularizer, name=f"slat_mesh_conv_{index + 1}")(structured)
        structured = keras.layers.Add(name=f"slat_mesh_residual_{index + 1}")([structured, residual])
    occupancy = keras.layers.Conv3D(1, 1, activation="sigmoid", name=STRUCTURE_OUTPUT_NAME, kernel_regularizer=regularizer)(structured)
    scalar = keras.layers.Conv3D(1, 1, activation="sigmoid", name=SCALAR_OUTPUT_NAME, kernel_regularizer=regularizer)(structured)
    deformation = keras.layers.Conv3D(3, 1, activation="tanh", name="raw_flexicubes_deformation", kernel_regularizer=regularizer)(structured)
    deformation = keras.layers.Rescaling(0.5, name=DEFORMATION_OUTPUT_NAME)(deformation)
    weights = keras.layers.Conv3D(1, 1, activation="softplus", name=WEIGHTS_OUTPUT_NAME, kernel_regularizer=regularizer)(structured)
    outputs = dict(base.output)
    outputs.update({STRUCTURE_OUTPUT_NAME: occupancy, SCALAR_OUTPUT_NAME: scalar, DEFORMATION_OUTPUT_NAME: deformation, WEIGHTS_OUTPUT_NAME: weights})
    return keras.Model(dict(base.input), outputs, name="rgbd_trellis_flexicubes_mesh")


def predict_flexicubes_mesh(model, rgb, depth, resolution, extent, raw=None):
    """Decodes dense scalar/deformation grids and extracts a triangle mesh."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise ImportError("Mesh previews require scikit-image") from error
    if raw is None:
        raw = model.predict(
            {"rgb": rgb[None], "depth": depth[None]}, verbose=0
        )
    scalar = np.asarray(raw[SCALAR_OUTPUT_NAME][0, ..., 0])
    deformation = np.asarray(raw[DEFORMATION_OUTPUT_NAME][0])
    weights = np.asarray(raw[WEIGHTS_OUTPUT_NAME][0, ..., 0])
    if not float(scalar.min()) <= 0.5 <= float(scalar.max()):
        raise ValueError(f"No scalar-grid surface crossing: [{scalar.min():.4f}, {scalar.max():.4f}]")
    voxel_size = 2.0 * extent / (resolution - 1)
    vertices, faces, normals, _ = marching_cubes(scalar, level=0.5, spacing=(voxel_size,) * 3)
    grid_indices = np.clip(np.rint(vertices / voxel_size).astype(np.int32), 0, resolution - 1)
    sampled_deformation = deformation[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    sampled_weights = weights[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], None]
    vertices += sampled_deformation * voxel_size * np.clip(sampled_weights, 0.5, 2.0)
    vertices -= extent
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/rgbd_trellis_simple_shapes"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--mesh-frame", choices=("world", "canonical"),
                        default="canonical")
    parser.add_argument("--structure-resolution", type=int, default=16)
    parser.add_argument("--structure-channels", type=int, default=32)
    parser.add_argument("--structure-surface-band", type=float, default=None)
    parser.add_argument("--structure-loss-weight", type=float, default=0.25)
    parser.add_argument("--mesh-extent", type=float, default=1.25)
    parser.add_argument("--scalar-loss-weight", type=float, default=1.0)
    parser.add_argument("--deformation-loss-weight", type=float, default=0.25)
    parser.add_argument("--weights-loss-weight", type=float, default=0.01)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--mesh-cache", type=Path, default=None)
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
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--preview-meshes", type=int, default=3)
    parser.add_argument("--mesh-resolution", type=int, default=48)
    parser.add_argument("--mesh-query-chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.structure_resolution, args.structure_channels,
        args.structure_loss_weight, args.mesh_extent,
        args.scalar_loss_weight, args.deformation_loss_weight,
        args.weights_loss_weight, args.distance_chunk_size,
        args.batch_size, args.epochs, args.max_depth, args.checkpoint_every,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes, ranges, and loss weights must be positive")
    if args.structure_resolution < 4:
        raise ValueError("structure resolution must be at least 4")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)

    records = load_records(args.dataset)
    shape_names = tuple(sorted({
        record["shape"]["type"] for record in records
    }))
    training_records, validation_records, test_records = split_records(
        records, args.seed, args.validation_split
    )
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    export_test_meshes(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", training_records, validation_records,
        test_records, args.seed, args.validation_split,
    )
    normalizer = GenericTargetNormalizer.fit(training_records, shape_names)
    normalizer.save(args.output / "normalization.json")
    voxel_size = 2.0 * args.mesh_extent / (args.structure_resolution - 1)
    surface_band = args.structure_surface_band
    if surface_band is None:
        surface_band = np.sqrt(3.0) * voxel_size / 2.0
    preprocessing = {
        "method": "TRELLIS-inspired structured 3D latent",
        "not_official_trellis": True,
        "depth_unit": "metres", "max_depth": args.max_depth,
        "shape_names": shape_names, "mesh_frame": args.mesh_frame,
        "structure_resolution": args.structure_resolution,
        "structure_channels": args.structure_channels,
        "structure_surface_band": surface_band,
        "mesh_extent": args.mesh_extent,
        "mesh_target_frame": args.mesh_frame,
        "stored_dataset_mesh_frame": "world",
        "mesh_spatial_features": (
            "final RGB/depth ResNet maps lifted to a 3D grid"
        ),
        "global_dense_grid_projection": False,
        "geometry_decoder": "structured scalar, deformation, and interpolation grids",
        "pointwise_sdf_decoder": False,
        "mesh_extraction": "scalar-grid isosurface with learned vertex deformation",
    }
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump(preprocessing, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    options = {
        "mesh_frame": args.mesh_frame,
        "distance_chunk_size": args.distance_chunk_size,
        "mesh_cache": args.mesh_cache or args.output / "mesh_grid_cache",
        "mesh_extent": args.mesh_extent,
        "structure_resolution": args.structure_resolution,
        "structure_surface_band": surface_band,
    }
    training = RGBDFlexiCubesDataset(
        training_records, *common, shuffle=True, seed=args.seed, **options
    )
    validation = RGBDFlexiCubesDataset(
        validation_records, *common, shuffle=False, seed=args.seed, **options
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_trellis_model(
        image_shape, len(shape_names), args.structure_resolution,
        args.structure_channels, args.mesh_extent, args.l2_regularization,
    )
    structure_loss = keras.losses.BinaryCrossentropy(name="surface_structure_bce")
    scalar_loss = keras.losses.BinaryCrossentropy(name="inside_outside_bce")
    deformation_loss = keras.losses.Huber(delta=0.1, name="vertex_deformation_huber")
    weights_loss = keras.losses.MeanSquaredError(name="interpolation_weights_mse")
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={
            "object_orientation_6d": generic_rotation_loss,
            STRUCTURE_OUTPUT_NAME: structure_loss,
            SCALAR_OUTPUT_NAME: scalar_loss,
            DEFORMATION_OUTPUT_NAME: deformation_loss,
            WEIGHTS_OUTPUT_NAME: weights_loss,
        },
        extra_metrics={
            "object_orientation_6d": [keras.metrics.MeanMetricWrapper(generic_rotation_loss, name="loss"), generic_geodesic_angle_degrees],
            STRUCTURE_OUTPUT_NAME: [keras.metrics.MeanMetricWrapper(structure_loss, name="loss"), keras.metrics.BinaryIoU(target_class_ids=(1,), threshold=0.5, name="surface_iou")],
            SCALAR_OUTPUT_NAME: [keras.metrics.MeanMetricWrapper(scalar_loss, name="loss"), keras.metrics.BinaryIoU(target_class_ids=(1,), threshold=0.5, name="volume_iou")],
            DEFORMATION_OUTPUT_NAME: [keras.metrics.MeanAbsoluteError(name="grid_deformation_mae")],
            WEIGHTS_OUTPUT_NAME: [keras.metrics.MeanAbsoluteError(name="interpolation_weight_mae")],
        },
        extra_loss_weights={
            STRUCTURE_OUTPUT_NAME: args.structure_loss_weight,
            SCALAR_OUTPUT_NAME: args.scalar_loss_weight,
            DEFORMATION_OUTPUT_NAME: args.deformation_loss_weight,
            WEIGHTS_OUTPUT_NAME: args.weights_loss_weight,
        },
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png",
            OUTPUT_NAMES + (STRUCTURE_OUTPUT_NAME, SCALAR_OUTPUT_NAME, DEFORMATION_OUTPUT_NAME, WEIGHTS_OUTPUT_NAME),
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
            monitor="val_loss", mode="min", factor=args.lr_reduction_factor,
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
        f"Training structured RGB-D mesh model for {shape_names} on {gpu}: "
        f"{len(training_records)} train, {len(validation_records)} validation, "
        f"{len(test_records)} test"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")

    preview_output = args.output / "structured_preview_meshes"
    preview_output.mkdir(parents=True, exist_ok=True)
    for record in test_records[:args.preview_meshes]:
        sample_id = Path(record["_metadata_path"]).stem
        rgb, depth = load_rgb_depth(record, args.max_depth)
        mesh = predict_flexicubes_mesh(
            model, rgb, depth, args.structure_resolution, args.mesh_extent
        )
        mesh.export(preview_output / f"{sample_id}.ply")
        print(f"Saved structured-latent mesh {sample_id}.ply")


if __name__ == "__main__":
    main()
