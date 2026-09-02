"""Train a TRELLIS-inspired model with canonical RGB-D voxel splatting.

Metric depth pixels are back-projected with the renderer camera convention,
mapped from camera coordinates into canonical object coordinates, and splatted
with their RGB values into a dense voxel grid. A 3D CNN consumes this grid for
mesh decoding, providing genuine pixel-to-voxel correspondence.

The generated datasets do not currently contain object masks. Consequently,
the canonical crop can include nearby floor points; the occupancy and RGB
channels allow the 3D CNN to learn to reject that plane.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.train_synthetic_rgbd_trellis_canonical_voxel \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --output experiments/rgbd_trellis_canonical_voxel
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
from pathlib import Path

import cv2
import jax
import keras
import numpy as np

from paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 import (
    OUTPUT_NAMES,
    PeriodicWeightsCheckpoint,
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
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    GenericTargetNormalizer,
    export_test_meshes,
    generic_geodesic_angle_degrees,
    generic_rotation_loss,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_voxel_sdf import (
    camera_ray_directions,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_trellis import (
    DEFORMATION_OUTPUT_NAME,
    SCALAR_OUTPUT_NAME,
    STRUCTURE_OUTPUT_NAME,
    WEIGHTS_OUTPUT_NAME,
    RGBDFlexiCubesDataset,
    RepeatGlobalGrid,
)


def rotation_6d_to_matrix(values, epsilon=1e-8):
    """Maps the saved two-axis representation to an SO(3) matrix."""
    values = np.asarray(values, dtype=np.float32)
    first, second = values[:3], values[3:]
    axis_x = first / max(float(np.linalg.norm(first)), epsilon)
    second = second - np.dot(axis_x, second) * axis_x
    axis_y = second / max(float(np.linalg.norm(second)), epsilon)
    axis_z = np.cross(axis_x, axis_y)
    return np.stack([axis_x, axis_y, axis_z], axis=-1)


def camera_to_canonical_points(record, camera_points):
    """Inverts canonical-to-camera scale, rotation, and translation."""
    orientation = record["object"]["orientation_camera_6d"]
    values = orientation["vector_a"] + orientation["vector_b"]
    rotation = rotation_6d_to_matrix(values)
    translation = np.asarray(
        record["object"]["translation_camera_xyz"], dtype=np.float32
    )
    scale = max(float(record["object"]["scale"]), 1e-6)
    return ((camera_points - translation) @ rotation) / scale


def splat_canonical_rgbd(rgb, depth, record, resolution, extent, max_depth,
                         y_fov):
    """Back-projects RGB-D and averages RGB values in canonical voxels."""
    valid = (
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    )
    directions = camera_ray_directions(*depth.shape, y_fov)
    camera_points = directions[valid] * depth[valid, None]
    canonical = camera_to_canonical_points(record, camera_points)
    normalized = (canonical + extent) / (2.0 * extent)
    inside = np.all((normalized >= 0.0) & (normalized <= 1.0), axis=-1)
    normalized = normalized[inside]
    colors = rgb[valid][inside]
    indices = np.floor(normalized * (resolution - 1)).astype(np.int32)
    flat_indices = (
        indices[:, 0] * resolution * resolution
        + indices[:, 1] * resolution + indices[:, 2]
    )
    cells = resolution ** 3
    counts = np.bincount(flat_indices, minlength=cells).astype(np.float32)
    color_sum = np.stack([
        np.bincount(flat_indices, weights=colors[:, channel], minlength=cells)
        for channel in range(3)
    ], axis=-1).astype(np.float32)
    nonempty = counts > 0.0
    color_sum[nonempty] /= counts[nonempty, None]
    occupancy = nonempty.astype(np.float32)[:, None]
    voxel = np.concatenate([occupancy, color_sum], axis=-1)
    return voxel.reshape((resolution,) * 3 + (4,))


class CanonicalVoxelMeshDataset(RGBDFlexiCubesDataset):
    """Adds cached canonical RGB-D feature voxels to mesh supervision."""

    def __init__(self, *args, y_fov, voxel_cache, **kwargs):
        self.y_fov = float(y_fov)
        self.voxel_cache = Path(voxel_cache)
        self.voxel_cache.mkdir(parents=True, exist_ok=True)
        # Shape names are already stored by GenericTargetNormalizer.
        # Consume this legacy option instead of forwarding it to RGBDDataset.
        kwargs.pop("shape_names", None)
        super().__init__(*args, **kwargs)
        self.canonical_voxels = np.stack([
            self.load_or_create_voxel(record) for record in self.records
        ])

    def load_or_create_voxel(self, record):
        sample_id = Path(record["_metadata_path"]).stem
        path = self.voxel_cache / (
            f"{sample_id}_canonical_rgbd_r{self.structure_resolution}"
            f"_e{self.mesh_extent:g}_fov{self.y_fov:g}.npy"
        )
        if path.is_file():
            return np.load(path)
        root = Path(record["_root"])
        bgr = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(root / record["rgb"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(root / record["depth"]).astype(np.float32)
        voxel = splat_canonical_rgbd(
            rgb, depth, record, self.structure_resolution, self.mesh_extent,
            self.max_depth, self.y_fov,
        )
        np.save(path, voxel)
        return voxel

    def __getitem__(self, batch_index):
        inputs, targets = super().__getitem__(batch_index)
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        inputs["canonical_rgbd_voxel"] = self.canonical_voxels[indices]
        return inputs, targets


def build_canonical_voxel_model(image_shape, num_shapes, resolution,
                                channels, l2_regularization):
    """Builds physical heads plus a canonical spatial 3D mesh decoder."""
    base = build_model(image_shape, l2_regularization, num_shapes)
    regularizer = keras.regularizers.L2(l2_regularization)
    voxel = keras.Input(
        (resolution,) * 3 + (4,), name="canonical_rgbd_voxel"
    )
    spatial = voxel
    for index, width in enumerate((channels // 2, channels, channels), 1):
        spatial = keras.layers.Conv3D(
            max(width, 8), 3, padding="same", activation="relu",
            kernel_regularizer=regularizer,
            name=f"canonical_voxel_conv3d_{index}",
        )(spatial)
    fused = base.get_layer("fused_512d_feature").output
    context = keras.layers.Dense(
        channels, activation="relu", kernel_regularizer=regularizer,
        name="canonical_mesh_global_context",
    )(fused)
    context = RepeatGlobalGrid(
        resolution, name="repeat_canonical_mesh_context"
    )(context)
    structured = keras.layers.Concatenate(
        name="canonical_voxel_global_fusion"
    )([spatial, context])
    for index in range(2):
        residual = keras.layers.Conv3D(
            channels, 1, padding="same", kernel_regularizer=regularizer,
            name=f"canonical_mesh_residual_projection_{index + 1}",
        )(structured)
        decoded = keras.layers.Conv3D(
            channels, 3, padding="same", activation="relu",
            kernel_regularizer=regularizer,
            name=f"canonical_mesh_decoder_conv_{index + 1}",
        )(structured)
        structured = keras.layers.Add(
            name=f"canonical_mesh_decoder_residual_{index + 1}"
        )([decoded, residual])
    outputs = dict(base.output)
    outputs[STRUCTURE_OUTPUT_NAME] = keras.layers.Conv3D(
        1, 1, activation="sigmoid", name=STRUCTURE_OUTPUT_NAME
    )(structured)
    outputs[SCALAR_OUTPUT_NAME] = keras.layers.Conv3D(
        1, 1, activation="sigmoid", name=SCALAR_OUTPUT_NAME
    )(structured)
    raw_deformation = keras.layers.Conv3D(
        3, 1, activation="tanh", name="raw_canonical_deformation"
    )(structured)
    outputs[DEFORMATION_OUTPUT_NAME] = keras.layers.Rescaling(
        0.5, name=DEFORMATION_OUTPUT_NAME
    )(raw_deformation)
    outputs[WEIGHTS_OUTPUT_NAME] = keras.layers.Conv3D(
        1, 1, activation="softplus", name=WEIGHTS_OUTPUT_NAME
    )(structured)
    inputs = dict(base.input)
    inputs["canonical_rgbd_voxel"] = voxel
    return keras.Model(
        inputs, outputs, name="rgbd_trellis_canonical_voxel_mesh"
    )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "experiments/rgbd_trellis_canonical_voxel"
    ))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--structure-resolution", type=int, default=16)
    parser.add_argument("--structure-channels", type=int, default=32)
    parser.add_argument("--mesh-extent", type=float, default=1.25)
    parser.add_argument("--structure-surface-band", type=float, default=None)
    parser.add_argument("--structure-loss-weight", type=float, default=0.25)
    parser.add_argument("--scalar-loss-weight", type=float, default=1.0)
    parser.add_argument("--deformation-loss-weight", type=float, default=0.25)
    parser.add_argument("--weights-loss-weight", type=float, default=0.01)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--mesh-cache", type=Path, default=None)
    parser.add_argument("--voxel-cache", type=Path, default=None)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.structure_resolution, args.structure_channels, args.mesh_extent,
        args.distance_chunk_size, args.y_fov, args.max_depth,
        args.batch_size, args.epochs, args.checkpoint_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sizes and ranges must be positive")
    if args.structure_resolution < 4:
        raise ValueError("structure resolution must be at least 4")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)

    records = load_records(args.dataset)
    shape_names = tuple(sorted({r["shape"]["type"] for r in records}))
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
        "method": "TRELLIS-inspired canonical RGB-D voxel splatting",
        "mesh_target_frame": "canonical",
        "stored_dataset_mesh_frame": "world",
        "canonical_splat_uses_object_pose": True,
        "inference_requires_predicted_or_known_object_pose": True,
        "voxel_channels": ["occupancy", "color_r", "color_g", "color_b"],
        "structure_resolution": args.structure_resolution,
        "structure_channels": args.structure_channels,
        "mesh_extent": args.mesh_extent,
        "structure_surface_band": surface_band,
        "max_depth": args.max_depth,
        "y_fov_degrees": args.y_fov,
        "mesh_frame": "canonical",
    }
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump(preprocessing, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    options = {
        "mesh_frame": "canonical",
        "mesh_extent": args.mesh_extent,
        "structure_resolution": args.structure_resolution,
        "structure_surface_band": surface_band,
        "distance_chunk_size": args.distance_chunk_size,
        "mesh_cache": args.mesh_cache or args.output / "mesh_grid_cache",
        "voxel_cache": args.voxel_cache or args.output / "voxel_cache",
        "y_fov": args.y_fov,
    }
    training = CanonicalVoxelMeshDataset(
        training_records, *common, shuffle=True, seed=args.seed, **options
    )
    validation = CanonicalVoxelMeshDataset(
        validation_records, *common, shuffle=False, seed=args.seed, **options
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_canonical_voxel_model(
        image_shape, len(shape_names), args.structure_resolution,
        args.structure_channels, args.l2_regularization,
    )
    structure_loss = keras.losses.BinaryCrossentropy(
        name="surface_structure_bce"
    )
    scalar_loss = keras.losses.BinaryCrossentropy(
        name="inside_outside_bce"
    )
    deformation_loss = keras.losses.Huber(
        delta=0.1, name="vertex_deformation_huber"
    )
    weights_loss = keras.losses.MeanSquaredError(
        name="interpolation_weights_mse"
    )
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
            "object_orientation_6d": [
                keras.metrics.MeanMetricWrapper(
                    generic_rotation_loss, name="loss"
                ),
                generic_geodesic_angle_degrees,
            ],
            STRUCTURE_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(
                    structure_loss, name="loss"
                ),
                keras.metrics.BinaryIoU(
                    target_class_ids=(1,), threshold=0.5,
                    name="surface_iou",
                )
            ],
            SCALAR_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(
                    scalar_loss, name="loss"
                ),
                keras.metrics.BinaryIoU(
                    target_class_ids=(1,), threshold=0.5,
                    name="volume_iou",
                )
            ],
            DEFORMATION_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(
                    deformation_loss, name="loss"
                ),
                keras.metrics.MeanAbsoluteError(name="deformation_mae")
            ],
            WEIGHTS_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(
                    weights_loss, name="loss"
                ),
                keras.metrics.MeanAbsoluteError(name="weight_mae")
            ],
        },
        extra_loss_weights={
            STRUCTURE_OUTPUT_NAME: args.structure_loss_weight,
            SCALAR_OUTPUT_NAME: args.scalar_loss_weight,
            DEFORMATION_OUTPUT_NAME: args.deformation_loss_weight,
            WEIGHTS_OUTPUT_NAME: args.weights_loss_weight,
        },
    )
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png",
            OUTPUT_NAMES + (
                STRUCTURE_OUTPUT_NAME, SCALAR_OUTPUT_NAME,
                DEFORMATION_OUTPUT_NAME, WEIGHTS_OUTPUT_NAME,
            ),
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
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, mode="min", verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, start_from_epoch=20,
            restore_best_weights=True, mode="min", verbose=1,
        ),
    ]
    model.summary()
    print(
        f"Training canonical-voxel mesh model on {gpu}: "
        f"{len(training_records)} train, {len(validation_records)} "
        f"validation, {len(test_records)} test"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
