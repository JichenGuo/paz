"""Train a TRELLIS-inspired structured-latent RGB-D mesh reconstructor.

This is a compact PAZ adaptation, not Microsoft's pretrained TRELLIS model.
It predicts a surface occupancy structure and feature grid from RGB-D, samples
local structured features at XYZ queries, decodes an SDF, and extracts its zero
level set as a triangle mesh.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_trellis \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/rgbd_trellis_simple_shapes
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
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
    CanonicalSDFMAE,
    GenericTargetNormalizer,
    RGBDSDFDataset,
    RepeatGeometryLatent,
    SDF_OUTPUT_NAME,
    export_test_meshes,
    generic_geodesic_angle_degrees,
    generic_rotation_loss,
    load_canonical_mesh,
    load_rgb_depth,
    predict_sdf_mesh,
)


STRUCTURE_OUTPUT_NAME = "structure_occupancy"


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


class RGBDStructuredSDFDataset(RGBDSDFDataset):
    """Adds a cached sparse surface structure target to SDF supervision."""

    def __init__(self, *args, structure_resolution, structure_surface_band,
                 **kwargs):
        self.structure_resolution = structure_resolution
        self.structure_surface_band = structure_surface_band
        super().__init__(*args, **kwargs)
        self.structure_targets = np.stack([
            self.load_or_create_structure(record) for record in self.records
        ])

    def load_or_create_structure(self, record):
        sample_id = Path(record["_metadata_path"]).stem
        key = (
            f"{sample_id}_structure_r{self.structure_resolution}"
            f"_e{self.sdf_extent:g}_b{self.structure_surface_band:g}"
            f"_{self.mesh_frame}.npy"
        )
        path = self.cache_dir / key
        if path.is_file():
            return np.load(path)
        target = build_structure_target(
            record, self.structure_resolution, self.sdf_extent,
            self.structure_surface_band, self.mesh_frame,
            self.distance_chunk_size,
        )
        np.save(path, target)
        return target

    def __getitem__(self, batch_index):
        inputs, targets = super().__getitem__(batch_index)
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        targets[STRUCTURE_OUTPUT_NAME] = self.structure_targets[indices]
        return inputs, targets


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


def build_trellis_model(image_shape, num_shapes, structure_resolution,
                        structure_channels, sdf_extent, l2_regularization):
    """Builds structured 3D latent, occupancy, SDF, and physical heads."""
    base = build_model(image_shape, l2_regularization, num_shapes)
    regularizer = keras.regularizers.L2(l2_regularization)
    fused = base.get_layer("fused_512d_feature").output
    latent_units = structure_resolution ** 3 * structure_channels
    structured = keras.layers.Dense(
        latent_units, activation="relu", kernel_regularizer=regularizer,
        name="structured_latent_projection",
    )(fused)
    structured = keras.layers.Reshape(
        (structure_resolution,) * 3 + (structure_channels,),
        name="structured_latent_grid",
    )(structured)
    structured = keras.layers.Conv3D(
        structure_channels, 3, padding="same", activation="relu",
        kernel_regularizer=regularizer, name="structured_latent_refinement",
    )(structured)
    occupancy = keras.layers.Conv3D(
        1, 1, activation="sigmoid", name=STRUCTURE_OUTPUT_NAME,
        kernel_regularizer=regularizer,
    )(structured)
    gated = keras.layers.Multiply(name="sparse_structured_latent")([
        structured, occupancy
    ])
    query_points = keras.Input((None, 3), name="sdf_query_points")
    local = TrilinearStructuredSampler(
        structure_resolution, sdf_extent, name="structured_feature_sampler"
    )([gated, query_points])
    global_latent = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=regularizer,
        name="global_geometry_condition",
    )(fused)
    global_repeated = RepeatGeometryLatent(name="repeat_global_geometry")(
        [global_latent, query_points]
    )
    decoded = keras.layers.Concatenate(name="structured_sdf_conditioning")([
        local, global_repeated, query_points
    ])
    for index, width in enumerate((256, 128, 64), start=1):
        decoded = keras.layers.Dense(
            width, activation="relu", kernel_regularizer=regularizer,
            name=f"structured_sdf_dense{index}",
        )(decoded)
    sdf = keras.layers.Dense(
        1, activation="tanh", kernel_regularizer=regularizer,
        name=SDF_OUTPUT_NAME,
    )(decoded)
    inputs = dict(base.input)
    inputs["sdf_query_points"] = query_points
    outputs = dict(base.output)
    outputs[STRUCTURE_OUTPUT_NAME] = occupancy
    outputs[SDF_OUTPUT_NAME] = sdf
    return keras.Model(inputs, outputs, name="rgbd_trellis_structured_sdf")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/rgbd_trellis_simple_shapes"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--mesh-frame", choices=("world", "canonical"),
                        default="world")
    parser.add_argument("--structure-resolution", type=int, default=8)
    parser.add_argument("--structure-channels", type=int, default=32)
    parser.add_argument("--structure-surface-band", type=float, default=None)
    parser.add_argument("--structure-loss-weight", type=float, default=0.25)
    parser.add_argument("--num-sdf-queries", type=int, default=2048)
    parser.add_argument("--sdf-extent", type=float, default=1.25)
    parser.add_argument("--sdf-truncation", type=float, default=0.2)
    parser.add_argument("--sdf-loss-weight", type=float, default=1.0)
    parser.add_argument("--distance-chunk-size", type=int, default=32)
    parser.add_argument("--sdf-cache", type=Path, default=None)
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
        args.structure_loss_weight, args.num_sdf_queries, args.sdf_extent,
        args.sdf_truncation, args.sdf_loss_weight, args.distance_chunk_size,
        args.batch_size, args.epochs, args.max_depth, args.checkpoint_every,
        args.mesh_resolution, args.mesh_query_chunk,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes, ranges, and loss weights must be positive")
    if args.mesh_resolution < 4 or args.structure_resolution < 2:
        raise ValueError("mesh resolution >= 4 and structure resolution >= 2")
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
    voxel_size = 2.0 * args.sdf_extent / (args.structure_resolution - 1)
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
        "num_sdf_queries": args.num_sdf_queries,
        "sdf_extent": args.sdf_extent,
        "sdf_truncation": args.sdf_truncation,
        "mesh_extraction": "marching tetrahedra at SDF zero level",
    }
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump(preprocessing, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    options = {
        "num_sdf_queries": args.num_sdf_queries,
        "sdf_extent": args.sdf_extent,
        "sdf_truncation": args.sdf_truncation,
        "shape_names": shape_names,
        "mesh_frame": args.mesh_frame,
        "distance_chunk_size": args.distance_chunk_size,
        "cache_dir": args.sdf_cache or args.output / "sdf_cache",
        "structure_resolution": args.structure_resolution,
        "structure_surface_band": surface_band,
    }
    training = RGBDStructuredSDFDataset(
        training_records, *common, shuffle=True, seed=args.seed, **options
    )
    validation = RGBDStructuredSDFDataset(
        validation_records, *common, shuffle=False, seed=args.seed, **options
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_trellis_model(
        image_shape, len(shape_names), args.structure_resolution,
        args.structure_channels, args.sdf_extent, args.l2_regularization,
    )
    sdf_loss = keras.losses.Huber(delta=0.1, name="truncated_sdf_huber")
    structure_loss = keras.losses.BinaryCrossentropy(
        name="surface_structure_bce"
    )
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={
            "object_orientation_6d": generic_rotation_loss,
            STRUCTURE_OUTPUT_NAME: structure_loss,
            SDF_OUTPUT_NAME: sdf_loss,
        },
        extra_metrics={
            "object_orientation_6d": [
                keras.metrics.MeanMetricWrapper(
                    generic_rotation_loss, name="loss"
                ),
                generic_geodesic_angle_degrees,
            ],
            STRUCTURE_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(structure_loss, name="loss"),
                keras.metrics.BinaryIoU(
                    target_class_ids=(1,), threshold=0.5,
                    name="surface_iou",
                ),
            ],
            SDF_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(sdf_loss, name="loss"),
                CanonicalSDFMAE(args.sdf_truncation),
            ],
        },
        extra_loss_weights={
            STRUCTURE_OUTPUT_NAME: args.structure_loss_weight,
            SDF_OUTPUT_NAME: args.sdf_loss_weight,
        },
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png",
            OUTPUT_NAMES + (STRUCTURE_OUTPUT_NAME, SDF_OUTPUT_NAME),
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
        mesh = predict_sdf_mesh(
            model, rgb, depth, args.mesh_resolution, args.sdf_extent,
            args.mesh_query_chunk,
        )
        mesh.export(preview_output / f"{sample_id}.ply")
        print(f"Saved structured-latent mesh {sample_id}.ply")


if __name__ == "__main__":
    main()
