"""Train the physical RGB-D ResNet-18 with an additional mesh-point head.

Ground-truth PLY surfaces are sampled deterministically and transformed into
the camera frame. A fixed-size point-set head uses symmetric Chamfer distance,
so vertex ordering and primitive topology do not need to match.

Example:
    KERAS_BACKEND=jax python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/resnet18_rgbd_mesh_validation
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import shutil
from pathlib import Path

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


MESH_OUTPUT_NAME = "mesh_points"


def world_to_camera_matrix(camera):
    """Builds the PAZ/OpenGL-style world-to-camera matrix in NumPy."""
    position = np.asarray(camera["position_world_xyz"], dtype=np.float64)
    target = np.asarray(camera["target_world_xyz"], dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    up = np.cross(left, forward)
    rotation = np.stack([left, up, -forward])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ position
    return matrix


def sample_mesh_surface(mesh_path, num_points, seed):
    """Samples an area-weighted PLY surface deterministically."""
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0.0:
        raise ValueError(f"Mesh has no valid surface area: {mesh_path}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), num_points, p=areas / areas.sum())
    chosen = triangles[selected]
    first = np.sqrt(rng.random((num_points, 1)))
    second = rng.random((num_points, 1))
    points = ((1.0 - first) * chosen[:, 0]
              + first * (1.0 - second) * chosen[:, 1]
              + first * second * chosen[:, 2])
    return points.astype(np.float32)


def load_camera_mesh_points(record, num_points):
    """Loads surface points and transforms world coordinates to camera."""
    stem = Path(record["_metadata_path"]).stem
    mesh_path = Path(record["_root"]) / "meshes" / f"{stem}.ply"
    points = sample_mesh_surface(mesh_path, num_points, int(stem))
    transform = world_to_camera_matrix(record["camera"])
    homogeneous = np.concatenate(
        [points, np.ones((num_points, 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def fit_mesh_statistics(records, num_points):
    """Computes coordinate statistics from training meshes only."""
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    for record in records:
        points = load_camera_mesh_points(record, num_points)
        count += len(points)
        total += points.sum(axis=0)
        total_squared += np.square(points).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    standard_deviation = np.sqrt(variance)
    standard_deviation = np.where(
        standard_deviation < 1e-6, 1.0, standard_deviation
    )
    return {
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
    }


class RGBDMeshDataset(RGBDDataset):
    """Adds standardized camera-frame surface points to RGB-D batches."""

    def __init__(self, *args, num_mesh_points, mesh_statistics, **kwargs):
        self.num_mesh_points = num_mesh_points
        self.mesh_mean = np.asarray(mesh_statistics["mean"], np.float32)
        self.mesh_std = np.asarray(
            mesh_statistics["standard_deviation"], np.float32
        )
        super().__init__(*args, **kwargs)

    def __getitem__(self, batch_index):
        inputs, targets = super().__getitem__(batch_index)
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        points = np.stack([
            load_camera_mesh_points(self.records[index], self.num_mesh_points)
            for index in indices
        ])
        targets[MESH_OUTPUT_NAME] = (
            points - self.mesh_mean
        ) / self.mesh_std
        return inputs, targets


@keras.saving.register_keras_serializable(package="paz")
class PhysicalChamferDistance(keras.losses.Loss):
    """Symmetric squared Chamfer distance after restoring metres."""

    def __init__(self, standard_deviation, name="physical_chamfer_m2",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(x) for x in standard_deviation)

    def call(self, target, prediction):
        scale = keras.ops.convert_to_tensor(self.standard_deviation)
        differences = (
            prediction[:, :, None, :] - target[:, None, :, :]
        ) * scale
        squared = keras.ops.sum(keras.ops.square(differences), axis=-1)
        prediction_to_target = keras.ops.min(squared, axis=2)
        target_to_prediction = keras.ops.min(squared, axis=1)
        return 0.5 * (
            keras.ops.mean(prediction_to_target, axis=1)
            + keras.ops.mean(target_to_prediction, axis=1)
        )

    def get_config(self):
        config = super().get_config()
        config["standard_deviation"] = self.standard_deviation
        return config


def build_mesh_model(image_shape=(256, 256), num_points=256,
                     l2_regularization=1e-4):
    """Extends the physical model with a fused-feature point-cloud head."""
    base = build_model(image_shape, l2_regularization)
    regularizer = keras.regularizers.L2(l2_regularization)
    fused = base.get_layer("fused_512d_feature").output
    mesh_features = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=regularizer,
        name="mesh_mlp",
    )(fused)
    mesh_features = keras.layers.Dropout(0.2, name="mesh_dropout")(
        mesh_features
    )
    flat_points = keras.layers.Dense(
        num_points * 3, kernel_regularizer=regularizer,
        name="mesh_point_coordinates",
    )(mesh_features)
    mesh_points = keras.layers.Reshape(
        (num_points, 3), name=MESH_OUTPUT_NAME
    )(flat_points)
    outputs = dict(base.output)
    outputs[MESH_OUTPUT_NAME] = mesh_points
    return keras.Model(
        base.input, outputs, name="physical_parameter_rgbd_resnet18_mesh"
    )


def export_test_meshes(records, destination):
    """Copies held-out PLY meshes beside exported RGB-D test data."""
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
                        default=Path("experiments/resnet18_rgbd_mesh"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--num-mesh-points", type=int, default=256)
    parser.add_argument("--mesh-loss-weight", type=float, default=1.0)
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
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=1e-3)
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.num_mesh_points, args.batch_size, args.epochs,
        args.checkpoint_every, args.max_depth, args.mesh_loss_weight,
        args.lr_reduction_patience, args.early_stopping_patience,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sizes, ranges, weights, and patience must be positive")
    if not 0.0 < args.lr_reduction_factor < 1.0:
        raise ValueError("LR reduction factor must be between zero and one")
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
    mesh_statistics = fit_mesh_statistics(
        train_records, args.num_mesh_points
    )
    normalizer.statistics[MESH_OUTPUT_NAME] = mesh_statistics
    normalizer.save(args.output / "normalization.json")
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "depth_unit": "metres",
            "max_depth": args.max_depth,
            "mesh_representation": "camera-frame surface point cloud",
            "num_mesh_points": args.num_mesh_points,
            "mesh_loss": "symmetric squared Chamfer distance in metres",
        }, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    mesh_options = {
        "num_mesh_points": args.num_mesh_points,
        "mesh_statistics": mesh_statistics,
    }
    training = RGBDMeshDataset(
        train_records, *common, shuffle=True, seed=args.seed, **mesh_options
    )
    validation = RGBDMeshDataset(
        validation_records, *common, shuffle=False, seed=args.seed,
        **mesh_options,
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_mesh_model(
        image_shape, args.num_mesh_points, args.l2_regularization
    )
    chamfer = PhysicalChamferDistance(
        mesh_statistics["standard_deviation"]
    )
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={MESH_OUTPUT_NAME: chamfer},
        extra_metrics={MESH_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(chamfer, name="loss")
        ]},
        extra_loss_weights={MESH_OUTPUT_NAME: args.mesh_loss_weight},
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png", OUTPUT_NAMES + (MESH_OUTPUT_NAME,)
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
        f"Training RGB-D mesh model on {gpu}: {len(train_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test; "
        f"{args.num_mesh_points} surface points per mesh"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
