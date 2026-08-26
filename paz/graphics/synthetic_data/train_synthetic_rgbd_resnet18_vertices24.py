"""Train the physical RGB-D ResNet-18 with a 24x3 mesh-vertex head.

Native cube, cylinder, and sphere meshes have incompatible topology. Each PLY
is therefore represented by 24 deterministic, well-spaced surface vertices:
area-weighted dense sampling followed by farthest-point sampling. Symmetric
Chamfer distance supervises the unordered vertices without false index-wise
correspondence between shapes.

Example:
    KERAS_BACKEND=jax python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_vertices24 \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --output experiments/resnet18_rgbd_vertices24
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
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh import (
    PhysicalChamferDistance,
    sample_mesh_surface,
    world_to_camera_matrix,
)


NUM_MESH_VERTICES = 24
VERTEX_OUTPUT_NAME = "mesh_vertices"


def farthest_point_sample(points, num_vertices):
    """Selects a deterministic, spatially well-distributed point subset."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < num_vertices:
        raise ValueError("candidate point count is smaller than target count")
    selected = np.empty(num_vertices, dtype=np.int64)
    centroid = points.mean(axis=0)
    selected[0] = int(np.argmax(np.sum(np.square(points - centroid), axis=1)))
    minimum_squared_distance = np.sum(
        np.square(points - points[selected[0]]), axis=1
    )
    for index in range(1, num_vertices):
        selected[index] = int(np.argmax(minimum_squared_distance))
        squared_distance = np.sum(
            np.square(points - points[selected[index]]), axis=1
        )
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, squared_distance
        )
    return points[selected]


def load_camera_vertices(record, num_vertices=NUM_MESH_VERTICES,
                         candidate_multiplier=64):
    """Builds fixed-size, well-spaced camera-frame vertices from one PLY."""
    stem = Path(record["_metadata_path"]).stem
    mesh_path = Path(record["_root"]) / "meshes" / f"{stem}.ply"
    candidate_count = max(num_vertices * candidate_multiplier, num_vertices)
    candidates = sample_mesh_surface(mesh_path, candidate_count, int(stem))
    world_vertices = farthest_point_sample(candidates, num_vertices)
    homogeneous = np.concatenate([
        world_vertices, np.ones((num_vertices, 1), dtype=np.float32)
    ], axis=1)
    transform = world_to_camera_matrix(record["camera"])
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def fit_vertex_statistics(records, num_vertices=NUM_MESH_VERTICES):
    """Fits coordinate normalization using training vertices only."""
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    for record in records:
        vertices = load_camera_vertices(record, num_vertices)
        count += len(vertices)
        total += vertices.sum(axis=0)
        total_squared += np.square(vertices).sum(axis=0)
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


class RGBDVertexDataset(RGBDDataset):
    """Adds normalized 24x3 camera-frame vertex targets to RGB-D batches."""

    def __init__(self, *args, vertex_statistics, **kwargs):
        self.vertex_mean = np.asarray(vertex_statistics["mean"], np.float32)
        self.vertex_std = np.asarray(
            vertex_statistics["standard_deviation"], np.float32
        )
        super().__init__(*args, **kwargs)
        self.vertex_targets = np.stack([
            load_camera_vertices(record) for record in self.records
        ])

    def __getitem__(self, batch_index):
        inputs, targets = super().__getitem__(batch_index)
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        vertices = self.vertex_targets[indices]
        targets[VERTEX_OUTPUT_NAME] = (
            vertices - self.vertex_mean
        ) / self.vertex_std
        return inputs, targets


def build_vertex_model(image_shape=(256, 256), l2_regularization=1e-4):
    """Adds a 24x3 vertex MLP to the dual-stream physical model."""
    base = build_model(image_shape, l2_regularization)
    regularizer = keras.regularizers.L2(l2_regularization)
    fused = base.get_layer("fused_512d_feature").output
    features = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=regularizer,
        name="mesh_vertex_mlp",
    )(fused)
    features = keras.layers.Dropout(0.2, name="mesh_vertex_dropout")(
        features
    )
    flat_vertices = keras.layers.Dense(
        NUM_MESH_VERTICES * 3, kernel_regularizer=regularizer,
        name="mesh_vertex_coordinates",
    )(features)
    vertices = keras.layers.Reshape(
        (NUM_MESH_VERTICES, 3), name=VERTEX_OUTPUT_NAME
    )(flat_vertices)
    outputs = dict(base.output)
    outputs[VERTEX_OUTPUT_NAME] = vertices
    return keras.Model(
        base.input, outputs, name="physical_parameter_rgbd_resnet18_vertices24"
    )


def export_test_meshes(records, destination):
    """Copies the original held-out PLY meshes for later evaluation."""
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
                        default=Path("experiments/resnet18_rgbd_vertices24"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--vertex-loss-weight", type=float, default=1.0)
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
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.vertex_loss_weight, args.batch_size, args.epochs,
        args.max_depth, args.checkpoint_every, args.lr_reduction_patience,
        args.early_stopping_patience,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("weights, sizes, ranges, and patience must be positive")
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
    vertex_statistics = fit_vertex_statistics(train_records)
    normalizer.statistics[VERTEX_OUTPUT_NAME] = vertex_statistics
    normalizer.save(args.output / "normalization.json")
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "depth_unit": "metres",
            "max_depth": args.max_depth,
            "mesh_representation": "24 camera-frame surface vertices",
            "num_mesh_vertices": NUM_MESH_VERTICES,
            "target_sampling": (
                "area-weighted dense sampling plus farthest-point sampling"
            ),
            "vertex_loss": "symmetric squared Chamfer distance in metres",
        }, file, indent=2)

    common = (normalizer, args.batch_size, args.max_depth)
    training = RGBDVertexDataset(
        train_records, *common, vertex_statistics=vertex_statistics,
        shuffle=True, seed=args.seed,
    )
    validation = RGBDVertexDataset(
        validation_records, *common, vertex_statistics=vertex_statistics,
        shuffle=False, seed=args.seed,
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_vertex_model(image_shape, args.l2_regularization)
    chamfer = PhysicalChamferDistance(
        vertex_statistics["standard_deviation"], name="vertex_chamfer_m2"
    )
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={VERTEX_OUTPUT_NAME: chamfer},
        extra_metrics={VERTEX_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(chamfer, name="loss")
        ]},
        extra_loss_weights={VERTEX_OUTPUT_NAME: args.vertex_loss_weight},
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png", OUTPUT_NAMES + (VERTEX_OUTPUT_NAME,)
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
        f"Training 24x3 vertex model on {gpu}: {len(train_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
