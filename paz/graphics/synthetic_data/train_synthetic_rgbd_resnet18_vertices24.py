"""Train the physical RGB-D ResNet-18 with a 24x3 mesh-vertex head.

Cube, cylinder, and sphere each use a fixed canonical 24-vertex template and a
fixed triangle list. Ground-truth pose and scale transform the corresponding
template into the camera frame. Ordered physical MSE therefore preserves the
semantic vertex indices needed to select connectivity from the predicted shape.

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
    world_to_camera_matrix,
)


NUM_MESH_VERTICES = 24
VERTEX_OUTPUT_NAME = "mesh_vertices"


def triangulate_quads(num_quads):
    faces = []
    for quad in range(num_quads):
        start = 4 * quad
        faces.extend(((start, start + 1, start + 2),
                      (start, start + 2, start + 3)))
    return np.asarray(faces, dtype=np.int32)


def build_cube_template():
    """Builds six independent four-corner faces matching a unit PAZ cube."""
    quads = [
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],
        [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)],
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
        [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)],
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    ]
    return np.asarray(quads, np.float32).reshape(24, 3), triangulate_quads(6)


def build_cylinder_template():
    """Builds two 12-vertex rings and fixed side/cap triangles."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    bottom = np.stack([ring[:, 0], np.full(12, -1.0), ring[:, 1]], axis=-1)
    top = np.stack([ring[:, 0], np.full(12, 1.0), ring[:, 1]], axis=-1)
    vertices = np.concatenate([bottom, top]).astype(np.float32)
    faces = []
    for index in range(12):
        following = (index + 1) % 12
        faces.extend(((index, following, 12 + following),
                      (index, 12 + following, 12 + index)))
    for index in range(1, 11):
        faces.append((0, index + 1, index))
        faces.append((12, 12 + index, 12 + index + 1))
    return vertices, np.asarray(faces, dtype=np.int32)


def build_sphere_template():
    """Builds two 11-vertex latitude rings plus north/south poles."""
    angles = np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)
    latitude = 0.4
    radius = np.sqrt(1.0 - latitude ** 2)
    upper = np.stack([
        radius * np.cos(angles), np.full(11, latitude),
        radius * np.sin(angles),
    ], axis=-1)
    lower = np.stack([
        radius * np.cos(angles), np.full(11, -latitude),
        radius * np.sin(angles),
    ], axis=-1)
    vertices = np.concatenate([
        np.asarray([[0.0, 1.0, 0.0]]), upper, lower,
        np.asarray([[0.0, -1.0, 0.0]]),
    ]).astype(np.float32)
    faces = []
    for index in range(11):
        following = (index + 1) % 11
        upper_index, upper_next = 1 + index, 1 + following
        lower_index, lower_next = 12 + index, 12 + following
        faces.extend(((0, upper_index, upper_next),
                      (upper_index, lower_index, lower_next),
                      (upper_index, lower_next, upper_next),
                      (23, lower_next, lower_index)))
    return vertices, np.asarray(faces, dtype=np.int32)


def build_templates():
    builders = {
        "cube": build_cube_template,
        "cylinder": build_cylinder_template,
        "sphere": build_sphere_template,
    }
    templates = {}
    for name, builder in builders.items():
        vertices, faces = builder()
        if vertices.shape != (NUM_MESH_VERTICES, 3):
            raise ValueError(f"{name} template must contain 24 XYZ vertices")
        templates[name] = {"vertices": vertices, "faces": faces}
    return templates


MESH_TEMPLATES = build_templates()


def recover_object_world_rotation(record):
    """Recovers the generator's upright yaw rotation from saved camera axes."""
    camera_linear = world_to_camera_matrix(record["camera"])[:3, :3]
    camera_axis_x = np.asarray(
        record["object"]["orientation_camera_6d"]["vector_a"],
        dtype=np.float64,
    )
    world_axis_x = np.linalg.solve(camera_linear, camera_axis_x)
    world_axis_x /= np.linalg.norm(world_axis_x)
    world_axis_y = np.asarray([0.0, 1.0, 0.0])
    world_axis_z = np.cross(world_axis_x, world_axis_y)
    return np.column_stack([
        world_axis_x, world_axis_y, world_axis_z
    ])


def load_camera_vertices(record, num_vertices=NUM_MESH_VERTICES):
    """Transforms the shape's ordered canonical template into camera space."""
    if num_vertices != NUM_MESH_VERTICES:
        raise ValueError("fixed templates require exactly 24 vertices")
    shape = record["shape"]["type"]
    canonical = MESH_TEMPLATES[shape]["vertices"]
    scale = float(record["object"]["scale"])
    rotation = recover_object_world_rotation(record)
    translation = np.asarray([0.0, scale, 0.0])
    world_vertices = scale * (canonical @ rotation.T) + translation
    homogeneous = np.concatenate([
        world_vertices, np.ones((NUM_MESH_VERTICES, 1))
    ], axis=1)
    transform = world_to_camera_matrix(record["camera"])
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


@keras.saving.register_keras_serializable(package="paz")
class PhysicalVertexMSE(keras.losses.Loss):
    """Ordered per-vertex squared Euclidean error in physical metres."""

    def __init__(self, standard_deviation, name="physical_vertex_mse",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(x) for x in standard_deviation)

    def call(self, target, prediction):
        scale = keras.ops.convert_to_tensor(self.standard_deviation)
        physical_difference = (prediction - target) * scale
        squared_distance = keras.ops.sum(
            keras.ops.square(physical_difference), axis=-1
        )
        return keras.ops.mean(squared_distance, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["standard_deviation"] = self.standard_deviation
        return config


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
    template_payload = {
        shape: {
            "vertices": template["vertices"].tolist(),
            "faces": template["faces"].tolist(),
        }
        for shape, template in MESH_TEMPLATES.items()
    }
    with (args.output / "mesh_templates.json").open("w") as file:
        json.dump(template_payload, file, indent=2)
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "depth_unit": "metres",
            "max_depth": args.max_depth,
            "mesh_representation": (
                "shape-conditioned ordered 24-vertex templates"
            ),
            "num_mesh_vertices": NUM_MESH_VERTICES,
            "connectivity": "mesh_templates.json selected by shape output",
            "vertex_loss": "ordered physical vertex MSE in square metres",
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
    vertex_mse = PhysicalVertexMSE(
        vertex_statistics["standard_deviation"]
    )
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={VERTEX_OUTPUT_NAME: vertex_mse},
        extra_metrics={VERTEX_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(vertex_mse, name="loss")
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
