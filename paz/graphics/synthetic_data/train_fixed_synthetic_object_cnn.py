"""Train a compact object-only CNN on fixed-camera/fixed-light RGB-D data.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.train_fixed_synthetic_object_cnn \
        --dataset datasets/synthetic_rgbd_fixed_1000 \
        --output experiments/fixed_camera_light_cnn
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import math
from pathlib import Path

import cv2
import jax
import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    CoordinateChannels,
    PhysicalEuclideanDistance,
    PhysicalMAE,
    PhysicalVectorMSE,
    SHAPE_NAMES,
    SpatialSoftArgmax2D,
    export_test_split,
    extract_targets,
    load_records,
    symmetry_geodesic_angle_degrees,
    symmetry_rotation_loss,
)


OBJECT_REGRESSION_NAMES = ("object_translation", "object_scale", "material")
OBJECT_OUTPUT_NAMES = (
    "object_translation", "object_orientation_6d", "object_scale", "shape",
    "material",
)


class ObjectNormalizer:
    """Standardizes object regression labels using training records only."""

    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def fit(cls, records):
        extracted = [extract_targets(record) for record in records]
        statistics = {}
        for name in OBJECT_REGRESSION_NAMES:
            values = np.stack([targets[name] for targets in extracted])
            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0)
            standard_deviation = np.where(
                standard_deviation < 1e-6, 1.0, standard_deviation
            )
            statistics[name] = {
                "mean": mean.tolist(),
                "standard_deviation": standard_deviation.tolist(),
            }
        return cls(statistics)

    def normalize(self, targets):
        targets = {
            name: np.asarray(targets[name], np.float32)
            for name in OBJECT_OUTPUT_NAMES
        }
        for name in OBJECT_REGRESSION_NAMES:
            statistics = self.statistics[name]
            mean = np.asarray(statistics["mean"], np.float32)
            std = np.asarray(
                statistics["standard_deviation"], np.float32
            )
            targets[name] = (targets[name] - mean) / std
        return targets

    def save(self, path):
        payload = {"shape_names": SHAPE_NAMES, "targets": self.statistics}
        with path.open("w") as file:
            json.dump(payload, file, indent=2)


class ObjectRGBDDataset(keras.utils.PyDataset):
    """Loads RGB-D inputs and object-only targets."""

    def __init__(self, records, normalizer, batch_size, max_depth,
                 shuffle=False, seed=0):
        super().__init__()
        self.records = list(records)
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        inputs, targets = [], []
        for index in indices:
            record = self.records[index]
            inputs.append(self.load_rgbd(record))
            targets.append(
                self.normalizer.normalize(extract_targets(record))
            )
        stacked_targets = {
            name: np.stack([target[name] for target in targets])
            for name in OBJECT_OUTPUT_NAMES
        }
        return np.stack(inputs), stacked_targets

    def load_rgbd(self, record):
        root = Path(record["_root"])
        rgb_path = root / record["rgb"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        depth = np.load(root / record["depth"]).astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth shapes must match")
        depth = np.clip(depth / self.max_depth, 0.0, 1.0)
        return np.concatenate([rgb, depth[..., None]], axis=-1)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class ObjectHistoryPlot(keras.callbacks.Callback):
    """Saves object-only train/validation loss curves after every epoch."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            self.history.setdefault(name, []).append(value)
        names = ("loss",) + tuple(
            f"{name}_loss" for name in OBJECT_OUTPUT_NAMES
        )
        figure, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
        epochs = np.arange(1, epoch + 2)
        for axis, name in zip(axes.flat, names):
            axis.plot(epochs, self.history.get(name, []), label="Training")
            validation = self.history.get(f"val_{name}")
            if validation is not None:
                axis.plot(epochs, validation, label="Validation")
            axis.set_title(name.replace("_", " ").title())
            axis.grid(alpha=0.3)
            axis.legend()
        figure.tight_layout()
        figure.savefig(self.path, dpi=150)
        plt.close(figure)


def convolution_block(inputs, filters, regularizer, stride=2):
    x = keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_regularizer=regularizer,
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def build_model(input_shape, l2_regularization=1e-4):
    """Builds a compact spatial pose and global object-property CNN."""
    inputs = keras.Input(input_shape, name="rgbd")
    regularizer = keras.regularizers.L2(l2_regularization)
    # Coordinates enter before downsampling so small image displacements are
    # available to every pose feature level.
    x = CoordinateChannels(name="input_coordinates")(inputs)
    x = convolution_block(x, 16, regularizer)
    x = convolution_block(x, 32, regularizer)
    pose_map = convolution_block(x, 64, regularizer)

    heatmaps = keras.layers.Conv2D(
        16, 1, kernel_regularizer=regularizer, name="object_heatmaps"
    )(pose_map)
    coordinates = SpatialSoftArgmax2D(
        name="object_spatial_soft_argmax"
    )(heatmaps)
    pose_context = keras.layers.GlobalAveragePooling2D()(pose_map)
    pose = keras.layers.Concatenate()([coordinates, pose_context])
    pose = keras.layers.Dense(
        64, activation="relu", kernel_regularizer=regularizer
    )(pose)
    pose = keras.layers.Dropout(0.2)(pose)

    appearance = convolution_block(pose_map, 96, regularizer)
    appearance = keras.layers.GlobalAveragePooling2D()(appearance)
    appearance = keras.layers.Dense(
        64, activation="relu", kernel_regularizer=regularizer
    )(appearance)
    appearance = keras.layers.Dropout(0.2)(appearance)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation",
            kernel_regularizer=regularizer)(pose),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d",
            kernel_regularizer=regularizer)(pose),
        "object_scale": keras.layers.Dense(
            1, name="object_scale", kernel_regularizer=regularizer)(pose),
        "shape": keras.layers.Dense(
            len(SHAPE_NAMES), activation="softmax", name="shape",
            kernel_regularizer=regularizer)(appearance),
        "material": keras.layers.Dense(
            7, name="material", kernel_regularizer=regularizer)(appearance),
    }
    return keras.Model(inputs, outputs, name="fixed_synthetic_object_cnn")


def compile_model(model, learning_rate, weight_decay, statistics):
    translation_std = statistics["object_translation"]["standard_deviation"]
    scale_std = statistics["object_scale"]["standard_deviation"]
    material_std = statistics["material"]["standard_deviation"]
    losses = {
        "object_translation": PhysicalVectorMSE(
            translation_std, name="physical_translation_mse"
        ),
        "object_orientation_6d": symmetry_rotation_loss,
        "object_scale": "mse",
        "shape": "categorical_crossentropy",
        "material": "mse",
    }
    metrics = {
        "object_translation": [
            PhysicalEuclideanDistance(
                translation_std, name="euclidean_distance_m"
            )
        ],
        "object_orientation_6d": [symmetry_geodesic_angle_degrees],
        "object_scale": [
            PhysicalMAE(scale_std, name="absolute_scale_error")
        ],
        "shape": ["accuracy"],
        "material": [
            PhysicalMAE(
                material_std, component_index=index,
                name=f"{name}_physical_mae",
            )
            for index, name in enumerate((
                "color_r", "color_g", "color_b", "ambient", "diffuse",
                "specular", "shininess",
            ))
        ],
    }
    optimizer = keras.optimizers.AdamW(
        learning_rate, weight_decay=weight_decay, global_clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)


def split_train_test(records, seed):
    """Randomly splits records into 80% training and 20% held-out test."""
    if len(records) < 2:
        raise ValueError("At least two generated samples are required")
    indices = np.random.default_rng(seed).permutation(len(records))
    test_count = max(1, round(len(records) * 0.2))
    training_count = len(records) - test_count
    training = [records[index] for index in indices[:training_count]]
    test = [records[index] for index in indices[training_count:]]
    return training, test


def save_split_manifest(path, training, test, seed):
    """Saves the reproducible 80/20 split without a validation partition."""
    def names(records):
        return [Path(record["_metadata_path"]).stem for record in records]

    payload = {
        "seed": seed,
        "ratio": {"train": 0.8, "test": 0.2},
        "train": names(training),
        "test": names(test),
    }
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/fixed_object_cnn"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.batch_size, args.epochs) < 1 or args.max_depth <= 0:
        raise ValueError("batch size, epochs, and max depth must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu)
    records = load_records(args.dataset)
    training_records, test_records = split_train_test(records, args.seed)
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", training_records, test_records, args.seed
    )
    normalizer = ObjectNormalizer.fit(training_records)
    normalizer.save(args.output / "normalization.json")
    training = ObjectRGBDDataset(
        training_records, normalizer, args.batch_size, args.max_depth,
        shuffle=True, seed=args.seed,
    )
    model = build_model(
        training[0][0].shape[1:], args.l2_regularization
    )
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        ObjectHistoryPlot(args.output / "loss.png"),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="loss", mode="min",
            save_best_only=True,
        ),
    ]
    print(
        f"Training on {cpu}: {len(training_records)} train, "
        f"{len(test_records)} held-out test; no validation split"
    )
    model.fit(
        training, epochs=args.epochs, callbacks=callbacks
    )
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
