"""Train a ResNet-18 to predict complete renderer parameters from RGB.

The 512-D encoder feature feeds separate geometry, material, and lighting
MLPs. The material head matches generator metadata: color RGB followed by
scalar diffuse, specular, and ambient coefficients.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 \
        --dataset datasets/synthetic_rgbd_1000_v3 \
        --output experiments/resnet18_physical_no_validation
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
    PhysicalEuclideanDistance,
    PhysicalMAE,
    PhysicalVectorMSE,
    SHAPE_NAMES,
    export_test_split,
    load_records,
    save_train_test_manifest,
    split_train_test,
    symmetry_geodesic_angle_degrees,
    symmetry_rotation_loss,
)


REGRESSION_NAMES = (
    "object_translation", "object_scale", "material", "light_position",
    "light_intensity",
)
OUTPUT_NAMES = (
    "object_translation", "object_orientation_6d", "object_scale", "shape",
    "material", "light_position", "light_intensity",
)
MATERIAL_NAMES = (
    "color_r", "color_g", "color_b", "diffuse", "specular", "ambient",
    "shininess",
)


def extract_targets(record):
    """Converts generator metadata to geometry/material/lighting targets."""
    shape_index = SHAPE_NAMES.index(record["shape"]["type"])
    shape = np.eye(len(SHAPE_NAMES), dtype=np.float32)[shape_index]
    orientation = record["object"]["orientation_camera_6d"]
    orientation_6d = np.asarray(
        orientation["vector_a"] + orientation["vector_b"], np.float32
    )
    source_material = record["material"]
    color = np.asarray(source_material["color_rgb"], dtype=np.float32)
    material = np.concatenate([
        color,
        np.asarray([
            source_material["diffuse"], source_material["specular"],
            source_material["ambient"], source_material["shininess"],
        ], dtype=np.float32),
    ])
    return {
        "object_translation": np.asarray(
            record["object"]["translation_camera_xyz"], np.float32
        ),
        # Shape is appended only for symmetry selection inside the loss.
        "object_orientation_6d": np.concatenate([orientation_6d, shape]),
        "object_scale": np.asarray(
            [record["object"]["scale"]], np.float32
        ),
        "shape": shape,
        "material": material,
        "light_position": np.asarray(
            record["light"]["position_camera_xyz"], np.float32
        ),
        "light_intensity": np.asarray(
            [record["light"]["intensity"]], np.float32
        ),
    }


class TargetNormalizer:
    """Standardizes all regression targets from training statistics."""

    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def fit(cls, records):
        targets = [extract_targets(record) for record in records]
        statistics = {}
        for name in REGRESSION_NAMES:
            values = np.stack([target[name] for target in targets])
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            statistics[name] = {
                "mean": mean.tolist(),
                "standard_deviation": std.tolist(),
            }
        return cls(statistics)

    def normalize(self, targets):
        normalized = {
            name: np.asarray(targets[name], np.float32)
            for name in OUTPUT_NAMES
        }
        for name in REGRESSION_NAMES:
            statistics = self.statistics[name]
            mean = np.asarray(statistics["mean"], np.float32)
            std = np.asarray(
                statistics["standard_deviation"], np.float32
            )
            normalized[name] = (normalized[name] - mean) / std
        return normalized

    def save(self, path):
        payload = {
            "shape_names": SHAPE_NAMES,
            "material_definition": {
                "values": [
                    "color_r", "color_g", "color_b", "diffuse", "specular",
                    "ambient", "shininess",
                ],
                "source": "generator metadata material fields",
            },
            "targets": self.statistics,
        }
        with path.open("w") as file:
            json.dump(payload, file, indent=2)


class RGBDataset(keras.utils.PyDataset):
    """Loads RGB images and complete renderer targets."""

    def __init__(self, records, normalizer, batch_size, shuffle=False, seed=0):
        super().__init__()
        self.records = list(records)
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        images, targets = [], []
        for index in indices:
            record = self.records[index]
            root = Path(record["_root"])
            image_path = root / record["rgb"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image.astype(np.float32) / 255.0)
            targets.append(
                self.normalizer.normalize(extract_targets(record))
            )
        stacked = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        return np.stack(images), stacked

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class TrainingPlot(keras.callbacks.Callback):
    """Saves total and per-head training losses after every epoch."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            self.history.setdefault(name, []).append(value)
        names = ("loss",) + tuple(f"{name}_loss" for name in OUTPUT_NAMES)
        figure, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True)
        epochs = np.arange(1, epoch + 2)
        for axis, name in zip(axes.flat, names):
            axis.plot(epochs, self.history.get(name, []))
            axis.set_title(name.replace("_", " ").title())
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.path, dpi=150)
        plt.close(figure)


def basic_block(inputs, filters, stride, regularizer, name):
    """Standard two-convolution ResNet basic block."""
    shortcut = inputs
    x = keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name=f"{name}_conv1",
    )(inputs)
    x = keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"{name}_relu1")(x)
    x = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name=f"{name}_conv2",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters, 1, strides=stride, use_bias=False,
            kernel_regularizer=regularizer, name=f"{name}_projection",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(shortcut)
    x = keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return keras.layers.Activation("relu", name=f"{name}_output")(x)


def build_encoder(inputs, regularizer):
    """Builds a standard ResNet-18 encoder returning a 512-D feature."""
    x = keras.layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name="stem_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation("relu", name="stem_relu")(x)
    x = keras.layers.MaxPooling2D(
        3, strides=2, padding="same", name="stem_pool"
    )(x)
    for stage, filters in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            stride = 2 if stage > 1 and block == 0 else 1
            x = basic_block(
                x, filters, stride, regularizer,
                name=f"stage{stage}_block{block + 1}",
            )
    return keras.layers.GlobalAveragePooling2D(
        name="resnet18_512d_feature"
    )(x)


def mlp_branch(features, name, regularizer):
    x = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=regularizer,
        name=f"{name}_mlp",
    )(features)
    return keras.layers.Dropout(0.2, name=f"{name}_dropout")(x)


def build_model(input_shape=(256, 256, 3), l2_regularization=1e-4):
    """Builds ResNet-18 plus geometry, material, and lighting MLPs."""
    inputs = keras.Input(input_shape, name="rgb")
    regularizer = keras.regularizers.L2(l2_regularization)
    features = build_encoder(inputs, regularizer)
    geometry = mlp_branch(features, "geometry", regularizer)
    material = mlp_branch(features, "material", regularizer)
    lighting = mlp_branch(features, "lighting", regularizer)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation",
            kernel_regularizer=regularizer)(geometry),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d",
            kernel_regularizer=regularizer)(geometry),
        "object_scale": keras.layers.Dense(
            1, name="object_scale",
            kernel_regularizer=regularizer)(geometry),
        "shape": keras.layers.Dense(
            len(SHAPE_NAMES), activation="softmax", name="shape",
            kernel_regularizer=regularizer)(geometry),
        "material": keras.layers.Dense(
            7, name="material", kernel_regularizer=regularizer)(material),
        "light_position": keras.layers.Dense(
            3, name="light_position",
            kernel_regularizer=regularizer)(lighting),
        "light_intensity": keras.layers.Dense(
            1, name="light_intensity",
            kernel_regularizer=regularizer)(lighting),
    }
    return keras.Model(inputs, outputs, name="physical_parameter_resnet18")


def compile_model(model, learning_rate, weight_decay, statistics):
    translation_std = statistics["object_translation"]["standard_deviation"]
    scale_std = statistics["object_scale"]["standard_deviation"]
    material_std = statistics["material"]["standard_deviation"]
    light_position_std = statistics["light_position"]["standard_deviation"]
    light_intensity_std = statistics["light_intensity"]["standard_deviation"]
    losses = {
        "object_translation": PhysicalVectorMSE(
            translation_std, name="physical_translation_mse"
        ),
        "object_orientation_6d": symmetry_rotation_loss,
        "object_scale": "mse",
        "shape": "categorical_crossentropy",
        "material": "mse",
        "light_position": keras.losses.Huber(delta=1.0),
        "light_intensity": PhysicalVectorMSE(
            light_intensity_std, name="physical_light_intensity_mse"
        ),
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
            for index, name in enumerate(MATERIAL_NAMES)
        ],
        "light_position": [
            PhysicalEuclideanDistance(
                light_position_std, name="euclidean_distance_m"
            )
        ],
        "light_intensity": [
            PhysicalMAE(light_intensity_std, name="physical_mae")
        ],
    }
    loss_weights = {name: 1.0 for name in OUTPUT_NAMES}
    loss_weights["light_position"] = 0.25
    optimizer = keras.optimizers.AdamW(
        learning_rate, weight_decay=weight_decay, global_clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer, loss=losses, loss_weights=loss_weights,
        metrics=metrics,
    )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/resnet18_physical"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch size and epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu)
    records = load_records(args.dataset)
    training_records, test_records = split_train_test(records, args.seed)
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    save_train_test_manifest(
        args.output / "split.json", training_records, test_records, args.seed
    )
    normalizer = TargetNormalizer.fit(training_records)
    normalizer.save(args.output / "normalization.json")
    training = RGBDataset(
        training_records, normalizer, args.batch_size, shuffle=True,
        seed=args.seed,
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
        TrainingPlot(args.output / "loss.png"),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", monitor="loss", mode="min",
            save_best_only=True,
        ),
    ]
    print(
        f"Training ResNet-18 on {cpu}: {len(training_records)} train, "
        f"{len(test_records)} held-out test; no validation split"
    )
    model.fit(training, epochs=args.epochs, callbacks=callbacks)
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
