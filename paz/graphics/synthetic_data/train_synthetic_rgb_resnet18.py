"""Train a two-stream ResNet-18 to predict parameters from RGB and depth.

Separate RGB and depth encoders produce 512-D features. Their fused 512-D
representation feeds separate geometry, material, and lighting MLPs. Depth is
converted from metres to [0, 1] by clipping to ``--max-depth``.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
        paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 \
        --dataset datasets/synthetic_rgbd_1000_v3 \
        --output experiments/resnet18_rgbd_physical_validation
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ["JAX_PLATFORMS"] = "cuda"

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


class RGBDDataset(keras.utils.PyDataset):
    """Loads separate RGB/depth inputs and complete renderer targets."""

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
        rgb_images, depth_images, targets = [], [], []
        for index in indices:
            record = self.records[index]
            root = Path(record["_root"])
            image_path = root / record["rgb"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = image.astype(np.float32) / 255.0
            depth_path = root / record["depth"]
            depth = np.load(depth_path).astype(np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(
                    f"RGB/depth shape mismatch for {image_path} and "
                    f"{depth_path}"
                )
            depth = np.clip(depth / self.max_depth, 0.0, 1.0)[..., None]
            rgb_images.append(rgb)
            depth_images.append(depth)
            targets.append(
                self.normalizer.normalize(extract_targets(record))
            )
        stacked = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        inputs = {
            "rgb": np.stack(rgb_images),
            "depth": np.stack(depth_images),
        }
        return inputs, stacked

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
            axis.plot(epochs, self.history.get(name, []), label="Train")
            validation_name = f"val_{name}"
            if validation_name in self.history:
                axis.plot(
                    epochs, self.history[validation_name], label="Validation"
                )
                axis.legend()
            axis.set_title(name.replace("_", " ").title())
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.path, dpi=150)
        plt.close(figure)


class PeriodicWeightsCheckpoint(keras.callbacks.Callback):
    """Saves model weights at a fixed epoch interval."""

    def __init__(self, directory, interval=10):
        super().__init__()
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")
        self.directory = Path(directory)
        self.interval = interval
        self.directory.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        completed_epoch = epoch + 1
        if completed_epoch % self.interval == 0:
            path = (
                self.directory
                / f"epoch_{completed_epoch:04d}.weights.h5"
            )
            self.model.save_weights(path)
            print(f"Saved periodic weights to {path}")


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


def build_encoder(inputs, regularizer, prefix):
    """Builds a standard ResNet-18 encoder returning a 512-D feature."""
    x = keras.layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name=f"{prefix}_stem_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(name=f"{prefix}_stem_bn")(x)
    x = keras.layers.Activation("relu", name=f"{prefix}_stem_relu")(x)
    x = keras.layers.MaxPooling2D(
        3, strides=2, padding="same", name=f"{prefix}_stem_pool"
    )(x)
    for stage, filters in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            stride = 2 if stage > 1 and block == 0 else 1
            x = basic_block(
                x, filters, stride, regularizer,
                name=f"{prefix}_stage{stage}_block{block + 1}",
            )
    return keras.layers.GlobalAveragePooling2D(
        name=f"{prefix}_resnet18_512d_feature"
    )(x)


def mlp_branch(features, name, regularizer):
    x = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=regularizer,
        name=f"{name}_mlp",
    )(features)
    return keras.layers.Dropout(0.2, name=f"{name}_dropout")(x)


def build_model(input_shape=(256, 256), l2_regularization=1e-4):
    """Builds separate RGB/depth ResNet-18 encoders and fused heads."""
    rgb = keras.Input((*input_shape, 3), name="rgb")
    depth = keras.Input((*input_shape, 1), name="depth")
    regularizer = keras.regularizers.L2(l2_regularization)
    rgb_features = build_encoder(rgb, regularizer, "rgb")
    depth_features = build_encoder(depth, regularizer, "depth")
    features = keras.layers.Concatenate(name="feature_concatenation")(
        [rgb_features, depth_features]
    )
    features = keras.layers.Dense(
        512, activation="relu", kernel_regularizer=regularizer,
        name="fused_512d_feature",
    )(features)
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
    return keras.Model(
        {"rgb": rgb, "depth": depth}, outputs,
        name="physical_parameter_rgb_depth_resnet18",
    )


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


def split_records(records, seed, validation_split=0.2, test_split=0.2):
    """Randomly splits records using fractions of the complete dataset."""
    if len(records) < 5:
        raise ValueError("At least five generated samples are required")
    if not 0.0 < validation_split < 1.0 - test_split:
        raise ValueError(
            f"validation split must be between 0 and {1.0 - test_split}"
        )
    indices = np.random.default_rng(seed).permutation(len(records))
    validation_count = max(1, round(len(records) * validation_split))
    test_count = max(1, round(len(records) * test_split))
    training_count = len(records) - validation_count - test_count
    if training_count < 1:
        raise ValueError("Split leaves no training samples")
    validation_end = training_count + validation_count
    train = [records[index] for index in indices[:training_count]]
    validation = [
        records[index] for index in indices[training_count:validation_end]
    ]
    test = [records[index] for index in indices[validation_end:]]
    return train, validation, test


def save_split_manifest(path, train, validation, test, seed,
                        validation_split, test_split=0.2):
    """Saves exact sample IDs and requested split fractions."""
    def sample_ids(records):
        return [Path(record["_metadata_path"]).stem for record in records]

    payload = {
        "seed": seed,
        "ratio": {
            "train": round(1.0 - validation_split - test_split, 10),
            "validation": validation_split,
            "test": test_split,
        },
        "train": sample_ids(train),
        "validation": sample_ids(validation),
        "test": sample_ids(test),
    }
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/resnet18_rgbd_physical"))
    parser.add_argument("--test-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5,
                        help="Factor applied when validation loss plateaus.")
    parser.add_argument("--lr-reduction-patience", type=int, default=7,
                        help="Plateau epochs before reducing learning rate.")
    parser.add_argument("--lr-min-delta", type=float, default=1e-3,
                        help="Minimum val-loss improvement to reset patience.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Unimproved validation epochs before stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=1e-3,
                        help="Minimum val-loss improvement for early stopping.")
    parser.add_argument("--early-stopping-start-epoch", type=int, default=20,
                        help="Epoch before which early stopping is disabled.")
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save weights every N completed epochs.")
    parser.add_argument("--max-depth", type=float, default=10.0,
                        help="Depth in metres mapped to 1.0 (farther clips).")
    parser.add_argument(
        "--validation-split", type=float, default=0.2,
        help=("Fraction of the complete dataset used for validation. "
              "Test always uses 0.2; the default gives 0.6/0.2/0.2."),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.batch_size, args.epochs, args.checkpoint_every) < 1:
        raise ValueError(
            "batch size, epochs, and checkpoint interval must be positive"
        )
    if args.lr_reduction_patience < 1:
        raise ValueError("LR-reduction patience must be positive")
    if not 0.0 < args.lr_reduction_factor < 1.0:
        raise ValueError("LR-reduction factor must be between 0 and 1")
    if args.lr_min_delta < 0.0 or args.min_learning_rate <= 0.0:
        raise ValueError("LR minimum delta and minimum rate must be valid")
    if args.early_stopping_patience < 1:
        raise ValueError("early-stopping patience must be positive")
    if (args.early_stopping_min_delta < 0.0
            or args.early_stopping_start_epoch < 0):
        raise ValueError("early-stopping delta and start epoch must be valid")
    if args.max_depth <= 0.0:
        raise ValueError("max depth must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = jax.devices("cuda")[0]
    jax.config.update("jax_default_device", gpu)
    records = load_records(args.dataset)
    training_records, validation_records, test_records = split_records(
        records, args.seed, args.validation_split
    )
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", training_records, validation_records,
        test_records, args.seed, args.validation_split,
    )
    normalizer = TargetNormalizer.fit(training_records)
    normalizer.save(args.output / "normalization.json")
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({"depth_unit": "metres", "max_depth": args.max_depth},
                  file, indent=2)
    training = RGBDDataset(
        training_records, normalizer, args.batch_size, args.max_depth,
        shuffle=True,
        seed=args.seed,
    )
    validation = RGBDDataset(
        validation_records, normalizer, args.batch_size, args.max_depth,
        shuffle=False, seed=args.seed,
    )
    rgb_shape = training[0][0]["rgb"].shape[1:3]
    model = build_model(rgb_shape, args.l2_regularization)
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(args.output / "loss.png"),
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
            min_delta=args.lr_min_delta,
            min_lr=args.min_learning_rate,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
            start_from_epoch=args.early_stopping_start_epoch,
            verbose=1,
        ),
    ]
    print(
        f"Training dual RGB/depth ResNet-18 on {gpu}: "
        f"{len(training_records)} train, "
        f"{len(validation_records)} validation, "
        f"{len(test_records)} held-out test"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
