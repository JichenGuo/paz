"""Train a multi-task CNN on data from generate_synthetic_rgbd.py.

Example:
    KERAS_BACKEND=jax python paz/graphics/train_synthetic_rgbd_cnn.py \
        --dataset synthetic_rgbd --output experiments/rgbd_cnn

RGB and metric depth are combined into a four-channel input. Continuous
targets, except the 6D orientation, are standardized using the training split.
The saved ``normalization.json`` is required to decode model predictions.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import math
from pathlib import Path

import cv2
import keras
import numpy as np


SHAPE_NAMES = ("cube", "cylinder", "sphere")
REGRESSION_NAMES = (
    "object_translation",
    "object_scale",
    "light_position",
    "light_intensity",
    "material",
)


def load_records(dataset_path):
    """Loads and validates all sample metadata files."""
    metadata_paths = sorted((dataset_path / "metadata").glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No metadata JSON files found in {dataset_path}")
    records = []
    for metadata_path in metadata_paths:
        with metadata_path.open() as file:
            record = json.load(file)
        record["_root"] = str(dataset_path)
        records.append(record)
    return records


def extract_targets(record):
    """Converts one metadata dictionary into model target arrays."""
    orientation = record["object"]["orientation_camera_6d"]
    material = record["material"]
    shape_arg = SHAPE_NAMES.index(record["shape"]["type"])
    return {
        "object_translation": np.asarray(
            record["object"]["translation_camera_xyz"], np.float32
        ),
        "object_orientation_6d": np.asarray(
            orientation["vector_a"] + orientation["vector_b"], np.float32
        ),
        "object_scale": np.asarray([record["object"]["scale"]], np.float32),
        "light_position": np.asarray(
            record["light"]["position_camera_xyz"], np.float32
        ),
        "light_intensity": np.asarray(
            [record["light"]["intensity"]], np.float32
        ),
        "shape": np.eye(len(SHAPE_NAMES), dtype=np.float32)[shape_arg],
        "material": np.asarray(
            material["color_rgb"]
            + [material["ambient"], material["diffuse"],
               material["specular"], material["shininess"]],
            np.float32,
        ),
    }


class TargetNormalizer:
    """Standardizes regression labels and serializes their statistics."""

    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def fit(cls, records):
        statistics = {}
        targets = [extract_targets(record) for record in records]
        for name in REGRESSION_NAMES:
            values = np.stack([target[name] for target in targets])
            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0)
            standard_deviation = np.where(standard_deviation < 1e-6,
                                          1.0, standard_deviation)
            statistics[name] = {
                "mean": mean.tolist(),
                "standard_deviation": standard_deviation.tolist(),
            }
        return cls(statistics)

    def normalize(self, targets):
        targets = dict(targets)
        for name in REGRESSION_NAMES:
            stats = self.statistics[name]
            mean = np.asarray(stats["mean"], np.float32)
            std = np.asarray(stats["standard_deviation"], np.float32)
            targets[name] = (targets[name] - mean) / std
        return targets

    def save(self, path):
        payload = {"shape_names": SHAPE_NAMES, "targets": self.statistics}
        with path.open("w") as file:
            json.dump(payload, file, indent=2)


class RGBDDataset(keras.utils.PyDataset):
    """Loads paired RGB/depth files and their multi-task labels."""

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
        inputs, batch_targets = [], []
        for index in indices:
            record = self.records[index]
            inputs.append(self.load_rgbd(record))
            targets = self.normalizer.normalize(extract_targets(record))
            batch_targets.append(targets)
        targets = {
            name: np.stack([target[name] for target in batch_targets])
            for name in batch_targets[0]
        }
        return np.stack(inputs), targets

    def load_rgbd(self, record):
        root = Path(record["_root"])
        rgb = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(root / record["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(root / record["depth"]).astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth shapes must match")
        depth = np.clip(depth / self.max_depth, 0.0, 1.0)
        return np.concatenate([rgb, depth[..., None]], axis=-1)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def convolution_block(inputs, filters, stride=2):
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same",
                            use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def build_model(input_shape, num_shapes=len(SHAPE_NAMES)):
    """Builds a shared CNN encoder with classification/regression heads."""
    inputs = keras.Input(input_shape, name="rgbd")
    x = convolution_block(inputs, 32)
    x = convolution_block(x, 64)
    x = convolution_block(x, 128)
    x = convolution_block(x, 256)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation")(x),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d")(x),
        "object_scale": keras.layers.Dense(1, name="object_scale")(x),
        "light_position": keras.layers.Dense(3, name="light_position")(x),
        "light_intensity": keras.layers.Dense(1, name="light_intensity")(x),
        "shape": keras.layers.Dense(
            num_shapes, activation="softmax", name="shape")(x),
        "material": keras.layers.Dense(7, name="material")(x),
    }
    return keras.Model(inputs, outputs, name="synthetic_rgbd_cnn")


def compile_model(model, learning_rate):
    losses = {name: "mse" for name in model.output_names}
    losses["shape"] = "categorical_crossentropy"
    metrics = {name: ["mae"] for name in model.output_names}
    metrics["shape"] = ["accuracy"]
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)


def split_records(records, validation_fraction, seed):
    if len(records) < 2:
        raise ValueError("At least two generated samples are required")
    indices = np.random.default_rng(seed).permutation(len(records))
    validation_count = max(1, round(len(records) * validation_fraction))
    validation_count = min(validation_count, len(records) - 1)
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    train = [records[index] for index in training_indices]
    valid = [records[index] for index in validation_indices]
    return train, valid


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/synthetic_rgbd_cnn"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1 or args.max_depth <= 0:
        raise ValueError("batch-size, epochs, and max-depth must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=True)
    records = load_records(args.dataset)
    train_records, valid_records = split_records(
        records, args.validation_fraction, args.seed
    )
    normalizer = TargetNormalizer.fit(train_records)
    normalizer.save(args.output / "normalization.json")
    train = RGBDDataset(train_records, normalizer, args.batch_size,
                        args.max_depth, shuffle=True, seed=args.seed)
    valid = RGBDDataset(valid_records, normalizer, args.batch_size,
                        args.max_depth)
    input_shape = train[0][0].shape[1:]
    model = build_model(input_shape)
    compile_model(model, args.learning_rate)
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        keras.callbacks.ModelCheckpoint(
            args.output / "best.keras", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=5, factor=0.5
        ),
    ]
    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks)
    model.save(args.output / "final.keras")


if __name__ == "__main__":
    main()
