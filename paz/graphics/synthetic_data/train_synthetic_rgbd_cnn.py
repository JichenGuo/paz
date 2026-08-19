"""Train a multi-task CNN on data from generate_synthetic_rgbd.py.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.train_synthetic_rgbd_cnn \
        --dataset synthetic_rgbd_1000 --output experiments/rgbd_cnn \
        --test-output synthetic_rgbd_1000_test

RGB and metric depth are combined into a four-channel input. Continuous
targets, except the 6D orientation, are standardized using the training split.
The saved ``normalization.json`` is required to decode model predictions.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import itertools
import json
import math
from pathlib import Path
import shutil

import cv2
import jax
import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHAPE_NAMES = ("cube", "cylinder", "sphere")
REGRESSION_NAMES = (
    "object_translation",
    "object_scale",
    "light_position",
    "light_intensity",
    "material",
)
LOSS_HEAD_NAMES = (
    "object_translation",
    "object_orientation_6d",
    "object_scale",
    "light_position",
    "light_intensity",
    "shape",
    "material",
)
METRIC_CURVES = (
    ("Object translation distance (m)",
     "object_translation_euclidean_distance_m"),
    ("Orientation geodesic angle (degrees)",
     "object_orientation_6d_symmetry_geodesic_angle_degrees"),
    ("Object absolute scale error", "object_scale_absolute_scale_error"),
    ("Light position distance (m)",
     "light_position_euclidean_distance_m"),
    ("Light intensity MAE", "light_intensity_physical_mae"),
    ("Shape accuracy", "shape_accuracy"),
    ("Material red MAE", "material_color_r_physical_mae"),
    ("Material green MAE", "material_color_g_physical_mae"),
    ("Material blue MAE", "material_color_b_physical_mae"),
    ("Material ambient MAE", "material_ambient_physical_mae"),
    ("Material diffuse MAE", "material_diffuse_physical_mae"),
    ("Material specular MAE", "material_specular_physical_mae"),
    ("Material shininess MAE", "material_shininess_physical_mae"),
)


def build_cube_symmetries():
    """Returns the 24 orientation-preserving symmetries of a cube."""
    symmetries = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float32)
            matrix[permutation, range(3)] = signs
            if np.linalg.det(matrix) > 0.0:
                symmetries.append(matrix)
    return np.stack(symmetries)


CUBE_SYMMETRIES = build_cube_symmetries()


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
        record["_metadata_path"] = str(metadata_path)
        records.append(record)
    return records


def extract_targets(record):
    """Converts one metadata dictionary into model target arrays."""
    orientation = record["object"]["orientation_camera_6d"]
    material = record["material"]
    shape_arg = SHAPE_NAMES.index(record["shape"]["type"])
    shape_target = np.eye(len(SHAPE_NAMES), dtype=np.float32)[shape_arg]
    orientation_target = np.asarray(
        orientation["vector_a"] + orientation["vector_b"], np.float32
    )
    return {
        "object_translation": np.asarray(
            record["object"]["translation_camera_xyz"], np.float32
        ),
        # Shape is appended only to let the loss choose a symmetry group.
        # The CNN orientation head still predicts exactly six values.
        "object_orientation_6d": np.concatenate(
            [orientation_target, shape_target]
        ),
        "object_scale": np.asarray([record["object"]["scale"]], np.float32),
        "light_position": np.asarray(
            record["light"]["position_camera_xyz"], np.float32
        ),
        "light_intensity": np.asarray(
            [record["light"]["intensity"]], np.float32
        ),
        "shape": shape_target,
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


class LossPlot(keras.callbacks.Callback):
    """Updates consolidated loss and physical-metric figures."""

    def __init__(self, path, metric_path=None):
        super().__init__()
        self.path = Path(path)
        self.metric_path = (
            Path(metric_path) if metric_path is not None
            else self.path.with_name("metrics.png")
        )
        self.training_loss = []
        self.validation_loss = []
        self.component_history = {
            name: {"train": [], "validation": []}
            for name in LOSS_HEAD_NAMES
        }
        self.metric_history = {
            log_name: {"train": [], "validation": []}
            for _, log_name in METRIC_CURVES
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.training_loss.append(logs.get("loss", np.nan))
        self.validation_loss.append(logs.get("val_loss", np.nan))
        for name, history in self.component_history.items():
            history["train"].append(logs.get(f"{name}_loss", np.nan))
            history["validation"].append(
                logs.get(f"val_{name}_loss", np.nan)
            )
        for name, history in self.metric_history.items():
            history["train"].append(logs.get(name, np.nan))
            history["validation"].append(
                logs.get(f"val_{name}", np.nan)
            )
        epochs = np.arange(1, len(self.training_loss) + 1)
        curves = [
            ("Total multi-task loss", self.training_loss,
             self.validation_loss)
        ]
        curves.extend([
            (name.replace("_", " ").title(), history["train"],
             history["validation"])
            for name, history in self.component_history.items()
        ])
        figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
        for axis, (title, training, validation) in zip(axes.flat, curves):
            axis.plot(epochs, training, label="Training")
            axis.plot(epochs, validation, label="Validation")
            axis.set(title=title, ylabel="Loss")
            axis.grid(alpha=0.3)
            axis.legend()
        for axis in axes[-1]:
            axis.set_xlabel("Epoch")
        figure.tight_layout()
        figure.savefig(self.path, dpi=150)
        plt.close(figure)

        figure, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True)
        for axis, (title, log_name) in zip(axes.flat, METRIC_CURVES):
            history = self.metric_history[log_name]
            axis.plot(epochs, history["train"], label="Training")
            axis.plot(epochs, history["validation"], label="Validation")
            axis.set(title=title, ylabel="Metric")
            axis.grid(alpha=0.3)
            axis.legend()
        for axis in axes.flat[len(METRIC_CURVES):]:
            axis.set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel("Epoch")
        figure.tight_layout()
        figure.savefig(self.metric_path, dpi=150)
        plt.close(figure)


def convolution_block(inputs, filters, regularizer, stride=2):
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same",
                            use_bias=False,
                            kernel_regularizer=regularizer)(inputs)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def normalize_vectors(vectors, epsilon=1e-8):
    """Normalizes vectors with finite gradients at zero magnitude."""
    squared_norm = keras.ops.sum(
        keras.ops.square(vectors), axis=-1, keepdims=True
    )
    safe_norm = keras.ops.sqrt(squared_norm + epsilon)
    return vectors / safe_norm


def stable_angle_from_cosine(cosine, epsilon=1e-6):
    """Computes acos away from its infinite-gradient endpoints."""
    cosine = keras.ops.clip(cosine, -1.0 + epsilon, 1.0 - epsilon)
    return keras.ops.arccos(cosine)


@keras.saving.register_keras_serializable("synthetic_rgbd")
def rotation_6d_to_matrix(rotation_6d):
    """Maps unconstrained (..., 6) vectors to (..., 3, 3) rotations."""
    vector_a = rotation_6d[..., :3]
    vector_b = rotation_6d[..., 3:]
    axis_x = normalize_vectors(vector_a)
    projection = keras.ops.sum(
        axis_x * vector_b, axis=-1, keepdims=True
    )
    axis_y = normalize_vectors(vector_b - projection * axis_x)
    axis_z = keras.ops.cross(axis_x, axis_y)
    return keras.ops.stack([axis_x, axis_y, axis_z], axis=-1)


@keras.saving.register_keras_serializable("synthetic_rgbd")
def symmetry_geodesic_angle(target_6d_and_shape, predicted_6d):
    """Minimum geodesic error under each primitive's symmetry group."""
    target_6d = target_6d_and_shape[..., :6]
    shape = target_6d_and_shape[..., 6:]
    target_rotation = rotation_6d_to_matrix(target_6d)
    predicted_rotation = rotation_6d_to_matrix(predicted_6d)

    cube_symmetries = keras.ops.cast(
        keras.ops.convert_to_tensor(CUBE_SYMMETRIES), target_rotation.dtype
    )
    equivalent_cubes = keras.ops.matmul(
        keras.ops.expand_dims(target_rotation, axis=1),
        keras.ops.expand_dims(cube_symmetries, axis=0),
    )
    equivalent_transpose = keras.ops.transpose(
        equivalent_cubes, (0, 1, 3, 2)
    )
    relative_cubes = keras.ops.matmul(
        equivalent_transpose,
        keras.ops.expand_dims(predicted_rotation, axis=1),
    )
    cube_traces = keras.ops.trace(relative_cubes, axis1=-2, axis2=-1)
    cube_cosines = (cube_traces - 1.0) / 2.0
    cube_angles = keras.ops.min(
        stable_angle_from_cosine(cube_cosines), axis=-1
    )

    target_axis = target_rotation[..., :, 1]
    predicted_axis = predicted_rotation[..., :, 1]
    axis_cosine = keras.ops.sum(target_axis * predicted_axis, axis=-1)
    # A plain, closed cylinder is unchanged by reversing its vertical axis.
    cylinder_cosine = keras.ops.abs(axis_cosine)
    cylinder_angles = stable_angle_from_cosine(cylinder_cosine)

    shape_arg = keras.ops.argmax(shape, axis=-1)
    angles = keras.ops.where(shape_arg == 0, cube_angles, cylinder_angles)
    return keras.ops.where(shape_arg == 2, keras.ops.zeros_like(angles), angles)


@keras.saving.register_keras_serializable("synthetic_rgbd")
def symmetry_rotation_loss(target_6d_and_shape, predicted_6d):
    """Squared minimum SO(3) geodesic distance under object symmetry."""
    angles = symmetry_geodesic_angle(target_6d_and_shape, predicted_6d)
    return keras.ops.square(angles)


@keras.saving.register_keras_serializable("synthetic_rgbd")
def symmetry_geodesic_angle_degrees(target_6d_and_shape, predicted_6d):
    """Symmetry-aware geodesic rotation error reported in degrees."""
    radians = symmetry_geodesic_angle(target_6d_and_shape, predicted_6d)
    return radians * (180.0 / np.pi)


@keras.saving.register_keras_serializable("synthetic_rgbd")
class PhysicalVectorMSE(keras.losses.Loss):
    """Squared Euclidean error after restoring physical target units."""

    def __init__(self, standard_deviation=(1.0,),
                 name="physical_vector_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(value)
                                        for value in standard_deviation)

    def call(self, target, prediction):
        scale = keras.ops.cast(
            keras.ops.convert_to_tensor(self.standard_deviation),
            prediction.dtype,
        )
        physical_error = (prediction - target) * scale
        return keras.ops.sum(keras.ops.square(physical_error), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"standard_deviation": self.standard_deviation})
        return config


@keras.saving.register_keras_serializable("synthetic_rgbd")
class PhysicalTranslationMSE(PhysicalVectorMSE):
    """Backward-compatible physical translation loss."""

    def __init__(self, standard_deviation=(1.0, 1.0, 1.0),
                 name="physical_translation_mse", **kwargs):
        super().__init__(standard_deviation, name=name, **kwargs)


class PhysicalMetric(keras.metrics.Metric):
    """Base class accumulating a mean metric in restored physical units."""

    def __init__(self, standard_deviation=(1.0,), component_index=None,
                 name="physical_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(value)
                                        for value in standard_deviation)
        self.component_index = component_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def physical_error(self, target, prediction):
        scale = keras.ops.cast(
            keras.ops.convert_to_tensor(self.standard_deviation),
            prediction.dtype,
        )
        error = (prediction - target) * scale
        if self.component_index is not None:
            error = error[..., self.component_index]
        return error

    def accumulate(self, values, sample_weight=None):
        values = keras.ops.cast(values, self.dtype)
        weights = keras.ops.ones_like(values)
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, self.dtype)
            weights = weights * sample_weight
        self.total.assign_add(keras.ops.sum(values * weights))
        self.count.assign_add(keras.ops.sum(weights))

    def result(self):
        return keras.ops.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "standard_deviation": self.standard_deviation,
            "component_index": self.component_index,
        })
        return config


@keras.saving.register_keras_serializable("synthetic_rgbd")
class PhysicalEuclideanDistance(PhysicalMetric):
    """Mean Euclidean vector error in restored physical units."""

    def __init__(self, standard_deviation=(1.0, 1.0, 1.0),
                 component_index=None, name="physical_distance", **kwargs):
        super().__init__(
            standard_deviation, component_index, name=name, **kwargs
        )

    def update_state(self, target, prediction, sample_weight=None):
        error = self.physical_error(target, prediction)
        distance = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(error), axis=-1)
        )
        self.accumulate(distance, sample_weight)


@keras.saving.register_keras_serializable("synthetic_rgbd")
class PhysicalMAE(PhysicalMetric):
    """Mean absolute component error in restored physical units."""

    def __init__(self, standard_deviation=(1.0,), component_index=None,
                 name="physical_mae", **kwargs):
        super().__init__(
            standard_deviation, component_index, name=name, **kwargs
        )

    def update_state(self, target, prediction, sample_weight=None):
        absolute_error = keras.ops.abs(
            self.physical_error(target, prediction)
        )
        self.accumulate(absolute_error, sample_weight)


def build_model(input_shape, num_shapes=len(SHAPE_NAMES),
                l2_regularization=1e-4):
    """Builds a shared CNN encoder with classification/regression heads."""
    inputs = keras.Input(input_shape, name="rgbd")
    regularizer = keras.regularizers.L2(l2_regularization)
    x = convolution_block(inputs, 16, regularizer)
    x = convolution_block(x, 32, regularizer)
    x = convolution_block(x, 64, regularizer)
    x = convolution_block(x, 128, regularizer)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation", kernel_regularizer=regularizer)(x),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d",
            kernel_regularizer=regularizer)(x),
        "object_scale": keras.layers.Dense(
            1, name="object_scale", kernel_regularizer=regularizer)(x),
        "light_position": keras.layers.Dense(
            3, name="light_position", kernel_regularizer=regularizer)(x),
        "light_intensity": keras.layers.Dense(
            1, name="light_intensity", kernel_regularizer=regularizer)(x),
        "shape": keras.layers.Dense(
            num_shapes, activation="softmax", name="shape",
            kernel_regularizer=regularizer)(x),
        "material": keras.layers.Dense(
            7, name="material", kernel_regularizer=regularizer)(x),
    }
    return keras.Model(inputs, outputs, name="synthetic_rgbd_cnn")


def compile_model(model, learning_rate, weight_decay=1e-4,
                  translation_standard_deviation=(1.0, 1.0, 1.0),
                  light_position_standard_deviation=(1.0, 1.0, 1.0),
                  light_intensity_standard_deviation=(1.0,),
                  object_scale_standard_deviation=(1.0,),
                  material_standard_deviation=(1.0,) * 7):
    losses = {name: "mse" for name in model.output_names}
    losses["shape"] = "categorical_crossentropy"
    losses["object_orientation_6d"] = symmetry_rotation_loss
    losses["object_translation"] = PhysicalVectorMSE(
        translation_standard_deviation, name="physical_translation_mse"
    )
    losses["light_position"] = PhysicalVectorMSE(
        light_position_standard_deviation,
        name="physical_light_position_mse",
    )
    losses["light_intensity"] = PhysicalVectorMSE(
        light_intensity_standard_deviation,
        name="physical_light_intensity_mse",
    )
    metrics = {
        "object_translation": [
            PhysicalEuclideanDistance(
                translation_standard_deviation,
                name="euclidean_distance_m",
            )
        ],
        "object_scale": [
            PhysicalMAE(
                object_scale_standard_deviation,
                name="absolute_scale_error",
            )
        ],
        "light_position": [
            PhysicalEuclideanDistance(
                light_position_standard_deviation,
                name="euclidean_distance_m",
            )
        ],
        "light_intensity": [
            PhysicalMAE(
                light_intensity_standard_deviation,
                name="physical_mae",
            )
        ],
        "material": [
            PhysicalMAE(
                material_standard_deviation, component_index=index,
                name=f"{name}_physical_mae",
            )
            for index, name in enumerate((
                "color_r", "color_g", "color_b", "ambient", "diffuse",
                "specular", "shininess",
            ))
        ],
    }
    metrics["shape"] = ["accuracy"]
    metrics["object_orientation_6d"] = [
        symmetry_geodesic_angle_degrees
    ]
    optimizer = keras.optimizers.AdamW(
        learning_rate, weight_decay=weight_decay, global_clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)


def split_records(records, seed):
    """Randomly splits records into 60% train, 20% validation, 20% test."""
    if len(records) < 5:
        raise ValueError("At least five generated samples are required")
    indices = np.random.default_rng(seed).permutation(len(records))
    validation_count = round(len(records) * 0.2)
    test_count = round(len(records) * 0.2)
    training_count = len(records) - validation_count - test_count
    training_indices = indices[:training_count]
    validation_end = training_count + validation_count
    validation_indices = indices[training_count:validation_end]
    test_indices = indices[validation_end:]
    train = [records[index] for index in training_indices]
    valid = [records[index] for index in validation_indices]
    test = [records[index] for index in test_indices]
    return train, valid, test


def export_test_split(records, destination):
    """Copies test RGB, depth, and metadata into a standalone directory."""
    for directory in ("rgb", "depth", "metadata"):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    for record in records:
        root = Path(record["_root"])
        rgb_source = root / record["rgb"]
        depth_source = root / record["depth"]
        metadata_source = Path(record["_metadata_path"])
        shutil.copy2(rgb_source, destination / "rgb" / rgb_source.name)
        shutil.copy2(depth_source, destination / "depth" / depth_source.name)
        shutil.copy2(
            metadata_source, destination / "metadata" / metadata_source.name
        )


def save_split_manifest(path, train, valid, test, seed):
    """Saves sample IDs so the random split can be reproduced exactly."""
    def names(records):
        return [Path(record["_metadata_path"]).stem for record in records]

    manifest = {
        "seed": seed,
        "ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "train": names(train),
        "validation": names(valid),
        "test": names(test),
    }
    with path.open("w") as file:
        json.dump(manifest, file, indent=2)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/synthetic_rgbd_cnn"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-output", type=Path, default=None,
        help="Test export directory; defaults to <output>/test_split.",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1 or args.max_depth <= 0:
        raise ValueError("batch-size, epochs, and max-depth must be positive")
    if args.l2_regularization < 0 or args.weight_decay < 0:
        raise ValueError("regularization and weight decay must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    cpu_device = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu_device)
    print(f"Training with Keras {keras.backend.backend()} on {cpu_device}")
    records = load_records(args.dataset)
    train_records, valid_records, test_records = split_records(
        records, args.seed
    )
    test_output = args.test_output or args.output / "test_split"
    export_test_split(test_records, test_output)
    save_split_manifest(
        args.output / "split.json", train_records, valid_records,
        test_records, args.seed,
    )
    print(f"Split: {len(train_records)} train, {len(valid_records)} "
          f"validation, {len(test_records)} test")
    print(f"Exported test split to {test_output}")
    normalizer = TargetNormalizer.fit(train_records)
    normalizer.save(args.output / "normalization.json")
    train = RGBDDataset(train_records, normalizer, args.batch_size,
                        args.max_depth, shuffle=True, seed=args.seed)
    valid = RGBDDataset(valid_records, normalizer, args.batch_size,
                        args.max_depth)
    input_shape = train[0][0].shape[1:]
    model = build_model(input_shape, l2_regularization=args.l2_regularization)
    translation_std = normalizer.statistics[
        "object_translation"
    ]["standard_deviation"]
    light_position_std = normalizer.statistics[
        "light_position"
    ]["standard_deviation"]
    light_intensity_std = normalizer.statistics[
        "light_intensity"
    ]["standard_deviation"]
    object_scale_std = normalizer.statistics[
        "object_scale"
    ]["standard_deviation"]
    material_std = normalizer.statistics[
        "material"
    ]["standard_deviation"]
    compile_model(
        model, args.learning_rate, args.weight_decay, translation_std,
        light_position_std, light_intensity_std, object_scale_std,
        material_std,
    )
    model.summary()
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        LossPlot(args.output / "loss.png"),
        keras.callbacks.TerminateOnNaN(),
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
