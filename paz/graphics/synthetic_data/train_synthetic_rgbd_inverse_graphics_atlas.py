"""Train a compact differentiable RGB-D inverse-graphics mesh model.

The default geometry decoder deforms a connected triangle template through a
small enclosing control cage. Fixed connectivity makes mesh extraction
immediate, while surface Chamfer supervision permits arbitrary category names
and does not require vertex correspondence or watertight ground-truth meshes.

By default, a network predicts a small enclosing cage whose mean-value
coordinates deform a fixed, connected triangle template.  The earlier Atlas
UV-patch representation remains available with ``--geometry-representation
atlas`` for controlled comparisons.

Canonical mesh vertices are transformed by predicted pose and metric scale,
then a lightweight differentiable surface-splat renderer produces object RGB,
depth, and silhouette maps. Reconstruction losses therefore update geometry,
pose, material, and lighting jointly without an SDF or Marching Cubes::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.train_synthetic_rgbd_inverse_graphics_atlas \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --output experiments/rgbd_inverse_graphics_cage
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import shutil
from pathlib import Path

import cv2
import jax
import keras
import numpy as np
import trimesh

from paz.backend import cage as cage_backend

from paz.graphics.synthetic_data.train_synthetic_rgb_resnet18 import (
    MATERIAL_NAMES,
    OUTPUT_NAMES,
    PeriodicWeightsCheckpoint,
    TrainingPlot,
    compile_model,
    save_split_manifest,
    split_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    export_test_split,
    load_records,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh import (
    load_camera_mesh_points,
    world_to_camera_matrix,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    GenericTargetNormalizer,
    canonical_to_camera_matrix,
    extract_generic_targets,
    export_test_meshes,
    generic_geodesic_angle_degrees,
    generic_rotation_loss,
)


ATLAS_OUTPUT_NAME = "atlas_vertices"
RENDERED_RGB_OUTPUT_NAME = "rendered_rgb"
RENDERED_DEPTH_OUTPUT_NAME = "rendered_depth"
RENDERED_SILHOUETTE_OUTPUT_NAME = "rendered_silhouette"


def build_atlas_topology(num_patches, patch_resolution):
    """Returns UV+patch coordinates and disconnected grid triangles."""
    axis = np.linspace(-1.0, 1.0, patch_resolution, dtype=np.float32)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    coordinates, faces, edges = [], [], []
    vertices_per_patch = patch_resolution ** 2
    for patch in range(num_patches):
        identity = np.zeros((vertices_per_patch, num_patches), np.float32)
        identity[:, patch] = 1.0
        coordinates.append(np.concatenate([uv, identity], axis=-1))
        offset = patch * vertices_per_patch
        for row in range(patch_resolution - 1):
            for column in range(patch_resolution - 1):
                first = offset + row * patch_resolution + column
                right = first + 1
                below = first + patch_resolution
                diagonal = below + 1
                faces.extend(((first, below, diagonal),
                              (first, diagonal, right)))
                edges.extend(((first, right), (first, below),
                              (first, diagonal), (right, diagonal),
                              (below, diagonal)))
    edges = np.unique(np.sort(np.asarray(edges, np.int32), axis=1), axis=0)
    return (np.concatenate(coordinates).astype(np.float32),
            np.asarray(faces, np.int32), edges)


def mesh_edges(faces):
    """Returns the unique undirected edges of a triangle array."""
    faces = np.asarray(faces, dtype=np.int32)
    edges = np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
    ])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def build_cage_topology(template_subdivisions=2, cage_subdivisions=1,
                        template_radius=1.0, cage_radius=1.25):
    """Builds a sphere template, enclosing cage, and fixed MVC weights."""
    if cage_radius <= template_radius:
        raise ValueError("cage radius must be larger than template radius")
    template = trimesh.creation.icosphere(
        subdivisions=template_subdivisions, radius=template_radius
    )
    cage = trimesh.creation.icosphere(
        subdivisions=cage_subdivisions, radius=cage_radius
    )
    template_vertices = np.asarray(template.vertices, dtype=np.float32)
    template_faces = np.asarray(template.faces, dtype=np.int32)
    cage_vertices = np.asarray(cage.vertices, dtype=np.float32)
    cage_faces = np.asarray(cage.faces, dtype=np.int32)
    weights = np.asarray(cage_backend.compute_mesh_weights(
        template_vertices, cage_vertices, cage_faces
    ), dtype=np.float32)
    totals = weights.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(weights)) or np.any(np.abs(totals) < 1e-8):
        raise ValueError("mean-value cage weights are not finite")
    weights = weights / totals
    return (template_vertices, template_faces, mesh_edges(template_faces),
            cage_vertices, cage_faces, weights)


@keras.saving.register_keras_serializable(package="paz")
class RepeatAtlasCoordinates(keras.layers.Layer):
    """Repeats a global feature and appends fixed UV/patch coordinates."""

    def __init__(self, coordinates, **kwargs):
        super().__init__(**kwargs)
        self.coordinates = np.asarray(coordinates, dtype=np.float32)

    def call(self, features):
        batch = keras.ops.shape(features)[0]
        count = len(self.coordinates)
        repeated = keras.ops.repeat(features[:, None, :], count, axis=1)
        coordinates = keras.ops.convert_to_tensor(
            self.coordinates, dtype=features.dtype
        )
        coordinates = keras.ops.broadcast_to(
            coordinates[None], (batch, count, self.coordinates.shape[1])
        )
        return keras.ops.concatenate([repeated, coordinates], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.coordinates),
                input_shape[-1] + self.coordinates.shape[-1])

    def get_config(self):
        config = super().get_config()
        config["coordinates"] = self.coordinates.tolist()
        return config


@keras.saving.register_keras_serializable(package="paz")
class CageMeshDeformation(keras.layers.Layer):
    """Maps bounded cage offsets to a dense mesh using fixed MVC weights."""

    def __init__(self, cage_vertices, mesh_weights, mesh_mean, mesh_deviation,
                 maximum_offset=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cage_vertices = np.asarray(cage_vertices, dtype=np.float32)
        self.mesh_weights = np.asarray(mesh_weights, dtype=np.float32)
        self.mesh_mean = np.asarray(mesh_mean, dtype=np.float32)
        self.mesh_deviation = np.asarray(mesh_deviation, dtype=np.float32)
        self.maximum_offset = float(maximum_offset)

    def call(self, raw_offsets):
        batch = keras.ops.shape(raw_offsets)[0]
        offsets = keras.ops.reshape(
            raw_offsets, (batch, len(self.cage_vertices), 3)
        )
        offsets = self.maximum_offset * keras.ops.tanh(offsets)
        cage = keras.ops.convert_to_tensor(
            self.cage_vertices, dtype=raw_offsets.dtype
        )[None] + offsets
        weights = keras.ops.convert_to_tensor(
            self.mesh_weights, dtype=raw_offsets.dtype
        )
        vertices = keras.ops.einsum("vc,bcd->bvd", weights, cage)
        mean = keras.ops.convert_to_tensor(
            self.mesh_mean, dtype=raw_offsets.dtype
        )
        deviation = keras.ops.convert_to_tensor(
            self.mesh_deviation, dtype=raw_offsets.dtype
        )
        return (vertices - mean) / deviation

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.mesh_weights), 3)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cage_vertices": self.cage_vertices.tolist(),
            "mesh_weights": self.mesh_weights.tolist(),
            "mesh_mean": self.mesh_mean.tolist(),
            "mesh_deviation": self.mesh_deviation.tolist(),
            "maximum_offset": self.maximum_offset,
        })
        return config


@keras.saving.register_keras_serializable(package="paz")
class PhysicalAtlasLoss(keras.losses.Loss):
    """Physical Chamfer distance plus inexpensive triangle regularization."""

    def __init__(self, standard_deviation, edges,
                 edge_uniformity_weight=0.02,
                 name="physical_atlas_chamfer_m2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(x) for x in standard_deviation)
        self.edges = np.asarray(edges, dtype=np.int32)
        self.edge_uniformity_weight = float(edge_uniformity_weight)

    def call(self, target, prediction):
        scale = keras.ops.convert_to_tensor(
            self.standard_deviation, dtype=prediction.dtype
        )
        differences = (
            prediction[:, :, None, :] - target[:, None, :, :]
        ) * scale
        squared = keras.ops.sum(keras.ops.square(differences), axis=-1)
        chamfer = 0.5 * (
            keras.ops.mean(keras.ops.min(squared, axis=2), axis=1)
            + keras.ops.mean(keras.ops.min(squared, axis=1), axis=1)
        )
        edge_indices = keras.ops.convert_to_tensor(self.edges, dtype="int32")
        physical = prediction * scale
        first = keras.ops.take(physical, edge_indices[:, 0], axis=1)
        second = keras.ops.take(physical, edge_indices[:, 1], axis=1)
        lengths = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(first - second), axis=-1) + 1e-8
        )
        mean_length = keras.ops.mean(lengths, axis=1, keepdims=True)
        uniformity = keras.ops.mean(
            keras.ops.square(lengths - mean_length), axis=1
        )
        return chamfer + self.edge_uniformity_weight * uniformity

    def get_config(self):
        config = super().get_config()
        config.update({
            "standard_deviation": self.standard_deviation,
            "edges": self.edges.tolist(),
            "edge_uniformity_weight": self.edge_uniformity_weight,
        })
        return config


@keras.saving.register_keras_serializable(package="paz")
class MaskedReconstructionHuber(keras.losses.Loss):
    """Huber loss normalized by non-background target elements."""

    def __init__(self, delta=0.1, name="masked_reconstruction_huber", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = float(delta)

    def call(self, target, prediction):
        if target.shape[-1] == 3:
            mask = keras.ops.max(keras.ops.abs(target), axis=-1, keepdims=True)
            mask = keras.ops.cast(mask > 1e-6, target.dtype)
        else:
            mask = keras.ops.cast(target > 1e-6, target.dtype)
        difference = keras.ops.abs(prediction - target)
        quadratic = keras.ops.minimum(difference, self.delta)
        linear = difference - quadratic
        loss = 0.5 * keras.ops.square(quadratic) \
            + self.delta * linear
        loss = keras.ops.mean(loss, axis=-1, keepdims=True) * mask
        axes = tuple(range(1, len(target.shape)))
        return keras.ops.sum(loss, axis=axes) / keras.ops.maximum(
            keras.ops.sum(mask, axis=axes), 1.0
        )

    def get_config(self):
        config = super().get_config()
        config["delta"] = self.delta
        return config


@keras.saving.register_keras_serializable(package="paz")
class SilhouetteBCEDiceLoss(keras.losses.Loss):
    """Combines pixel BCE with overlap loss for small object silhouettes."""

    def __init__(self, epsilon=1e-6, name="silhouette_bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = float(epsilon)

    def call(self, target, prediction):
        prediction = keras.ops.clip(prediction, self.epsilon, 1.0-self.epsilon)
        bce = -(target * keras.ops.log(prediction)
                + (1.0 - target) * keras.ops.log(1.0 - prediction))
        axes = tuple(range(1, len(target.shape)))
        intersection = keras.ops.sum(target * prediction, axis=axes)
        denominator = keras.ops.sum(target + prediction, axis=axes)
        dice_loss = 1.0 - (
            2.0 * intersection + self.epsilon
        ) / (denominator + self.epsilon)
        return keras.ops.mean(bce, axis=axes) + dice_loss

    def get_config(self):
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config


def transform_points(points, matrix):
    homogeneous = np.concatenate([
        points, np.ones((len(points), 1), dtype=np.float64)
    ], axis=-1)
    return (homogeneous @ np.asarray(matrix).T)[:, :3]


def load_canonical_mesh_points(record, num_points):
    """Samples an arbitrary world mesh and maps it to canonical coordinates."""
    camera_points = load_camera_mesh_points(record, num_points)
    camera_to_canonical = np.linalg.inv(canonical_to_camera_matrix(record))
    return transform_points(
        camera_points, camera_to_canonical
    ).astype(np.float32)


def fit_canonical_statistics(records, num_points):
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    for record in records:
        points = load_canonical_mesh_points(record, num_points)
        count += len(points)
        total += points.sum(axis=0)
        total_squared += np.square(points).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    deviation = np.sqrt(variance)
    deviation = np.where(deviation < 1e-6, 1.0, deviation)
    return {"mean": mean.tolist(), "standard_deviation": deviation.tolist()}


def camera_ray_directions(height, width, y_fov_degrees):
    y_fov = np.deg2rad(y_fov_degrees)
    aspect = width / height
    height_world = 2.0 * np.tan(y_fov / 2.0)
    width_world = height_world * aspect
    pixel_size = width_world / width
    x_offset = (np.arange(width) + 0.5) * pixel_size
    y_offset = (np.arange(height) + 0.5) * pixel_size
    x_grid, y_grid = np.meshgrid(
        (width_world / 2.0) - x_offset,
        (height_world / 2.0) - y_offset,
    )
    directions = np.stack([-x_grid, y_grid, -np.ones_like(x_grid)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def object_render_targets(record, rgb, depth, resolution, max_depth, y_fov,
                          floor_tolerance):
    """Builds object-only RGB, z-depth, and silhouette targets."""
    directions = camera_ray_directions(*depth.shape, y_fov)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    ray_distance = depth / np.maximum(-directions[..., 2], 1e-8)
    camera_points = directions * ray_distance[..., None]
    homogeneous = np.concatenate([
        camera_points,
        np.ones((*camera_points.shape[:2], 1), dtype=np.float64),
    ], axis=-1)
    camera_to_world = np.linalg.inv(world_to_camera_matrix(record["camera"]))
    world_points = homogeneous @ camera_to_world.T
    mask = valid & (world_points[..., 1] > floor_tolerance)
    size = (resolution, resolution)
    rgb_small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(
        mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST
    )[..., None]
    return (
        rgb_small.astype(np.float32) * mask_small,
        depth_small.astype(np.float32)[..., None] * mask_small,
        mask_small.astype(np.float32),
    )


class CompactAtlasDataset(keras.utils.PyDataset):
    """Loads normalized RGB-D and arbitrary-mesh surface supervision."""

    def __init__(self, records, normalizer, shape_names, batch_size,
                 max_depth, num_vertices, mesh_statistics,
                 render_resolution, y_fov, floor_tolerance,
                 shuffle=False, seed=0):
        super().__init__()
        self.records = list(records)
        self.normalizer = normalizer
        self.shape_names = tuple(shape_names)
        self.batch_size = int(batch_size)
        self.max_depth = float(max_depth)
        self.mesh_mean = np.asarray(mesh_statistics["mean"], np.float32)
        self.mesh_std = np.asarray(
            mesh_statistics["standard_deviation"], np.float32
        )
        self.render_resolution = int(render_resolution)
        self.y_fov = float(y_fov)
        self.floor_tolerance = float(floor_tolerance)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        self.mesh_targets = np.stack([
            load_canonical_mesh_points(record, num_vertices)
            for record in self.records
        ])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        rgb_images, depth_images, targets = [], [], []
        rendered_rgb, rendered_depth, rendered_silhouette = [], [], []
        for index in indices:
            record = self.records[index]
            root = Path(record["_root"])
            bgr = cv2.imread(str(root / record["rgb"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(root / record["rgb"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb.astype(np.float32) / 255.0)
            depth = np.load(root / record["depth"]).astype(np.float32)
            depth = np.where(
                np.isfinite(depth) & (depth > 0.0) &
                (depth <= self.max_depth), depth, 0.0,
            )
            depth_images.append((depth / self.max_depth)[..., None])
            render_targets = object_render_targets(
                record, rgb_images[-1], depth, self.render_resolution,
                self.max_depth, self.y_fov, self.floor_tolerance,
            )
            rendered_rgb.append(render_targets[0])
            rendered_depth.append(render_targets[1])
            rendered_silhouette.append(render_targets[2])
            targets.append(self.normalizer.normalize(
                extract_generic_targets(record, self.shape_names)
            ))
        inputs = {
            "rgb": np.stack(rgb_images),
            "depth": np.stack(depth_images),
        }
        outputs = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        vertices = self.mesh_targets[indices]
        outputs[ATLAS_OUTPUT_NAME] = (
            vertices - self.mesh_mean
        ) / self.mesh_std
        outputs[RENDERED_RGB_OUTPUT_NAME] = np.stack(rendered_rgb)
        outputs[RENDERED_DEPTH_OUTPUT_NAME] = np.stack(rendered_depth)
        outputs[RENDERED_SILHOUETTE_OUTPUT_NAME] = np.stack(
            rendered_silhouette
        )
        return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def compact_encoder(rgb, depth, regularizer, base_width):
    features = keras.layers.Concatenate(name="rgbd_concatenation")([rgb, depth])
    for index, width in enumerate([
            base_width, base_width * 2, base_width * 4, base_width * 8], 1):
        features = keras.layers.Conv2D(
            width, 3, strides=2, padding="same", use_bias=False,
            kernel_regularizer=regularizer, name=f"compact_conv_{index}",
        )(features)
        features = keras.layers.BatchNormalization(
            name=f"compact_bn_{index}"
        )(features)
        features = keras.layers.Activation(
            "relu", name=f"compact_relu_{index}"
        )(features)
    features = keras.layers.GlobalAveragePooling2D(
        name="compact_global_pool"
    )(features)
    return keras.layers.Dense(
        256, activation="relu", kernel_regularizer=regularizer,
        name="compact_global_feature",
    )(features)


@keras.saving.register_keras_serializable(package="paz")
class DifferentiableAtlasRenderer(keras.layers.Layer):
    """Soft-rasterizes shaded triangle centroids to RGB-D and silhouette."""

    def __init__(self, faces, render_resolution, y_fov_degrees,
                 target_statistics, mesh_statistics, splat_sigma=1.2,
                 depth_temperature=0.15, splat_strength=2.0, **kwargs):
        super().__init__(**kwargs)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.render_resolution = int(render_resolution)
        self.y_fov_degrees = float(y_fov_degrees)
        self.target_statistics = target_statistics
        self.mesh_statistics = mesh_statistics
        self.splat_sigma = float(splat_sigma)
        self.depth_temperature = float(depth_temperature)
        self.splat_strength = float(splat_strength)
        rows, columns = np.meshgrid(
            np.arange(self.render_resolution, dtype=np.float32),
            np.arange(self.render_resolution, dtype=np.float32),
            indexing="ij",
        )
        self.pixels = np.stack([rows, columns], axis=-1).reshape(-1, 2)

    def denormalize(self, values, name):
        statistics = self.target_statistics[name]
        mean = keras.ops.convert_to_tensor(
            statistics["mean"], dtype=values.dtype
        )
        deviation = keras.ops.convert_to_tensor(
            statistics["standard_deviation"], dtype=values.dtype
        )
        return values * deviation + mean

    @staticmethod
    def normalize(values, epsilon=1e-8):
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(values), axis=-1, keepdims=True)
            + epsilon
        )
        return values / norm

    @staticmethod
    def cross(first, second):
        return keras.ops.stack([
            first[..., 1] * second[..., 2]
            - first[..., 2] * second[..., 1],
            first[..., 2] * second[..., 0]
            - first[..., 0] * second[..., 2],
            first[..., 0] * second[..., 1]
            - first[..., 1] * second[..., 0],
        ], axis=-1)

    def call(self, inputs):
        (canonical, translation_raw, orientation, scale_raw, material_raw,
         light_position_raw, light_intensity_raw) = inputs
        mesh_mean = keras.ops.convert_to_tensor(
            self.mesh_statistics["mean"], dtype=canonical.dtype
        )
        mesh_deviation = keras.ops.convert_to_tensor(
            self.mesh_statistics["standard_deviation"],
            dtype=canonical.dtype,
        )
        canonical = canonical * mesh_deviation + mesh_mean
        translation = self.denormalize(translation_raw, "object_translation")
        scale = keras.ops.maximum(
            self.denormalize(scale_raw, "object_scale"), 1e-3
        )
        first = self.normalize(orientation[..., :3])
        second = orientation[..., 3:6]
        second = second - keras.ops.sum(
            first * second, axis=-1, keepdims=True
        ) * first
        second = self.normalize(second)
        third = self.cross(first, second)
        rotation = keras.ops.stack([first, second, third], axis=-1)
        vertices = keras.ops.matmul(
            canonical, keras.ops.transpose(rotation, (0, 2, 1))
        )
        vertices = vertices * scale[:, None, :] + translation[:, None, :]

        faces = keras.ops.convert_to_tensor(self.faces, dtype="int32")
        triangle_first = keras.ops.take(vertices, faces[:, 0], axis=1)
        triangle_second = keras.ops.take(vertices, faces[:, 1], axis=1)
        triangle_third = keras.ops.take(vertices, faces[:, 2], axis=1)
        samples = (triangle_first + triangle_second + triangle_third) / 3.0
        normals = self.normalize(self.cross(
            triangle_second - triangle_first,
            triangle_third - triangle_first,
        ))

        material = self.denormalize(material_raw, "material")
        color = keras.ops.clip(material[..., :3], 0.0, 1.0)
        diffuse = keras.ops.clip(material[..., 3:4], 0.0, 1.0)
        specular = keras.ops.clip(material[..., 4:5], 0.0, 1.0)
        ambient = keras.ops.clip(material[..., 5:6], 0.0, 1.0)
        shininess = keras.ops.clip(material[..., 6:7], 1.0, 128.0)
        light_position = self.denormalize(
            light_position_raw, "light_position"
        )
        light_intensity = keras.ops.clip(
            self.denormalize(light_intensity_raw, "light_intensity"),
            0.0, 5.0,
        )
        light_direction = self.normalize(
            light_position[:, None, :] - samples
        )
        view_direction = self.normalize(-samples)
        half_direction = self.normalize(light_direction + view_direction)
        # Atlas patches may learn either winding, so use two-sided shading.
        lambert = keras.ops.abs(keras.ops.sum(
            normals * light_direction, axis=-1, keepdims=True
        ))
        highlight = keras.ops.power(
            keras.ops.clip(keras.ops.abs(keras.ops.sum(
                normals * half_direction, axis=-1, keepdims=True
            )), 0.0, 1.0),
            shininess[:, None, :],
        )
        sample_colors = keras.ops.clip(
            ambient[:, None, :] * color[:, None, :]
            + light_intensity[:, None, :] * (
                diffuse[:, None, :] * lambert * color[:, None, :]
                + specular[:, None, :] * highlight
            ),
            0.0, 1.0,
        )

        depth = -samples[..., 2]
        valid = depth > 1e-4
        y_fov = np.deg2rad(self.y_fov_degrees)
        height_world = 2.0 * np.tan(y_fov / 2.0)
        width_world = height_world
        x_grid = samples[..., 0] / keras.ops.minimum(
            samples[..., 2], -1e-4
        )
        y_grid = samples[..., 1] / keras.ops.maximum(depth, 1e-4)
        columns = (width_world / 2.0 - x_grid) \
            * self.render_resolution / width_world
        rows = (height_world / 2.0 - y_grid) \
            * self.render_resolution / height_world
        projected = keras.ops.stack([rows, columns], axis=-1)
        pixels = keras.ops.convert_to_tensor(
            self.pixels, dtype=projected.dtype
        )
        squared_pixel_distance = keras.ops.sum(keras.ops.square(
            projected[:, :, None, :] - pixels[None, None, :, :]
        ), axis=-1)
        gaussian = keras.ops.exp(
            -squared_pixel_distance / (2.0 * self.splat_sigma ** 2)
        ) * keras.ops.cast(valid[..., None], projected.dtype)
        silhouette = 1.0 - keras.ops.exp(
            -self.splat_strength * keras.ops.sum(
                gaussian, axis=1, keepdims=False
            )
        )
        logits = (
            -squared_pixel_distance / (2.0 * self.splat_sigma ** 2)
            - depth[..., None] / self.depth_temperature
        )
        logits = logits - keras.ops.max(logits, axis=1, keepdims=True)
        visibility = keras.ops.exp(logits) * gaussian
        visibility = visibility / keras.ops.maximum(
            keras.ops.sum(visibility, axis=1, keepdims=True), 1e-8
        )
        rendered_depth = keras.ops.sum(
            visibility * depth[..., None], axis=1
        ) * silhouette
        rendered_rgb = keras.ops.sum(
            visibility[..., None] * sample_colors[:, :, None, :], axis=1
        ) * silhouette[..., None]
        shape = (-1, self.render_resolution, self.render_resolution)
        rendered_rgb = keras.ops.reshape(rendered_rgb, shape + (3,))
        rendered_depth = keras.ops.reshape(rendered_depth, shape + (1,))
        silhouette = keras.ops.reshape(silhouette, shape + (1,))
        return rendered_rgb, rendered_depth, silhouette

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        spatial = (batch, self.render_resolution, self.render_resolution)
        return spatial + (3,), spatial + (1,), spatial + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "faces": self.faces.tolist(),
            "render_resolution": self.render_resolution,
            "y_fov_degrees": self.y_fov_degrees,
            "target_statistics": self.target_statistics,
            "mesh_statistics": self.mesh_statistics,
            "splat_sigma": self.splat_sigma,
            "depth_temperature": self.depth_temperature,
            "splat_strength": self.splat_strength,
        })
        return config


def build_model(image_shape, num_shapes, coordinates, base_width,
                l2_regularization, faces, render_resolution, y_fov,
                target_statistics, mesh_statistics, splat_sigma,
                depth_temperature, geometry_representation="cage",
                cage_vertices=None, mesh_weights=None,
                cage_maximum_offset=1.0):
    """Builds a compact physical-parameter and deformable-mesh network."""
    regularizer = keras.regularizers.L2(l2_regularization)
    rgb = keras.Input((*image_shape, 3), name="rgb")
    depth = keras.Input((*image_shape, 1), name="depth")
    features = compact_encoder(rgb, depth, regularizer, base_width)
    geometry = keras.layers.Dense(
        192, activation="relu", kernel_regularizer=regularizer,
        name="geometry_mlp",
    )(features)
    material = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=regularizer,
        name="material_mlp",
    )(features)
    lighting = keras.layers.Dense(
        128, activation="relu", kernel_regularizer=regularizer,
        name="lighting_mlp",
    )(features)
    outputs = {
        "object_translation": keras.layers.Dense(
            3, name="object_translation")(geometry),
        "object_orientation_6d": keras.layers.Dense(
            6, name="object_orientation_6d")(geometry),
        "object_scale": keras.layers.Dense(1, name="object_scale")(geometry),
        "shape": keras.layers.Dense(
            num_shapes, activation="softmax", name="shape")(geometry),
        "material": keras.layers.Dense(
            len(MATERIAL_NAMES), name="material")(material),
        "light_position": keras.layers.Dense(
            3, name="light_position")(lighting),
        "light_intensity": keras.layers.Dense(
            1, name="light_intensity")(lighting),
    }
    if geometry_representation == "atlas":
        mesh = RepeatAtlasCoordinates(
            coordinates, name="repeat_atlas_coordinates"
        )(geometry)
        for index, width in enumerate((192, 128, 64), 1):
            mesh = keras.layers.Dense(
                width, activation="relu", kernel_regularizer=regularizer,
                name=f"atlas_decoder_{index}",
            )(mesh)
        outputs[ATLAS_OUTPUT_NAME] = keras.layers.Dense(
            3, name=ATLAS_OUTPUT_NAME, kernel_regularizer=regularizer
        )(mesh)
    elif geometry_representation == "cage":
        if cage_vertices is None or mesh_weights is None:
            raise ValueError("cage vertices and MVC weights are required")
        cage_feature = keras.layers.Dense(
            192, activation="relu", kernel_regularizer=regularizer,
            name="cage_decoder_1",
        )(geometry)
        cage_feature = keras.layers.Dense(
            128, activation="relu", kernel_regularizer=regularizer,
            name="cage_decoder_2",
        )(cage_feature)
        raw_offsets = keras.layers.Dense(
            len(cage_vertices) * 3, kernel_initializer="zeros",
            bias_initializer="zeros", kernel_regularizer=regularizer,
            name="cage_vertex_offsets",
        )(cage_feature)
        outputs[ATLAS_OUTPUT_NAME] = CageMeshDeformation(
            cage_vertices, mesh_weights, mesh_statistics["mean"],
            mesh_statistics["standard_deviation"], cage_maximum_offset,
            name=ATLAS_OUTPUT_NAME,
        )(raw_offsets)
    else:
        raise ValueError(
            f"unknown geometry representation: {geometry_representation}"
        )
    rendered = DifferentiableAtlasRenderer(
        faces, render_resolution, y_fov, target_statistics, mesh_statistics,
        splat_sigma=splat_sigma, depth_temperature=depth_temperature,
        name="differentiable_atlas_renderer",
    )([
        outputs[ATLAS_OUTPUT_NAME], outputs["object_translation"],
        outputs["object_orientation_6d"], outputs["object_scale"],
        outputs["material"], outputs["light_position"],
        outputs["light_intensity"],
    ])
    outputs[RENDERED_RGB_OUTPUT_NAME] = keras.layers.Activation(
        "linear", name=RENDERED_RGB_OUTPUT_NAME
    )(rendered[0])
    outputs[RENDERED_DEPTH_OUTPUT_NAME] = keras.layers.Activation(
        "linear", name=RENDERED_DEPTH_OUTPUT_NAME
    )(rendered[1])
    outputs[RENDERED_SILHOUETTE_OUTPUT_NAME] = keras.layers.Activation(
        "linear", name=RENDERED_SILHOUETTE_OUTPUT_NAME
    )(rendered[2])
    return keras.Model(
        {"rgb": rgb, "depth": depth}, outputs,
        name=f"compact_rgbd_{geometry_representation}_mesh",
    )


def denormalize_numpy(values, statistics):
    return (np.asarray(values) * statistics["standard_deviation"]
            + statistics["mean"])


def rotation_6d_to_matrix(values, epsilon=1e-8):
    first, second = np.asarray(values[:3]), np.asarray(values[3:6])
    first = first / max(float(np.linalg.norm(first)), epsilon)
    second = second - np.dot(first, second) * first
    second = second / max(float(np.linalg.norm(second)), epsilon)
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def save_preview_meshes(model, dataset, records, mesh_statistics,
                        target_statistics, faces, output, count):
    canonical_output = output / "canonical"
    camera_output = output / "camera"
    canonical_output.mkdir(parents=True, exist_ok=True)
    camera_output.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(mesh_statistics["mean"], np.float32)
    deviation = np.asarray(
        mesh_statistics["standard_deviation"], np.float32
    )
    saved = 0
    for batch_index in range(len(dataset)):
        inputs, _ = dataset[batch_index]
        predictions = model.predict(inputs, verbose=0)
        begin = batch_index * dataset.batch_size
        for offset, vertices in enumerate(predictions[ATLAS_OUTPUT_NAME]):
            if saved >= count:
                return
            record = records[begin + offset]
            sample_id = Path(record["_metadata_path"]).stem
            canonical = vertices * deviation + mean
            rotation = rotation_6d_to_matrix(
                predictions["object_orientation_6d"][offset]
            )
            translation = denormalize_numpy(
                predictions["object_translation"][offset],
                target_statistics["object_translation"],
            )
            scale = max(float(denormalize_numpy(
                predictions["object_scale"][offset],
                target_statistics["object_scale"],
            )[0]), 1e-3)
            camera = scale * (canonical @ rotation.T) + translation
            trimesh.Trimesh(canonical, faces, process=False).export(
                canonical_output / f"{sample_id}.ply"
            )
            trimesh.Trimesh(camera, faces, process=False).export(
                camera_output / f"{sample_id}.ply"
            )
            saved += 1


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/rgbd_inverse_graphics_cage"))
    parser.add_argument("--num-patches", type=int, default=4)
    parser.add_argument("--patch-resolution", type=int, default=8)
    parser.add_argument("--geometry-representation", choices=("cage", "atlas"),
                        default="cage")
    parser.add_argument("--template-subdivisions", type=int, default=2)
    parser.add_argument("--cage-subdivisions", type=int, default=1)
    parser.add_argument("--template-radius", type=float, default=1.0)
    parser.add_argument("--cage-radius", type=float, default=1.25)
    parser.add_argument("--cage-maximum-offset", type=float, default=1.0)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--mesh-loss-weight", type=float, default=2.0)
    parser.add_argument("--edge-uniformity-weight", type=float, default=0.02)
    parser.add_argument("--render-resolution", type=int, default=32)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--render-floor-tolerance", type=float, default=0.01)
    parser.add_argument("--splat-sigma", type=float, default=1.2)
    parser.add_argument("--depth-temperature", type=float, default=0.15)
    parser.add_argument("--rgb-reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--depth-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--silhouette-reconstruction-weight", type=float,
                        default=1.0)
    parser.add_argument("--preview-meshes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--l2-regularization", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.num_patches, args.patch_resolution, args.base_width,
        args.mesh_loss_weight, args.batch_size, args.epochs, args.max_depth,
        args.checkpoint_every, args.render_resolution, args.y_fov,
        args.splat_sigma, args.depth_temperature, args.template_radius,
        args.cage_radius, args.cage_maximum_offset,
    )
    if any(value <= 0 for value in positive) or args.preview_meshes < 0:
        raise ValueError("sizes, weights, and ranges must be positive")
    if args.patch_resolution < 2:
        raise ValueError("--patch-resolution must be at least two")
    if args.template_subdivisions < 0 or args.cage_subdivisions < 0:
        raise ValueError("subdivision counts cannot be negative")
    if args.edge_uniformity_weight < 0.0:
        raise ValueError("--edge-uniformity-weight cannot be negative")
    reconstruction_weights = (
        args.rgb_reconstruction_weight, args.depth_reconstruction_weight,
        args.silhouette_reconstruction_weight,
    )
    if min(reconstruction_weights) < 0.0:
        raise ValueError("reconstruction weights cannot be negative")
    if args.render_floor_tolerance < 0.0:
        raise ValueError("--render-floor-tolerance cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)

    devices = jax.devices("cuda")
    device = devices[0]
    jax.config.update("jax_default_device", device)
    records = load_records(args.dataset)
    shape_names = tuple(sorted({record["shape"]["type"] for record in records}))
    training_records, validation_records, test_records = split_records(
        records, args.seed, args.validation_split
    )
    export_test_split(test_records, args.output / "test_split")
    export_test_meshes(test_records, args.output / "test_split")
    save_split_manifest(
        args.output / "split.json", training_records, validation_records,
        test_records, args.seed, args.validation_split,
    )

    cage_vertices = cage_faces = mesh_weights = None
    if args.geometry_representation == "cage":
        (coordinates, faces, edges, cage_vertices, cage_faces,
         mesh_weights) = build_cage_topology(
            args.template_subdivisions, args.cage_subdivisions,
            args.template_radius, args.cage_radius,
        )
    else:
        coordinates, faces, edges = build_atlas_topology(
            args.num_patches, args.patch_resolution
        )
    num_vertices = len(coordinates)
    normalizer = GenericTargetNormalizer.fit(training_records, shape_names)
    mesh_statistics = fit_canonical_statistics(
        training_records, num_vertices
    )
    normalizer.statistics[ATLAS_OUTPUT_NAME] = mesh_statistics
    normalizer.save(args.output / "normalization.json")
    topology = {"coordinates": coordinates, "faces": faces, "edges": edges}
    if args.geometry_representation == "cage":
        topology.update({
            "cage_vertices": cage_vertices,
            "cage_faces": cage_faces,
            "mesh_weights": mesh_weights,
        })
    # Keep the historical filename so the existing evaluation renderer can
    # consume both Atlas and cage experiments.
    np.savez_compressed(args.output / "atlas_topology.npz", **topology)
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "method": f"compact {args.geometry_representation} mesh",
            "mesh_frame": "canonical transformed by predicted pose and scale",
            "mesh_representation": (
                "mean-value-coordinate cage deformation"
                if args.geometry_representation == "cage" else
                "direct vertices with fixed patch faces"
            ),
            "geometry_representation": args.geometry_representation,
            "num_patches": args.num_patches,
            "patch_resolution": args.patch_resolution,
            "template_subdivisions": args.template_subdivisions,
            "cage_subdivisions": args.cage_subdivisions,
            "template_radius": args.template_radius,
            "cage_radius": args.cage_radius,
            "cage_maximum_offset": args.cage_maximum_offset,
            "num_vertices": num_vertices,
            "num_faces": len(faces),
            "max_depth": args.max_depth,
            "y_fov_degrees": args.y_fov,
            "training_render_resolution": args.render_resolution,
            "differentiable_renderer": "soft shaded triangle-centroid splats",
            "render_floor_tolerance_metres": args.render_floor_tolerance,
            "splat_sigma_pixels": args.splat_sigma,
            "depth_temperature_metres": args.depth_temperature,
            "rgb_reconstruction_weight": args.rgb_reconstruction_weight,
            "depth_reconstruction_weight": args.depth_reconstruction_weight,
            "silhouette_reconstruction_weight": (
                args.silhouette_reconstruction_weight
            ),
            "rgb_range": [0.0, 1.0],
            "depth_range": [0.0, 1.0],
            "requires_watertight_meshes": False,
            "uses_sdf": False,
            "uses_marching_cubes": False,
        }, file, indent=2)

    common = {
        "normalizer": normalizer,
        "shape_names": shape_names,
        "batch_size": args.batch_size,
        "max_depth": args.max_depth,
        "num_vertices": num_vertices,
        "mesh_statistics": mesh_statistics,
        "render_resolution": args.render_resolution,
        "y_fov": args.y_fov,
        "floor_tolerance": args.render_floor_tolerance,
    }
    training = CompactAtlasDataset(
        training_records, shuffle=True, seed=args.seed, **common
    )
    validation = CompactAtlasDataset(
        validation_records, shuffle=False, seed=args.seed, **common
    )
    test = CompactAtlasDataset(
        test_records, shuffle=False, seed=args.seed, **common
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_model(
        image_shape, len(shape_names), coordinates, args.base_width,
        args.l2_regularization, faces, args.render_resolution, args.y_fov,
        normalizer.statistics, mesh_statistics, args.splat_sigma,
        args.depth_temperature, args.geometry_representation,
        cage_vertices, mesh_weights, args.cage_maximum_offset,
    )
    atlas_loss = PhysicalAtlasLoss(
        mesh_statistics["standard_deviation"], edges,
        args.edge_uniformity_weight,
    )
    rgb_loss = MaskedReconstructionHuber(
        delta=0.1, name="rendered_rgb_masked_huber"
    )
    depth_loss = MaskedReconstructionHuber(
        delta=0.1, name="rendered_metric_depth_masked_huber"
    )
    silhouette_loss = SilhouetteBCEDiceLoss()
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses={
            "object_orientation_6d": generic_rotation_loss,
            ATLAS_OUTPUT_NAME: atlas_loss,
            RENDERED_RGB_OUTPUT_NAME: rgb_loss,
            RENDERED_DEPTH_OUTPUT_NAME: depth_loss,
            RENDERED_SILHOUETTE_OUTPUT_NAME: silhouette_loss,
        },
        extra_metrics={
            "object_orientation_6d": [generic_geodesic_angle_degrees],
            ATLAS_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(atlas_loss, name="loss")
            ],
            RENDERED_RGB_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(rgb_loss, name="loss"),
                keras.metrics.MeanAbsoluteError(name="mae_0_1"),
            ],
            RENDERED_DEPTH_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(depth_loss, name="loss"),
                keras.metrics.MeanAbsoluteError(name="mae_m"),
            ],
            RENDERED_SILHOUETTE_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(
                    silhouette_loss, name="loss"
                ),
                keras.metrics.BinaryIoU(
                    target_class_ids=(1,), threshold=0.5, name="iou"
                ),
            ],
        },
        extra_loss_weights={
            ATLAS_OUTPUT_NAME: args.mesh_loss_weight,
            RENDERED_RGB_OUTPUT_NAME: args.rgb_reconstruction_weight,
            RENDERED_DEPTH_OUTPUT_NAME: args.depth_reconstruction_weight,
            RENDERED_SILHOUETTE_OUTPUT_NAME: (
                args.silhouette_reconstruction_weight
            ),
        },
    )
    callbacks = [
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png", OUTPUT_NAMES + (
                ATLAS_OUTPUT_NAME, RENDERED_RGB_OUTPUT_NAME,
                RENDERED_DEPTH_OUTPUT_NAME,
                RENDERED_SILHOUETTE_OUTPUT_NAME,
            )
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
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6,
            mode="min", verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, start_from_epoch=20,
            restore_best_weights=True, mode="min", verbose=1,
        ),
    ]
    model.summary()
    print(
        f"Training compact {args.geometry_representation} mesh on {device}: "
        f"{len(training_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test; "
        f"{num_vertices} vertices, {len(faces)} faces"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")
    save_preview_meshes(
        model, test, test_records, mesh_statistics, normalizer.statistics,
        faces,
        args.output / f"{args.geometry_representation}_preview_meshes",
        min(args.preview_meshes, len(test_records)),
    )


if __name__ == "__main__":
    main()
