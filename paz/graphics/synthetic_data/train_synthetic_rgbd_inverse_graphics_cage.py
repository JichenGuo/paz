"""Train a compact differentiable RGB-D inverse-graphics mesh model.

The default geometry decoder deforms a connected triangle template through a
small enclosing control cage. Fixed connectivity makes mesh extraction
immediate, while surface Chamfer supervision permits arbitrary category names
and does not require vertex correspondence or watertight ground-truth meshes.

A network predicts a small enclosing cage whose mean-value coordinates deform
a fixed, connected triangle template.

Canonical mesh vertices are transformed by predicted pose and metric scale,
then a lightweight differentiable surface-splat renderer produces object RGB,
depth, and silhouette maps. Reconstruction losses therefore update geometry,
pose, material, and lighting jointly without an SDF or Marching Cubes::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.train_synthetic_rgbd_inverse_graphics_cage \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --output experiments/rgbd_inverse_graphics_cage
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import shutil
from collections import namedtuple
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
    symmetry_geodesic_angle_degrees,
    symmetry_rotation_loss,
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


MESH_OUTPUT_NAME = "mesh_vertices"
CAGE_OUTPUT_NAME = "cage_vertices"
RENDERED_RGB_OUTPUT_NAME = "rendered_rgb"
RENDERED_DEPTH_OUTPUT_NAME = "rendered_depth"
RENDERED_SILHOUETTE_OUTPUT_NAME = "rendered_silhouette"
Objectives = namedtuple("Objectives", [
    "mesh", "mesh_normal", "rotation", "rotation_metric", "cage",
    "losses", "metrics", "weights",
])
CageDecoderArgs = namedtuple("CageDecoderArgs", [
    "coordinates", "vertices", "weights", "mesh_statistics", "max_offset",
    "y_fov", "max_depth", "target_statistics",
])
RenderArgs = namedtuple("RenderArgs", [
    "faces", "resolution", "y_fov", "target_statistics", "mesh_statistics",
    "splat_sigma", "depth_temperature", "splat_strength",
])


def normalize_tensor(values, epsilon=1e-8):
    squared = keras.ops.sum(keras.ops.square(values), axis=-1, keepdims=True)
    return values / keras.ops.maximum(keras.ops.sqrt(squared), epsilon)


def rotation_6d_to_matrix_ops(values):
    first = normalize_tensor(values[..., :3])
    second = values[..., 3:6]
    projection = keras.ops.sum(first * second, axis=-1, keepdims=True)
    second = normalize_tensor(second - projection * first)
    third = keras.ops.cross(first, second)
    return keras.ops.stack([first, second, third], axis=-1)


def denormalize_tensor(values, statistics, name):
    target = statistics[name]
    mean = keras.ops.convert_to_tensor(target["mean"], dtype=values.dtype)
    stdv = keras.ops.convert_to_tensor(
        target["standard_deviation"], dtype=values.dtype
    )
    return values * stdv + mean


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
class RepeatCageCoordinates(keras.layers.Layer):
    """Repeats a global feature and appends fixed cage coordinates."""

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
class CameraGeometryChannels(keras.layers.Layer):
    """Builds pose-aligned geometry channels from object-only depth.

    The input depth is ``metric_depth / max_depth``. Consequently, the
    returned XYZ coordinates are camera-frame points in max-depth units. This
    keeps their numerical range comparable to CNN activations while retaining
    projective geometry.
    """

    def __init__(self, height, width, y_fov_degrees, max_depth,
                 target_statistics, **kwargs):
        super().__init__(**kwargs)
        self.height = int(height)
        self.width = int(width)
        self.y_fov_degrees = float(y_fov_degrees)
        self.max_depth = float(max_depth)
        self.target_statistics = target_statistics
        y_fov = np.deg2rad(self.y_fov_degrees)
        aspect = self.width / self.height
        half_height = np.tan(y_fov / 2.0)
        half_width = aspect * half_height
        x = ((np.arange(self.width, dtype=np.float32) + 0.5)
             / self.width * 2.0 - 1.0)
        y = 1.0 - ((np.arange(self.height, dtype=np.float32) + 0.5)
                   / self.height * 2.0)
        x_grid, y_grid = np.meshgrid(x, y)
        # The renderer convention looks down negative camera Z. Keep both
        # normalized image coordinates and normalized ray directions.
        rays = np.stack([
            x_grid * half_width,
            y_grid * half_height,
            -np.ones_like(x_grid),
        ], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        self.image_coordinates = np.stack([x_grid, y_grid], axis=-1)
        self.rays = rays.astype(np.float32)

    def call(self, inputs):
        (normalized_depth, object_mask, translation_raw, orientation,
         scale_raw) = inputs
        depth = keras.ops.image.resize(
            normalized_depth, (self.height, self.width),
            interpolation="bilinear",
        )
        valid_depth = keras.ops.image.resize(
            keras.ops.cast(normalized_depth > 0.0, normalized_depth.dtype),
            (self.height, self.width), interpolation="nearest",
        )
        object_mask = keras.ops.image.resize(
            object_mask, (self.height, self.width), interpolation="nearest"
        )
        valid = valid_depth * keras.ops.cast(object_mask > 0.5, depth.dtype)
        depth = depth * valid
        rays = keras.ops.convert_to_tensor(self.rays, dtype=depth.dtype)
        rays = keras.ops.broadcast_to(
            rays[None], (keras.ops.shape(depth)[0], self.height, self.width, 3)
        )
        image_coordinates = keras.ops.convert_to_tensor(
            self.image_coordinates, dtype=depth.dtype
        )
        image_coordinates = keras.ops.broadcast_to(
            image_coordinates[None],
            (keras.ops.shape(depth)[0], self.height, self.width, 2),
        )
        ray_distance = depth / keras.ops.maximum(
            -rays[..., 2:3], 1e-6
        )
        camera_points = rays * ray_distance * valid * self.max_depth
        translation = denormalize_tensor(
            translation_raw, self.target_statistics, "object_translation"
        )
        scale = keras.ops.maximum(
            denormalize_tensor(
                scale_raw, self.target_statistics, "object_scale"
            ), 1e-3
        )
        rotation = rotation_6d_to_matrix_ops(orientation)
        centered = camera_points - translation[:, None, None, :]
        canonical_points = keras.ops.einsum(
            "bhwi,bij->bhwj", centered, rotation
        ) / scale[:, None, None, :]
        canonical_points = canonical_points * valid
        return keras.ops.concatenate([
            image_coordinates, rays, camera_points, canonical_points,
            depth, valid,
        ], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.height, self.width, 13)

    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
            "y_fov_degrees": self.y_fov_degrees,
            "max_depth": self.max_depth,
            "target_statistics": self.target_statistics,
        })
        return config


@keras.saving.register_keras_serializable(package="paz")
class CageMeshDeformation(keras.layers.Layer):
    """Maps bounded cage offsets to a dense mesh using fixed MVC weights."""

    def __init__(self, cage_vertices, mesh_weights, mesh_mean, mesh_deviation,
                 maximum_offset=1.0, reference_vertices=None, **kwargs):
        super().__init__(**kwargs)
        self.cage_vertices = np.asarray(cage_vertices, dtype=np.float32)
        self.mesh_weights = np.asarray(mesh_weights, dtype=np.float32)
        if reference_vertices is None:
            reference_vertices = self.mesh_weights @ self.cage_vertices
        self.reference_vertices = np.asarray(
            reference_vertices, dtype=np.float32
        )
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
        reference = keras.ops.convert_to_tensor(
            self.reference_vertices, dtype=raw_offsets.dtype
        )
        # Residual form guarantees exact identity at zero offsets. It is
        # algebraically equivalent to W @ (C0 + dC) only when W @ C0 == V0.
        vertices = reference[None] + keras.ops.einsum(
            "vc,bcd->bvd", weights, offsets
        )
        mean = keras.ops.convert_to_tensor(
            self.mesh_mean, dtype=raw_offsets.dtype
        )
        deviation = keras.ops.convert_to_tensor(
            self.mesh_deviation, dtype=raw_offsets.dtype
        )
        return (vertices - mean) / deviation, cage

    def compute_output_shape(self, input_shape):
        return ((input_shape[0], len(self.mesh_weights), 3),
                (input_shape[0], len(self.cage_vertices), 3))

    def get_config(self):
        config = super().get_config()
        config.update({
            "reference_vertices": self.reference_vertices.tolist(),
            "cage_vertices": self.cage_vertices.tolist(),
            "mesh_weights": self.mesh_weights.tolist(),
            "mesh_mean": self.mesh_mean.tolist(),
            "mesh_deviation": self.mesh_deviation.tolist(),
            "maximum_offset": self.maximum_offset,
        })
        return config


@keras.saving.register_keras_serializable(package="paz")
class CageValidityLoss(keras.losses.Loss):
    """Keeps a predicted cage close, non-collapsed, and consistently wound."""

    def __init__(self, faces, edges, offset_weight=0.1,
                 edge_weight=0.2, orientation_weight=1.0,
                 area_weight=0.5, volume_weight=0.1,
                 minimum_area_ratio=0.15, orientation_margin=0.05,
                 name="cage_validity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.faces = np.asarray(faces, np.int32)
        self.edges = np.asarray(edges, np.int32)
        self.offset_weight = float(offset_weight)
        self.edge_weight = float(edge_weight)
        self.orientation_weight = float(orientation_weight)
        self.area_weight = float(area_weight)
        self.volume_weight = float(volume_weight)
        self.minimum_area_ratio = float(minimum_area_ratio)
        self.orientation_margin = float(orientation_margin)

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

    def call(self, target, prediction):
        faces = keras.ops.convert_to_tensor(self.faces, dtype="int32")
        edges = keras.ops.convert_to_tensor(self.edges, dtype="int32")
        offset = keras.ops.mean(keras.ops.square(prediction - target),
                                axis=(1, 2))
        first = keras.ops.take(prediction, edges[:, 0], axis=1)
        second = keras.ops.take(prediction, edges[:, 1], axis=1)
        target_first = keras.ops.take(target, edges[:, 0], axis=1)
        target_second = keras.ops.take(target, edges[:, 1], axis=1)
        length = keras.ops.sqrt(keras.ops.sum(
            keras.ops.square(first - second), axis=-1) + 1e-8)
        target_length = keras.ops.sqrt(keras.ops.sum(
            keras.ops.square(target_first - target_second), axis=-1) + 1e-8)
        edge = keras.ops.mean(keras.ops.square(
            length / keras.ops.maximum(target_length, 1e-6) - 1.0
        ), axis=1)

        triangle = keras.ops.take(prediction, faces, axis=1)
        target_triangle = keras.ops.take(target, faces, axis=1)
        cross = self.cross(
            triangle[:, :, 1] - triangle[:, :, 0],
            triangle[:, :, 2] - triangle[:, :, 0],
        )
        target_cross = self.cross(
            target_triangle[:, :, 1] - target_triangle[:, :, 0],
            target_triangle[:, :, 2] - target_triangle[:, :, 0],
        )
        area = keras.ops.sqrt(keras.ops.sum(
            keras.ops.square(cross), axis=-1) + 1e-8)
        target_area = keras.ops.sqrt(keras.ops.sum(
            keras.ops.square(target_cross), axis=-1) + 1e-8)
        area_ratio = area / keras.ops.maximum(target_area, 1e-6)
        area_barrier = keras.ops.mean(keras.ops.square(keras.ops.maximum(
            self.minimum_area_ratio - area_ratio, 0.0
        )), axis=1)
        center = keras.ops.mean(prediction, axis=1, keepdims=True)
        face_center = keras.ops.mean(triangle, axis=2)
        target_center = keras.ops.mean(target, axis=1, keepdims=True)
        target_face_center = keras.ops.mean(target_triangle, axis=2)
        signed = keras.ops.sum(cross * (face_center - center), axis=-1)
        target_signed = keras.ops.sum(
            target_cross * (target_face_center - target_center), axis=-1
        )
        signed_ratio = signed / keras.ops.maximum(
            keras.ops.abs(target_signed), 1e-6
        )
        orientation = keras.ops.mean(keras.ops.square(keras.ops.maximum(
            self.orientation_margin - signed_ratio, 0.0
        )), axis=1)
        volume = keras.ops.sum(keras.ops.sum(
            triangle[:, :, 0] * self.cross(
                triangle[:, :, 1], triangle[:, :, 2]
            ), axis=-1
        ), axis=1) / 6.0
        target_volume = keras.ops.sum(keras.ops.sum(
            target_triangle[:, :, 0] * self.cross(
                target_triangle[:, :, 1], target_triangle[:, :, 2]
            ), axis=-1
        ), axis=1) / 6.0
        volume = keras.ops.square(
            volume / keras.ops.maximum(keras.ops.abs(target_volume), 1e-6)
            - keras.ops.sign(target_volume)
        )
        return (self.offset_weight * offset + self.edge_weight * edge
                + self.orientation_weight * orientation
                + self.area_weight * area_barrier
                + self.volume_weight * volume)

    def get_config(self):
        config = super().get_config()
        config.update({
            "faces": self.faces.tolist(), "edges": self.edges.tolist(),
            "offset_weight": self.offset_weight,
            "edge_weight": self.edge_weight,
            "orientation_weight": self.orientation_weight,
            "area_weight": self.area_weight,
            "volume_weight": self.volume_weight,
            "minimum_area_ratio": self.minimum_area_ratio,
            "orientation_margin": self.orientation_margin,
        })
        return config


def cage_validity_metrics(cage_faces):
    """Returns separately logged terms of the cage validity objective."""
    edges = mesh_edges(cage_faces)
    specifications = {
        "offset_mse": (1.0, 0.0, 0.0, 0.0, 0.0),
        "edge_distortion": (0.0, 1.0, 0.0, 0.0, 0.0),
        "orientation_barrier": (0.0, 0.0, 1.0, 0.0, 0.0),
        "area_barrier": (0.0, 0.0, 0.0, 1.0, 0.0),
        "volume_error": (0.0, 0.0, 0.0, 0.0, 1.0),
    }
    metrics = []
    for name, weights in specifications.items():
        term = CageValidityLoss(
            cage_faces, edges, offset_weight=weights[0],
            edge_weight=weights[1], orientation_weight=weights[2],
            area_weight=weights[3], volume_weight=weights[4],
            name=f"cage_{name}",
        )
        metrics.append(keras.metrics.MeanMetricWrapper(term, name=name))
    return metrics


class WeightedLossLogger(keras.callbacks.Callback):
    """Adds explicit weighted loss contributions to epoch logs and CSV."""

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = dict(loss_weights)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for output, weight in self.loss_weights.items():
            for prefix in ("", "val_"):
                source = f"{prefix}{output}_loss"
                if source in logs:
                    logs[f"{prefix}{output}_weighted_loss"] = (
                        float(weight) * float(logs[source])
                    )


@keras.saving.register_keras_serializable(package="paz")
class PhysicalMeshLoss(keras.losses.Loss):
    """Surface Chamfer, normal consistency, and edge regularization."""

    def __init__(self, standard_deviation, faces, edges,
                 edge_uniformity_weight=0.02,
                 normal_consistency_weight=0.1,
                 point_to_plane_weight=0.5,
                 name="physical_surface_chamfer_m2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.standard_deviation = tuple(float(x) for x in standard_deviation)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.edges = np.asarray(edges, dtype=np.int32)
        self.edge_uniformity_weight = float(edge_uniformity_weight)
        self.normal_consistency_weight = float(normal_consistency_weight)
        self.point_to_plane_weight = float(point_to_plane_weight)
        # Fixed deterministic samples cover triangle interiors, unlike a loss
        # on template vertices alone. The centroid and three inset corners
        # avoid duplicating samples along shared edges.
        self.barycentrics = np.asarray([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
        ], dtype=np.float32)

    def components(self, target, prediction):
        """Returns per-example Chamfer, normal, and edge terms."""
        scale = keras.ops.convert_to_tensor(
            self.standard_deviation, dtype=prediction.dtype
        )
        physical = prediction * scale
        target_physical = target[..., :3] * scale
        target_normals = target[..., 3:6]
        target_normals = target_normals / keras.ops.maximum(
            keras.ops.sqrt(keras.ops.sum(
                keras.ops.square(target_normals), axis=-1, keepdims=True
            )), 1e-8,
        )
        faces = keras.ops.convert_to_tensor(self.faces, dtype="int32")
        triangles = keras.ops.take(physical, faces, axis=1)
        barycentrics = keras.ops.convert_to_tensor(
            self.barycentrics, dtype=prediction.dtype
        )
        surface = keras.ops.einsum("sk,bfkd->bfsd", barycentrics, triangles)
        surface = keras.ops.reshape(
            surface, (-1, len(self.faces) * len(self.barycentrics), 3)
        )
        face_normals = keras.ops.cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
        )
        face_normals = face_normals / keras.ops.maximum(
            keras.ops.sqrt(keras.ops.sum(
                keras.ops.square(face_normals), axis=-1, keepdims=True
            )), 1e-8,
        )
        surface_normals = keras.ops.repeat(
            face_normals, len(self.barycentrics), axis=1
        )
        differences = surface[:, :, None, :] - target_physical[:, None, :, :]
        squared = keras.ops.sum(keras.ops.square(differences), axis=-1)
        chamfer = 0.5 * (
            keras.ops.mean(keras.ops.min(squared, axis=2), axis=1)
            + keras.ops.mean(keras.ops.min(squared, axis=1), axis=1)
        )
        surface_to_target = keras.ops.argmin(squared, axis=2)
        target_to_surface = keras.ops.argmin(squared, axis=1)
        matched_target_normals = keras.ops.take_along_axis(
            target_normals,
            keras.ops.repeat(surface_to_target[..., None], 3, axis=-1),
            axis=1,
        )
        matched_surface_normals = keras.ops.take_along_axis(
            surface_normals,
            keras.ops.repeat(target_to_surface[..., None], 3, axis=-1),
            axis=1,
        )
        predicted_alignment = keras.ops.abs(keras.ops.sum(
            surface_normals * matched_target_normals, axis=-1
        ))
        target_alignment = keras.ops.abs(keras.ops.sum(
            target_normals * matched_surface_normals, axis=-1
        ))
        normal_error = 0.5 * (
            keras.ops.mean(1.0 - predicted_alignment, axis=1)
            + keras.ops.mean(1.0 - target_alignment, axis=1)
        )
        edge_indices = keras.ops.convert_to_tensor(self.edges, dtype="int32")
        first = keras.ops.take(physical, edge_indices[:, 0], axis=1)
        second = keras.ops.take(physical, edge_indices[:, 1], axis=1)
        lengths = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(first - second), axis=-1) + 1e-8
        )
        mean_length = keras.ops.mean(lengths, axis=1, keepdims=True)
        uniformity = keras.ops.mean(
            keras.ops.square(lengths - mean_length), axis=1
        )
        surface_indices = keras.ops.repeat(
            surface_to_target[..., None, None], 3, axis=-1
        )
        surface_offsets = keras.ops.take_along_axis(
            differences, surface_indices, axis=2
        )[:, :, 0]
        target_differences = keras.ops.transpose(differences, (0, 2, 1, 3))
        target_indices = keras.ops.repeat(
            target_to_surface[..., None, None], 3, axis=-1
        )
        target_offsets = keras.ops.take_along_axis(
            target_differences, target_indices, axis=2
        )[:, :, 0]
        surface_plane = keras.ops.square(keras.ops.sum(
            surface_offsets * matched_target_normals, axis=-1
        ))
        target_plane = keras.ops.square(keras.ops.sum(
            target_offsets * matched_surface_normals, axis=-1
        ))
        point_to_plane = 0.5 * (
            keras.ops.mean(surface_plane, axis=1)
            + keras.ops.mean(target_plane, axis=1)
        )
        return chamfer, normal_error, uniformity, point_to_plane

    def call(self, target, prediction):
        chamfer, normal_error, uniformity, point_to_plane = self.components(
            target, prediction
        )
        return (chamfer
                + self.normal_consistency_weight * normal_error
                + self.edge_uniformity_weight * uniformity
                + self.point_to_plane_weight * point_to_plane)

    def normal_error(self, target, prediction):
        """Returns unsigned nearest-surface normal error in [0, 1]."""
        return self.components(target, prediction)[1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "standard_deviation": self.standard_deviation,
            "faces": self.faces.tolist(),
            "edges": self.edges.tolist(),
            "edge_uniformity_weight": self.edge_uniformity_weight,
            "normal_consistency_weight": self.normal_consistency_weight,
            "point_to_plane_weight": self.point_to_plane_weight,
        })
        return config


@keras.saving.register_keras_serializable(package="paz")
class PhysicalMeshNormalError(PhysicalMeshLoss):
    """Separately logged unsigned nearest-surface normal inconsistency."""

    def __init__(self, standard_deviation, faces, edges,
                 edge_uniformity_weight=0.0,
                 normal_consistency_weight=0.0,
                 point_to_plane_weight=0.0,
                 name="surface_normal_error", **kwargs):
        super().__init__(
            standard_deviation, faces, edges, edge_uniformity_weight,
            normal_consistency_weight, point_to_plane_weight,
            name=name, **kwargs,
        )

    def call(self, target, prediction):
        return self.components(target, prediction)[1]


@keras.saving.register_keras_serializable(package="paz")
class PhysicalMeshPointPlaneError(PhysicalMeshLoss):

    def __init__(self, standard_deviation, faces, edges,
                 name="surface_point_to_plane_m2", **kwargs):
        super().__init__(
            standard_deviation, faces, edges, 0.0, 0.0, 0.0,
            name=name, **kwargs,
        )

    def call(self, target, prediction):
        return self.components(target, prediction)[3]


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
class ForegroundMeanAbsoluteError(keras.metrics.Metric):

    def __init__(self, name="foreground_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, target, prediction, sample_weight=None):
        mask = keras.ops.max(keras.ops.abs(target), axis=-1, keepdims=True)
        mask = keras.ops.cast(mask > 1e-6, prediction.dtype)
        error = keras.ops.mean(keras.ops.abs(prediction - target), axis=-1)
        self.total.assign_add(keras.ops.sum(error * mask[..., 0]))
        self.count.assign_add(keras.ops.sum(mask))

    def result(self):
        return self.total / keras.ops.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


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


def load_canonical_mesh_samples(record, num_points):
    """Samples aligned canonical surface points and unsigned face normals."""
    stem = Path(record["_metadata_path"]).stem
    mesh_path = Path(record["_root"]) / "meshes" / f"{stem}.ply"
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    twice_areas = np.linalg.norm(crosses, axis=1)
    if not np.isfinite(twice_areas).all() or twice_areas.sum() <= 0.0:
        raise ValueError(f"Mesh has no valid surface area: {mesh_path}")
    face_normals = crosses / np.maximum(twice_areas[:, None], 1e-12)
    rng = np.random.default_rng(int(stem))
    selected = rng.choice(
        len(faces), num_points, p=twice_areas / twice_areas.sum()
    )
    chosen = triangles[selected]
    first = np.sqrt(rng.random((num_points, 1)))
    second = rng.random((num_points, 1))
    points = ((1.0 - first) * chosen[:, 0]
              + first * (1.0 - second) * chosen[:, 1]
              + first * second * chosen[:, 2])
    world_to_canonical = (
        np.linalg.inv(canonical_to_camera_matrix(record))
        @ world_to_camera_matrix(record["camera"])
    )
    canonical_points = transform_points(points, world_to_canonical)
    normal_matrix = np.linalg.inv(world_to_canonical[:3, :3]).T
    canonical_normals = face_normals[selected] @ normal_matrix.T
    canonical_normals /= np.maximum(
        np.linalg.norm(canonical_normals, axis=1, keepdims=True), 1e-12
    )
    return (canonical_points.astype(np.float32),
            canonical_normals.astype(np.float32))


def load_canonical_mesh_points(record, num_points):
    """Samples arbitrary-mesh surface points in canonical coordinates."""
    return load_canonical_mesh_samples(record, num_points)[0]


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


def object_foreground_mask(record, depth, max_depth, y_fov,
                           floor_tolerance):
    """Returns an object-only mask by rejecting invalid and floor points."""
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
    return valid & (world_points[..., 1] > floor_tolerance)


def object_render_targets(record, rgb, depth, resolution, max_depth, y_fov,
                          floor_tolerance):
    """Builds object-only RGB, z-depth, and silhouette targets."""
    mask = object_foreground_mask(
        record, depth, max_depth, y_fov, floor_tolerance
    )
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


class CompactCageDataset(keras.utils.PyDataset):
    """Loads normalized RGB-D and arbitrary-mesh surface supervision."""

    def __init__(self, records, normalizer, shape_names, batch_size,
                 max_depth, num_mesh_points, mesh_statistics,
                 render_resolution, y_fov, floor_tolerance,
                 reference_cage=None,
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
        self.reference_cage = (None if reference_cage is None else
                               np.asarray(reference_cage, np.float32))
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        self.mesh_targets = []
        for record in self.records:
            points, normals = load_canonical_mesh_samples(
                record, num_mesh_points
            )
            normalized_points = (points - self.mesh_mean) / self.mesh_std
            self.mesh_targets.append(np.concatenate(
                [normalized_points, normals], axis=-1
            ))
        self.mesh_targets = np.stack(self.mesh_targets)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, batch_index):
        begin = batch_index * self.batch_size
        indices = self.indices[begin:begin + self.batch_size]
        rgb_images, depth_images, object_masks, targets = [], [], [], []
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
            object_masks.append(object_foreground_mask(
                record, depth, self.max_depth, self.y_fov,
                self.floor_tolerance,
            )[..., None].astype(np.float32))
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
            "object_mask": np.stack(object_masks),
        }
        outputs = {
            name: np.stack([target[name] for target in targets])
            for name in OUTPUT_NAMES
        }
        outputs[MESH_OUTPUT_NAME] = self.mesh_targets[indices]
        if self.reference_cage is not None:
            outputs[CAGE_OUTPUT_NAME] = np.broadcast_to(
                self.reference_cage,
                (len(indices),) + self.reference_cage.shape,
            ).copy()
        outputs[RENDERED_RGB_OUTPUT_NAME] = np.stack(rendered_rgb)
        outputs[RENDERED_DEPTH_OUTPUT_NAME] = np.stack(rendered_depth)
        outputs[RENDERED_SILHOUETTE_OUTPUT_NAME] = np.stack(
            rendered_silhouette
        )
        return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class OutputSubsetDataset(keras.utils.PyDataset):
    """Presents selected targets from another dataset without copying data."""

    def __init__(self, dataset, output_names):
        super().__init__()
        self.dataset = dataset
        self.output_names = tuple(output_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, {name: targets[name] for name in self.output_names}

    def on_epoch_end(self):
        self.dataset.on_epoch_end()


def compact_encoder(rgb, depth, object_mask, regularizer, base_width):
    """Returns global context and the final spatial RGB-D feature map."""
    masked_rgb = keras.layers.Multiply(name="masked_rgb")([rgb, object_mask])
    masked_depth = keras.layers.Multiply(
        name="masked_depth"
    )([depth, object_mask])
    encoder_inputs = [rgb, depth, object_mask, masked_rgb, masked_depth]
    features = keras.layers.Concatenate(
        name="rgbd_concatenation"
    )(encoder_inputs)
    for index, width in enumerate([
            base_width, base_width * 2, base_width * 4, base_width * 8], 1):
        features = keras.layers.Conv2D(
            width, 3, strides=2, padding="same", use_bias=False,
            kernel_regularizer=regularizer, name=f"compact_conv_{index}",
        )(features)
        features = keras.layers.GroupNormalization(
            groups=min(8, width), name=f"compact_group_norm_{index}"
        )(features)
        features = keras.layers.Activation(
            "relu", name=f"compact_relu_{index}"
        )(features)
    spatial_features = features
    pooled = keras.layers.GlobalAveragePooling2D(
        name="compact_global_pool"
    )(spatial_features)
    global_features = keras.layers.Dense(
        256, activation="relu", kernel_regularizer=regularizer,
        name="compact_global_feature",
    )(pooled)
    return global_features, spatial_features


@keras.saving.register_keras_serializable(package="paz")
class DifferentiableCageRenderer(keras.layers.Layer):
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
        translation = denormalize_tensor(
            translation_raw, self.target_statistics, "object_translation"
        )
        scale = keras.ops.maximum(
            denormalize_tensor(
                scale_raw, self.target_statistics, "object_scale"
            ), 1e-3
        )
        rotation = rotation_6d_to_matrix_ops(orientation)
        vertices = keras.ops.matmul(
            canonical, keras.ops.transpose(rotation, (0, 2, 1))
        )
        vertices = vertices * scale[:, None, :] + translation[:, None, :]

        faces = keras.ops.convert_to_tensor(self.faces, dtype="int32")
        triangle_first = keras.ops.take(vertices, faces[:, 0], axis=1)
        triangle_second = keras.ops.take(vertices, faces[:, 1], axis=1)
        triangle_third = keras.ops.take(vertices, faces[:, 2], axis=1)
        triangles = keras.ops.stack(
            [triangle_first, triangle_second, triangle_third], axis=2
        )
        barycentrics = keras.ops.convert_to_tensor(np.asarray([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
        ], np.float32), dtype=vertices.dtype)
        samples = keras.ops.einsum(
            "sk,bfkd->bfsd", barycentrics, triangles
        )
        samples_per_face = 4
        samples = keras.ops.reshape(
            samples, (-1, len(self.faces) * samples_per_face, 3)
        )
        face_cross = keras.ops.cross(
            triangle_second - triangle_first,
            triangle_third - triangle_first,
        )
        face_area = 0.5 * keras.ops.sqrt(keras.ops.sum(
            keras.ops.square(face_cross), axis=-1
        ) + 1e-8)
        normals = normalize_tensor(face_cross)
        normals = keras.ops.repeat(normals, samples_per_face, axis=1)
        area_weight = face_area / keras.ops.maximum(
            keras.ops.mean(face_area, axis=1, keepdims=True), 1e-8
        )
        area_weight = keras.ops.clip(area_weight, 0.25, 4.0)
        area_weight = keras.ops.repeat(
            area_weight, samples_per_face, axis=1
        )

        material = denormalize_tensor(
            material_raw, self.target_statistics, "material"
        )
        color = keras.ops.clip(material[..., :3], 0.0, 1.0)
        diffuse = keras.ops.clip(material[..., 3:4], 0.0, 1.0)
        specular = keras.ops.clip(material[..., 4:5], 0.0, 1.0)
        ambient = keras.ops.clip(material[..., 5:6], 0.0, 1.0)
        shininess = keras.ops.clip(material[..., 6:7], 1.0, 128.0)
        light_position = denormalize_tensor(
            light_position_raw, self.target_statistics, "light_position"
        )
        light_intensity = keras.ops.clip(
            denormalize_tensor(
                light_intensity_raw, self.target_statistics,
                "light_intensity"
            ),
            0.0, 5.0,
        )
        light_direction = normalize_tensor(
            light_position[:, None, :] - samples
        )
        view_direction = normalize_tensor(-samples)
        half_direction = normalize_tensor(light_direction + view_direction)
        lambert = keras.ops.maximum(keras.ops.sum(
            normals * light_direction, axis=-1, keepdims=True
        ), 0.0)
        highlight = keras.ops.power(
            keras.ops.clip(keras.ops.sum(
                normals * half_direction, axis=-1, keepdims=True
            ), 0.0, 1.0),
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
        ) * keras.ops.cast(valid[..., None], projected.dtype) \
            * area_weight[..., None]
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


def build_parameter_heads(features, regularizer, num_shapes):
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
    return geometry, outputs


def build_cage_queries(geometry, spatial_features, depth, object_mask,
                       outputs, regularizer, config):
    width = 128
    queries = RepeatCageCoordinates(
        config.vertices, name="cage_vertex_queries"
    )(geometry)
    queries = keras.layers.Dense(
        width, activation="relu", kernel_regularizer=regularizer,
        name="cage_query_projection",
    )(queries)
    height, image_width = spatial_features.shape[1:3]
    camera = CameraGeometryChannels(
        int(height), int(image_width), config.y_fov, config.max_depth,
        config.target_statistics, name="cage_camera_geometry_channels",
    )([depth, object_mask, outputs["object_translation"],
       outputs["object_orientation_6d"], outputs["object_scale"]])
    tokens = keras.layers.Concatenate(
        name="cage_position_aware_features"
    )([spatial_features, camera])
    tokens = keras.layers.Conv2D(
        width, 1, padding="same", activation="relu",
        kernel_regularizer=regularizer, name="cage_spatial_projection",
    )(tokens)
    tokens = keras.layers.Reshape(
        (-1, width), name="cage_spatial_tokens"
    )(tokens)
    attended = keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=width // 4, kernel_regularizer=regularizer,
        name="cage_spatial_attention",
    )(query=queries, value=tokens, key=tokens)
    features = keras.layers.Add(
        name="cage_attention_residual"
    )([queries, attended])
    return keras.layers.LayerNormalization(
        name="cage_attention_norm"
    )(features)


def build_cage_outputs(features, outputs, regularizer, config):
    feed_forward = keras.layers.Dense(
        256, activation="gelu", kernel_regularizer=regularizer,
        name="cage_query_feed_forward_1",
    )(features)
    feed_forward = keras.layers.Dense(
        128, kernel_regularizer=regularizer,
        name="cage_query_feed_forward_2",
    )(feed_forward)
    features = keras.layers.Add(
        name="cage_query_residual"
    )([features, feed_forward])
    features = keras.layers.LayerNormalization(
        name="cage_query_norm"
    )(features)
    initializer = keras.initializers.RandomNormal(stddev=1e-3)
    offsets = keras.layers.Dense(
        3, kernel_initializer=initializer, bias_initializer="zeros",
        kernel_regularizer=regularizer, name="cage_vertex_offsets",
    )(features)
    mesh, cage = CageMeshDeformation(
        config.vertices, config.weights, config.mesh_statistics["mean"],
        config.mesh_statistics["standard_deviation"], config.max_offset,
        reference_vertices=config.coordinates, name="cage_deformation",
    )(offsets)
    outputs[MESH_OUTPUT_NAME] = keras.layers.Activation(
        "linear", name=MESH_OUTPUT_NAME
    )(mesh)
    outputs[CAGE_OUTPUT_NAME] = keras.layers.Activation(
        "linear", name=CAGE_OUTPUT_NAME
    )(cage)


def build_render_outputs(outputs, config):
    rendered = DifferentiableCageRenderer(
        config.faces, config.resolution, config.y_fov,
        config.target_statistics, config.mesh_statistics,
        splat_sigma=config.splat_sigma,
        depth_temperature=config.depth_temperature,
        splat_strength=config.splat_strength,
        name="differentiable_cage_renderer",
    )([
        outputs[MESH_OUTPUT_NAME], outputs["object_translation"],
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


def build_model(image_shape, num_shapes, coordinates, base_width,
                l2_regularization, faces, render_resolution, y_fov,
                target_statistics, mesh_statistics, splat_sigma,
                depth_temperature, max_depth, cage_vertices, mesh_weights,
                cage_maximum_offset=1.0, splat_strength=0.5):
    """Builds a compact physical-parameter and cage-deformation network."""
    regularizer = keras.regularizers.L2(l2_regularization)
    rgb = keras.Input((*image_shape, 3), name="rgb")
    depth = keras.Input((*image_shape, 1), name="depth")
    object_mask = keras.Input((*image_shape, 1), name="object_mask")
    features, spatial = compact_encoder(
        rgb, depth, object_mask, regularizer, base_width
    )
    geometry, outputs = build_parameter_heads(
        features, regularizer, num_shapes
    )
    decoder = CageDecoderArgs(
        coordinates, cage_vertices, mesh_weights, mesh_statistics,
        cage_maximum_offset, y_fov, max_depth, target_statistics,
    )
    cage_features = build_cage_queries(
        geometry, spatial, depth, object_mask, outputs, regularizer, decoder
    )
    build_cage_outputs(cage_features, outputs, regularizer, decoder)
    renderer = RenderArgs(
        faces, render_resolution, y_fov, target_statistics, mesh_statistics,
        splat_sigma, depth_temperature, splat_strength,
    )
    build_render_outputs(outputs, renderer)
    return keras.Model(
        {"rgb": rgb, "depth": depth, "object_mask": object_mask}, outputs,
        name="compact_rgbd_cage_mesh",
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
        for offset, vertices in enumerate(predictions[MESH_OUTPUT_NAME]):
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
    parser.add_argument("--template-subdivisions", type=int, default=2)
    parser.add_argument("--cage-subdivisions", type=int, default=2)
    parser.add_argument("--template-radius", type=float, default=1.0)
    parser.add_argument("--cage-radius", type=float, default=1.25)
    parser.add_argument("--cage-maximum-offset", type=float, default=0.65)
    parser.add_argument("--cage-validity-weight", type=float, default=1.0)
    parser.add_argument("--cage-offset-weight", type=float, default=0.1)
    parser.add_argument("--cage-edge-weight", type=float, default=0.2)
    parser.add_argument("--cage-orientation-weight", type=float, default=1.0)
    parser.add_argument("--cage-area-weight", type=float, default=0.5)
    parser.add_argument("--cage-volume-weight", type=float, default=0.1)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--mesh-loss-weight", type=float, default=2.0)
    parser.add_argument("--mesh-target-points", type=int, default=1024)
    parser.add_argument("--edge-uniformity-weight", type=float, default=0.005)
    parser.add_argument("--normal-consistency-weight", type=float, default=0.1,
                        help="Unsigned nearest-surface normal loss weight.")
    parser.add_argument("--point-to-plane-weight", type=float, default=0.5)
    parser.add_argument("--joint-pose-loss-weight", type=float, default=0.25)
    parser.add_argument("--render-resolution", type=int, default=32)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--render-floor-tolerance", type=float, default=0.01)
    parser.add_argument("--splat-sigma", type=float, default=0.3)
    parser.add_argument("--splat-strength", type=float, default=0.5)
    parser.add_argument("--depth-temperature", type=float, default=0.15)
    parser.add_argument("--rgb-reconstruction-weight", type=float, default=0.25)
    weight_config = {"type": float, "default": 1.0}
    parser.add_argument("--depth-reconstruction-weight", **weight_config)
    parser.add_argument("--silhouette-reconstruction-weight", type=float,
                        default=0.1)
    parser.add_argument("--geometry-warmup-epochs", type=int, default=15,
                        help=("Geometry and pose-alignment epochs before "
                              "joint optimization."))
    parser.add_argument(
        "--geometry-only", action="store_true",
        help=("Train mesh, cage, translation, rotation, and scale for "
              "--epochs epochs, then stop."),
    )
    parser.add_argument(
        "--overfit-sample-ids", nargs="+", default=None,
        help="Diagnostic: use only these sample IDs for train and validation.",
    )
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


def validate_args(args):
    positive = (
        args.base_width, args.mesh_loss_weight, args.mesh_target_points,
        args.batch_size, args.epochs, args.max_depth, args.checkpoint_every,
        args.render_resolution, args.y_fov, args.splat_sigma,
        args.splat_strength, args.depth_temperature, args.template_radius,
        args.cage_radius, args.cage_maximum_offset,
    )
    if min(positive) <= 0 or args.preview_meshes < 0:
        raise ValueError("sizes, weights, and ranges must be positive")
    subdivisions = args.template_subdivisions, args.cage_subdivisions
    if min(subdivisions) < 0 or args.geometry_warmup_epochs < 0:
        raise ValueError("epoch and subdivision counts cannot be negative")
    cage_weights = (
        args.cage_validity_weight, args.cage_offset_weight,
        args.cage_edge_weight, args.cage_orientation_weight,
        args.cage_area_weight, args.cage_volume_weight,
    )
    reconstruction_weights = (
        args.rgb_reconstruction_weight, args.depth_reconstruction_weight,
        args.silhouette_reconstruction_weight,
    )
    other_weights = (
        args.edge_uniformity_weight, args.normal_consistency_weight,
        args.point_to_plane_weight, args.joint_pose_loss_weight,
        args.render_floor_tolerance,
    )
    if min(cage_weights + reconstruction_weights + other_weights) < 0.0:
        raise ValueError("loss weights and floor tolerance cannot be negative")


def select_records(args):
    records = load_records(args.dataset)
    splits = split_records(records, args.seed, args.validation_split)
    if not args.overfit_sample_ids:
        return records, splits
    records_by_id = {
        Path(record["_metadata_path"]).stem: record for record in records
    }
    missing = set(args.overfit_sample_ids).difference(records_by_id)
    if missing:
        raise ValueError(f"Unknown overfit sample IDs: {sorted(missing)}")
    selected = [records_by_id[name] for name in args.overfit_sample_ids]
    print(f"Diagnostic overfit set: {args.overfit_sample_ids}")
    return records, (selected, list(selected), splits[2])


def build_objectives(args, mesh_statistics, faces, edges, cage_faces,
                     shape_names):
    mesh = PhysicalMeshLoss(
        mesh_statistics["standard_deviation"], faces, edges,
        args.edge_uniformity_weight, args.normal_consistency_weight,
        args.point_to_plane_weight,
    )
    mesh_normal = PhysicalMeshNormalError(
        mesh_statistics["standard_deviation"], faces, edges
    )
    point_plane = PhysicalMeshPointPlaneError(
        mesh_statistics["standard_deviation"], faces, edges
    )
    primitive_shapes = shape_names == ("cube", "cylinder", "sphere")
    if primitive_shapes:
        rotation = symmetry_rotation_loss
    else:
        rotation = generic_rotation_loss
    rotation_metric = (symmetry_geodesic_angle_degrees if primitive_shapes
                       else generic_geodesic_angle_degrees)
    cage = CageValidityLoss(
        cage_faces, mesh_edges(cage_faces),
        offset_weight=args.cage_offset_weight,
        edge_weight=args.cage_edge_weight,
        orientation_weight=args.cage_orientation_weight,
        area_weight=args.cage_area_weight,
        volume_weight=args.cage_volume_weight,
    )
    rgb = MaskedReconstructionHuber(
        delta=0.1, name="rendered_rgb_masked_huber"
    )
    depth = MaskedReconstructionHuber(
        delta=0.1, name="rendered_metric_depth_masked_huber"
    )
    silhouette = SilhouetteBCEDiceLoss()
    losses = {
        "object_orientation_6d": rotation,
        MESH_OUTPUT_NAME: mesh,
        CAGE_OUTPUT_NAME: cage,
        RENDERED_RGB_OUTPUT_NAME: rgb,
        RENDERED_DEPTH_OUTPUT_NAME: depth,
        RENDERED_SILHOUETTE_OUTPUT_NAME: silhouette,
    }
    metrics = {
        "object_orientation_6d": [
            keras.metrics.MeanMetricWrapper(rotation, name="loss"),
            rotation_metric,
        ],
        MESH_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(mesh, name="loss"),
            keras.metrics.MeanMetricWrapper(
                mesh_normal, name="surface_normal_error"
            ),
            keras.metrics.MeanMetricWrapper(
                point_plane, name="surface_point_to_plane_m2"
            ),
        ],
        CAGE_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(cage, name="loss")
        ] + cage_validity_metrics(cage_faces),
        RENDERED_RGB_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(rgb, name="loss"),
            ForegroundMeanAbsoluteError(name="foreground_mae_0_1"),
        ],
        RENDERED_DEPTH_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(depth, name="loss"),
            ForegroundMeanAbsoluteError(name="foreground_mae_m"),
        ],
        RENDERED_SILHOUETTE_OUTPUT_NAME: [
            keras.metrics.MeanMetricWrapper(silhouette, name="loss"),
            keras.metrics.BinaryIoU(
                target_class_ids=(1,), threshold=0.5, name="iou"
            ),
        ],
    }
    weights = {
        MESH_OUTPUT_NAME: args.mesh_loss_weight,
        CAGE_OUTPUT_NAME: args.cage_validity_weight,
        RENDERED_RGB_OUTPUT_NAME: args.rgb_reconstruction_weight,
        RENDERED_DEPTH_OUTPUT_NAME: args.depth_reconstruction_weight,
        RENDERED_SILHOUETTE_OUTPUT_NAME:
            args.silhouette_reconstruction_weight,
    }
    return Objectives(mesh, mesh_normal, rotation, rotation_metric, cage,
                      losses, metrics, weights)


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    # Respect JAX_PLATFORMS: CUDA is used on the server, while explicit CPU
    # runs remain possible for focused diagnostics and unit tests.
    devices = jax.devices()
    device = devices[0]
    jax.config.update("jax_default_device", device)
    records, splits = select_records(args)
    shape_names = tuple(sorted({record["shape"]["type"] for record in records}))
    training_records, validation_records, test_records = splits
    export_test_split(test_records, args.output / "test_split")
    export_test_meshes(test_records, args.output / "test_split")
    save_split_manifest(
        args.output / "split.json", training_records, validation_records,
        test_records, args.seed, args.validation_split,
    )

    (coordinates, faces, edges, cage_vertices, cage_faces,
     mesh_weights) = build_cage_topology(
        args.template_subdivisions, args.cage_subdivisions,
        args.template_radius, args.cage_radius,
    )
    num_vertices = len(coordinates)
    normalizer = GenericTargetNormalizer.fit(training_records, shape_names)
    mesh_statistics = fit_canonical_statistics(
        training_records, args.mesh_target_points
    )
    normalizer.statistics[MESH_OUTPUT_NAME] = mesh_statistics
    normalizer.save(args.output / "normalization.json")
    topology = {
        "coordinates": coordinates, "faces": faces, "edges": edges,
        "cage_vertices": cage_vertices, "cage_faces": cage_faces,
        "mesh_weights": mesh_weights,
    }
    np.savez_compressed(args.output / "mesh_topology.npz", **topology)
    with (args.output / "input_preprocessing.json").open("w") as file:
        json.dump({
            "method": "compact cage-deformed mesh",
            "mesh_frame": "canonical transformed by predicted pose and scale",
            "mesh_representation": "mean-value-coordinate cage deformation",
            "geometry_representation": "cage",
            "cage_decoder": (
                "per-vertex queries with four-head attention over RGB-D "
                "features augmented by pixel XY, camera rays, back-projected "
                "camera XYZ, normalized depth, and a valid-depth mask"
            ),
            "spatial_token_frame": "camera and predicted canonical",
            "spatial_point_units": "metric units",
            "canonical_alignment": (
                "inverse predicted translation, SO(3) rotation, and metric "
                "scale"
            ),
            "attention_mask": "object-only depth excluding floor",
            "template_subdivisions": args.template_subdivisions,
            "cage_subdivisions": args.cage_subdivisions,
            "template_radius": args.template_radius,
            "cage_radius": args.cage_radius,
            "cage_maximum_offset": args.cage_maximum_offset,
            "cage_validity_weight": args.cage_validity_weight,
            "normal_consistency_weight": args.normal_consistency_weight,
            "point_to_plane_weight": args.point_to_plane_weight,
            "joint_pose_loss_weight": args.joint_pose_loss_weight,
            "encoder_mask_conditioning": (
                "RGB-D, object mask, masked RGB, and masked depth"
            ),
            "num_vertices": num_vertices,
            "num_faces": len(faces),
            "mesh_target_points": args.mesh_target_points,
            "max_depth": args.max_depth,
            "y_fov_degrees": args.y_fov,
            "training_render_resolution": args.render_resolution,
            "differentiable_renderer": (
                "soft shaded four-samples-per-triangle splats"
            ),
            "render_floor_tolerance_metres": args.render_floor_tolerance,
            "splat_sigma_pixels": args.splat_sigma,
            "splat_strength": args.splat_strength,
            "depth_temperature_metres": args.depth_temperature,
            "rgb_reconstruction_weight": args.rgb_reconstruction_weight,
            "depth_reconstruction_weight": args.depth_reconstruction_weight,
            "silhouette_reconstruction_weight": (
                args.silhouette_reconstruction_weight
            ),
            "geometry_warmup_epochs": args.geometry_warmup_epochs,
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
        "num_mesh_points": args.mesh_target_points,
        "mesh_statistics": mesh_statistics,
        "render_resolution": args.render_resolution,
        "y_fov": args.y_fov,
        "floor_tolerance": args.render_floor_tolerance,
        "reference_cage": cage_vertices,
    }
    training = CompactCageDataset(
        training_records, shuffle=True, seed=args.seed, **common
    )
    validation = CompactCageDataset(
        validation_records, shuffle=False, seed=args.seed, **common
    )
    test = CompactCageDataset(
        test_records, shuffle=False, seed=args.seed, **common
    )
    image_shape = training[0][0]["rgb"].shape[1:3]
    model = build_model(
        image_shape, len(shape_names), coordinates, args.base_width,
        args.l2_regularization, faces, args.render_resolution, args.y_fov,
        normalizer.statistics, mesh_statistics, args.splat_sigma,
        args.depth_temperature, args.max_depth, cage_vertices, mesh_weights,
        args.cage_maximum_offset,
        args.splat_strength,
    )
    objectives = build_objectives(
        args, mesh_statistics, faces, edges, cage_faces, shape_names
    )
    warmup_epochs = (args.epochs if args.geometry_only else
                     args.geometry_warmup_epochs)
    if warmup_epochs:
        geometry_names = [
            MESH_OUTPUT_NAME, "object_translation",
            "object_orientation_6d", "object_scale",
        ]
        vector_mse = keras.losses.MeanSquaredError()
        geometry_losses = {
            MESH_OUTPUT_NAME: objectives.mesh,
            "object_translation": vector_mse,
            "object_orientation_6d": objectives.rotation,
            "object_scale": vector_mse,
        }
        geometry_weights = {
            MESH_OUTPUT_NAME: args.mesh_loss_weight,
            "object_translation": 1.0,
            "object_orientation_6d": 1.0,
            "object_scale": 1.0,
        }
        geometry_metrics = {
            MESH_OUTPUT_NAME: [
                keras.metrics.MeanMetricWrapper(objectives.mesh, name="loss"),
                keras.metrics.MeanMetricWrapper(
                    objectives.mesh_normal, name="surface_normal_error"
                ),
            ],
            "object_translation": [
                keras.metrics.MeanSquaredError(name="loss")
            ],
            "object_orientation_6d": [
                keras.metrics.MeanMetricWrapper(
                    objectives.rotation, name="loss"
                ),
                objectives.rotation_metric,
            ],
            "object_scale": [keras.metrics.MeanSquaredError(name="loss")],
        }
        geometry_names.append(CAGE_OUTPUT_NAME)
        geometry_losses[CAGE_OUTPUT_NAME] = objectives.cage
        geometry_weights[CAGE_OUTPUT_NAME] = args.cage_validity_weight
        geometry_metrics[CAGE_OUTPUT_NAME] = [
            keras.metrics.MeanMetricWrapper(objectives.cage, name="loss")
        ] + cage_validity_metrics(cage_faces)
        geometry_model = keras.Model(
            model.inputs,
            {name: model.get_layer(name).output for name in geometry_names},
            name="cage_geometry_warmup",
        )
        geometry_model.compile(
            optimizer=keras.optimizers.AdamW(
                args.learning_rate, weight_decay=args.weight_decay,
                global_clipnorm=1.0,
            ),
            loss=geometry_losses, loss_weights=geometry_weights,
            metrics=geometry_metrics,
        )
        print(
            f"Geometry-only training for {warmup_epochs} epochs"
        )
        geometry_model.fit(
            OutputSubsetDataset(training, geometry_names),
            validation_data=OutputSubsetDataset(validation, geometry_names),
            epochs=warmup_epochs,
            callbacks=[
                WeightedLossLogger(geometry_weights),
                keras.callbacks.CSVLogger(
                    args.output / "geometry_warmup.csv"
                ),
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ModelCheckpoint(
                    args.output / "geometry_warmup_best.weights.h5",
                    monitor="val_loss", mode="min", save_best_only=True,
                    save_weights_only=True,
                ),
            ],
        )
        geometry_model.load_weights(
            args.output / "geometry_warmup_best.weights.h5"
        )
        if args.geometry_only:
            model.save(args.output / "geometry_only.keras")
            save_preview_meshes(
                model, test, test_records, mesh_statistics,
                normalizer.statistics, faces,
                args.output / "geometry_only_preview_meshes",
                min(args.preview_meshes, len(test_records)),
            )
            return
    pose_weights = {
        "object_translation": args.joint_pose_loss_weight,
        "object_orientation_6d": args.joint_pose_loss_weight,
        "object_scale": args.joint_pose_loss_weight,
    }
    compile_model(
        model, args.learning_rate, args.weight_decay, normalizer.statistics,
        extra_losses=objectives.losses, extra_metrics=objectives.metrics,
        extra_loss_weights=objectives.weights,
        loss_weight_overrides=pose_weights,
    )
    logged_weights = {name: 1.0 for name in OUTPUT_NAMES}
    logged_weights.update(objectives.weights)
    logged_weights.update(pose_weights)
    callbacks = [
        WeightedLossLogger(logged_weights),
        keras.callbacks.CSVLogger(args.output / "training.csv"),
        TrainingPlot(
            args.output / "loss.png", OUTPUT_NAMES + (
                CAGE_OUTPUT_NAME,
                MESH_OUTPUT_NAME, RENDERED_RGB_OUTPUT_NAME,
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
        f"Training compact cage mesh on {device}: "
        f"{len(training_records)} train, "
        f"{len(validation_records)} validation, {len(test_records)} test; "
        f"{num_vertices} vertices, {len(faces)} faces"
    )
    model.fit(
        training, validation_data=validation, epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")
    best_model = keras.models.load_model(
        args.output / "best.keras", compile=False
    )
    save_preview_meshes(
        best_model, test, test_records, mesh_statistics,
        normalizer.statistics,
        faces,
        args.output / "cage_preview_meshes",
        min(args.preview_meshes, len(test_records)),
    )


if __name__ == "__main__":
    main()
