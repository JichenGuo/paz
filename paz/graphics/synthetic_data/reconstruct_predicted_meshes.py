"""Triangulate predicted point clouds and overlay them on original RGB images.

The current synthetic primitives are convex, so reconstruction uses the 3D
convex hull of each predicted camera-frame point cloud. Camera- and world-frame
PLY meshes plus filled and wireframe RGB comparisons are saved.

Example:
    KERAS_BACKEND=jax python -m \
        paz.graphics.synthetic_data.reconstruct_predicted_meshes \
        --experiment experiments/resnet18_rgbd_val_mesh_HPC
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
from pathlib import Path

import cv2
import keras
import numpy as np
import trimesh

# Registers the serialized mesh loss before loading a complete Keras model.
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_mesh import (
    MESH_OUTPUT_NAME,
    world_to_camera_matrix,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import load_records


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def load_rgb_depth(record, max_depth):
    """Loads RGB in [0, 1] and normalized single-channel depth."""
    root = Path(record["_root"])
    rgb_path = root / record["rgb"]
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(root / record["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    depth = np.clip(depth / max_depth, 0.0, 1.0)[..., None]
    return rgb, depth


def denormalize_points(points, statistics):
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    standard_deviation = np.asarray(
        statistics["standard_deviation"], dtype=np.float32
    )
    return np.asarray(points, dtype=np.float32) * standard_deviation + mean


def reconstruct_convex_mesh(points):
    """Returns a watertight triangular convex hull of predicted points."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4 or points.shape[1:] != (3,):
        raise ValueError("At least four XYZ points are required")
    if not np.isfinite(points).all():
        raise ValueError("Predicted point cloud contains non-finite values")
    mesh = trimesh.convex.convex_hull(points)
    if len(mesh.faces) == 0 or not mesh.is_watertight:
        raise ValueError("Convex hull did not produce a watertight mesh")
    return mesh


def camera_intrinsics(image_shape, y_fov_degrees):
    """Returns PAZ pinhole focal lengths and principal point in pixels."""
    height, width = image_shape[:2]
    y_fov = np.deg2rad(y_fov_degrees)
    focal_y = 0.5 * height / np.tan(0.5 * y_fov)
    focal_x = focal_y
    return focal_x, focal_y, 0.5 * width, 0.5 * height


def project_camera_points(points, image_shape, y_fov_degrees):
    """Projects PAZ camera coordinates, whose forward axis is negative Z."""
    focal_x, focal_y, center_x, center_y = camera_intrinsics(
        image_shape, y_fov_degrees
    )
    forward_depth = -points[:, 2]
    safe_depth = np.maximum(forward_depth, 1e-6)
    horizontal = focal_x * points[:, 0] / safe_depth + center_x
    vertical = center_y - focal_y * points[:, 1] / safe_depth
    pixels = np.stack([horizontal, vertical], axis=-1)
    return pixels, forward_depth


def rasterize_mesh(mesh, image_shape, y_fov_degrees):
    """Rasterizes a binary projected surface and triangle-edge image."""
    height, width = image_shape[:2]
    pixels, depths = project_camera_points(
        mesh.vertices, image_shape, y_fov_degrees
    )
    surface = np.zeros((height, width), dtype=np.uint8)
    wireframe = np.zeros((height, width), dtype=np.uint8)
    face_depths = depths[mesh.faces].mean(axis=1)
    # Painter ordering is not needed for a binary mask, but makes the function
    # ready for colored faces and avoids drawing invalid behind-camera faces.
    for face_index in np.argsort(face_depths)[::-1]:
        face = mesh.faces[face_index]
        if np.any(depths[face] <= 1e-6):
            continue
        polygon = np.rint(pixels[face]).astype(np.int32)
        cv2.fillConvexPoly(surface, polygon, 255, lineType=cv2.LINE_AA)
        cv2.polylines(
            wireframe, [polygon], True, 255, 1, lineType=cv2.LINE_AA
        )
    return surface, wireframe


def colored_overlay(rgb_uint8, mask, color, alpha):
    """Alpha blends one RGB color wherever a uint8 mask is nonzero."""
    overlay = rgb_uint8.copy().astype(np.float32)
    selected = mask > 0
    overlay[selected] = (
        (1.0 - alpha) * overlay[selected]
        + alpha * np.asarray(color, dtype=np.float32)
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def camera_mesh_to_world(mesh, camera):
    """Transforms reconstructed camera vertices into world coordinates."""
    world_mesh = mesh.copy()
    world_mesh.apply_transform(np.linalg.inv(world_to_camera_matrix(camera)))
    return world_mesh


def save_comparison(path, rgb, surface_mask, wireframe_mask):
    """Saves original, filled reconstruction, and triangle-wireframe panels."""
    rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    surface = colored_overlay(
        rgb_uint8, surface_mask, [255, 140, 0], alpha=0.45
    )
    wireframe = colored_overlay(
        rgb_uint8, wireframe_mask, [0, 255, 0], alpha=0.9
    )
    separator = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([
        rgb_uint8, separator, surface, separator, wireframe
    ], axis=1)
    # OpenCV writes BGR while arrays above are RGB.
    cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None,
                        help="Full .keras model; defaults to best.keras.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--y-fov", type=float, default=45.0,
                        help="Vertical field of view used during generation.")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.y_fov <= 0.0 or args.y_fov >= 180.0:
        raise ValueError("field of view must be between zero and 180 degrees")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    test_split = args.test_split or args.experiment / "test_split"
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "mesh_reconstruction"
    camera_output = output / "meshes_camera"
    world_output = output / "meshes_world"
    comparison_output = output / "comparisons"
    mask_output = output / "projected_masks"
    for directory in (
        camera_output, world_output, comparison_output, mask_output
    ):
        directory.mkdir(parents=True, exist_ok=True)

    preprocessing = load_json(args.experiment / "input_preprocessing.json")
    normalization = load_json(args.experiment / "normalization.json")
    mesh_statistics = normalization["targets"][MESH_OUTPUT_NAME]
    records = load_records(test_split)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if not records:
        raise ValueError(f"No test records found in {test_split}")
    model = keras.models.load_model(model_path, compile=False)
    if MESH_OUTPUT_NAME not in model.output_names:
        raise ValueError(f"{model_path} has no {MESH_OUTPUT_NAME} output")

    manifest = {
        "model": str(model_path),
        "reconstruction": "3D convex hull",
        "coordinate_frames": ["camera", "world"],
        "comparison_panels": [
            "original RGB", "filled projected surface", "triangle wireframe"
        ],
        "y_fov_degrees": args.y_fov,
        "samples": [],
    }
    print(f"Reconstructing {len(records)} predicted point clouds")
    for index, record in enumerate(records):
        sample_id = Path(record["_metadata_path"]).stem
        rgb, depth = load_rgb_depth(record, preprocessing["max_depth"])
        predictions = model.predict(
            {"rgb": rgb[None], "depth": depth[None]}, verbose=0
        )
        camera_points = denormalize_points(
            predictions[MESH_OUTPUT_NAME][0], mesh_statistics
        )
        camera_mesh = reconstruct_convex_mesh(camera_points)
        world_mesh = camera_mesh_to_world(camera_mesh, record["camera"])
        surface, wireframe = rasterize_mesh(
            camera_mesh, rgb.shape[:2], args.y_fov
        )

        camera_path = camera_output / f"{sample_id}.ply"
        world_path = world_output / f"{sample_id}.ply"
        comparison_path = comparison_output / f"{sample_id}.png"
        mask_path = mask_output / f"{sample_id}.png"
        camera_mesh.export(camera_path)
        world_mesh.export(world_path)
        save_comparison(comparison_path, rgb, surface, wireframe)
        cv2.imwrite(str(mask_path), surface)
        manifest["samples"].append({
            "sample_id": sample_id,
            "num_input_points": int(len(camera_points)),
            "num_mesh_vertices": int(len(camera_mesh.vertices)),
            "num_mesh_faces": int(len(camera_mesh.faces)),
            "camera_mesh": str(camera_path.relative_to(output)),
            "world_mesh": str(world_path.relative_to(output)),
            "comparison": str(comparison_path.relative_to(output)),
        })
        print(
            f"Reconstructed {index + 1}/{len(records)}: {sample_id}, "
            f"{len(camera_mesh.vertices)} vertices, "
            f"{len(camera_mesh.faces)} triangles"
        )

    with (output / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved reconstructed meshes and comparisons to {output}")


if __name__ == "__main__":
    main()
