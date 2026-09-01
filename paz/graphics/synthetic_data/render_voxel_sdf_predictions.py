"""Render camera-frame Marching Cubes meshes from depth-voxel SDF models.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.render_voxel_sdf_predictions \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --experiment experiments/resnet18_depth_voxel_sdf
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jp
import keras
import numpy as np
import trimesh

import paz
from paz.graphics.synthetic_data.render_sdf_mesh_predictions import (
    decode_prediction,
    evaluate_render,
    jsonify,
    save_comparison,
    summarize,
    unique_edges,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_voxel_sdf import (
    SDF_OUTPUT_NAME,
    depth_to_voxel,
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def camera_transform(camera):
    return paz.SE3.view_transform(
        jp.asarray(camera["position_world_xyz"]),
        jp.asarray(camera["target_world_xyz"]),
        jp.asarray([0.0, 1.0, 0.0]),
    )


def load_rgb_depth(metadata, dataset, max_depth):
    rgb_path = dataset / metadata["rgb"]
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(dataset / metadata["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    depth = np.where(
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth),
        depth, 0.0,
    ).astype(np.float32)
    return rgb, depth


def load_camera_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh found in {path}")
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except ModuleNotFoundError:
        pass
    return mesh


def build_scene(camera_mesh, prediction, world_to_camera):
    """Places an already posed camera-frame mesh into the PAZ world frame."""
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera))
    values = prediction["material"]
    material = paz.graphics.Material(
        jp.asarray(values["color_rgb"]), values["ambient"],
        values["diffuse"], values["specular"], values["shininess"],
    )
    vertices = np.asarray(camera_mesh.vertices, dtype=np.float32)
    faces = np.asarray(camera_mesh.faces, dtype=np.int32)
    colors = np.repeat(
        np.asarray(values["color_rgb"], dtype=np.float32)[None],
        len(vertices), axis=0,
    )
    mesh = paz.graphics.Mesh(
        jp.asarray(vertices), jp.asarray(colors), jp.asarray(camera_to_world),
        material, jp.asarray(faces), jp.asarray(unique_edges(faces)),
    )
    floor_material = paz.graphics.Material(
        jp.asarray([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    return paz.graphics.Scene([
        paz.graphics.Plane(material=floor_material), mesh
    ]), camera_to_world


def render_prediction(camera_mesh, prediction, camera, image_shape, y_fov,
                      tiles, chunk_size, face_chunk_size, shadows):
    world_to_camera = camera_transform(camera)
    scene, camera_to_world = build_scene(
        camera_mesh, prediction, world_to_camera
    )
    light_camera = np.append(
        prediction["light_position_camera_xyz"], 1.0
    )
    light_world = np.asarray(camera_to_world) @ light_camera
    light = paz.graphics.PointLight(
        jp.full(3, prediction["light_intensity"]), jp.asarray(light_world[:3])
    )
    rgb, depth = paz.graphics.render(
        image_shape, np.deg2rad(y_fov), world_to_camera, scene, None, light,
        tiles=(tiles, tiles), chunk_size=chunk_size,
        face_chunk_size=face_chunk_size, shadows=shadows,
    )
    return np.asarray(rgb), np.asarray(depth), camera_to_world, light_world[:3]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/synthetic_rgbd_1000_v4"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, default=None,
                        help="Defaults to <experiment>/marching_cubes_meshes.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Defaults to <experiment>/best.keras.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--clear-caches-every", type=int, default=0)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.tiles, args.chunk_size, args.face_chunk_size) <= 0:
        raise ValueError("render chunk and tile sizes must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    if args.clear_caches_every < 0:
        raise ValueError("cache interval cannot be negative")
    meshes = args.meshes or args.experiment / "marching_cubes_meshes"
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "voxel_sdf_render_predictions"
    for name in ("rendered_rgb", "rendered_depth", "comparisons", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)
    mesh_paths = sorted(meshes.glob("*.ply"))
    if args.max_samples is not None:
        mesh_paths = mesh_paths[:args.max_samples]
    if not mesh_paths:
        raise ValueError(f"No predicted PLY meshes found in {meshes}")

    normalization = load_json(args.experiment / "normalization.json")
    preprocessing = load_json(args.experiment / "input_preprocessing.json")
    statistics = normalization["targets"]
    shape_names = normalization["shape_names"]
    voxel_resolution = int(preprocessing["voxel_resolution"])
    voxel_bounds = preprocessing["voxel_bounds_camera_xyz"]
    y_fov = float(preprocessing["y_fov_degrees"])
    max_depth = float(preprocessing["max_depth"])
    model = keras.models.load_model(model_path, compile=False)
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    expected = {"rgb", "depth_voxel", "sdf_query_points"}
    if input_names != expected or SDF_OUTPUT_NAME not in model.output_names:
        raise ValueError(
            f"Expected voxel-SDF inputs {sorted(expected)}, got "
            f"{sorted(input_names)}"
        )

    rows = []
    print(f"Loaded {model_path}; rendering {len(mesh_paths)} camera meshes")
    for index, mesh_path in enumerate(mesh_paths):
        sample_id = mesh_path.stem
        metadata = load_json(
            args.dataset / "metadata" / f"{sample_id}.json"
        )
        rgb, target_depth = load_rgb_depth(metadata, args.dataset, max_depth)
        voxel = depth_to_voxel(
            target_depth, voxel_bounds, voxel_resolution, y_fov
        )
        pipeline_start = time.perf_counter()
        model_start = time.perf_counter()
        raw = model.predict({
            "rgb": rgb[None], "depth_voxel": voxel[None],
            "sdf_query_points": np.zeros((1, 1, 3), dtype=np.float32),
        }, verbose=0)
        jax.tree.map(lambda value: jax.block_until_ready(value), raw)
        model_time_ms = (time.perf_counter() - model_start) * 1000.0
        prediction = decode_prediction(
            raw, statistics, shape_names,
            normalization.get("material_definition"),
        )
        mesh_start = time.perf_counter()
        camera_mesh = load_camera_mesh(mesh_path)
        mesh_time_ms = (time.perf_counter() - mesh_start) * 1000.0
        render_start = time.perf_counter()
        rendered_rgb, rendered_depth, camera_to_world, light_world = (
            render_prediction(
                camera_mesh, prediction, metadata["camera"], rgb.shape[:2],
                y_fov, args.tiles, args.chunk_size, args.face_chunk_size,
                not args.no_shadows,
            )
        )
        jax.block_until_ready(rendered_rgb)
        jax.block_until_ready(rendered_depth)
        render_time_ms = (time.perf_counter() - render_start) * 1000.0
        rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)
        rendered_depth = np.where(
            (rendered_depth > 0.0) & (rendered_depth <= max_depth),
            rendered_depth, 0.0,
        ).astype(np.float32)
        pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000.0
        metrics = evaluate_render(
            rendered_rgb, rgb, rendered_depth, target_depth
        )
        metrics.update({
            "model_inference_time_ms": model_time_ms,
            "mesh_preparation_time_ms": mesh_time_ms,
            "render_time_ms": render_time_ms,
            "prediction_render_pipeline_time_ms": pipeline_time_ms,
        })
        rows.append({"sample_id": sample_id, **metrics})
        paz.image.write(
            str(output / "rendered_rgb" / f"{sample_id}.png"),
            (rendered_rgb * 255.0).astype(np.uint8),
        )
        np.save(output / "rendered_depth" / f"{sample_id}.npy",
                rendered_depth)
        save_comparison(
            output / "comparisons" / f"{sample_id}.png", rgb,
            rendered_rgb, target_depth, rendered_depth, metrics,
        )
        payload = {
            "source_sample": sample_id,
            "source_camera_frame_mesh": str(mesh_path),
            "prediction": prediction,
            "camera": metadata["camera"],
            "camera_to_world_4x4": camera_to_world,
            "light_position_world_xyz": light_world,
            "metrics": metrics,
        }
        with (output / "metadata" / f"{sample_id}.json").open("w") as file:
            json.dump(jsonify(payload), file, indent=2)
        print(
            f"Rendered {index + 1}/{len(mesh_paths)}: {sample_id}, "
            f"RGB MAE={metrics['rgb_mae_0_1']:.4f}, "
            f"depth MAE={metrics['depth_mae_m']:.4f} m, "
            f"model={model_time_ms:.1f} ms, render={render_time_ms:.1f} ms"
        )
        if (args.clear_caches_every > 0
                and (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["mesh_coordinate_frame"] = "camera"
    summary["timing_excludes_marching_cubes_extraction"] = True
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(f"Saved voxel-SDF render evaluation to {output}")


if __name__ == "__main__":
    main()
