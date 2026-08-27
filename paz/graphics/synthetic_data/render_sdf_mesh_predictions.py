"""Render predicted SDF meshes with predicted physical scene parameters.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.render_sdf_mesh_predictions \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --experiment experiments/resnet18_rgbd_sdf
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import jax
import jax.numpy as jp
import keras
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paz
# Register the custom SDF layer and metric before loading a saved model.
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_sdf import (
    SDF_OUTPUT_NAME,
)


MATERIAL_NAMES = (
    "color_r", "color_g", "color_b", "ambient", "diffuse", "specular",
    "shininess",
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def denormalize(values, statistics):
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    deviation = np.asarray(
        statistics["standard_deviation"], dtype=np.float32
    )
    return np.asarray(values, dtype=np.float32) * deviation + mean


def rotation_6d_to_matrix(values, epsilon=1e-8):
    first, second = values[:3], values[3:]
    axis_x = first / max(np.linalg.norm(first), epsilon)
    second = second - np.dot(axis_x, second) * axis_x
    axis_y = second / max(np.linalg.norm(second), epsilon)
    axis_z = np.cross(axis_x, axis_y)
    return np.stack([axis_x, axis_y, axis_z], axis=-1)


def decode_prediction(raw, statistics, shape_names,
                      material_definition=None):
    translation = denormalize(
        raw["object_translation"][0], statistics["object_translation"]
    )
    orientation = np.asarray(raw["object_orientation_6d"][0])
    scale = max(float(denormalize(
        raw["object_scale"][0], statistics["object_scale"]
    )[0]), 1e-3)
    shape_probabilities = np.asarray(raw["shape"][0])
    material_values = denormalize(
        raw["material"][0], statistics["material"]
    )
    if material_definition and "values" in material_definition:
        by_name = dict(zip(material_definition["values"], material_values))
        material_values = np.asarray([
            by_name[name] for name in MATERIAL_NAMES
        ], dtype=np.float32)
    material_values[:6] = np.clip(material_values[:6], 0.0, 1.0)
    material_values[6] = np.clip(material_values[6], 1.0, 500.0)
    light_position = denormalize(
        raw["light_position"][0], statistics["light_position"]
    )
    light_intensity = max(float(denormalize(
        raw["light_intensity"][0], statistics["light_intensity"]
    )[0]), 0.0)
    return {
        "object_translation_camera_xyz": translation,
        "object_orientation_camera_6d": orientation,
        "object_rotation_camera_3x3": rotation_6d_to_matrix(orientation),
        "object_scale": scale,
        "shape": shape_names[int(np.argmax(shape_probabilities))],
        "shape_probabilities": shape_probabilities,
        "material": {
            "color_rgb": material_values[:3],
            "ambient": float(material_values[3]),
            "diffuse": float(material_values[4]),
            "specular": float(material_values[5]),
            "shininess": float(material_values[6]),
        },
        "light_position_camera_xyz": light_position,
        "light_intensity": light_intensity,
    }


def load_inputs(metadata, dataset, max_depth):
    bgr = cv2.imread(str(dataset / metadata["rgb"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(dataset / metadata["rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(dataset / metadata["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth dimensions do not match")
    normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)[..., None]
    return rgb, depth, normalized_depth


def camera_transform(camera):
    return paz.SE3.view_transform(
        jp.asarray(camera["position_world_xyz"]),
        jp.asarray(camera["target_world_xyz"]),
        jp.asarray([0.0, 1.0, 0.0]),
    )


def load_sdf_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangles found in {path}")
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except ModuleNotFoundError:
        # trimesh uses optional networkx for connected-component winding.
        pass
    return mesh


def unique_edges(faces):
    edges = np.concatenate([
        faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]
    ])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def build_scene(mesh, prediction, world_to_camera):
    object_to_camera = np.eye(4, dtype=np.float32)
    object_to_camera[:3, :3] = (
        prediction["object_rotation_camera_3x3"]
        * prediction["object_scale"]
    )
    object_to_camera[:3, 3] = prediction[
        "object_translation_camera_xyz"
    ]
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera))
    object_to_world = camera_to_world @ object_to_camera
    values = prediction["material"]
    material = paz.graphics.Material(
        jp.asarray(values["color_rgb"]), values["ambient"],
        values["diffuse"], values["specular"], values["shininess"],
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = np.repeat(
        np.asarray(values["color_rgb"], dtype=np.float32)[None],
        len(vertices), axis=0,
    )
    predicted_mesh = paz.graphics.Mesh(
        jp.asarray(vertices), jp.asarray(colors), jp.asarray(object_to_world),
        material, jp.asarray(faces), jp.asarray(unique_edges(faces)),
    )
    floor_material = paz.graphics.Material(
        jp.asarray([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    scene = paz.graphics.Scene([
        paz.graphics.Plane(material=floor_material), predicted_mesh
    ])
    return scene, object_to_camera, object_to_world


def render(mesh, prediction, camera, image_shape, y_fov, tiles, chunk_size,
           face_chunk_size, shadows):
    world_to_camera = camera_transform(camera)
    scene, object_to_camera, object_to_world = build_scene(
        mesh, prediction, world_to_camera
    )
    light_camera = np.append(
        prediction["light_position_camera_xyz"], 1.0
    )
    light_world = np.linalg.inv(np.asarray(world_to_camera)) @ light_camera
    light = paz.graphics.PointLight(
        jp.full(3, prediction["light_intensity"]), jp.asarray(light_world[:3])
    )
    rgb, depth = paz.graphics.render(
        image_shape, np.deg2rad(y_fov), world_to_camera, scene, None, light,
        tiles=(tiles, tiles), chunk_size=chunk_size,
        face_chunk_size=face_chunk_size, shadows=shadows,
    )
    return (np.asarray(rgb), np.asarray(depth), object_to_camera,
            object_to_world, light_world[:3])


def evaluate_render(rendered_rgb, target_rgb, rendered_depth, target_depth):
    rgb_difference = rendered_rgb.astype(np.float64) - target_rgb
    rgb_mse = float(np.mean(np.square(rgb_difference)))
    target_valid = target_depth > 0.0
    rendered_valid = rendered_depth > 0.0
    intersection = target_valid & rendered_valid
    union = target_valid | rendered_valid
    if np.any(intersection):
        depth_difference = rendered_depth[intersection] - target_depth[
            intersection
        ]
        depth_mae = float(np.mean(np.abs(depth_difference)))
        depth_rmse = float(np.sqrt(np.mean(np.square(depth_difference))))
    else:
        depth_mae = depth_rmse = float("nan")
    return {
        "rgb_mae_0_1": float(np.mean(np.abs(rgb_difference))),
        "rgb_psnr_db": (100.0 if rgb_mse < 1e-10 else
                        float(-10.0 * np.log10(rgb_mse))),
        "depth_mae_m": depth_mae,
        "depth_rmse_m": depth_rmse,
        "valid_depth_iou": float(
            np.sum(intersection) / max(np.sum(union), 1)
        ),
    }


def save_comparison(path, target_rgb, rendered_rgb, target_depth,
                    rendered_depth, metrics):
    rgb_error = np.mean(np.abs(rendered_rgb - target_rgb), axis=-1)
    valid = (target_depth > 0.0) & (rendered_depth > 0.0)
    depth_error = np.zeros_like(target_depth)
    depth_error[valid] = np.abs(rendered_depth[valid] - target_depth[valid])
    positive_depth = np.concatenate([
        target_depth[target_depth > 0.0], rendered_depth[rendered_depth > 0.0]
    ])
    depth_max = float(np.percentile(positive_depth, 99.0)) \
        if positive_depth.size else 1.0
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 1].imshow(np.clip(rendered_rgb, 0.0, 1.0))
    axes[0, 1].set_title("Rendered prediction")
    image = axes[0, 2].imshow(rgb_error, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(f"RGB absolute error\nMAE={metrics['rgb_mae_0_1']:.4f}")
    figure.colorbar(image, ax=axes[0, 2], fraction=0.046)
    image = axes[1, 0].imshow(target_depth, cmap="viridis", vmin=0.0,
                              vmax=depth_max)
    axes[1, 0].set_title("Original depth [m]")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
    image = axes[1, 1].imshow(rendered_depth, cmap="viridis", vmin=0.0,
                              vmax=depth_max)
    axes[1, 1].set_title("Rendered depth [m]")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    image = axes[1, 2].imshow(depth_error, cmap="magma", vmin=0.0,
                              vmax=max(float(np.percentile(
                                  depth_error[valid], 99.0
                              )) if np.any(valid) else 1.0, 1e-6))
    axes[1, 2].set_title(
        f"Depth absolute error [m]\nMAE={metrics['depth_mae_m']:.4f}"
    )
    figure.colorbar(image, ax=axes[1, 2], fraction=0.046)
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def jsonify(value):
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, (np.ndarray, jp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(rows):
    result = {"num_samples": len(rows)}
    for key in rows[0]:
        if key == "sample_id":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            result[key] = {
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "standard_deviation": float(finite.std()),
            }
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/synthetic_rgbd_1000_v4"))
    parser.add_argument("--experiment", type=Path,
                        default=Path("experiments/resnet18_rgbd_sdf"))
    parser.add_argument("--meshes", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--clear-caches-every", type=int, default=1)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (args.max_depth, args.y_fov, args.tiles, args.chunk_size,
                args.face_chunk_size)
    if any(value <= 0 for value in positive):
        raise ValueError("render sizes and ranges must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    meshes = args.meshes or args.experiment / "sdf_preview_meshes"
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "sdf_render_predictions"
    for name in ("rendered_rgb", "rendered_depth", "comparisons", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)
    mesh_paths = sorted(meshes.glob("*.ply"))
    if args.max_samples is not None:
        mesh_paths = mesh_paths[:args.max_samples]
    if not mesh_paths:
        raise ValueError(f"No predicted meshes found in {meshes}")

    normalization = load_json(args.experiment / "normalization.json")
    statistics = normalization["targets"]
    shape_names = normalization["shape_names"]
    model = keras.models.load_model(model_path, compile=False)
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    expected = {"rgb", "depth", "sdf_query_points"}
    if input_names != expected or SDF_OUTPUT_NAME not in model.output_names:
        raise ValueError(
            f"Expected SDF model inputs {sorted(expected)}, got "
            f"{sorted(input_names)}"
        )
    rows = []
    print(f"Loaded {model_path}; rendering {len(mesh_paths)} SDF meshes")
    for index, mesh_path in enumerate(mesh_paths):
        sample_id = mesh_path.stem
        metadata = load_json(
            args.dataset / "metadata" / f"{sample_id}.json"
        )
        input_rgb, target_depth, normalized_depth = load_inputs(
            metadata, args.dataset, args.max_depth
        )
        raw = model.predict({
            "rgb": input_rgb[None],
            "depth": normalized_depth[None],
            "sdf_query_points": np.zeros((1, 1, 3), dtype=np.float32),
        }, verbose=0)
        prediction = decode_prediction(
            raw, statistics, shape_names,
            normalization.get("material_definition"),
        )
        mesh = load_sdf_mesh(mesh_path)
        (rendered_rgb, rendered_depth, object_to_camera, object_to_world,
         light_world) = render(
            mesh, prediction, metadata["camera"], input_rgb.shape[:2],
            args.y_fov, args.tiles, args.chunk_size, args.face_chunk_size,
            not args.no_shadows,
        )
        rendered_depth = np.where(
            (rendered_depth > 0.0) & (rendered_depth <= args.max_depth),
            rendered_depth, 0.0,
        ).astype(np.float32)
        rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)
        metrics = evaluate_render(
            rendered_rgb, input_rgb, rendered_depth, target_depth
        )
        rows.append({"sample_id": sample_id, **metrics})
        rendered_uint8 = (rendered_rgb * 255.0).astype(np.uint8)
        paz.image.write(
            str(output / "rendered_rgb" / f"{sample_id}.png"),
            rendered_uint8,
        )
        np.save(output / "rendered_depth" / f"{sample_id}.npy",
                rendered_depth)
        save_comparison(
            output / "comparisons" / f"{sample_id}.png", input_rgb,
            rendered_rgb, target_depth, rendered_depth, metrics,
        )
        payload = {
            "source_sample": sample_id,
            "source_sdf_mesh": str(mesh_path),
            "prediction": prediction,
            "camera": metadata["camera"],
            "object_to_camera_4x4": object_to_camera,
            "object_to_world_4x4": object_to_world,
            "light_position_world_xyz": light_world,
            "metrics": metrics,
        }
        with (output / "metadata" / f"{sample_id}.json").open("w") as file:
            json.dump(jsonify(payload), file, indent=2)
        print(
            f"Rendered {index + 1}/{len(mesh_paths)}: {sample_id}, "
            f"RGB MAE={metrics['rgb_mae_0_1']:.4f}, "
            f"depth MAE={metrics['depth_mae_m']:.4f} m"
        )
        if (args.clear_caches_every > 0
                and (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summarize(rows)), file, indent=2)
    print(f"Saved rendered comparisons to {output}")


if __name__ == "__main__":
    main()
