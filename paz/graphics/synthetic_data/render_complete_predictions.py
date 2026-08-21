"""Render held-out RGB-D samples from all predicted physical parameters.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.render_complete_predictions \
        --dataset datasets/synthetic_rgbd_1000_v3 \
        --experiment experiments/02_complete_physical_cnn_no_validation

Object pose and light position are predicted in camera coordinates. The saved
camera position and target convert them back to world coordinates so the
object, floor, lighting, and viewpoint can be rendered in the original frame.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import jax
import jax.numpy as jp
import keras
import numpy as np

import paz
from paz.graphics.synthetic_data.generate_synthetic_rgbd import (
    SHAPE_BUILDERS,
    jsonify,
    rotation_6d_to_matrix,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import SHAPE_NAMES
from paz.graphics.synthetic_data.train_synthetic_rgbd_cnn import (
    CUBE_SYMMETRIES,
)


MATERIAL_NAMES = (
    "color_r", "color_g", "color_b", "ambient", "diffuse", "specular",
    "shininess",
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def denormalize(prediction, statistics):
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    standard_deviation = np.asarray(
        statistics["standard_deviation"], dtype=np.float32
    )
    return np.asarray(prediction) * standard_deviation + mean


def load_rgbd(record, test_split, max_depth):
    rgb_path = test_split / record["rgb"]
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    depth = np.load(test_split / record["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)
    return (
        np.concatenate([rgb, normalized_depth[..., None]], axis=-1),
        rgb,
        depth,
    )


def decode_predictions(raw, statistics):
    translation = denormalize(
        raw["object_translation"][0], statistics["object_translation"]
    )
    scale = float(denormalize(
        raw["object_scale"][0], statistics["object_scale"]
    )[0])
    scale = max(scale, 1e-3)
    orientation_6d = np.asarray(
        raw["object_orientation_6d"][0], dtype=np.float64
    )
    rotation = rotation_6d_to_matrix(
        orientation_6d[:3], orientation_6d[3:]
    )
    shape_probabilities = np.asarray(raw["shape"][0], dtype=np.float32)
    shape = SHAPE_NAMES[int(np.argmax(shape_probabilities))]

    material = denormalize(raw["material"][0], statistics["material"])
    material[:6] = np.clip(material[:6], 0.0, 1.0)
    material[6] = np.clip(material[6], 1.0, 500.0)
    light_position = denormalize(
        raw["light_position"][0], statistics["light_position"]
    )
    light_intensity = float(denormalize(
        raw["light_intensity"][0], statistics["light_intensity"]
    )[0])
    light_intensity = max(light_intensity, 0.0)
    return {
        "object": {
            "translation_camera_xyz": translation,
            "orientation_camera_6d": orientation_6d,
            "rotation_camera_3x3": rotation,
            "scale": scale,
        },
        "shape": {
            "type": shape,
            "probabilities": shape_probabilities,
        },
        "material": {
            "color_rgb": material[:3],
            "ambient": float(material[3]),
            "diffuse": float(material[4]),
            "specular": float(material[5]),
            "shininess": float(material[6]),
        },
        "light": {
            "position_camera_xyz": light_position,
            "intensity": light_intensity,
        },
    }


def camera_transform(camera):
    """Returns the world-to-camera transform from saved look-at parameters."""
    return paz.SE3.view_transform(
        jp.array(camera["position_world_xyz"]),
        jp.array(camera["target_world_xyz"]),
        jp.array([0.0, 1.0, 0.0]),
    )


def build_scene(prediction, world_to_camera):
    """Converts the camera-frame prediction into a world-frame PAZ scene."""
    object_prediction = prediction["object"]
    object_to_camera = np.eye(4, dtype=np.float32)
    object_to_camera[:3, :3] = (
        object_prediction["rotation_camera_3x3"]
        * object_prediction["scale"]
    )
    object_to_camera[:3, 3] = object_prediction[
        "translation_camera_xyz"
    ]
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera))
    object_to_world = camera_to_world @ object_to_camera
    values = prediction["material"]
    material = paz.graphics.Material(
        jp.array(values["color_rgb"]), values["ambient"], values["diffuse"],
        values["specular"], values["shininess"],
    )
    primitive = SHAPE_BUILDERS[prediction["shape"]["type"]](
        jp.array(object_to_world), material
    )
    floor_material = paz.graphics.Material(
        jp.array([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    scene = paz.graphics.Scene([
        paz.graphics.Plane(material=floor_material), primitive
    ])
    return scene, object_to_camera, object_to_world


def render_prediction(prediction, camera, image_size, y_fov, shadows, tiles,
                      chunk_size):
    world_to_camera = camera_transform(camera)
    scene, object_to_camera, object_to_world = build_scene(
        prediction, world_to_camera
    )
    light_prediction = prediction["light"]
    light_camera = np.append(
        light_prediction["position_camera_xyz"], 1.0
    )
    light_world = np.linalg.inv(np.asarray(world_to_camera)) @ light_camera
    light = paz.graphics.PointLight(
        jp.full(3, light_prediction["intensity"]),
        jp.array(light_world[:3]),
    )
    rgb, depth = paz.graphics.render(
        image_size, np.deg2rad(y_fov), world_to_camera, scene, None, light,
        shadows=shadows, tiles=(tiles, tiles), chunk_size=chunk_size,
    )
    return (
        np.asarray(rgb), np.asarray(depth), object_to_camera,
        object_to_world, light_world[:3],
    )


def stable_rotation_angle(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def symmetry_rotation_error_degrees(predicted, target, shape):
    """Returns the minimum physically distinct rotation error in degrees."""
    if shape == "sphere":
        return 0.0
    if shape == "cube":
        angles = [
            stable_rotation_angle((target @ symmetry).T @ predicted)
            for symmetry in CUBE_SYMMETRIES
        ]
        return float(np.rad2deg(min(angles)))
    # A closed cylinder is invariant to rotation around its vertical axis and
    # to reversing that axis, so only the unoriented Y axis is observable.
    cosine = np.clip(
        abs(np.dot(target[:, 1], predicted[:, 1])), -1.0, 1.0
    )
    return float(np.rad2deg(np.arccos(cosine)))


def evaluate_parameters(prediction, ground_truth):
    """Computes unit-aware errors for every predicted physical parameter."""
    target_orientation = ground_truth["object"]["orientation_camera_6d"]
    target_rotation = rotation_6d_to_matrix(
        target_orientation["vector_a"], target_orientation["vector_b"]
    )
    predicted_object = prediction["object"]
    target_translation = np.asarray(
        ground_truth["object"]["translation_camera_xyz"]
    )
    predicted_translation = np.asarray(
        predicted_object["translation_camera_xyz"]
    )
    target_light = np.asarray(
        ground_truth["light"]["position_camera_xyz"]
    )
    predicted_light = np.asarray(
        prediction["light"]["position_camera_xyz"]
    )
    target_material = ground_truth["material"]
    predicted_material = prediction["material"]
    metrics = {
        "translation_error_m": float(np.linalg.norm(
            predicted_translation - target_translation
        )),
        "orientation_error_degrees": symmetry_rotation_error_degrees(
            predicted_object["rotation_camera_3x3"], target_rotation,
            ground_truth["shape"]["type"],
        ),
        "scale_absolute_error": float(abs(
            predicted_object["scale"] - ground_truth["object"]["scale"]
        )),
        "shape_correct": int(
            prediction["shape"]["type"] == ground_truth["shape"]["type"]
        ),
        "light_position_error_m": float(np.linalg.norm(
            predicted_light - target_light
        )),
        "light_intensity_absolute_error": float(abs(
            prediction["light"]["intensity"]
            - ground_truth["light"]["intensity"]
        )),
    }
    target_values = (
        list(target_material["color_rgb"])
        + [target_material["ambient"], target_material["diffuse"],
           target_material["specular"], target_material["shininess"]]
    )
    predicted_values = (
        list(predicted_material["color_rgb"])
        + [predicted_material["ambient"], predicted_material["diffuse"],
           predicted_material["specular"], predicted_material["shininess"]]
    )
    for name, predicted_value, target_value in zip(
            MATERIAL_NAMES, predicted_values, target_values):
        metrics[f"material_{name}_absolute_error"] = float(
            abs(predicted_value - target_value)
        )
    return metrics


def evaluate_render(rendered_rgb, target_rgb, rendered_depth, target_depth):
    """Computes image fidelity and metric-depth reconstruction errors."""
    rgb_error = rendered_rgb.astype(np.float64) - target_rgb.astype(np.float64)
    rgb_mse = float(np.mean(np.square(rgb_error)))
    valid_depth = (target_depth > 0.0) & (rendered_depth > 0.0)
    if np.any(valid_depth):
        depth_error = (
            rendered_depth[valid_depth] - target_depth[valid_depth]
        )
        depth_mae = float(np.mean(np.abs(depth_error)))
        depth_rmse = float(np.sqrt(np.mean(np.square(depth_error))))
    else:
        depth_mae, depth_rmse = float("nan"), float("nan")
    return {
        "rgb_mae_0_1": float(np.mean(np.abs(rgb_error))),
        "rgb_psnr_db": (
            100.0 if rgb_mse < 1e-10 else float(-10.0 * np.log10(rgb_mse))
        ),
        "depth_mae_m": depth_mae,
        "depth_rmse_m": depth_rmse,
    }


def summarize_metrics(rows):
    """Aggregates continuous errors and classification accuracy."""
    summary = {"num_samples": len(rows)}
    for name in rows[0]:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if name == "shape_correct":
            summary["shape_accuracy"] = float(np.mean(values))
        elif finite.size:
            summary[name] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "standard_deviation": float(np.std(finite)),
            }
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Original dataset; used as camera fallback.")
    parser.add_argument("--test-split", type=Path, default=None,
                        help="Defaults to <experiment>/test_split.")
    parser.add_argument("--model", type=Path, default=None,
                        help="Defaults to <experiment>/best.keras.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <experiment>/test_predictions.")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--clear-caches-every", type=int, default=10)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.max_depth, args.y_fov, args.tiles, args.chunk_size) <= 0:
        raise ValueError("depth, field of view, tiles, and chunk must be positive")
    if args.clear_caches_every < 0:
        raise ValueError("cache interval must be nonnegative")
    test_split = args.test_split or args.experiment / "test_split"
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "test_predictions"
    for name in ("rendered_rgb", "rendered_depth", "comparison", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted((test_split / "metadata").glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No test metadata found in {test_split}")
    statistics = load_json(
        args.experiment / "normalization.json"
    )["targets"]
    model = keras.models.load_model(model_path, compile=False)
    print(f"Loaded {model_path}; rendering {len(metadata_paths)} test samples")
    metric_rows = []

    for item_index, metadata_path in enumerate(metadata_paths):
        sample_id = metadata_path.stem
        record = load_json(metadata_path)
        if "camera" not in record:
            if args.dataset is None:
                raise ValueError(
                    f"{metadata_path} has no camera metadata; pass --dataset"
                )
            source_metadata = (
                args.dataset / "metadata" / f"{sample_id}.json"
            )
            record["camera"] = load_json(source_metadata)["camera"]
        rgbd, input_rgb, target_depth = load_rgbd(
            record, test_split, args.max_depth
        )
        raw = model.predict(rgbd[None], verbose=0)
        prediction = decode_predictions(raw, statistics)
        (rendered_rgb, rendered_depth, object_to_camera, object_to_world,
         light_world) = render_prediction(
            prediction, record["camera"], input_rgb.shape[:2], args.y_fov,
            not args.no_shadows, args.tiles, args.chunk_size,
        )
        rendered_uint8 = np.clip(
            rendered_rgb * 255.0, 0.0, 255.0
        ).astype(np.uint8)
        input_uint8 = np.clip(input_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        comparison = np.concatenate([input_uint8, rendered_uint8], axis=1)
        metrics = evaluate_parameters(prediction, record)
        metrics.update(evaluate_render(
            rendered_rgb, input_rgb, rendered_depth, target_depth
        ))
        metric_rows.append({"sample_id": sample_id, **metrics})
        paz.image.write(
            str(output / "rendered_rgb" / f"{sample_id}.png"),
            rendered_uint8,
        )
        paz.image.write(
            str(output / "comparison" / f"{sample_id}.png"), comparison
        )
        np.save(
            output / "rendered_depth" / f"{sample_id}.npy",
            rendered_depth.astype(np.float32),
        )
        payload = {
            "source_sample": sample_id,
            "prediction": prediction,
            "camera": record["camera"],
            "object_to_camera_4x4": object_to_camera,
            "object_to_world_4x4": object_to_world,
            "light_position_world_xyz": light_world,
            "render_coordinate_frame": "world",
            "metrics": metrics,
            "rendered_rgb": f"rendered_rgb/{sample_id}.png",
            "rendered_depth": f"rendered_depth/{sample_id}.npy",
            "comparison": f"comparison/{sample_id}.png",
        }
        with (output / "metadata" / f"{sample_id}.json").open("w") as file:
            json.dump(jsonify(payload), file, indent=2)
        print(f"Rendered {item_index + 1}/{len(metadata_paths)}: {sample_id}")
        if (args.clear_caches_every > 0
                and (item_index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = summarize_metrics([
        {key: value for key, value in row.items() if key != "sample_id"}
        for row in metric_rows
    ])
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(f"Saved evaluation summary to {output / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
