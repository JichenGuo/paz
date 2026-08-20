"""Render held-out test objects predicted by the fixed-condition CNN.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.render_fixed_object_predictions \
        --dataset datasets/synthetic_rgbd_fixed_1000 \
        --experiment experiments/fixed_camera_light_cnn_no_validation

The script reads test IDs from the experiment split, loads best.keras,
decodes normalized predictions, and renders them with the fixed acquisition
configuration saved by generate_fixed_synthetic_rgbd.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
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
from paz.graphics.synthetic_data.train_fixed_synthetic_object_cnn import (
    SHAPE_NAMES,
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


def load_rgbd(record, dataset, max_depth):
    rgb_path = dataset / record["rgb"]
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    depth = np.load(dataset / record["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)
    rgbd = np.concatenate([rgb, normalized_depth[..., None]], axis=-1)
    return rgbd, rgb


def decode_predictions(raw_predictions, statistics):
    """Restores physical units and constructs a valid camera-frame pose."""
    translation = denormalize(
        raw_predictions["object_translation"][0],
        statistics["object_translation"],
    )
    scale = float(denormalize(
        raw_predictions["object_scale"][0],
        statistics["object_scale"],
    )[0])
    scale = max(scale, 1e-3)
    material = denormalize(
        raw_predictions["material"][0], statistics["material"]
    )
    material[:3] = np.clip(material[:3], 0.0, 1.0)
    material[3:6] = np.clip(material[3:6], 0.0, 1.0)
    material[6] = np.clip(material[6], 1.0, 500.0)
    orientation_6d = np.asarray(
        raw_predictions["object_orientation_6d"][0], dtype=np.float64
    )
    rotation = rotation_6d_to_matrix(
        orientation_6d[:3], orientation_6d[3:]
    )
    shape_probabilities = np.asarray(raw_predictions["shape"][0])
    shape_index = int(np.argmax(shape_probabilities))
    return {
        "translation_camera_xyz": translation,
        "rotation_camera_3x3": rotation,
        "orientation_camera_6d": orientation_6d,
        "scale": scale,
        "shape": SHAPE_NAMES[shape_index],
        "shape_probabilities": shape_probabilities,
        "material": material,
    }


def build_predicted_scene(prediction, world_to_camera):
    """Builds a PAZ scene from a decoded camera-frame object prediction."""
    object_to_camera = np.eye(4, dtype=np.float32)
    object_to_camera[:3, :3] = (
        prediction["rotation_camera_3x3"] * prediction["scale"]
    )
    object_to_camera[:3, 3] = prediction["translation_camera_xyz"]
    object_to_world = np.linalg.inv(world_to_camera) @ object_to_camera

    values = prediction["material"]
    material = paz.graphics.Material(
        jp.array(values[:3]), float(values[3]), float(values[4]),
        float(values[5]), float(values[6]),
    )
    primitive = SHAPE_BUILDERS[prediction["shape"]](
        jp.array(object_to_world), material
    )
    floor_material = paz.graphics.Material(
        jp.array([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    scene = paz.graphics.Scene([
        paz.graphics.Plane(material=floor_material), primitive
    ])
    return scene, object_to_world


def render_prediction(prediction, configuration, shadows, tiles, chunk_size):
    camera_position = jp.array(configuration["camera_position"])
    camera_target = jp.array(configuration["camera_target"])
    world_to_camera = paz.SE3.view_transform(
        camera_position, camera_target, jp.array([0.0, 1.0, 0.0])
    )
    scene, object_to_world = build_predicted_scene(
        prediction, np.asarray(world_to_camera)
    )
    intensity = jp.full(3, configuration["light_intensity"])
    light = paz.graphics.PointLight(
        intensity, jp.array(configuration["light_position"])
    )
    height, width = configuration["image_size"]
    rgb, depth = paz.graphics.render(
        (height, width), np.deg2rad(configuration["y_fov_degrees"]),
        world_to_camera, scene, None, light, shadows=shadows,
        tiles=(tiles, tiles), chunk_size=chunk_size,
    )
    return np.asarray(rgb), np.asarray(depth), object_to_world


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None,
                        help="Defaults to <experiment>/best.keras.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <experiment>/test_predictions.")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--clear-caches-every", type=int, default=10)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.max_depth, args.tiles, args.chunk_size) <= 0:
        raise ValueError("depth, tiles, and chunk size must be positive")
    if args.clear_caches_every < 0:
        raise ValueError("cache interval must be nonnegative")
    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "test_predictions"
    for name in ("rendered_rgb", "rendered_depth", "comparison", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)

    split = load_json(args.experiment / "split.json")
    if "test" not in split:
        raise ValueError("Experiment split.json does not contain a test split")
    statistics = load_json(
        args.experiment / "normalization.json"
    )["targets"]
    configuration = load_json(args.dataset / "fixed_configuration.json")
    model = keras.models.load_model(model_path, compile=False)
    print(f"Loaded {model_path}; rendering {len(split['test'])} test samples")

    for item_index, sample_id in enumerate(split["test"]):
        record = load_json(args.dataset / "metadata" / f"{sample_id}.json")
        rgbd, input_rgb = load_rgbd(record, args.dataset, args.max_depth)
        raw_predictions = model.predict(rgbd[None], verbose=0)
        prediction = decode_predictions(raw_predictions, statistics)
        rendered_rgb, rendered_depth, object_to_world = render_prediction(
            prediction, configuration, not args.no_shadows, args.tiles,
            args.chunk_size,
        )
        rendered_uint8 = np.clip(
            rendered_rgb * 255.0, 0.0, 255.0
        ).astype(np.uint8)
        input_uint8 = np.clip(input_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        comparison = np.concatenate([input_uint8, rendered_uint8], axis=1)
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
            "object_to_world_4x4": object_to_world,
            "rendered_rgb": f"rendered_rgb/{sample_id}.png",
            "rendered_depth": f"rendered_depth/{sample_id}.npy",
            "comparison": f"comparison/{sample_id}.png",
        }
        with (output / "metadata" / f"{sample_id}.json").open("w") as file:
            json.dump(jsonify(payload), file, indent=2)
        print(f"Rendered {item_index + 1}/{len(split['test'])}: {sample_id}")
        if (args.clear_caches_every > 0
                and (item_index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
