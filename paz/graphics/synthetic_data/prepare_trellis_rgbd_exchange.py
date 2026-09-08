"""Prepare RGB images and metric RGB-D alignment for official TRELLIS.

Run this script in the PAZ/Keras environment.  Its output is deliberately a
plain file-based interface so that TRELLIS can run in a separate environment::

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
      paz.graphics.synthetic_data.prepare_trellis_rgbd_exchange \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --alignment-experiment experiments/resnet18_rgbd_physical_validation \
      --output experiments/trellis_exchange

The exchange contains ``manifest.json``, copied RGB images, and per-sample JSON
files with translation, rotation, scale, camera parameters, and ground truth.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import keras
import numpy as np


FORMAT_NAME = "paz-trellis-rgbd-exchange"
FORMAT_VERSION = 1


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def jsonify(value):
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def denormalize(values, statistics):
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    deviation = np.asarray(
        statistics["standard_deviation"], dtype=np.float32
    )
    return np.asarray(values, dtype=np.float32) * deviation + mean


def rotation_6d_to_matrix(values, epsilon=1e-8):
    """Map two predicted axes to a valid rotation with Gram--Schmidt."""
    values = np.asarray(values, dtype=np.float64)
    first, second = values[:3], values[3:]
    axis_x = first / max(float(np.linalg.norm(first)), epsilon)
    second = second - np.dot(axis_x, second) * axis_x
    axis_y = second / max(float(np.linalg.norm(second)), epsilon)
    axis_z = np.cross(axis_x, axis_y)
    return np.stack([axis_x, axis_y, axis_z], axis=-1)


def prediction_dict(model, prediction):
    if isinstance(prediction, dict):
        return prediction
    if not isinstance(prediction, (list, tuple)):
        prediction = [prediction]
    return dict(zip(model.output_names, prediction))


def load_rgbd(metadata, dataset, max_depth):
    rgb_path = dataset / metadata["rgb"]
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_path = dataset / metadata["depth"]
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    depth = np.where(valid, depth, 0.0)
    return rgb, np.clip(depth / max_depth, 0.0, 1.0)[..., None], rgb_path


def predict_alignment(model, rgb, depth, statistics):
    raw = prediction_dict(model, model.predict({
        "rgb": rgb.astype(np.float32)[None] / 255.0,
        "depth": depth[None],
    }, verbose=0))
    required = {
        "object_translation", "object_orientation_6d", "object_scale"
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Alignment model lacks outputs: {sorted(missing)}")
    translation = denormalize(
        raw["object_translation"][0], statistics["object_translation"]
    )
    orientation = np.asarray(raw["object_orientation_6d"][0])
    scale = float(denormalize(
        raw["object_scale"][0], statistics["object_scale"]
    )[0])
    return {
        "translation_camera_xyz": translation,
        "orientation_camera_6d": orientation,
        "rotation_camera_3x3": rotation_6d_to_matrix(orientation),
        "scale_metres_per_canonical_unit": max(scale, 1e-4),
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--alignment-experiment", type=Path, required=True)
    parser.add_argument("--alignment-model", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"),
        default="test",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    experiment = args.alignment_experiment.resolve()
    output = args.output.resolve()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    rgb_output = output / "rgb"
    alignment_output = output / "alignment"
    rgb_output.mkdir(parents=True, exist_ok=True)
    alignment_output.mkdir(parents=True, exist_ok=True)

    normalization = load_json(experiment / "normalization.json")
    preprocessing_path = experiment / "input_preprocessing.json"
    preprocessing = (load_json(preprocessing_path)
                     if preprocessing_path.is_file() else {})
    max_depth = float(preprocessing.get("max_depth", 10.0))
    model_path = (args.alignment_model.resolve()
                  if args.alignment_model else experiment / "best.keras")
    model = keras.models.load_model(model_path, compile=False)
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    if input_names != {"rgb", "depth"}:
        raise ValueError(
            "Expected an RGB-D physical model with inputs ['depth', 'rgb']; "
            f"got {sorted(input_names)}"
        )

    split_path = experiment / "split.json"
    split_data = load_json(split_path) if split_path.is_file() else None
    if args.split == "all":
        sample_ids = sorted(
            path.stem for path in (dataset / "metadata").glob("*.json")
        )
    elif split_data is not None:
        if args.split not in split_data:
            raise KeyError(f"Split {args.split!r} is absent from {split_path}")
        sample_ids = list(split_data[args.split])
    else:
        raise FileNotFoundError(
            f"{split_path} is required unless --split all is selected"
        )
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]

    samples, failures = [], []
    for index, sample_id in enumerate(sample_ids):
        try:
            target_json = alignment_output / f"{sample_id}.json"
            target_rgb = rgb_output / f"{sample_id}.png"
            if target_json.exists() and not args.overwrite:
                payload = load_json(target_json)
            else:
                metadata_path = dataset / "metadata" / f"{sample_id}.json"
                metadata = load_json(metadata_path)
                rgb, depth, source_rgb = load_rgbd(
                    metadata, dataset, max_depth
                )
                start = time.perf_counter()
                alignment = predict_alignment(
                    model, rgb, depth, normalization["targets"]
                )
                inference_ms = (time.perf_counter() - start) * 1000.0
                shutil.copy2(source_rgb, target_rgb)
                payload = {
                    "sample_id": sample_id,
                    "rgb": f"rgb/{sample_id}.png",
                    "metric_alignment": alignment,
                    "camera": metadata["camera"],
                    "ground_truth": {
                        "object": metadata.get("object"),
                        "shape": metadata.get("shape"),
                    },
                    "alignment_inference_time_ms": inference_ms,
                }
                with target_json.open("w") as file:
                    json.dump(jsonify(payload), file, indent=2)
            samples.append({
                "sample_id": sample_id,
                "rgb": payload["rgb"],
                "alignment": f"alignment/{sample_id}.json",
            })
            print(f"Prepared {index + 1}/{len(sample_ids)}: {sample_id}")
        except Exception as error:
            failures.append({
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
            print(f"Skipped {sample_id}: {type(error).__name__}: {error}")

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "split": args.split,
        "alignment_model": str(model_path),
        "max_depth_metres": max_depth,
        "num_requested": len(sample_ids),
        "num_prepared": len(samples),
        "samples": samples,
        "failures": failures,
    }
    with (output / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved {len(samples)} exchange samples to {output}")


if __name__ == "__main__":
    main()
