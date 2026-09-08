"""Combine official TRELLIS shape with RGB-D metric pose and scale.

This is an inference baseline, not a reimplementation of TRELLIS. The official
pretrained TRELLIS image-to-3D pipeline predicts an object-centric mesh from
RGB. A separately trained PAZ RGB-D physical model predicts camera-frame
translation, 6D orientation, and metric scale. The normalized TRELLIS mesh is
then placed with the predicted similarity transform::

    p_camera = scale * rotation @ p_canonical + translation

Recommended execution keeps Keras/JAX on CPU and reserves CUDA for TRELLIS::

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
      paz.graphics.synthetic_data.run_original_trellis_rgbd_alignment \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --alignment-experiment experiments/resnet18_rgbd_physical_validation \
      --output experiments/original_trellis_rgbd_alignment

The official TRELLIS repository and its dependencies must be installed in the
active environment. Model weights are loaded by the official pipeline from a
local path or from ``microsoft/TRELLIS-image-large``.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import keras
import numpy as np
import trimesh
from PIL import Image


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
    """Maps two predicted axes to SO(3) using Gram--Schmidt."""
    values = np.asarray(values, dtype=np.float64)
    first, second = values[:3], values[3:]
    axis_x = first / max(float(np.linalg.norm(first)), epsilon)
    second = second - np.dot(axis_x, second) * axis_x
    axis_y = second / max(float(np.linalg.norm(second)), epsilon)
    axis_z = np.cross(axis_x, axis_y)
    return np.stack([axis_x, axis_y, axis_z], axis=-1)


def decode_alignment(raw, statistics):
    translation = denormalize(
        raw["object_translation"][0], statistics["object_translation"]
    )
    orientation = np.asarray(
        raw["object_orientation_6d"][0], dtype=np.float32
    )
    scale = float(denormalize(
        raw["object_scale"][0], statistics["object_scale"]
    )[0])
    return {
        "translation_camera_xyz": translation,
        "orientation_camera_6d": orientation,
        "rotation_camera_3x3": rotation_6d_to_matrix(orientation),
        "scale_metres_per_canonical_unit": max(scale, 1e-4),
    }


def model_prediction_dict(model, prediction):
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
    depth = np.load(dataset / metadata["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
    depth = np.where(
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth),
        depth, 0.0,
    )
    normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)[..., None]
    return rgb, normalized_depth, rgb_path


def predict_metric_alignment(model, rgb, normalized_depth, statistics):
    inputs = {
        "rgb": rgb.astype(np.float32)[None] / 255.0,
        "depth": normalized_depth[None],
    }
    raw = model_prediction_dict(model, model.predict(inputs, verbose=0))
    required = {
        "object_translation", "object_orientation_6d", "object_scale"
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(
            f"Alignment model lacks outputs: {sorted(missing)}"
        )
    return decode_alignment(raw, statistics)


def trellis_mesh_to_numpy(mesh_result):
    """Converts the official MeshExtractResult into NumPy arrays."""
    vertices = mesh_result.vertices
    faces = mesh_result.faces
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().float().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected TRELLIS vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected TRELLIS faces: {faces.shape}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def convert_up_axis(mesh, source_up):
    """Converts a Z-up TRELLIS mesh to the Y-up scene convention."""
    if source_up == "y":
        return mesh
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    mesh.apply_transform(transform)
    return mesh


def normalize_canonical_mesh(mesh, mode):
    """Removes TRELLIS translation/scale while preserving proportions."""
    center = np.asarray(mesh.bounds).mean(axis=0)
    mesh.vertices -= center
    if mode == "none":
        factor = 1.0
    elif mode == "max-abs":
        factor = float(np.max(np.abs(mesh.vertices)))
    elif mode == "max-extent":
        factor = float(np.max(mesh.extents)) / 2.0
    else:
        raise ValueError(f"Unknown canonical normalization: {mode}")
    if not np.isfinite(factor) or factor <= 1e-8:
        raise ValueError("TRELLIS returned a degenerate mesh")
    mesh.vertices /= factor
    return mesh, center, factor


def view_transform(camera):
    """Returns the renderer-compatible world-to-camera look-at matrix."""
    eye = np.asarray(camera["position_world_xyz"], dtype=np.float64)
    target = np.asarray(camera["target_world_xyz"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    # Match the repository view transform exactly; its camera-left and
    # derived up vectors are intentionally not normalized.
    left = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    up = np.cross(left, forward)
    rotation = np.stack([left, up, -forward], axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ eye
    return transform


def place_mesh(canonical_mesh, alignment, camera):
    object_to_camera = np.eye(4, dtype=np.float64)
    object_to_camera[:3, :3] = (
        alignment["rotation_camera_3x3"]
        * alignment["scale_metres_per_canonical_unit"]
    )
    object_to_camera[:3, 3] = alignment["translation_camera_xyz"]
    world_to_camera = view_transform(camera)
    object_to_world = np.linalg.inv(world_to_camera) @ object_to_camera
    camera_mesh = canonical_mesh.copy()
    camera_mesh.apply_transform(object_to_camera)
    world_mesh = canonical_mesh.copy()
    world_mesh.apply_transform(object_to_world)
    return camera_mesh, world_mesh, object_to_camera, object_to_world


def rotation_angle_degrees(predicted, target):
    relative = np.asarray(target).T @ np.asarray(predicted)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def alignment_errors(alignment, metadata):
    target_translation = np.asarray(
        metadata["object"]["translation_camera_xyz"], dtype=np.float64
    )
    target_orientation = metadata["object"]["orientation_camera_6d"]
    target_values = np.asarray(
        target_orientation["vector_a"] + target_orientation["vector_b"]
    )
    target_rotation = rotation_6d_to_matrix(target_values)
    target_scale = float(metadata["object"]["scale"])
    return {
        "translation_error_m": float(np.linalg.norm(
            alignment["translation_camera_xyz"] - target_translation
        )),
        "orientation_error_degrees": rotation_angle_degrees(
            alignment["rotation_camera_3x3"], target_rotation
        ),
        "scale_absolute_error": abs(
            alignment["scale_metres_per_canonical_unit"] - target_scale
        ),
    }


def summarize(rows):
    summary = {"num_samples": len(rows)}
    if not rows:
        return summary
    for key in rows[0]:
        if key == "sample_id":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            summary[key] = {
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "standard_deviation": float(finite.std()),
            }
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--alignment-experiment", type=Path, required=True)
    parser.add_argument("--alignment-model", type=Path, default=None,
                        help="Defaults to <alignment-experiment>/best.keras.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trellis-model", default="microsoft/TRELLIS-image-large",
        help="Official Hugging Face model ID or local pretrained directory.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--sparse-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument("--trellis-up-axis", choices=("y", "z"),
                        default="z")
    parser.add_argument(
        "--canonical-normalization",
        choices=("max-abs", "max-extent", "none"), default="max-abs",
    )
    parser.add_argument("--no-trellis-preprocess", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    if min(args.sparse_steps, args.slat_steps) < 1:
        raise ValueError("TRELLIS sampling steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("canonical_meshes", "camera_meshes", "world_meshes",
                 "metadata"):
        (args.output / name).mkdir(parents=True, exist_ok=True)

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as error:
        raise ImportError(
            "Official TRELLIS is not installed. Clone microsoft/TRELLIS "
            "with its submodules, install its requirements, and make its "
            "repository importable before running this baseline."
        ) from error

    alignment_model_path = args.alignment_model or (
        args.alignment_experiment / "best.keras"
    )
    normalization = load_json(
        args.alignment_experiment / "normalization.json"
    )
    preprocessing_path = (
        args.alignment_experiment / "input_preprocessing.json"
    )
    preprocessing = load_json(preprocessing_path) \
        if preprocessing_path.is_file() else {}
    max_depth = float(preprocessing.get("max_depth", 10.0))
    alignment_model = keras.models.load_model(
        alignment_model_path, compile=False
    )
    input_names = {
        tensor.name.split(":")[0] for tensor in alignment_model.inputs
    }
    if input_names != {"rgb", "depth"}:
        raise ValueError(
            "The metric alignment model must have exactly the inputs "
            f"['depth', 'rgb']; got {sorted(input_names)}. Use a physical "
            "RGB-D model rather than a mesh-decoder model."
        )

    split_path = args.alignment_experiment / "split.json"
    if split_path.is_file():
        sample_ids = list(load_json(split_path)["test"])
    else:
        sample_ids = sorted(
            path.stem for path in (args.dataset / "metadata").glob("*.json")
        )
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model)
    pipeline.cuda()
    rows, failures = [], []
    for index, sample_id in enumerate(sample_ids):
        try:
            metadata = load_json(
                args.dataset / "metadata" / f"{sample_id}.json"
            )
            rgb, normalized_depth, rgb_path = load_rgbd(
                metadata, args.dataset, max_depth
            )
            alignment_start = time.perf_counter()
            alignment = predict_metric_alignment(
                alignment_model, rgb, normalized_depth,
                normalization["targets"],
            )
            alignment_ms = (time.perf_counter() - alignment_start) * 1000.0

            trellis_start = time.perf_counter()
            trellis_output = pipeline.run(
                Image.open(rgb_path).convert("RGB"),
                seed=args.seed + index,
                formats=["mesh"],
                preprocess_image=not args.no_trellis_preprocess,
                sparse_structure_sampler_params={
                    "steps": args.sparse_steps,
                    "cfg_strength": args.sparse_cfg,
                },
                slat_sampler_params={
                    "steps": args.slat_steps,
                    "cfg_strength": args.slat_cfg,
                },
            )
            trellis_ms = (time.perf_counter() - trellis_start) * 1000.0
            canonical = trellis_mesh_to_numpy(trellis_output["mesh"][0])
            canonical = convert_up_axis(canonical, args.trellis_up_axis)
            canonical, original_center, normalization_factor = (
                normalize_canonical_mesh(
                    canonical, args.canonical_normalization
                )
            )
            camera_mesh, world_mesh, object_to_camera, object_to_world = (
                place_mesh(canonical, alignment, metadata["camera"])
            )
            canonical_path = (
                args.output / "canonical_meshes" / f"{sample_id}.ply"
            )
            camera_path = args.output / "camera_meshes" / f"{sample_id}.ply"
            world_path = args.output / "world_meshes" / f"{sample_id}.ply"
            canonical.export(canonical_path)
            camera_mesh.export(camera_path)
            world_mesh.export(world_path)
            errors = alignment_errors(alignment, metadata)
            row = {
                "sample_id": sample_id,
                **errors,
                "alignment_inference_time_ms": alignment_ms,
                "trellis_inference_time_ms": trellis_ms,
                "num_vertices": len(canonical.vertices),
                "num_faces": len(canonical.faces),
            }
            rows.append(row)
            payload = {
                "sample_id": sample_id,
                "source_rgb": str(rgb_path),
                "trellis_model": args.trellis_model,
                "trellis_seed": args.seed + index,
                "trellis_up_axis": args.trellis_up_axis,
                "canonical_normalization": args.canonical_normalization,
                "trellis_original_bbox_center": original_center,
                "trellis_normalization_factor": normalization_factor,
                "metric_alignment": alignment,
                "object_to_camera_4x4": object_to_camera,
                "object_to_world_4x4": object_to_world,
                "paths": {
                    "canonical_mesh": str(canonical_path),
                    "camera_mesh": str(camera_path),
                    "world_mesh": str(world_path),
                },
                "metrics": row,
            }
            with (args.output / "metadata" / f"{sample_id}.json").open(
                    "w") as file:
                json.dump(jsonify(payload), file, indent=2)
            print(
                f"Processed {index + 1}/{len(sample_ids)}: {sample_id}; "
                f"TRELLIS={trellis_ms / 1000.0:.2f} s, "
                f"translation={errors['translation_error_m']:.3f} m, "
                f"scale={errors['scale_absolute_error']:.3f}"
            )
        except Exception as error:
            failures.append({
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
            print(f"Skipped {sample_id}: {type(error).__name__}: {error}")

    if rows:
        with (args.output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    summary.update({
        "num_requested": len(sample_ids),
        "num_failed": len(failures),
        "shape_source": "official pretrained TRELLIS image-to-3D",
        "metric_alignment_source": str(alignment_model_path),
        "failures": failures,
    })
    with (args.output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(
        f"Saved {len(rows)} aligned TRELLIS meshes to {args.output}; "
        f"{len(failures)} failures"
    )


if __name__ == "__main__":
    main()
