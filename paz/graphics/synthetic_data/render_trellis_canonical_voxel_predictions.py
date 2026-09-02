"""Render predictions from the canonical RGB-D voxel TRELLIS model.

Inference is deliberately two-pass. The first pass uses an empty canonical
voxel to predict object pose and scale from the RGB-D encoders. That prediction
is then used to transform the observed colored point cloud into canonical
coordinates. The second pass decodes the resulting canonical voxel into a
mesh. Ground-truth object pose is never used to construct the inference voxel.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
      paz.graphics.synthetic_data.render_trellis_canonical_voxel_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/rgbd_trellis_canonical_voxel
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import jax
import keras
import numpy as np

import paz
from paz.graphics.synthetic_data.render_sdf_mesh_predictions import (
    decode_prediction,
    evaluate_render,
    jsonify,
    load_inputs,
    render,
    save_comparison,
    summarize,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_resnet18_voxel_sdf import (
    camera_ray_directions,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_trellis import (
    DEFORMATION_OUTPUT_NAME,
    SCALAR_OUTPUT_NAME,
    STRUCTURE_OUTPUT_NAME,
    WEIGHTS_OUTPUT_NAME,
    predict_flexicubes_mesh,
)
# Registers CanonicalVoxelMeshDataset model layers before load_model().
from paz.graphics.synthetic_data.train_synthetic_rgbd_trellis_canonical_voxel import (
    rotation_6d_to_matrix,
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def splat_with_predicted_pose(rgb, depth, prediction, resolution, extent,
                              max_depth, y_fov):
    """Splats RGB-D into canonical voxels using predicted pose and scale."""
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    directions = camera_ray_directions(*depth.shape, y_fov)
    camera_points = directions[valid] * depth[valid, None]
    translation = np.asarray(
        prediction["object_translation_camera_xyz"], dtype=np.float32
    )
    rotation = rotation_6d_to_matrix(
        prediction["object_orientation_camera_6d"]
    )
    scale = max(float(prediction["object_scale"]), 1e-6)
    canonical = ((camera_points - translation) @ rotation) / scale
    normalized = (canonical + extent) / (2.0 * extent)
    inside = np.all((normalized >= 0.0) & (normalized <= 1.0), axis=-1)
    normalized = normalized[inside]
    colors = rgb[valid][inside]
    cells = resolution ** 3
    if not len(normalized):
        return np.zeros((resolution,) * 3 + (4,), dtype=np.float32)
    indices = np.floor(normalized * (resolution - 1)).astype(np.int32)
    flat_indices = (
        indices[:, 0] * resolution * resolution
        + indices[:, 1] * resolution + indices[:, 2]
    )
    counts = np.bincount(flat_indices, minlength=cells).astype(np.float32)
    color_sum = np.stack([
        np.bincount(
            flat_indices, weights=colors[:, channel], minlength=cells
        )
        for channel in range(3)
    ], axis=-1).astype(np.float32)
    occupied = counts > 0.0
    color_sum[occupied] /= counts[occupied, None]
    voxel = np.concatenate([
        occupied.astype(np.float32)[:, None], color_sum
    ], axis=-1)
    return voxel.reshape((resolution,) * 3 + (4,))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/synthetic_rgbd_1000_v4"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None,
                        help="Defaults to <experiment>/best.keras.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--clear-caches-every", type=int, default=1)
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

    model_path = args.model or args.experiment / "best.keras"
    output = args.output or (
        args.experiment / "canonical_voxel_render_predictions"
    )
    for name in ("meshes", "rendered_rgb", "rendered_depth", "comparisons",
                 "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)

    normalization = load_json(args.experiment / "normalization.json")
    preprocessing = load_json(args.experiment / "input_preprocessing.json")
    split = load_json(args.experiment / "split.json")
    sample_ids = list(split["test"])
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]
    if not sample_ids:
        raise ValueError("The experiment test split is empty")
    statistics = normalization["targets"]
    shape_names = normalization["shape_names"]
    max_depth = float(preprocessing["max_depth"])
    y_fov = float(preprocessing["y_fov_degrees"])
    resolution = int(preprocessing["structure_resolution"])
    extent = float(preprocessing["mesh_extent"])

    model = keras.models.load_model(model_path, compile=False)
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    expected_inputs = {"rgb", "depth", "canonical_rgbd_voxel"}
    if input_names != expected_inputs:
        raise ValueError(
            f"Expected model inputs {sorted(expected_inputs)}, got "
            f"{sorted(input_names)}"
        )
    required_outputs = {
        STRUCTURE_OUTPUT_NAME, SCALAR_OUTPUT_NAME,
        DEFORMATION_OUTPUT_NAME, WEIGHTS_OUTPUT_NAME,
    }
    missing = required_outputs.difference(model.output_names)
    if missing:
        raise ValueError(f"Model lacks mesh outputs: {sorted(missing)}")

    empty_voxel = np.zeros(
        (1, resolution, resolution, resolution, 4), dtype=np.float32
    )
    rows, failures = [], []
    print(
        f"Loaded {model_path}; two-pass rendering {len(sample_ids)} samples"
    )
    for index, sample_id in enumerate(sample_ids):
        try:
            metadata = load_json(
                args.dataset / "metadata" / f"{sample_id}.json"
            )
            rgb, target_depth, normalized_depth = load_inputs(
                metadata, args.dataset, max_depth
            )
            pipeline_start = time.perf_counter()

            pose_start = time.perf_counter()
            first_raw = model.predict({
                "rgb": rgb[None],
                "depth": normalized_depth[None],
                "canonical_rgbd_voxel": empty_voxel,
            }, verbose=0)
            jax.tree.map(lambda value: jax.block_until_ready(value), first_raw)
            pose_ms = (time.perf_counter() - pose_start) * 1000.0
            first_prediction = decode_prediction(
                first_raw, statistics, shape_names,
                normalization.get("material_definition"),
            )

            splat_start = time.perf_counter()
            canonical_voxel = splat_with_predicted_pose(
                rgb, target_depth, first_prediction, resolution, extent,
                max_depth, y_fov,
            )
            splat_ms = (time.perf_counter() - splat_start) * 1000.0

            mesh_inference_start = time.perf_counter()
            raw = model.predict({
                "rgb": rgb[None],
                "depth": normalized_depth[None],
                "canonical_rgbd_voxel": canonical_voxel[None],
            }, verbose=0)
            jax.tree.map(lambda value: jax.block_until_ready(value), raw)
            mesh_inference_ms = (
                time.perf_counter() - mesh_inference_start
            ) * 1000.0
            prediction = decode_prediction(
                raw, statistics, shape_names,
                normalization.get("material_definition"),
            )

            extraction_start = time.perf_counter()
            mesh = predict_flexicubes_mesh(
                model, rgb, normalized_depth, resolution, extent, raw=raw
            )
            extraction_ms = (
                time.perf_counter() - extraction_start
            ) * 1000.0
            mesh_path = output / "meshes" / f"{sample_id}.ply"
            mesh.export(mesh_path)

            render_start = time.perf_counter()
            rendered = render(
                mesh, prediction, metadata["camera"], rgb.shape[:2],
                y_fov, args.tiles, args.chunk_size, args.face_chunk_size,
                not args.no_shadows,
            )
            (rendered_rgb, rendered_depth, object_to_camera,
             object_to_world, light_world) = rendered
            jax.block_until_ready(rendered_rgb)
            jax.block_until_ready(rendered_depth)
            render_ms = (time.perf_counter() - render_start) * 1000.0
            rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)
            rendered_depth = np.where(
                (rendered_depth > 0.0) & (rendered_depth <= max_depth),
                rendered_depth, 0.0,
            ).astype(np.float32)
            pipeline_ms = (
                time.perf_counter() - pipeline_start
            ) * 1000.0
            metrics = evaluate_render(
                rendered_rgb, rgb, rendered_depth, target_depth
            )
            metrics.update({
                "pose_pass_time_ms": pose_ms,
                "canonical_splat_time_ms": splat_ms,
                "mesh_pass_time_ms": mesh_inference_ms,
                "mesh_extraction_time_ms": extraction_ms,
                "render_time_ms": render_ms,
                "total_pipeline_time_ms": pipeline_ms,
                "canonical_voxel_occupied_cells": int(
                    np.count_nonzero(canonical_voxel[..., 0])
                ),
            })
            rows.append({"sample_id": sample_id, **metrics})
            paz.image.write(
                str(output / "rendered_rgb" / f"{sample_id}.png"),
                (rendered_rgb * 255.0).astype(np.uint8),
            )
            np.save(
                output / "rendered_depth" / f"{sample_id}.npy",
                rendered_depth,
            )
            save_comparison(
                output / "comparisons" / f"{sample_id}.png", rgb,
                rendered_rgb, target_depth, rendered_depth, metrics,
            )
            payload = {
                "source_sample": sample_id,
                "predicted_mesh": str(mesh_path),
                "inference_pose_source": "first model pass",
                "first_pass_prediction": first_prediction,
                "second_pass_prediction": prediction,
                "camera": metadata["camera"],
                "object_to_camera_4x4": object_to_camera,
                "object_to_world_4x4": object_to_world,
                "light_position_world_xyz": light_world,
                "metrics": metrics,
            }
            with (output / "metadata" / f"{sample_id}.json").open("w") \
                    as file:
                json.dump(jsonify(payload), file, indent=2)
            print(
                f"Rendered {index + 1}/{len(sample_ids)}: {sample_id}, "
                f"RGB MAE={metrics['rgb_mae_0_1']:.4f}, "
                f"depth MAE={metrics['depth_mae_m']:.4f} m, "
                f"pose={pose_ms:.1f} ms, mesh={mesh_inference_ms:.1f} ms"
            )
        except Exception as error:
            failures.append({
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
            print(f"Skipped {sample_id}: {type(error).__name__}: {error}")
        if (args.clear_caches_every > 0 and
                (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    if rows:
        with (output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows)
    else:
        summary = {"num_samples": 0}
    summary.update({
        "num_requested": len(sample_ids),
        "num_failed": len(failures),
        "inference_mode": "two-pass predicted-pose canonical splatting",
        "ground_truth_object_pose_used": False,
        "failures": failures,
    })
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(
        f"Saved {len(rows)} renders to {output}; "
        f"{len(failures)} samples failed"
    )


if __name__ == "__main__":
    main()
