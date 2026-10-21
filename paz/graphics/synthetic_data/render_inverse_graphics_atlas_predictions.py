"""Render and compare held-out inverse-graphics Atlas predictions.

This loads a trained ``train_synthetic_rgbd_inverse_graphics_atlas`` model,
predicts canonical geometry and physical parameters, applies the predicted
similarity transform, and renders the actual fixed-connectivity triangle mesh
at the original RGB-D resolution with the PAZ renderer.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.render_inverse_graphics_atlas_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/rgbd_inverse_graphics_atlas
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
import keras
import numpy as np
import trimesh

# Register every custom layer/loss before loading the saved Keras model.
from paz.graphics.synthetic_data.train_synthetic_rgbd_inverse_graphics_atlas import (
    ATLAS_OUTPUT_NAME,
    DifferentiableAtlasRenderer,
    MaskedReconstructionHuber,
    PhysicalAtlasLoss,
    RepeatAtlasCoordinates,
    SilhouetteBCEDiceLoss,
    rotation_6d_to_matrix,
)
from paz.graphics.synthetic_data.render_sdf_mesh_predictions import (
    evaluate_render,
    jsonify,
    render,
    save_comparison,
    summarize,
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def denormalize(values, statistics):
    return (np.asarray(values, dtype=np.float32)
            * np.asarray(statistics["standard_deviation"], np.float32)
            + np.asarray(statistics["mean"], np.float32))


def load_inputs(metadata, dataset, max_depth):
    path = dataset / metadata["rgb"]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.load(dataset / metadata["depth"]).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {path}")
    depth = np.where(
        np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth),
        depth, 0.0,
    )
    return rgb, depth, (depth / max_depth)[..., None]


def decode_prediction(raw, statistics, mesh_statistics, shape_names,
                      material_definition):
    canonical = denormalize(raw[ATLAS_OUTPUT_NAME][0], mesh_statistics)
    translation = denormalize(
        raw["object_translation"][0], statistics["object_translation"]
    )
    orientation = np.asarray(raw["object_orientation_6d"][0], np.float32)
    rotation = rotation_6d_to_matrix(orientation)
    scale = max(float(denormalize(
        raw["object_scale"][0], statistics["object_scale"]
    )[0]), 1e-3)
    material_values = denormalize(
        raw["material"][0], statistics["material"]
    )
    names = material_definition.get("values", [])
    if names:
        by_name = dict(zip(names, material_values))
        material = {
            "color_rgb": np.asarray([
                by_name["color_r"], by_name["color_g"], by_name["color_b"]
            ], np.float32),
            "diffuse": float(by_name["diffuse"]),
            "specular": float(by_name["specular"]),
            "ambient": float(by_name["ambient"]),
            "shininess": float(by_name["shininess"]),
        }
    else:
        material = {
            "color_rgb": material_values[:3],
            "diffuse": float(material_values[3]),
            "specular": float(material_values[4]),
            "ambient": float(material_values[5]),
            "shininess": float(material_values[6]),
        }
    material["color_rgb"] = np.clip(material["color_rgb"], 0.0, 1.0)
    for name in ("diffuse", "specular", "ambient"):
        material[name] = float(np.clip(material[name], 0.0, 1.0))
    material["shininess"] = float(np.clip(
        material["shininess"], 1.0, 500.0
    ))
    light_position = denormalize(
        raw["light_position"][0], statistics["light_position"]
    )
    light_intensity = max(float(denormalize(
        raw["light_intensity"][0], statistics["light_intensity"]
    )[0]), 0.0)
    probabilities = np.asarray(raw["shape"][0])
    return canonical, {
        "object_translation_camera_xyz": translation,
        "object_orientation_camera_6d": orientation,
        "object_rotation_camera_3x3": rotation,
        "object_scale": scale,
        "shape": shape_names[int(np.argmax(probabilities))],
        "shape_probabilities": probabilities,
        "material": material,
        "light_position_camera_xyz": light_position,
        "light_intensity": light_intensity,
    }


def inference_model(model):
    required = (
        ATLAS_OUTPUT_NAME, "object_translation", "object_orientation_6d",
        "object_scale", "shape", "material", "light_position",
        "light_intensity",
    )
    outputs = dict(zip(model.output_names, model.outputs))
    missing = set(required).difference(outputs)
    if missing:
        raise ValueError(f"Model lacks outputs: {sorted(missing)}")
    return keras.Model(model.inputs, {name: outputs[name] for name in required})


def write_rgb(path, image):
    bgr = cv2.cvtColor(
        (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(str(path), bgr)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
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
        raise ValueError("render tile and chunk sizes must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.clear_caches_every < 0:
        raise ValueError("--clear-caches-every cannot be negative")

    dataset = args.dataset.resolve()
    experiment = args.experiment.resolve()
    model_path = args.model or experiment / "best.keras"
    output = args.output or experiment / "test_render_comparisons"
    for name in (
        "rendered_rgb", "rendered_depth", "comparisons",
        "canonical_meshes", "camera_meshes", "world_meshes", "metadata",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)

    normalization = load_json(experiment / "normalization.json")
    preprocessing = load_json(experiment / "input_preprocessing.json")
    split = load_json(experiment / "split.json")
    topology = np.load(experiment / "atlas_topology.npz")
    faces = topology["faces"].astype(np.int32)
    statistics = normalization["targets"]
    mesh_statistics = statistics[ATLAS_OUTPUT_NAME]
    shape_names = tuple(normalization["shape_names"])
    material_definition = normalization.get("material_definition", {})
    max_depth = float(preprocessing.get("max_depth", 10.0))
    y_fov = float(preprocessing.get("y_fov_degrees", 45.0))
    sample_ids = list(split["test"])
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]

    complete_model = keras.models.load_model(model_path, compile=False)
    model = inference_model(complete_model)
    rows, failures = [], []
    print(f"Loaded {model_path}; rendering {len(sample_ids)} test samples")
    for index, sample_id in enumerate(sample_ids):
        try:
            metadata = load_json(
                dataset / "metadata" / f"{sample_id}.json"
            )
            target_rgb, target_depth, normalized_depth = load_inputs(
                metadata, dataset, max_depth
            )
            start = time.perf_counter()
            raw = model.predict({
                "rgb": target_rgb[None], "depth": normalized_depth[None]
            }, verbose=0)
            jax.tree.map(jax.block_until_ready, raw)
            inference_ms = (time.perf_counter() - start) * 1000.0
            canonical_vertices, prediction = decode_prediction(
                raw, statistics, mesh_statistics, shape_names,
                material_definition,
            )
            canonical_mesh = trimesh.Trimesh(
                canonical_vertices, faces, process=False
            )
            render_start = time.perf_counter()
            rendered_rgb, rendered_depth, object_to_camera, object_to_world, \
                light_world = render(
                    canonical_mesh, prediction, metadata["camera"],
                    target_rgb.shape[:2], y_fov, args.tiles,
                    args.chunk_size, args.face_chunk_size,
                    not args.no_shadows,
                )
            jax.block_until_ready(rendered_rgb)
            jax.block_until_ready(rendered_depth)
            render_ms = (time.perf_counter() - render_start) * 1000.0
            rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)
            rendered_depth = np.where(
                (rendered_depth > 0.0) & (rendered_depth <= max_depth),
                rendered_depth, 0.0,
            ).astype(np.float32)
            metrics = evaluate_render(
                rendered_rgb, target_rgb, rendered_depth, target_depth
            )
            metrics.update({
                "model_inference_time_ms": inference_ms,
                "render_time_ms": render_ms,
            })
            rows.append({"sample_id": sample_id, **metrics})

            camera_mesh = canonical_mesh.copy()
            camera_mesh.apply_transform(object_to_camera)
            world_mesh = canonical_mesh.copy()
            world_mesh.apply_transform(object_to_world)
            canonical_mesh.export(
                output / "canonical_meshes" / f"{sample_id}.ply"
            )
            camera_mesh.export(output / "camera_meshes" / f"{sample_id}.ply")
            world_mesh.export(output / "world_meshes" / f"{sample_id}.ply")
            write_rgb(output / "rendered_rgb" / f"{sample_id}.png",
                      rendered_rgb)
            np.save(output / "rendered_depth" / f"{sample_id}.npy",
                    rendered_depth)
            save_comparison(
                output / "comparisons" / f"{sample_id}.png",
                target_rgb, rendered_rgb, target_depth, rendered_depth,
                metrics,
            )
            payload = {
                "sample_id": sample_id,
                "prediction": prediction,
                "object_to_camera_4x4": object_to_camera,
                "object_to_world_4x4": object_to_world,
                "light_position_world_xyz": light_world,
                "metrics": metrics,
            }
            with (output / "metadata" / f"{sample_id}.json").open("w") as file:
                json.dump(jsonify(payload), file, indent=2)
            print(
                f"Rendered {index + 1}/{len(sample_ids)}: {sample_id}; "
                f"RGB MAE={metrics['rgb_mae_0_1']:.4f}, "
                f"depth MAE={metrics['depth_mae_m']:.4f} m"
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
    summary = summarize(rows) if rows else {"num_samples": 0}
    summary.update({
        "num_requested": len(sample_ids),
        "num_failed": len(failures),
        "model": str(model_path),
        "render_resolution": "original input resolution",
        "failures": failures,
    })
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(f"Saved {len(rows)} test comparisons to {output}; "
          f"{len(failures)} failures")


if __name__ == "__main__":
    main()
