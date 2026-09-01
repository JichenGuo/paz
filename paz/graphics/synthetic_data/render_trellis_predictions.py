"""Render meshes decoded by ``train_synthetic_rgbd_trellis`` models.

Example:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python -m \
        paz.graphics.synthetic_data.render_trellis_predictions \
        --dataset datasets/synthetic_rgbd_1000_v4 \
        --experiment experiments/rgbd_trellis_simple_shapes
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
import jax.numpy as jp
import keras
import numpy as np

import paz
from paz.graphics.synthetic_data.render_sdf_mesh_predictions import (
    camera_transform,
    decode_prediction,
    evaluate_render,
    jsonify,
    load_inputs,
    render,
    save_comparison,
    summarize,
    unique_edges,
)
from paz.graphics.synthetic_data.train_synthetic_rgbd_trellis import (
    DEFORMATION_OUTPUT_NAME,
    SCALAR_OUTPUT_NAME,
    STRUCTURE_OUTPUT_NAME,
    WEIGHTS_OUTPUT_NAME,
    predict_flexicubes_mesh,
)


def load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def render_world_mesh(mesh, prediction, camera, image_shape, y_fov, tiles,
                      chunk_size, face_chunk_size, shadows):
    """Renders a decoder mesh whose vertices are already in world space."""
    world_to_camera = camera_transform(camera)
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
    identity = np.eye(4, dtype=np.float32)
    predicted_mesh = paz.graphics.Mesh(
        jp.asarray(vertices), jp.asarray(colors), jp.asarray(identity),
        material, jp.asarray(faces), jp.asarray(unique_edges(faces)),
    )
    floor_material = paz.graphics.Material(
        jp.asarray([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    scene = paz.graphics.Scene([
        paz.graphics.Plane(material=floor_material), predicted_mesh
    ])
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera))
    light_camera = np.append(prediction["light_position_camera_xyz"], 1.0)
    light_world = camera_to_world @ light_camera
    light = paz.graphics.PointLight(
        jp.full(3, prediction["light_intensity"]), jp.asarray(light_world[:3])
    )
    rgb, depth = paz.graphics.render(
        image_shape, np.deg2rad(y_fov), world_to_camera, scene, None, light,
        tiles=(tiles, tiles), chunk_size=chunk_size,
        face_chunk_size=face_chunk_size, shadows=shadows,
    )
    return (np.asarray(rgb), np.asarray(depth), world_to_camera, identity,
            light_world[:3])


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("datasets/synthetic_rgbd_1000_v4"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None,
                        help="Defaults to <experiment>/best.keras.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--clear-caches-every", type=int, default=1)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.y_fov, args.tiles, args.chunk_size,
           args.face_chunk_size) <= 0:
        raise ValueError("render sizes and ranges must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max samples must be positive")
    if args.clear_caches_every < 0:
        raise ValueError("cache interval cannot be negative")

    model_path = args.model or args.experiment / "best.keras"
    output = args.output or args.experiment / "trellis_render_predictions"
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
    resolution = int(preprocessing["structure_resolution"])
    extent = float(preprocessing["mesh_extent"])
    mesh_frame = preprocessing.get("mesh_frame", "canonical")
    model = keras.models.load_model(model_path, compile=False)
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    required_outputs = {
        STRUCTURE_OUTPUT_NAME, SCALAR_OUTPUT_NAME,
        DEFORMATION_OUTPUT_NAME, WEIGHTS_OUTPUT_NAME,
    }
    if input_names != {"rgb", "depth"}:
        raise ValueError(
            f"Expected RGB-D model inputs, got {sorted(input_names)}"
        )
    missing = required_outputs.difference(model.output_names)
    if missing:
        raise ValueError(f"Model lacks mesh-decoder outputs: {sorted(missing)}")

    rows = []
    print(f"Loaded {model_path}; decoding and rendering {len(sample_ids)} meshes")
    for index, sample_id in enumerate(sample_ids):
        metadata = load_json(args.dataset / "metadata" / f"{sample_id}.json")
        rgb, target_depth, normalized_depth = load_inputs(
            metadata, args.dataset, max_depth
        )
        pipeline_start = time.perf_counter()
        inference_start = time.perf_counter()
        raw = model.predict({
            "rgb": rgb[None], "depth": normalized_depth[None]
        }, verbose=0)
        jax.tree.map(lambda value: jax.block_until_ready(value), raw)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        prediction = decode_prediction(
            raw, statistics, shape_names,
            normalization.get("material_definition"),
        )

        extraction_start = time.perf_counter()
        mesh = predict_flexicubes_mesh(
            model, rgb, normalized_depth, resolution, extent, raw=raw
        )
        extraction_ms = (time.perf_counter() - extraction_start) * 1000.0
        mesh_path = output / "meshes" / f"{sample_id}.ply"
        mesh.export(mesh_path)

        render_start = time.perf_counter()
        if mesh_frame == "world":
            rendered = render_world_mesh(
                mesh, prediction, metadata["camera"], rgb.shape[:2],
                args.y_fov, args.tiles, args.chunk_size,
                args.face_chunk_size, not args.no_shadows,
            )
        else:
            rendered = render(
                mesh, prediction, metadata["camera"], rgb.shape[:2],
                args.y_fov, args.tiles, args.chunk_size,
                args.face_chunk_size, not args.no_shadows,
            )
        rendered_rgb, rendered_depth, first_transform, object_to_world, \
            light_world = rendered
        jax.block_until_ready(rendered_rgb)
        jax.block_until_ready(rendered_depth)
        render_ms = (time.perf_counter() - render_start) * 1000.0
        rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)
        rendered_depth = np.where(
            (rendered_depth > 0.0) & (rendered_depth <= max_depth),
            rendered_depth, 0.0,
        ).astype(np.float32)
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
        metrics = evaluate_render(
            rendered_rgb, rgb, rendered_depth, target_depth
        )
        metrics.update({
            "model_inference_time_ms": inference_ms,
            "mesh_extraction_time_ms": extraction_ms,
            "render_time_ms": render_ms,
            "prediction_render_pipeline_time_ms": pipeline_ms,
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
            "predicted_mesh": str(mesh_path),
            "mesh_frame": mesh_frame,
            "prediction": prediction,
            "camera": metadata["camera"],
            "first_render_transform_4x4": first_transform,
            "object_to_world_4x4": object_to_world,
            "light_position_world_xyz": light_world,
            "metrics": metrics,
        }
        with (output / "metadata" / f"{sample_id}.json").open("w") as file:
            json.dump(jsonify(payload), file, indent=2)
        print(
            f"Rendered {index + 1}/{len(sample_ids)}: {sample_id}, "
            f"RGB MAE={metrics['rgb_mae_0_1']:.4f}, "
            f"depth MAE={metrics['depth_mae_m']:.4f} m, "
            f"inference={inference_ms:.1f} ms, "
            f"extraction={extraction_ms:.1f} ms, render={render_ms:.1f} ms"
        )
        if (args.clear_caches_every > 0 and
                (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["mesh_frame"] = mesh_frame
    summary["timing_includes_mesh_extraction"] = True
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(f"Saved TRELLIS mesh renders and comparisons to {output}")


if __name__ == "__main__":
    main()
