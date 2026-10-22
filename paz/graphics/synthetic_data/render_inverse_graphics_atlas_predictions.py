"""Render and compare held-out inverse-graphics mesh predictions.

This loads a trained ``train_synthetic_rgbd_inverse_graphics_atlas`` model,
predicts canonical geometry and physical parameters, applies the predicted
similarity transform, and renders the actual fixed-connectivity triangle mesh
at the original RGB-D resolution with the PAZ renderer. Both the legacy Atlas
patch representation and mean-value-coordinate cage deformation are supported.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.render_inverse_graphics_atlas_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/rgbd_inverse_graphics_cage
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
    CageMeshDeformation,
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


def load_mesh_topology(experiment, preprocessing):
    """Loads and validates either a cage or legacy Atlas topology."""
    candidates = (
        experiment / "mesh_topology.npz",
        experiment / "atlas_topology.npz",
    )
    topology_path = next((path for path in candidates if path.is_file()), None)
    if topology_path is None:
        raise FileNotFoundError(
            "Missing mesh topology; expected mesh_topology.npz or "
            "atlas_topology.npz"
        )
    with np.load(topology_path) as archive:
        topology = {name: archive[name] for name in archive.files}
    required = {"coordinates", "faces"}
    missing = required.difference(topology)
    if missing:
        raise ValueError(
            f"{topology_path} lacks topology arrays: {sorted(missing)}"
        )
    representation = preprocessing.get("geometry_representation")
    inferred = "cage" if {
        "cage_vertices", "cage_faces", "mesh_weights"
    }.issubset(topology) else "atlas"
    if representation is None:
        representation = inferred
    if representation not in ("cage", "atlas"):
        raise ValueError(f"Unsupported geometry representation: {representation}")
    if representation != inferred:
        raise ValueError(
            f"Metadata says {representation!r}, but {topology_path.name} "
            f"contains {inferred!r} topology"
        )
    faces = np.asarray(topology["faces"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0):
        raise ValueError("mesh faces must have shape (F, 3) and be nonnegative")
    return topology_path, topology, representation


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


def inference_model(model, geometry_representation):
    required = (
        ATLAS_OUTPUT_NAME, "object_translation", "object_orientation_6d",
        "object_scale", "shape", "material", "light_position",
        "light_intensity",
    )
    outputs = dict(zip(model.output_names, model.outputs))
    missing = set(required).difference(outputs)
    if missing:
        raise ValueError(f"Model lacks outputs: {sorted(missing)}")
    selected = {name: outputs[name] for name in required}
    if geometry_representation == "cage":
        try:
            selected["cage_vertex_offsets_raw"] = model.get_layer(
                "cage_vertex_offsets"
            ).output
        except ValueError as error:
            raise ValueError(
                "Cage experiment model lacks the cage_vertex_offsets layer"
            ) from error
    return keras.Model(model.inputs, selected)


def decode_cage_vertices(raw, topology, preprocessing):
    """Reconstructs the predicted canonical control cage for inspection."""
    base = np.asarray(topology["cage_vertices"], dtype=np.float32)
    offsets = np.asarray(raw["cage_vertex_offsets_raw"][0], np.float32)
    offsets = offsets.reshape(base.shape)
    maximum = float(preprocessing.get("cage_maximum_offset", 1.0))
    return base + maximum * np.tanh(offsets)


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
    normalization = load_json(experiment / "normalization.json")
    preprocessing = load_json(experiment / "input_preprocessing.json")
    split = load_json(experiment / "split.json")
    topology_path, topology, geometry_representation = load_mesh_topology(
        experiment, preprocessing
    )
    faces = np.asarray(topology["faces"], dtype=np.int32)
    vertex_count = len(topology["coordinates"])
    if len(faces) and int(faces.max()) >= vertex_count:
        raise ValueError(
            f"Topology references vertex {int(faces.max())}, but only "
            f"{vertex_count} template vertices are defined"
        )
    directories = [
        "rendered_rgb", "rendered_depth", "comparisons",
        "canonical_meshes", "camera_meshes", "world_meshes", "metadata",
    ]
    if geometry_representation == "cage":
        directories.extend((
            "canonical_cages", "camera_cages", "world_cages",
        ))
    for name in directories:
        (output / name).mkdir(parents=True, exist_ok=True)
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
    model = inference_model(complete_model, geometry_representation)
    rows, failures = [], []
    print(
        f"Loaded {model_path}; representation={geometry_representation}, "
        f"topology={topology_path.name}, vertices={vertex_count}, "
        f"faces={len(faces)}; rendering {len(sample_ids)} test samples"
    )
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
            if canonical_vertices.shape != (vertex_count, 3):
                raise ValueError(
                    f"Predicted vertices have shape {canonical_vertices.shape}; "
                    f"expected ({vertex_count}, 3) from {topology_path.name}"
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
            cage_paths = None
            if geometry_representation == "cage":
                cage_vertices = decode_cage_vertices(
                    raw, topology, preprocessing
                )
                cage_faces = np.asarray(topology["cage_faces"], np.int32)
                canonical_cage = trimesh.Trimesh(
                    cage_vertices, cage_faces, process=False
                )
                camera_cage = canonical_cage.copy()
                camera_cage.apply_transform(object_to_camera)
                world_cage = canonical_cage.copy()
                world_cage.apply_transform(object_to_world)
                cage_paths = {
                    "canonical": f"canonical_cages/{sample_id}.ply",
                    "camera": f"camera_cages/{sample_id}.ply",
                    "world": f"world_cages/{sample_id}.ply",
                }
                canonical_cage.export(output / cage_paths["canonical"])
                camera_cage.export(output / cage_paths["camera"])
                world_cage.export(output / cage_paths["world"])
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
                "geometry_representation": geometry_representation,
                "prediction": prediction,
                "object_to_camera_4x4": object_to_camera,
                "object_to_world_4x4": object_to_world,
                "light_position_world_xyz": light_world,
                "metrics": metrics,
            }
            if cage_paths is not None:
                payload["predicted_control_cages"] = cage_paths
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
        "geometry_representation": geometry_representation,
        "topology": str(topology_path),
        "num_mesh_vertices": vertex_count,
        "num_mesh_faces": len(faces),
        "render_resolution": "original input resolution",
        "failures": failures,
    })
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)
    print(f"Saved {len(rows)} test comparisons to {output}; "
          f"{len(failures)} failures")


if __name__ == "__main__":
    main()
