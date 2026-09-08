"""Run official TRELLIS from a PAZ RGB-D exchange directory.

This file intentionally imports neither PAZ nor Keras. Run it directly from
the TRELLIS environment (using its filesystem path, not ``python -m paz``)::

    conda activate trellis
    export ATTN_BACKEND=xformers
    export SPCONV_ALGO=native
    python /path/to/paz/paz/graphics/synthetic_data/run_trellis_exchange.py \
      --exchange /path/to/experiments/trellis_exchange \
      --output /path/to/experiments/trellis_exchange_results
"""

import os

os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


FORMAT_NAME = "paz-trellis-rgbd-exchange"


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


def trellis_mesh_to_trimesh(result):
    vertices, faces = result.vertices, result.faces
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
    if source_up == "y":
        return mesh
    transform = np.eye(4)
    transform[:3, :3] = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    mesh.apply_transform(transform)
    return mesh


def normalize_mesh(mesh, mode):
    center = np.asarray(mesh.bounds).mean(axis=0)
    mesh.vertices -= center
    if mode == "none":
        factor = 1.0
    elif mode == "max-abs":
        factor = float(np.max(np.abs(mesh.vertices)))
    elif mode == "max-extent":
        factor = float(np.max(mesh.extents)) / 2.0
    else:
        raise ValueError(mode)
    if not np.isfinite(factor) or factor <= 1e-8:
        raise ValueError("TRELLIS returned a degenerate mesh")
    mesh.vertices /= factor
    return center, factor


def world_to_camera(camera):
    eye = np.asarray(camera["position_world_xyz"], dtype=np.float64)
    target = np.asarray(camera["target_world_xyz"], dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm <= 1e-8:
        raise ValueError("Camera position and target coincide")
    forward /= norm
    left = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    up = np.cross(left, forward)
    rotation = np.stack([left, up, -forward], axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ eye
    return transform


def place_mesh(mesh, alignment, camera):
    rotation = np.asarray(alignment["rotation_camera_3x3"])
    translation = np.asarray(alignment["translation_camera_xyz"])
    scale = float(alignment["scale_metres_per_canonical_unit"])
    object_to_camera = np.eye(4)
    object_to_camera[:3, :3] = scale * rotation
    object_to_camera[:3, 3] = translation
    object_to_world = np.linalg.inv(world_to_camera(camera)) @ object_to_camera
    camera_mesh, world_mesh = mesh.copy(), mesh.copy()
    camera_mesh.apply_transform(object_to_camera)
    world_mesh.apply_transform(object_to_world)
    return camera_mesh, world_mesh, object_to_camera, object_to_world


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trellis-model", default="microsoft/TRELLIS-image-large"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--sparse-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument("--trellis-up-axis", choices=("y", "z"), default="z")
    parser.add_argument(
        "--canonical-normalization",
        choices=("max-abs", "max-extent", "none"), default="max-abs",
    )
    parser.add_argument("--no-trellis-preprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    exchange, output = args.exchange.resolve(), args.output.resolve()
    manifest = load_json(exchange / "manifest.json")
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError("Not a PAZ--TRELLIS RGB-D exchange directory")
    samples = list(manifest["samples"])
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        samples = samples[:args.max_samples]
    for directory in ("canonical_meshes", "camera_meshes", "world_meshes",
                      "metadata"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as error:
        raise ImportError(
            "Official TRELLIS is not importable. Run this file directly "
            "inside the TRELLIS environment and repository."
        ) from error
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model)
    pipeline.cuda()

    completed, failures = [], []
    for index, item in enumerate(samples):
        sample_id = item["sample_id"]
        result_json = output / "metadata" / f"{sample_id}.json"
        try:
            if result_json.exists() and not args.overwrite:
                completed.append(load_json(result_json))
                print(f"Existing {index + 1}/{len(samples)}: {sample_id}")
                continue
            exchange_data = load_json(exchange / item["alignment"])
            image = Image.open(exchange / item["rgb"]).convert("RGB")
            start = time.perf_counter()
            result = pipeline.run(
                image,
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
            inference_ms = (time.perf_counter() - start) * 1000.0
            canonical = trellis_mesh_to_trimesh(result["mesh"][0])
            convert_up_axis(canonical, args.trellis_up_axis)
            center, factor = normalize_mesh(
                canonical, args.canonical_normalization
            )
            camera_mesh, world_mesh, to_camera, to_world = place_mesh(
                canonical, exchange_data["metric_alignment"],
                exchange_data["camera"],
            )
            paths = {
                "canonical_mesh": f"canonical_meshes/{sample_id}.ply",
                "camera_mesh": f"camera_meshes/{sample_id}.ply",
                "world_mesh": f"world_meshes/{sample_id}.ply",
            }
            canonical.export(output / paths["canonical_mesh"])
            camera_mesh.export(output / paths["camera_mesh"])
            world_mesh.export(output / paths["world_mesh"])
            payload = {
                "sample_id": sample_id,
                "source_exchange": str(exchange),
                "trellis_model": args.trellis_model,
                "trellis_seed": args.seed + index,
                "trellis_inference_time_ms": inference_ms,
                "trellis_up_axis": args.trellis_up_axis,
                "canonical_normalization": args.canonical_normalization,
                "original_bbox_center": center,
                "normalization_factor": factor,
                "object_to_camera_4x4": to_camera,
                "object_to_world_4x4": to_world,
                "num_vertices": len(canonical.vertices),
                "num_faces": len(canonical.faces),
                "paths": paths,
                "exchange_alignment": exchange_data,
            }
            payload = jsonify(payload)
            with result_json.open("w") as file:
                json.dump(payload, file, indent=2)
            completed.append(payload)
            print(
                f"Processed {index + 1}/{len(samples)}: {sample_id}; "
                f"TRELLIS={inference_ms / 1000.0:.2f} s"
            )
        except Exception as error:
            failure = {
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            print(f"Skipped {sample_id}: {type(error).__name__}: {error}")

    summary = {
        "format": "paz-trellis-results",
        "version": 1,
        "source_exchange": str(exchange),
        "trellis_model": args.trellis_model,
        "num_requested": len(samples),
        "num_completed": len(completed),
        "num_failed": len(failures),
        "mean_trellis_inference_time_ms": (
            float(np.mean([row["trellis_inference_time_ms"]
                           for row in completed])) if completed else None
        ),
        "failures": failures,
    }
    with (output / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved {len(completed)} results to {output}; "
          f"{len(failures)} failures")


if __name__ == "__main__":
    main()
