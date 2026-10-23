"""Refine neural cage predictions with differentiable RGB-D rendering.

The trained network provides geometry, pose, material, and lighting. For each
test image, only the cage vertices are optimized against foreground RGB,
metric depth, and silhouette observations using the PAZ triangle renderer.

Example::

    KERAS_BACKEND=jax JAX_PLATFORMS=cuda python -m \
      paz.graphics.synthetic_data.refine_inverse_graphics_cage_predictions \
      --dataset datasets/synthetic_rgbd_1000_v4 \
      --experiment experiments/cage_joint_v3_dense \
      --refinement-steps 100 --max-samples 20
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import csv
import json
import time
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jp
import keras
import numpy as np
import optax
import paz
import trimesh

from paz.backend.mesh import build_laplacian, compute_volume
from .render_inverse_graphics_cage_predictions import (
    decode_cage_vertices,
    decode_prediction,
    inference_model,
    load_inputs,
    load_json,
    load_mesh_topology,
    mesh_validity_metrics,
    object_render_metrics,
)
from paz.graphics.synthetic_data.render_sdf_mesh_predictions import (
    camera_transform,
    evaluate_render,
    jsonify,
    render,
    save_comparison,
    summarize,
    unique_edges,
)


LossWeights = namedtuple(
    "LossWeights", "rgb depth silhouette smooth volume displacement"
)
RenderConfig = namedtuple(
    "RenderConfig", "shape y_fov pose tiles chunk_size face_chunk depth"
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refinement-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--image-size-ratio", type=float, default=0.25)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=100.0)
    parser.add_argument("--silhouette-weight", type=float, default=10.0)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--volume-weight", type=float, default=1.0)
    parser.add_argument("--displacement-weight", type=float, default=1e-2)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--face-chunk-size", type=int, default=128)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def validate_args(args):
    positive = [args.refinement_steps, args.learning_rate]
    positive += [args.image_size_ratio, args.tiles, args.chunk_size]
    positive.append(args.face_chunk_size)
    if min(positive) <= 0:
        raise ValueError("Refinement and rendering arguments must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")


def choose_model(args, experiment):
    if args.model is not None:
        return args.model.resolve()
    for name in ("best.keras", "geometry_only.keras", "final.keras"):
        path = experiment / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No complete model found in {experiment}")


def resize_observations(rgb, depth, mask, ratio):
    H, W = depth.shape[:2]
    shape = max(1, round(H * ratio)), max(1, round(W * ratio))
    rgb = jax.image.resize(jp.asarray(rgb), (*shape, 3), "bilinear")
    depth = jax.image.resize(jp.asarray(depth), (*shape, 1), "nearest")
    mask = jax.image.resize(jp.asarray(mask), (*shape, 1), "nearest")
    return rgb, depth, mask, shape


def build_material(prediction):
    values = prediction["material"]
    args = values["ambient"], values["diffuse"], values["specular"]
    return paz.graphics.Material(
        jp.asarray(values["color_rgb"]), *args, values["shininess"]
    )


def build_light(prediction, world_to_camera):
    camera = np.append(prediction["light_position_camera_xyz"], 1.0)
    world = np.linalg.inv(np.asarray(world_to_camera)) @ camera
    intensity = jp.full(3, prediction["light_intensity"])
    return paz.graphics.PointLight(intensity, jp.asarray(world[:3]))


def object_transform(prediction, world_to_camera):
    transform = np.eye(4, dtype=np.float32)
    rotation = prediction["object_rotation_camera_3x3"]
    transform[:3, :3] = rotation * prediction["object_scale"]
    transform[:3, 3] = prediction["object_translation_camera_xyz"]
    return jp.asarray(np.linalg.inv(np.asarray(world_to_camera)) @ transform)


def build_render_model(initial_vertices, initial_cage, weights, faces,
                       prediction, config):
    colors = jp.repeat(
        jp.asarray(prediction["material"]["color_rgb"])[None],
        len(initial_vertices), axis=0,
    )
    material = build_material(prediction)
    light = build_light(prediction, config.pose)
    transform = object_transform(prediction, config.pose)
    edges = jp.asarray(unique_edges(np.asarray(faces)))
    initial_vertices = jp.asarray(initial_vertices)
    initial_cage = jp.asarray(initial_cage)
    weights = jp.asarray(weights)
    faces = jp.asarray(faces)

    def model(cage):
        offsets = cage - initial_cage
        vertices = initial_vertices + weights @ offsets
        mesh_args = vertices, colors, transform, material, faces, edges
        scene = paz.graphics.Scene([paz.graphics.Mesh(*mesh_args)])
        render_args = config.shape, config.y_fov, config.pose, scene
        render_args += None, light, config.tiles, config.chunk_size
        image, depth = paz.graphics.render(
            *render_args, shadows=False,
            face_chunk_size=config.face_chunk,
        )
        mask = paz.depth.to_soft_mask(depth, *config.depth)[..., None]
        return image, depth[..., None], mask, vertices

    return model


def build_objective(model, observations, geometry, loss_weights):
    target_rgb, target_depth, target_mask = observations
    initial_cage, laplacian, faces, initial_volume = geometry

    def terms(cage):
        rgb, depth, mask, vertices = model(cage)
        values = {
            "rgb": paz.losses.masked_mae(
                target_rgb, rgb, target_mask
            ),
            "depth": paz.losses.masked_mae(
                target_depth, depth, target_mask
            ),
            "silhouette": paz.losses.mae(target_mask, mask),
            "smooth": paz.losses.laplacian_smoothing(vertices, laplacian),
            "volume": paz.losses.volume_matching(
                vertices, faces, initial_volume
            ),
            "displacement": jp.mean((cage - initial_cage) ** 2),
        }
        return values

    def loss(cage):
        values = terms(cage)
        losses = [values[name] for name in LossWeights._fields]
        return paz.losses.weight(losses, loss_weights)

    return loss, terms


def refine_cage(initial_cage, loss, terms, args):
    trace = []
    callback = paz.optimization.TraceParameters(trace)
    optimizer = optax.adamw(args.learning_rate)
    stop = paz.optimization.patience_stop(
        args.early_stop_min_delta, args.early_stop_patience
    )
    minimize_args = jp.asarray(initial_cage), loss, optimizer
    minimize_args += args.refinement_steps,
    kwargs = dict(stop_fn=stop, callbacks=[callback], verbose=True)
    start = time.perf_counter()
    _, last_cage, history = paz.minimize(*minimize_args, **kwargs)
    jax.block_until_ready(last_cage)
    elapsed = time.perf_counter() - start
    count = int(history.stop_step)
    losses = np.asarray(history.losses[:count])
    cages = trace[:count]
    last_loss = float(loss(last_cage))
    losses = np.append(losses, last_loss)
    cages.append(np.asarray(last_cage))
    best = int(np.argmin(losses))
    best_cage = np.asarray(cages[best])
    values = jax.device_get(terms(jp.asarray(best_cage)))
    values = {name: float(value) for name, value in values.items()}
    values["total"] = float(losses[best])
    values["best_step"] = best
    return best_cage, losses.tolist(), values, elapsed


def refined_vertices(initial_vertices, initial_cage, refined_cage, weights):
    offsets = np.asarray(refined_cage) - np.asarray(initial_cage)
    return np.asarray(initial_vertices) + np.asarray(weights) @ offsets


def render_mesh(vertices, faces, prediction, metadata, image_shape, args,
                y_fov):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    render_args = mesh, prediction, metadata["camera"], image_shape, y_fov
    render_args += args.tiles, args.chunk_size, args.face_chunk_size
    rendered = render(*render_args, shadows=not args.no_shadows)
    return mesh, rendered


def evaluate_pair(targets, rendered, metadata, y_fov, floor_tolerance):
    target_rgb, target_depth = targets
    rgb, depth = np.asarray(rendered[0]), np.asarray(rendered[1])
    metrics = evaluate_render(rgb, target_rgb, depth, target_depth)
    object_args = target_rgb, rgb, target_depth, depth
    object_args += metadata["camera"], y_fov, floor_tolerance
    metrics.update(object_render_metrics(*object_args))
    return rgb, depth, metrics


def write_loss_trace(path, losses):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows(enumerate(losses))


def make_output_directories(output):
    names = [
        "initial_comparisons", "refined_comparisons", "refined_meshes",
        "refined_cages", "losses", "metadata",
    ]
    for name in names:
        (output / name).mkdir(parents=True, exist_ok=True)


def save_rows(output, rows):
    if not rows:
        return
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"num_samples": len(rows)}
    for prefix in ("initial_", "refined_"):
        selected = [
            {key[len(prefix):]: value for key, value in row.items()
             if key.startswith(prefix)} for row in rows
        ]
        summary[prefix[:-1]] = summarize(selected)
    with (output / "metrics_summary.json").open("w") as file:
        json.dump(jsonify(summary), file, indent=2)


def process_sample(sample_id, context):
    args, model, dataset, output = context[:4]
    statistics, mesh_statistics, shape_names = context[4:7]
    material_definition, topology, preprocessing = context[7:10]
    metadata = load_json(dataset / "metadata" / f"{sample_id}.json")
    max_depth = float(preprocessing.get("max_depth", 10.0))
    y_fov = float(preprocessing.get("y_fov_degrees", 45.0))
    floor = float(preprocessing.get("render_floor_tolerance_metres", 0.01))
    target = load_inputs(metadata, dataset, max_depth, y_fov, floor)
    target_rgb, target_depth, normalized_depth, object_mask = target
    inputs = {
        "rgb": target_rgb[None], "depth": normalized_depth[None],
        "object_mask": object_mask[None],
    }
    raw = model.predict(inputs, verbose=0)
    initial_vertices, prediction = decode_prediction(
        raw, statistics, mesh_statistics, shape_names, material_definition
    )
    initial_cage = decode_cage_vertices(raw, topology, preprocessing)
    faces = np.asarray(topology["faces"], np.int32)
    weights = np.asarray(topology["mesh_weights"], np.float32)
    observations = resize_observations(
        target_rgb, target_depth[..., None], object_mask,
        args.image_size_ratio,
    )
    target_small = observations[:3]
    image_shape = observations[3]
    world_to_camera = camera_transform(metadata["camera"])
    config = RenderConfig(
        image_shape, np.deg2rad(y_fov), world_to_camera,
        (args.tiles, args.tiles), args.chunk_size, args.face_chunk_size,
        (1e-3, max_depth),
    )
    render_model = build_render_model(
        initial_vertices, initial_cage, weights, faces,
        prediction, config,
    )
    laplacian = build_laplacian(initial_vertices, faces)
    initial_volume = compute_volume(jp.asarray(initial_vertices), faces)
    geometry = initial_cage, laplacian, jp.asarray(faces), initial_volume
    loss_weights = LossWeights(
        args.rgb_weight, args.depth_weight, args.silhouette_weight,
        args.smooth_weight, args.volume_weight, args.displacement_weight,
    )
    loss, terms = build_objective(
        render_model, target_small, geometry, loss_weights
    )
    refined_cage, losses, values, elapsed = refine_cage(
        initial_cage, loss, terms, args
    )
    vertices = refined_vertices(
        initial_vertices, initial_cage, refined_cage, weights
    )
    render_args = prediction, metadata, target_rgb.shape[:2], args, y_fov
    initial_mesh, initial_render = render_mesh(
        initial_vertices, faces, *render_args
    )
    refined_mesh, refined_render = render_mesh(vertices, faces, *render_args)
    targets = target_rgb, target_depth
    initial = evaluate_pair(targets, initial_render, metadata, y_fov, floor)
    refined = evaluate_pair(targets, refined_render, metadata, y_fov, floor)
    save_comparison(
        output / "initial_comparisons" / f"{sample_id}.png",
        target_rgb, initial[0], target_depth, initial[1], initial[2],
    )
    save_comparison(
        output / "refined_comparisons" / f"{sample_id}.png",
        target_rgb, refined[0], target_depth, refined[1], refined[2],
    )
    refined_mesh.export(output / "refined_meshes" / f"{sample_id}.ply")
    cage_faces = np.asarray(topology["cage_faces"], np.int32)
    cage_mesh = trimesh.Trimesh(refined_cage, cage_faces, process=False)
    cage_mesh.export(output / "refined_cages" / f"{sample_id}.ply")
    write_loss_trace(output / "losses" / f"{sample_id}.csv", losses)
    payload = {
        "sample_id": sample_id, "refinement_seconds": elapsed,
        "loss_terms": values, "prediction": jsonify(prediction),
        "initial_validity": mesh_validity_metrics(
            initial_mesh.vertices, faces, "mesh"
        ),
        "refined_validity": mesh_validity_metrics(vertices, faces, "mesh"),
    }
    with (output / "metadata" / f"{sample_id}.json").open("w") as file:
        json.dump(jsonify(payload), file, indent=2)
    row = {"sample_id": sample_id, "refinement_seconds": elapsed}
    row.update({f"initial_{key}": value for key, value in initial[2].items()})
    row.update({f"refined_{key}": value for key, value in refined[2].items()})
    return row


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    dataset = args.dataset.resolve()
    experiment = args.experiment.resolve()
    output = args.output or experiment / "cage_refinement"
    output.mkdir(parents=True, exist_ok=True)
    make_output_directories(output)
    normalization = load_json(experiment / "normalization.json")
    preprocessing = load_json(experiment / "input_preprocessing.json")
    split = load_json(experiment / "split.json")
    _, topology, representation = load_mesh_topology(
        experiment, preprocessing
    )
    complete = keras.models.load_model(
        choose_model(args, experiment), compile=False
    )
    model = inference_model(complete, representation)
    sample_ids = list(split["test"])
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]
    statistics = normalization["targets"]
    mesh_statistics = statistics["mesh_vertices"]
    shape_names = tuple(normalization.get("shape_names", ()))
    material = normalization.get("material_definition", {})
    context = (
        args, model, dataset, output, statistics, mesh_statistics,
        shape_names, material, topology, preprocessing,
    )
    rows, failures = [], []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"Refining {sample_id} ({index}/{len(sample_ids)})")
        try:
            rows.append(process_sample(sample_id, context))
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": str(error)})
            print(f"Failed {sample_id}: {error}")
    save_rows(output, rows)
    with (output / "failures.json").open("w") as file:
        json.dump(failures, file, indent=2)
    print(f"Saved {len(rows)} refinements to {output}")
    if failures:
        print(f"Failed samples: {len(failures)}")


if __name__ == "__main__":
    main()
