"""Generate primitive RGB-D data under one fixed camera and point light.

Example:
    python -m paz.graphics.synthetic_data.generate_fixed_synthetic_rgbd \
        --output datasets/synthetic_rgbd_fixed_1000 \
        --num-samples 1000 --shapes cube cylinder sphere

Camera pose, camera target, light position, and light intensity remain fixed.
Object shape, floor-plane location, yaw, scale, and material are sampled.
"""

import argparse
import gc
import json
from pathlib import Path

import jax
import numpy as np

from paz.graphics.synthetic_data.generate_synthetic_rgbd import (
    COLORS,
    SHAPE_BUILDERS,
    generate_sample,
    jsonify,
    sample_is_complete,
)


def sample_fixed_parameters(rng, shape_names, camera_position, camera_target,
                            light_position, light_intensity,
                            object_position_range):
    """Samples object properties while copying fixed acquisition parameters."""
    object_scale = float(rng.uniform(0.35, 0.65))
    object_x, object_z = rng.uniform(
        -object_position_range, object_position_range, size=2
    )
    return {
        "shape": str(rng.choice(shape_names)),
        "object_scale": object_scale,
        "object_yaw": float(rng.uniform(-np.pi, np.pi)),
        "object_translation_world": np.array(
            [object_x, object_scale, object_z], dtype=np.float32
        ),
        "color": COLORS[rng.integers(len(COLORS))],
        "ambient": float(rng.uniform(0.05, 0.25)),
        "diffuse": float(rng.uniform(0.55, 0.95)),
        "specular": float(rng.uniform(0.0, 0.8)),
        "shininess": float(rng.uniform(20.0, 200.0)),
        "camera_position": np.asarray(camera_position, dtype=np.float32),
        "camera_target": np.asarray(camera_target, dtype=np.float32),
        "light_position": np.asarray(light_position, dtype=np.float32),
        "light_intensity": float(light_intensity),
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("synthetic_rgbd_fixed"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--y-fov", type=float, default=45.0)
    parser.add_argument("--shapes", nargs="+", choices=SHAPE_BUILDERS,
                        default=list(SHAPE_BUILDERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-position", nargs=3, type=float,
                        default=(0.0, 3.5, 4.5), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-target", nargs=3, type=float,
                        default=(0.0, 0.5, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--light-position", nargs=3, type=float,
                        default=(-3.0, 5.0, 3.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--light-intensity", type=float, default=1.3)
    parser.add_argument("--object-position-range", type=float, default=0.5,
                        help="Uniform X/Z object displacement in metres.")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--clear-caches-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    positive = (
        args.num_samples, args.height, args.width, args.tiles,
        args.chunk_size, args.light_intensity,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sizes and light intensity must be positive")
    if args.object_position_range < 0 or args.clear_caches_every < 0:
        raise ValueError("ranges and cache interval must be nonnegative")

    for directory in ("rgb", "depth", "meshes", "metadata"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    configuration = {
        "camera_position": args.camera_position,
        "camera_target": args.camera_target,
        "light_position": args.light_position,
        "light_intensity": args.light_intensity,
        "object_position_range": args.object_position_range,
        "y_fov_degrees": args.y_fov,
        "image_size": [args.height, args.width],
    }
    with (args.output / "fixed_configuration.json").open("w") as file:
        json.dump(jsonify(configuration), file, indent=2)

    rng = np.random.default_rng(args.seed)
    for index in range(args.num_samples):
        parameters = sample_fixed_parameters(
            rng, args.shapes, args.camera_position, args.camera_target,
            args.light_position, args.light_intensity,
            args.object_position_range,
        )
        if args.resume and sample_is_complete(args.output, index):
            print(f"Skipped {index + 1}/{args.num_samples} (already complete)")
            continue
        generate_sample(
            args.output, index, parameters, (args.height, args.width),
            np.deg2rad(args.y_fov), not args.no_shadows,
            (args.tiles, args.tiles), args.chunk_size,
        )
        print(f"Generated {index + 1}/{args.num_samples}")
        if (args.clear_caches_every > 0
                and (index + 1) % args.clear_caches_every == 0):
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
