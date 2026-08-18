"""Generate RGB-D observations of one primitive resting on a floor.

Example:
    python -m paz.graphics.generate_synthetic_rgbd \
        --output synthetic_rgbd --num-samples 100 --shapes cube cylinder sphere

Each sample contains an RGB PNG, metric depth as ``.npy``, the object mesh as
PLY, and JSON metadata containing the camera intrinsics/extrinsics, object
pose, and lighting parameters.  Depth value zero denotes no intersection.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

import paz


COLORS = np.array(
    [[0.85, 0.25, 0.20], [0.20, 0.55, 0.90], [0.25, 0.75, 0.40]],
    dtype=np.float32,
)
SHAPE_BUILDERS = {
    "cube": paz.graphics.Cube,
    "cylinder": paz.graphics.Cylinder,
    "sphere": paz.graphics.Sphere,
}


def sample_parameters(rng, shape_names):
    """Samples one camera, object, and point-light configuration."""
    shape_name = str(rng.choice(shape_names))
    object_scale = float(rng.uniform(0.35, 0.65))
    object_yaw = float(rng.uniform(-np.pi, np.pi))

    azimuth = float(rng.uniform(-np.pi, np.pi))
    elevation = float(rng.uniform(np.deg2rad(20.0), np.deg2rad(65.0)))
    distance = float(rng.uniform(3.0, 5.5))
    target = np.array([0.0, object_scale, 0.0])
    camera = target + distance * np.array(
        [np.cos(elevation) * np.sin(azimuth),
         np.sin(elevation),
         np.cos(elevation) * np.cos(azimuth)]
    )

    light_azimuth = float(rng.uniform(-np.pi, np.pi))
    light_radius = float(rng.uniform(2.0, 4.5))
    light_height = float(rng.uniform(3.0, 6.0))
    light = np.array(
        [light_radius * np.sin(light_azimuth), light_height,
         light_radius * np.cos(light_azimuth)]
    )
    return {
        "shape": shape_name,
        "object_scale": object_scale,
        "object_yaw": object_yaw,
        "color": COLORS[rng.integers(len(COLORS))],
        "ambient": float(rng.uniform(0.05, 0.25)),
        "diffuse": float(rng.uniform(0.55, 0.95)),
        "specular": float(rng.uniform(0.0, 0.8)),
        "shininess": float(rng.uniform(20.0, 200.0)),
        "camera_position": camera,
        "camera_target": target,
        "light_position": light,
        "light_intensity": float(rng.uniform(0.8, 1.8)),
    }


def build_scene(parameters):
    """Builds an analytic PAZ scene and returns it with the object transform."""
    object_scale = parameters["object_scale"]
    translation = paz.SE3.translation(jp.array([0.0, object_scale, 0.0]))
    rotation = paz.SE3.rotation_y(parameters["object_yaw"])
    scaling = paz.SE3.scaling(jp.full(3, object_scale))
    object_transform = translation @ rotation @ scaling
    material = paz.graphics.Material(
        jp.array(parameters["color"]), parameters["ambient"],
        parameters["diffuse"], parameters["specular"],
        parameters["shininess"],
    )
    primitive = SHAPE_BUILDERS[parameters["shape"]](
        object_transform, material
    )
    floor_material = paz.graphics.Material(
        jp.array([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0
    )
    return paz.graphics.Scene([paz.graphics.Plane(material=floor_material),
                               primitive]), object_transform
def build_mesh(shape_name, transform, sections=64):
    """Creates a triangle mesh matching the rendered analytic primitive."""
    if shape_name == "cube":
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    elif shape_name == "cylinder":
        mesh = trimesh.creation.cylinder(radius=1.0, height=2.0,
                                         sections=sections)
        # trimesh cylinders use Z as the vertical axis; PAZ uses Y.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            -np.pi / 2.0, [1.0, 0.0, 0.0]
        ))
    elif shape_name == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")
    mesh.apply_transform(np.asarray(transform))
    return mesh


def jsonify(value):
    """Converts nested NumPy values into values supported by JSON."""
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, jp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def decompose_transform(transform):
    """Returns translation, XYZ Euler orientation, and scale from a transform."""
    transform = np.asarray(transform, dtype=np.float64)
    linear = transform[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    rotation = linear / scale
    euler_xyz = Rotation.from_matrix(rotation).as_euler("xyz")
    return {
        "translation_xyz": transform[:3, 3],
        "orientation_euler_xyz_radians": euler_xyz,
        "scale_xyz": scale,
    }


def rotation_matrix_to_6d(rotation):
    """Encodes a rotation as its first two object axes in the target frame."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return rotation[:, 0], rotation[:, 1]


def rotation_6d_to_matrix(vector_a, vector_b, epsilon=1e-8):
    """Maps two unconstrained vectors to SO(3) using Gram--Schmidt."""
    vector_a = np.asarray(vector_a, dtype=np.float64)
    vector_b = np.asarray(vector_b, dtype=np.float64)
    norm_a = np.linalg.norm(vector_a)
    if norm_a < epsilon:
        raise ValueError("vector_a must be non-zero")
    axis_x = vector_a / norm_a
    orthogonal_b = vector_b - np.dot(axis_x, vector_b) * axis_x
    norm_b = np.linalg.norm(orthogonal_b)
    if norm_b < epsilon:
        raise ValueError("vector_b must not be parallel to vector_a")
    axis_y = orthogonal_b / norm_b
    axis_z = np.cross(axis_x, axis_y)
    return np.column_stack([axis_x, axis_y, axis_z])


def generate_sample(output, index, parameters, image_size, y_fov,
                    shadows=True):
    """Renders and writes a single dataset sample."""
    scene, object_transform = build_scene(parameters)
    world_to_camera = paz.SE3.view_transform(
        jp.array(parameters["camera_position"]),
        jp.array(parameters["camera_target"]), jp.array([0.0, 1.0, 0.0])
    )
    intensity = jp.full(3, parameters["light_intensity"])
    light = paz.graphics.PointLight(
        intensity, jp.array(parameters["light_position"])
    )
    rgb, depth = paz.graphics.render(
        image_size, y_fov, world_to_camera, scene, None, light,
        tiles=(1, 1), chunk_size=1024, shadows=shadows,
    )

    stem = f"{index:06d}"
    rgb_uint8 = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    depth_float = np.asarray(depth, dtype=np.float32)
    paz.image.write(str(output / "rgb" / f"{stem}.png"), rgb_uint8)
    np.save(output / "depth" / f"{stem}.npy", depth_float)
    mesh = build_mesh(parameters["shape"], object_transform)
    mesh.export(output / "meshes" / f"{stem}.ply")

    height, width = image_size
    intrinsics = paz.graphics.camera.compute_intrinsics(y_fov, height, width)
    world_to_camera_parameters = decompose_transform(world_to_camera)
    object_to_camera = np.asarray(world_to_camera) @ np.asarray(object_transform)
    object_to_camera_parameters = decompose_transform(object_to_camera)
    object_scale = object_to_camera_parameters["scale_xyz"]
    object_rotation = object_to_camera[:3, :3] / object_scale
    vector_a, vector_b = rotation_matrix_to_6d(object_rotation)
    metadata = {
        "location": {
            "camera_position": parameters["camera_position"],
            "camera_target": parameters["camera_target"],
            "world_to_camera": {
                "matrix_4x4": world_to_camera,
                **world_to_camera_parameters,
            },
            "object_in_camera": {
                "translation_xyz": object_to_camera_parameters[
                    "translation_xyz"
                ],
                "orientation_6d": {
                    "vector_a": vector_a,
                    "vector_b": vector_b,
                },
                "scale_xyz": object_scale,
            },
        },
        "light": {
            "type": "point",
            "position": parameters["light_position"],
            "intensity_rgb": np.full(3, parameters["light_intensity"]),
        },
        "shape": {
            "type": parameters["shape"],
            "object_scale": parameters["object_scale"],
        },
        "material": {
            "model": "phong",
            "color_rgb": parameters["color"],
            "ambient": parameters["ambient"],
            "diffuse": parameters["diffuse"],
            "specular": parameters["specular"],
            "shininess": parameters["shininess"],
        },
        "rgb": f"rgb/{stem}.png",
        "depth": f"depth/{stem}.npy",
        "mesh": f"meshes/{stem}.ply",
        "depth_unit": "metre",
        "camera_intrinsics_3x4": np.asarray(intrinsics).tolist(),
    }
    with (output / "metadata" / f"{stem}.json").open("w") as file:
        json.dump(jsonify(metadata), file, indent=2)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_rgbd"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--y-fov", type=float, default=45.0,
                        help="Vertical field of view in degrees.")
    parser.add_argument("--shapes", nargs="+", choices=SHAPE_BUILDERS,
                        default=list(SHAPE_BUILDERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shadows", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.num_samples < 1 or args.height < 1 or args.width < 1:
        raise ValueError("num-samples, height, and width must be positive")
    for directory in ("rgb", "depth", "meshes", "metadata"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.num_samples):
        parameters = sample_parameters(rng, args.shapes)
        generate_sample(args.output, index, parameters,
                        (args.height, args.width), np.deg2rad(args.y_fov),
                        not args.no_shadows)
        print(f"Generated {index + 1}/{args.num_samples}")


if __name__ == "__main__":
    main()
