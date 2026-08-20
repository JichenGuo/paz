import numpy as np

from paz.graphics.synthetic_data.generate_synthetic_rgbd import (
    build_mesh,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    sample_is_complete,
    sample_parameters,
)


def test_sample_parameters_is_reproducible():
    first = sample_parameters(np.random.default_rng(7), ["cube"])
    second = sample_parameters(np.random.default_rng(7), ["cube"])
    assert first["shape"] == second["shape"]
    assert first["object_scale"] == second["object_scale"]
    assert np.allclose(first["camera_position"], second["camera_position"])
    assert np.allclose(first["light_position"], second["light_position"])


def test_camera_target_is_offset_in_nominal_image_plane():
    parameters = sample_parameters(np.random.default_rng(9), ["sphere"])
    center = np.array([0.0, parameters["object_scale"], 0.0])
    offset = parameters["camera_target"] - center
    view_direction = center - parameters["camera_position"]

    assert not np.allclose(offset, 0.0)
    assert np.isclose(np.dot(offset, view_direction), 0.0, atol=1e-7)
    assert np.linalg.norm(offset) <= np.sqrt(0.5) * parameters["object_scale"]


def test_camera_target_offsets_vary_in_two_dimensions():
    offsets = []
    for seed in range(32):
        parameters = sample_parameters(np.random.default_rng(seed), ["cube"])
        center = np.array([0.0, parameters["object_scale"], 0.0])
        view = center - parameters["camera_position"]
        view /= np.linalg.norm(view)
        right = np.cross(view, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        offset = parameters["camera_target"] - center
        offsets.append([np.dot(offset, right), np.dot(offset, up)])

    assert np.all(np.std(offsets, axis=0) > 0.05)


def test_rotation_6d_round_trip():
    angle = 0.7
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    vector_a, vector_b = rotation_matrix_to_6d(rotation)
    reconstructed = rotation_6d_to_matrix(vector_a, vector_b)
    assert np.allclose(reconstructed, rotation)


def test_rotation_6d_orthogonalizes_raw_vectors():
    rotation = rotation_6d_to_matrix([2.0, 0.0, 0.0], [1.0, 3.0, 0.0])
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_mesh_rests_on_floor():
    parameters = sample_parameters(np.random.default_rng(3), ["cylinder"])
    from paz.graphics.synthetic_data.generate_synthetic_rgbd import build_scene

    _, transform = build_scene(parameters)
    mesh = build_mesh("cylinder", transform)
    assert np.isclose(mesh.bounds[0, 1], 0.0, atol=1e-6)


def test_sample_is_complete_requires_every_output(tmp_path):
    paths = [
        tmp_path / "rgb" / "000000.png",
        tmp_path / "depth" / "000000.npy",
        tmp_path / "meshes" / "000000.ply",
        tmp_path / "metadata" / "000000.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert sample_is_complete(tmp_path, 0)
    paths[-1].unlink()
    assert not sample_is_complete(tmp_path, 0)
