import numpy as np

from paz.graphics.generate_synthetic_rgbd import (
    build_mesh,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    sample_parameters,
)


def test_sample_parameters_is_reproducible():
    first = sample_parameters(np.random.default_rng(7), ["cube"])
    second = sample_parameters(np.random.default_rng(7), ["cube"])
    assert first["shape"] == second["shape"]
    assert first["object_scale"] == second["object_scale"]
    assert np.allclose(first["camera_position"], second["camera_position"])
    assert np.allclose(first["light_position"], second["light_position"])


def test_camera_targets_object_center():
    parameters = sample_parameters(np.random.default_rng(9), ["sphere"])
    expected = np.array([0.0, parameters["object_scale"], 0.0])
    assert np.allclose(parameters["camera_target"], expected)


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
    from paz.graphics.generate_synthetic_rgbd import build_scene

    _, transform = build_scene(parameters)
    mesh = build_mesh("cylinder", transform)
    assert np.isclose(mesh.bounds[0, 1], 0.0, atol=1e-6)
