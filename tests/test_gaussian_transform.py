"""CPU-only geometric regressions; no inference, weights or model imports needed.

Run: python -m unittest discover -s tests -v
"""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData
import torch


# Import this small module without initializing idol's model dependencies.
spec = importlib.util.spec_from_file_location('idol_gaussian', Path(__file__).parents[1] / 'idol' / 'gaussian.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Gaussian = module.Gaussian


def rotation_matrix(q):
    """Independent quaternion-vector rotation used by the covariance oracle."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, vector = q[0], q[1:]
    eye = np.eye(3)
    columns = [(w*w - np.dot(vector, vector)) * v + 2 * np.dot(vector, v) * vector
               + 2 * w * np.cross(vector, v) for v in eye]
    return np.stack(columns, axis=1)


def covariance(gaussian):
    matrices = np.stack([rotation_matrix(q) for q in gaussian.rotation.cpu().numpy()])
    factors = matrices * gaussian.scaling.cpu().numpy()[:, None, :]
    return factors @ factors.transpose(0, 2, 1)


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(982)
    raw = torch.randn(11, 3, generator=generator, dtype=dtype)
    result = Gaussian(raw, torch.randn(11, 1, 3, generator=generator, dtype=dtype),
                      torch.randn(11, 3, generator=generator, dtype=dtype),
                      torch.randn(11, 4, generator=generator, dtype=dtype),
                      torch.randn(11, 1, generator=generator, dtype=dtype), .01, .1, .001)
    # Exercise isotropic, highly anisotropic and 180-degree rotations.
    result.scaling[:3] = torch.tensor([[.02, .02, .02], [.001, .07, .03], [.1, .02, .003]], dtype=dtype)
    result.rotation[:4] = torch.eye(4, dtype=dtype)
    for name in ('xyz', 'scaling', 'rotation', 'features_dc', 'opacity'):
        setattr(result, name, getattr(result, name).to(dtype=dtype))
    return result


class GaussianTransformTests(unittest.TestCase):
    def test_affine_centers_and_full_covariance(self):
        gaussian = fixture()
        a = np.array([[1.2, .45, -.2], [.1, .7, .3], [.0, .2, 1.8]])
        translation = np.array([.4, -.2, .8])
        edited = gaussian.transformed(a, translation)
        np.testing.assert_allclose(edited.xyz.numpy(), gaussian.xyz.numpy() @ a.T + translation, atol=1e-12)
        np.testing.assert_allclose(covariance(edited), a @ covariance(gaussian) @ a.T, atol=1e-12)
        np.testing.assert_allclose(torch.linalg.vector_norm(edited.rotation, dim=-1), 1, atol=1e-12)
        np.testing.assert_array_equal(edited.features_dc, gaussian.features_dc)
        np.testing.assert_array_equal(edited.opacity, gaussian.opacity)

    def test_reflection_and_inverse_roundtrip(self):
        gaussian = fixture()
        a = np.array([[-1., .6, 0], [0, 1.3, .1], [0, 0, .7]])
        t = np.array([2., -1., .3])
        transformed = gaussian.transformed(a, t)
        back = transformed.transformed(np.linalg.inv(a), -np.linalg.inv(a) @ t)
        np.testing.assert_allclose(back.xyz.numpy(), gaussian.xyz.numpy(), atol=1e-12)
        np.testing.assert_allclose(covariance(back), covariance(gaussian), atol=1e-12)
        for quaternion in transformed.rotation:
            self.assertAlmostEqual(np.linalg.det(rotation_matrix(quaternion)), 1., places=12)

    def test_identity_handles_repeated_singular_values(self):
        gaussian = fixture()
        edited = gaussian.transformed(np.eye(3))
        np.testing.assert_allclose(edited.xyz, gaussian.xyz, atol=0)
        np.testing.assert_allclose(covariance(edited), covariance(gaussian), atol=1e-12)

    def test_selection_preserves_unselected_parts_and_input(self):
        gaussian = fixture()
        original = {name: getattr(gaussian, name).clone() for name in
                    ('xyz', 'scaling', 'rotation', 'features_dc', 'opacity')}
        mask = torch.tensor([True, False, True] + [False] * 8)
        a = np.array([[1., .4, 0], [0, 1., 0], [0, 0, 1.]])
        edited = gaussian.transformed(a, [1, 2, 3], mask=mask)
        for name, value in original.items():
            self.assertTrue(torch.equal(getattr(gaussian, name), value), name)
            self.assertTrue(torch.equal(getattr(edited, name)[~mask], value[~mask]), name)
            self.assertNotEqual(getattr(edited, name).data_ptr(), getattr(gaussian, name).data_ptr())
        np.testing.assert_allclose(covariance(edited)[mask], a @ covariance(gaussian)[mask] @ a.T, atol=1e-12)
        edited.xyz.zero_()
        edited.features_dc.zero_()
        self.assertTrue(torch.equal(gaussian.xyz, original['xyz']))
        self.assertTrue(torch.equal(gaussian.features_dc, original['features_dc']))

    def test_empty_selection_and_empty_object(self):
        gaussian = fixture()
        unchanged = gaussian.transformed(np.eye(3), mask=torch.zeros(11, dtype=torch.bool))
        self.assertTrue(torch.equal(unchanged.rotation, gaussian.rotation))
        for name in ('xyz', 'scaling', 'rotation', 'features_dc', 'opacity'):
            setattr(gaussian, name, getattr(gaussian, name)[:0])
        self.assertEqual(gaussian.transformed(np.eye(3)).xyz.shape, (0, 3))

    def test_half_geometry_promoted_for_cpu_svd(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                gaussian = fixture(dtype)
                a = np.array([[1, .3, 0], [0, 1, .2], [0, 0, 1]])
                edited = gaussian.transformed(a)
                self.assertEqual(edited.xyz.dtype, torch.float32)
                self.assertEqual(edited.scaling.dtype, torch.float32)
                self.assertEqual(edited.rotation.dtype, torch.float32)
                # Convert only for the independent NumPy oracle (no bfloat16).
                gaussian.rotation = gaussian.rotation.float()
                gaussian.scaling = gaussian.scaling.float()
                np.testing.assert_allclose(covariance(edited), a @ covariance(gaussian) @ a.T, rtol=1e-5, atol=1e-8)

    def test_ply_export_preserves_edited_covariance_in_y_up(self):
        gaussian = fixture(torch.float32).transformed([[1, .4, .1], [0, 1, .3], [0, 0, 1]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'edited.ply'
            gaussian.save_ply(path)
            vertices = PlyData.read(path)['vertex'].data
        p = np.stack([vertices[name] for name in ('x', 'y', 'z')], axis=1)
        scale = np.exp(np.stack([vertices[f'scale_{i}'] for i in range(3)], axis=1))
        quaternion = np.stack([vertices[f'rot_{i}'] for i in range(4)], axis=1)
        rotations = np.stack([rotation_matrix(q) for q in quaternion])
        factors = rotations * scale[:, None, :]
        actual_covariance = factors @ factors.transpose(0, 2, 1)
        z_to_y = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(p, gaussian.xyz.numpy() @ z_to_y.T, atol=1e-7)
        np.testing.assert_allclose(actual_covariance, z_to_y @ covariance(gaussian) @ z_to_y.T, rtol=1e-5, atol=1e-8)
        self.assertTrue(all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names))

    def test_known_planar_axes_remove_shear_without_moving_other_parts(self):
        # Rounded edge directions from an observed sheared reconstruction. This
        # is a numeric regression, not an image-based or ground-truth benchmark.
        u = np.array([-.9408, .6611, 0.])
        v = np.array([-.0379, -.3358, 0.])
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)
        u_hat = u / np.linalg.norm(u)
        v_perpendicular = v - np.dot(v, u_hat) * u_hat
        v_perpendicular *= np.linalg.norm(v) / np.linalg.norm(v_perpendicular)
        a = np.column_stack([u, v_perpendicular, normal]) @ np.linalg.inv(np.column_stack([u, v, normal]))
        origin = np.array([.1, -.2, .3])
        gaussian = fixture()
        corners = origin + np.stack([np.zeros(3), u, u+v, v])
        gaussian.xyz[:4] = torch.from_numpy(corners)
        mask = torch.arange(11) < 4
        edited = gaussian.transformed(a, origin - a @ origin, mask=mask)
        points = edited.xyz[:4].numpy()
        edges = np.roll(points, -1, axis=0) - points
        angles = [math.degrees(math.acos(np.clip(np.dot(-edges[i-1], edges[i]) /
                  (np.linalg.norm(edges[i-1]) * np.linalg.norm(edges[i])), -1, 1))) for i in range(4)]
        np.testing.assert_allclose(angles, [90.] * 4, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(edges, axis=1),
                                   [np.linalg.norm(u), np.linalg.norm(v)] * 2, atol=1e-12)
        np.testing.assert_allclose(covariance(edited)[:4], a @ covariance(gaussian)[:4] @ a.T, atol=1e-12)
        self.assertTrue(torch.equal(edited.xyz[4:], gaussian.xyz[4:]))

    def test_invalid_transforms_and_selection_are_rejected(self):
        gaussian = fixture()
        for linear in (np.zeros((3, 3)), np.diag([1, 1, 0]), np.ones((3, 4)),
                       np.full((3, 3), float('nan')), np.full((3, 3), float('inf'))):
            with self.subTest(linear=linear), self.assertRaises(ValueError):
                gaussian.transformed(linear)
        for translation in ([1, 2], [1, 2, float('inf')]):
            with self.subTest(translation=translation), self.assertRaises(ValueError):
                gaussian.transformed(np.eye(3), translation)
        for mask in ([True], [1] * 11, np.zeros((11, 1), dtype=bool)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                gaussian.transformed(np.eye(3), mask=mask)

    def test_invalid_geometry_is_rejected(self):
        for attribute, value in (('xyz', float('nan')), ('scaling', 0.), ('rotation', 0.)):
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                gaussian = fixture()
                getattr(gaussian, attribute)[0] = value
                gaussian.transformed(np.eye(3))

    def test_finite_inputs_that_overflow_are_rejected(self):
        for attribute in ('xyz', 'scaling'):
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                gaussian = fixture(torch.float32)
                getattr(gaussian, attribute)[0] = 2e38
                gaussian.transformed(np.eye(3) * 4)


if __name__ == '__main__':
    unittest.main()
