import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Gaussian:
    """Built from the Gaussian decoder's raw outputs; the activated values are what renderers and PLY files use.

    xyz [N, 3] positions, features_dc [N, 1, 3] colour (degree-0 spherical harmonics), scaling [N, 3],
    rotation [N, 4] unit quaternions (w, x, y, z), opacity [N, 1]. The object is z-up, inside [-0.5, 0.5]^3."""

    def __init__(self, raw_xyz, features_dc, raw_scaling, raw_rotation, raw_opacity,
                 scaling_bias: float, opacity_bias: float, minimum_kernel_size: float):
        device = raw_xyz.device
        aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        scaling_offset = inverse_softplus(torch.tensor(scaling_bias)).to(device)
        opacity_offset = inverse_sigmoid(torch.tensor(opacity_bias)).to(device)
        rotation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        self.xyz = raw_xyz * aabb[None, 3:] + aabb[None, :3]
        self.features_dc = features_dc
        scaling = F.softplus(raw_scaling + scaling_offset)
        self.scaling = torch.sqrt(torch.square(scaling) + minimum_kernel_size ** 2)
        self.rotation = F.normalize(raw_rotation + rotation_offset[None, :])
        self.opacity = torch.sigmoid(raw_opacity + opacity_offset)

    @torch.no_grad()
    def transformed(self, linear, translation=None, *, mask=None):
        """Return a copy with an affine edit, in the SDK's z-up coordinates.

        ``linear`` is [3, 3] and ``translation`` is [3]: column-vector positions
        become ``linear @ xyz + translation``. An optional boolean [N] ``mask``
        selects the Gaussians to edit; all others are copied unchanged.

        The covariance is transformed as ``A @ covariance @ A.T``, including for
        shear and nonuniform scale. Colours (degree-0 SH) and peak opacities are
        preserved, not renormalized as probability densities. The input is never
        mutated. Geometry is computed and returned in float32 (float64 for a
        float64 input); this is an inference-time edit, not a differentiable op.

        This does not infer a correction or segment an object. Use a selected
        part and a measured transform when correcting geometry. A partial edit
        can create gaps at its boundary; a global shear can distort other parts.
        """
        dtype = torch.float64 if self.xyz.dtype == torch.float64 else torch.float32
        device = self.xyz.device
        a = torch.as_tensor(linear, dtype=dtype, device=device)
        t = torch.zeros(3, dtype=dtype, device=device) if translation is None else torch.as_tensor(
            translation, dtype=dtype, device=device)
        if a.shape != (3, 3) or not torch.isfinite(a).all():
            raise ValueError('linear must be a finite [3, 3] matrix')
        if t.shape != (3,) or not torch.isfinite(t).all():
            raise ValueError('translation must be a finite [3] vector')
        singular_values = torch.linalg.svdvals(a)
        if singular_values[-1] <= 3 * torch.finfo(dtype).eps * singular_values[0]:
            raise ValueError('linear must be nonsingular at the working precision')
        if mask is None:
            selected = torch.ones(len(self.xyz), dtype=torch.bool, device=device)
        else:
            selected = torch.as_tensor(mask, device=device)
            if selected.dtype != torch.bool or selected.shape != (len(self.xyz),):
                raise ValueError('mask must be a boolean [N] array')

        # The constructor takes decoder logits; these are already activated.
        result = object.__new__(Gaussian)
        for name in ('xyz', 'scaling', 'rotation'):
            setattr(result, name, getattr(self, name).detach().to(dtype=dtype).clone())
        result.features_dc = self.features_dc.detach().clone()
        result.opacity = self.opacity.detach().clone()
        if not selected.any():
            return result

        xyz, scale, quat = result.xyz[selected], result.scaling[selected], result.rotation[selected]
        if (not torch.isfinite(xyz).all() or not torch.isfinite(scale).all() or
                not torch.isfinite(quat).all() or (scale <= 0).any() or
                (torch.linalg.vector_norm(quat, dim=-1) == 0).any()):
            raise ValueError('selected Gaussians must have finite geometry, positive scales and nonzero rotations')
        rotation = _quaternion_to_matrix(quat)
        # SVD of A R diag(s), rather than eigendecomposing its square, avoids
        # squaring the condition number. U diag(s_new**2) U.T is the covariance.
        factor = a @ (rotation * scale[:, None, :])
        if not torch.isfinite(factor).all():
            raise ValueError('transform produced unrepresentable geometry')
        u, scale_new, _ = torch.linalg.svd(factor, full_matrices=False)
        # A covariance eigenbasis can have either handedness. Quaternion rotations
        # must be proper, so flip one column without changing the covariance.
        u[:, :, -1] *= torch.where(torch.linalg.det(u) < 0, -1.0, 1.0)[:, None]
        xyz_new = xyz @ a.T + t
        if not torch.isfinite(xyz_new).all() or not torch.isfinite(scale_new).all() or (scale_new <= 0).any():
            raise ValueError('transform produced unrepresentable geometry')
        result.xyz[selected] = xyz_new
        result.scaling[selected] = scale_new
        result.rotation[selected] = _matrix_to_quaternion(u)
        return result

    def save_ply(self, path):
        """Write the standard 3D Gaussian Splatting PLY (z-up object turned to the y-up convention viewers use)."""
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # Clamp before the inverse activations: a saturated opacity (exactly 1.0 after the sigmoid) or a zero
        # scale would otherwise be written as +/-inf, which some PLY viewers reject.
        eps = 1e-6
        opacities = inverse_sigmoid(self.opacity.clamp(eps, 1 - eps)).detach().cpu().numpy()
        scale = torch.log(self.scaling.clamp_min(eps)).detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # z-up -> y-up: rotate +90 degrees about x, i.e. (x, y, z) -> (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
        rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)

        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1).astype(np.float32)
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        elements = np.empty(attributes.shape[0], dtype=[(name, 'f4') for name in names])
        for column, name in enumerate(names):
            elements[name] = attributes[:, column]
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def _quaternion_to_matrix(quaternion):
    """Batch of (w, x, y, z) quaternions to rotation matrices."""
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


def _matrix_to_quaternion(matrix):
    """Proper rotation matrices to unit (w, x, y, z) quaternions.

    Choose the largest component before division, including at 180 degrees.
    """
    r00, r01, r02 = matrix[:, 0].unbind(-1)
    r10, r11, r12 = matrix[:, 1].unbind(-1)
    r20, r21, r22 = matrix[:, 2].unbind(-1)
    candidates = torch.stack([
        torch.stack([1+r00+r11+r22, r21-r12, r02-r20, r10-r01], dim=-1),
        torch.stack([r21-r12, 1+r00-r11-r22, r01+r10, r02+r20], dim=-1),
        torch.stack([r02-r20, r01+r10, 1-r00+r11-r22, r12+r21], dim=-1),
        torch.stack([r10-r01, r02+r20, r12+r21, 1-r00-r11+r22], dim=-1),
    ], dim=1)
    largest = candidates.diagonal(dim1=1, dim2=2).argmax(dim=-1)
    quaternion = candidates[torch.arange(len(matrix), device=matrix.device), largest]
    return F.normalize(quaternion, dim=-1)


def quaternion_multiply(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions; a is one quaternion, b is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)
