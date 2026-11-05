# Affine edits of generated Gaussians

A single-image reconstruction can reproduce the input appearance while giving a
rigid object an incorrect shape in other views. For example, a rectangular table
may have a sheared top. Changing the display camera cannot fix that geometry.
Likewise, moving Gaussian centers alone leaves their anisotropic shapes behind.

`Gaussian.transformed()` provides an **explicit, optional postprocessing edit**:

```python
import torch

# A is a 3 x 3 linear map and t is a length-3 translation.
# Both use the SDK's z-up object coordinates, BEFORE save_ply's y-up conversion.
A = torch.tensor([[1.0, -0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
edited = output.gaussian.transformed(A, [0.0, 0.0, 0.0])
edited.save_ply("edited.ply")
output.gaussian.save_ply("original.ply")  # Original is unchanged.
```

For column-vector positions, the mapping is `x_new = A @ x + t`. For a Gaussian
with covariance `Sigma`, the new covariance is `A @ Sigma @ A.T`. The implementation
uses an SVD of `A @ R @ diag(scale)` to recover positive scales and a proper
quaternion rotation, including for shear, nonuniform scaling and reflection.
Degree-0 SH colors and peak opacities are copied unchanged. These are rendering
primitives, not normalized probability densities: no determinant factor is
applied to opacity. The apparent opacity of an overlapping rendered collection
is not guaranteed to remain identical after an edit.

Geometry is returned in float32, or float64 when the original positions are
float64. This avoids unsupported half-precision SVD and reduces precision loss.
The operation is for inference-time editing, with no gradient tracking. Finite,
nonsingular transforms are required; singularity is checked at working precision.

## Correct a known pair of planar axes

When two **measured 3D edge directions** are known to represent perpendicular
edges, an explicit affine map can remove their shear. This example keeps the
first edge fixed, makes the second perpendicular while preserving its length,
and preserves the plane normal. It fixes `origin` in place:

```python
import torch

# Illustrative model-space measurements, not directions read from a 2D photo.
u = torch.tensor([-0.9408, 0.6611, 0.0], dtype=torch.float64)
v = torch.tensor([-0.0379, -0.3358, 0.0], dtype=torch.float64)
origin = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
u_hat = u / torch.linalg.vector_norm(u)
normal = torch.linalg.cross(u, v)
normal /= torch.linalg.vector_norm(normal)
v_orthogonal = v - torch.dot(v, u_hat) * u_hat
v_orthogonal *= torch.linalg.vector_norm(v) / torch.linalg.vector_norm(v_orthogonal)

source_basis = torch.stack([u, v, normal], dim=1)
target_basis = torch.stack([u, v_orthogonal, normal], dim=1)
A = target_basis @ torch.linalg.inv(source_basis)
t = origin - A @ origin

# part_mask must be a boolean [N] array from a separately reviewed selection.
# The SDK does not infer this selection from the example edge measurements.
edited = output.gaussian.transformed(A, t, mask=part_mask)
edited.save_ply("part-edited.ply")
```

The vectors must be finite, nonzero and not collinear. In this numeric example
the adjacent angle is approximately 118.7 degrees before correction and 90
degrees afterwards. The regression test checks the corrected corners, edge
lengths, Gaussian covariances and unchanged unselected Gaussians. This is a
controlled geometric test, **not** an image reconstruction accuracy benchmark.

## Limits and validation

- This does not change the model weights, infer true dimensions, detect shear,
  segment a tabletop, or guarantee a correct reconstruction from one photograph.
- Apply right-angle constraints only to parts known to be rectangular. Curved
  lamps, angled legs, organic shapes and perspective image edges do not justify
  that assumption.
- A single affine map cannot fix bowed edges, taper, missing surfaces, topology
  errors, or a quadrilateral that is not a parallelogram. A selected edit may
  create seams or detach supporting parts; inspect and handle those separately.
- A global correction also deforms all other parts, including props on a table.
  A boolean mask preserves the unselected primitives exactly, but is not an
  automatic blending or attachment constraint.
- The predicted input camera is not recalibrated. A nonrigid edit may change the
  match to the source photograph; compare the source view and orthographic
  top/front/side views, not just a single attractive perspective.
- Keep the original output and record the transform and selection with the edited
  artifact. Do not label a manually constrained result as unmodified inference.
- The existing PLY writer exports float32 and clamps scales to at least `1e-6`.
  Edits at a much smaller scale, or float64 precision beyond that format, will
  not survive export exactly.

The CPU regression suite needs only PyTorch, NumPy and plyfile and does not load
models or weights:

```bash
python -m unittest discover -s tests -v
```
