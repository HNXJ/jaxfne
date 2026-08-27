# jax-fem interoperability (scoping stage)

**Status: schema/contract only.** `JaxFemFieldBridge` verifies the optional
dependency is installed and returns a status contract — it does not run a
real field solve yet. This page documents the current scope and the plan,
not a working feature; see `artifacts/publication/publication_evidence_index.json`
for the tracked field-solver backlog entry this serves.

## Why this bridge exists

jaxfne's laminar source-to-field readout (`jaxfne.fields.project_laminar_sources`)
is currently a static, non-learned linear projection
(`field_solver_status = "linear_solver"`) — not a differentiable elliptic/PDE
solve. [jax-fem](https://github.com/deepmodeling/jax-fem) is a real, actively
maintained differentiable GPU-accelerated FEM library built on JAX, proving
that a differentiable elliptic solve composing with jaxfne's existing
differentiable spiking/HDP pipeline is feasible — the goal this bridge works
toward is a real differentiable volume-conductor solve for the laminar column
geometry, not a name.

## Licensing — read this before installing

jax-fem is **GPLv3-licensed**, with commercial licensing available separately
directly from the author (see jax-fem's own README/LICENSE). It is
deliberately kept as an **optional extra**, never a core jaxfne dependency,
and is **not** included in the `jaxfne[all]` bundle — installing it is a
separate, explicit choice:

```bash
pip install jaxfne[jax-fem]
```

jaxfne's own core stays MIT-licensed and usable by anyone who doesn't opt
into this specific bridge. If you plan to use jax-fem in a commercial
product, review its license (and consider contacting its author about a
commercial license) independently of jaxfne's own licensing.

## Current status

```python
import jaxfne as jtfne

bridge = jtfne.JaxFemFieldBridge(geometry="laminar_column", n_layers=6)
spec = bridge.to_spec().to_dict()
# {"name": "jax_fem_field_bridge", "backend": "jax_fem",
#  "status": "schema_only_no_field_solve", "physical_amplitude_calibrated": False, ...}

bridge.construct()  # raises ImportError with an install hint if jax-fem isn't installed
```

## Scoping notes toward a real solve (not yet implemented)

- jax-fem's typical demos generate meshes via `gmsh` (an external mesh-generation
  tool), but `jax_fem.generate_mesh.Mesh` itself only needs raw `(points, cells)`
  arrays, and a gmsh-free `box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)`
  helper already exists in the library — a layered laminar-column geometry
  (a thin box, `Nx=Ny=1`, `Nz`=number of depth layers) does not require
  adopting `gmsh` as a dependency.
- The real solve would subclass `jax_fem.problem.Problem`, defining
  `get_tensor_map()` (the flux/conductivity term — piecewise-constant per
  layer, matching jaxfne's existing layer geometry) and `get_mass_map()`
  (the source term, fed from jaxfne's per-neuron source proxy), then call
  `jax_fem.solver.solver(problem)`.
- Validation target: the classical "method of images" closed-form solution
  for a point current source in a planar-layered volume conductor with
  piecewise-constant conductivity — the same problem family referenced in
  Pettersen/Einevoll-style forward-CSD-modeling literature. See
  `novelty::tfne-analytic-ground-truth-validation` in the tracked backlog.

## See also

- [Source and Field Equations](../source_field_equations.md) — the current
  `linear_solver`/`proxy_no_field_solve` default this bridge targets replacing.
- [Jaxley interoperability](jaxley_interop.md) — the analogous, fully-working
  optional-bridge pattern this one is modeled on.
