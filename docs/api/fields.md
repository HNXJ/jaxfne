# Fields API

Field solvers, source projection, and spatial operators for neural field computation.

## Overview

The fields module provides spatial projection operators to transform point-source neural currents into field readouts (LFP, CSD, EEG, MEG). All field computations are **proxy approximations** suitable for tutorial-scale simulations, not full PDE solutions.

```
Emitter currents → Source projection → Field convolution → Probe readouts
```

---

## Source Projection

### LaminarSourceGeometry

```python
jaxfne.LaminarSourceGeometry
```

*(Defined in `jaxfne/core.py`, not `jaxfne/fields/` — re-exported at package
level as `jaxfne.LaminarSourceGeometry`.)*

A frozen dataclass grouping named `LaminarPopulation` descriptors and
materializing a deterministic `(n_units_total, 3)` positions array for use in
`project_laminar_sources`/`project_sources_to_laminar_field`. Depths are
proxy-normalized coordinates in `[0, 1]`, not physical microns; no
physical-amplitude, PDE, or calibration claim is made.

### Attributes

- `populations` (`tuple[LaminarPopulation, ...]`): ordered named population descriptors
- `n_units_total` (`int`): total neuron count across all populations
- `position_units` (`str`, default `"relative_laminar_depth_proxy"`)
- `source_calibration_status` (`str`, default `"uncalibrated_izhikevich_native_current"`)
- `physical_amplitude_calibrated` (`bool`, default `False`)
- `claim_level` (`str`, default `"computational_scaffold"`)

`LaminarPopulation` (the element type of `populations`) is itself a frozen
dataclass: `name: str`, `cell_type: str`, `layer: str`, `depth_min: float`,
`depth_max: float`, `n_units: int`, plus the same
`source_calibration_status`/`physical_amplitude_calibrated`/`claim_level`
trio.

### Methods

#### `validate() -> dict`

Checks population sum equals `n_units_total`, that
`physical_amplitude_calibrated` stays `False`, and validates each contained
`LaminarPopulation`. Returns `{"valid": bool, "issues": list[str], "n_populations": int}`.

#### `to_dict() -> dict`

JSON-safe dict: `type`, `n_units_total`, `n_populations`, `position_units`,
`source_calibration_status`, `physical_amplitude_calibrated`, `claim_level`,
and `populations` (each population's own `to_dict()`).

#### `population_slices() -> dict[str, slice]`

Maps each population `name` to the `slice` of neuron indices it occupies
(populations are laid out contiguously in `populations` order).

#### `positions_array(dtype: str = "float32") -> jax.Array`

Returns a deterministic `(n_units_total, 3)` array: `x=0`, `y=0`, `z`
linearly spaced within each population's `[depth_min, depth_max]` range, in
population order. No random sampling — feed this directly into
`project_laminar_sources(sources, positions=...)`.

### Building a `LaminarSourceGeometry`

There is no `from_dict` constructor. Build one from a sequence of
`LaminarPopulation` objects, either directly or via the
`laminar_source_geometry()` helper (also `jaxfne.core`), which validates each
population and computes `n_units_total` for you:

```python
from jaxfne.core import LaminarPopulation, laminar_source_geometry

pops = [
    LaminarPopulation(name="L4_E", cell_type="E", layer="L4",
                       depth_min=0.4, depth_max=0.6, n_units=200),
    LaminarPopulation(name="L2_PV", cell_type="PV", layer="L2",
                       depth_min=0.1, depth_max=0.3, n_units=50),
]
geom = laminar_source_geometry(pops)
positions = geom.positions_array()  # (250, 3), feed into project_laminar_sources
```

---

### `project_laminar_sources(sources, positions, *, n_contacts=16, width=0.10, mode="density_preserving", dtype="float32") -> FieldOutput`

Project emitter sources into laminar contact space.

**Parameters (keyword-only after `positions`):**
- `sources` (jax.Array): Emitter source signals `[time, n_emitters]`
- `positions` (jax.Array): Emitter positions `[n_emitters, 3]`
- `n_contacts` (int): Number of laminar contacts, default `16`
- `width` (float): Gaussian kernel width, default `0.10`
- `mode` (str): `"density_preserving"` (default) preserves absolute scale; `"row_normalize"` erases attenuation for contacts outside the modeled population
- `dtype` (str): Output dtype, default `"float32"`

**Returns:** `FieldOutput`

**Description:**
Transforms point-emitter sources into distributed laminar-contact density via a Gaussian-kernel proxy projection — not a full dipole/PDE solve.

**Example:**
```python
field = jtfne.project_laminar_sources(sources, positions, n_contacts=16)
```

---

### `project_sources_to_laminar_field(sources, positions, n_contacts=16, *, mode="density_preserving", dtype="float32") -> FieldOutput`

Convenience wrapper over `project_laminar_sources` with `n_contacts` as a
positional parameter.

**Parameters:**
- `sources` (jax.Array): Source density `[time, n_emitters]`
- `positions` (jax.Array): Emitter positions `[n_emitters, 3]`
- `n_contacts` (int, positional): Number of laminar contacts, default `16`
- `mode` (str, keyword-only): `"density_preserving"` (default) | `"row_normalize"`
- `dtype` (str, keyword-only): Output dtype, default `"float32"`

**Returns:** `FieldOutput` containing LFP and CSD proxy arrays

**Example:**
```python
field = jtfne.project_sources_to_laminar_field(sources, positions, n_contacts=16, mode="density_preserving")
```

---

## FieldOutput

```python
jaxfne.FieldOutput
```

Frozen dataclass container for laminar proxy field/readout arrays, returned
by `project_laminar_sources` / `project_sources_to_laminar_field`.

### Attributes (dataclass fields)

- `source_proxy` (jax.Array): source density projected onto contacts, `[T, n_contacts]`
- `phi_e_proxy` (jax.Array): extracellular potential proxy, `[T, n_contacts]`
- `csd_proxy` (jax.Array): current source density proxy, `[T, n_contacts]`
- `lfp_proxy` (jax.Array): local field potential proxy, `[T, n_contacts]` (currently identical to `phi_e_proxy` — see below)
- `kernel` (jax.Array): the projection kernel/leadfield matrix, `[n_contacts, n_emitters]`
- `contact_depths` (jax.Array): normalized `[0, 1]` depth of each contact, `[n_contacts]`
- `diagnostics` (dict): field solution metadata and validation flags (see below)

### Read-only properties (aliases)

- `.phi_e` -> `phi_e_proxy`
- `.csd` -> `csd_proxy`
- `.lfp` -> `lfp_proxy`
- `.kernel_matrix` -> `kernel`

**Note:** `lfp_proxy` and `phi_e_proxy` are currently the same array
(`jaxfne/fields/proxy.py:169-170` sets `lfp_proxy = source_proxy` then
`phi_e_proxy = lfp_proxy`) — the module does not yet distinguish an LFP
readout from the raw potential proxy. `csd_proxy` is the genuinely distinct
quantity, computed from `phi_e_proxy` via `csd_tensor`.

There is no `FieldOutput.to_dict()` method and no `source` attribute (the
field is named `source_proxy`). To get a JSON-safe dict of the diagnostics,
use `jaxfne.io.json_safe`:

```python
import json
from jaxfne.io import json_safe

field_dict = json_safe(field.diagnostics)
```

### Diagnostics Dictionary

`diagnostics` is produced by `validate_projection_invariants` (structural
invariants) merged with a field-solution report from
`_make_field_solution_report`, plus a few extra keys — 40 keys total as of
this writing (verified via `sorted(field.diagnostics.keys())` on a live
`project_laminar_sources` call). Every key below is required — it is always
present on the returned dict, regardless of which projection `mode` was
used. The keys relevant to truth-gate/claim status:

| Field | Type | Purpose |
|-------|------|---------|
| `field_solver_status` | str | Solver type: always `"linear_solver"` for these proxy operators |
| `solver_name` | str | Human-readable ID: `"laminar_proxy"` |
| `boundary_condition` | str | BC declaration: `"mean_zero_neumann"` |
| `gauge` | str | Gauge convention: `"mean_zero"` |
| `csd_sign_convention` | str | **Canonical:** `"positive_equals_extracellular_source"` |
| `current_density_layout` | str | J_e status: `"not_applicable"` |
| `solver_residual_l2_relative` | float \| None | PDE residual; `None` for proxy |
| `n_iterations` | int \| None | Solver iteration count; `None` for proxy |
| `converged` | bool \| None | Convergence flag; `None` for proxy |
| `finite_phi_e` / `finite_phi_e_proxy` | bool | Is potential array finite? |
| `finite_J_e` | bool | Is current density finite? Always `False` (J_e never computed) |
| `finite_CSD` / `finite_csd_proxy` | bool | Is CSD array finite? |
| `finite_source_proxy`, `finite_lfp_proxy`, `finite_sources`, `finite_positions`, `finite_kernel` | bool | Per-array finiteness checks |
| `field_claim_level` | str | Statement authority: `"proxy_readout"` |
| `physical_amplitude_calibrated` | bool | Can statement physical units? Always `False` for proxy |
| `source_projection_mode` | str | How sources map to field: `"proxy_no_field_solve"` |
| `source_current_conservation_status` | str | Conservation test: `"not_applicable_proxy_mode"` |
| `source_conservation_tested` | bool | Was conservation validated? Always `False` for proxy |
| `source_conservation_claim_allowed` | bool | Can statement conserved sources? Always `False` for proxy |
| `source_calibration_status` | str | `"uncalibrated_izhikevich_native_current"` |
| `source_decomposition` | str | `"proxy_reduced_emitter"` |
| `field_admissibility` | dict | Nested dict of kernel-normalization/finiteness sub-checks |
| `warnings` | list[str] | Any structural-invariant warnings raised during validation |
| `*_shape` keys | tuple[int, ...] | Shapes of `source`/`positions`/`kernel`/`source_proxy`/`phi_e_proxy`/`csd_proxy`/`lfp_proxy` |
| `dtype` | str | dtype of the input `sources` array |

There is no `amplitude_status` or `field_model_status` key (the doc
previously invented these) — the real names are `physical_amplitude_calibrated`
and `field_claim_level`. Similarly, `source_conservation_claim_allowed` is the
real key in place of the invented `source_conservation_status`.

---

## Field Solvers

### `probe_laminar_modes(field_output, modes=("source", "phi_e", "CSD", "LFP")) -> dict`

Extract the requested proxy readouts from a `FieldOutput`.

**Parameters:**
- `field_output` (`FieldOutput`): result of `project_laminar_sources` / `project_sources_to_laminar_field`.
- `modes` (`Sequence[str]`, default `("source", "phi_e", "CSD", "LFP")`): which
  proxies to extract. Valid values: `"source"`/`"sources"`, `"phi_e"`, `"CSD"`,
  `"LFP"`, `"J_e"`. The default extracts the full proxy set.

**Returns:** a `dict` mapping each requested proxy to its `*_proxy` array (plus a
`readout_metadata` entry). Requesting `"J_e"` returns a status string only —
current density is never synthesized without a real field solver.

**Description:**
A thin accessor over the laminar proxy outputs. All returned arrays are proxy
readouts (`field_solver_status = "linear_solver"`), not physical signals.

**Example:**
```python
fo = jtfne.project_laminar_sources(sources, positions)
out = jtfne.probe_laminar_modes(fo)                  # default: source, phi_e, CSD, LFP
csd = out["csd_proxy"]; lfp = out["lfp_proxy"]
```

---

### `construct_source_tensor(*, mode="total_membrane_current_proxy", ...) -> (array, metadata)`

Assemble a proxy source tensor for laminar projection from the named `mode`.

**Parameters:**
- `mode` (str, default `"total_membrane_current_proxy"`): source basis. Also
  `"decomposed_cap_ion_plus_synaptic_proxy"` and `"spike_proxy"`. The array the
  chosen mode needs must be supplied (e.g. `total_membrane_current=...`).
- `total_membrane_current`, `decomposed_cap_ion`, `synaptic_current`,
  `spike_proxy` (`jax.Array`, optional): the per-mode source arrays.
- `scale` (float, default `1.0`): multiplicative scale.

**Returns:** `(source_array, metadata)` — the proxy source array plus a JSON-safe
metadata dict recording the mode and proxy status. The result is a proxy source
basis, not a physical current density.

**Example:**
```python
src, meta = jtfne.construct_source_tensor(total_membrane_current=I_mem)  # default mode
```

---

### `cable_filter_tau(cell_type, depth_z, *, tau_e_superficial_ms=1.0, tau_e_deep_ms=5.0, tau_pv_ms=0.5, tau_sst_ms=2.0, tau_vip_ms=2.0) -> tau_s`

Build the per-neuron cable time constant array consumed by `cable_filter_sources`.

**Parameters:**
- `cell_type` (`Sequence[str]`): per-neuron cell-type labels, length `[N]`.
- `depth_z` (`jax.Array`): per-neuron normalized laminar depth in `[0, 1]`,
  length `[N]` (0=superficial, 1=deep).
- `tau_e_superficial_ms`, `tau_e_deep_ms` (float): `E`-cell tau is linearly
  interpolated between these two values by `depth_z`.
- `tau_pv_ms`, `tau_sst_ms`, `tau_vip_ms` (float): fixed tau per interneuron
  subtype. Any other/unrecognized cell type falls back to `tau_sst_ms`.

**Returns:** `tau_s` (`jax.Array`, seconds, shape `[N]`).

**Description:**
Defaults are the numerically-swept operating point used by `cable_filter_sources` below
(`order=2`). `PV` gets the shortest tau (highest cutoff, passes gamma at
every depth); `E` cells get a depth-graded tau (long apical dendrites on deep
pyramidal cells => longer tau => lower cutoff).

**Example:**
```python
nt = model.neuron_table()
tau_s = jtfne.cable_filter_tau(nt["cell_type"], nt["z"])
```

---

### `cable_filter_sources(sources, tau_s, dt_ms, *, order=2) -> filtered_sources`

Apply a depth/cell-type-dependent passive-cable low-pass **tensor** to
per-neuron source-proxy traces. Standard pipeline stage:

```
emitter -> (source_scale gain tensor) -> source
        -> cable_filter_sources (this tensor)
        -> readout (project_laminar_sources / eeg_proxy_transform / meg_proxy_transform)
```

**Parameters:**
- `sources` (`jax.Array`): source-proxy traces, shape `[T, N]`.
- `tau_s` (`jax.Array`): per-neuron cable time constant in seconds, shape
  `[N]`. Build with `cable_filter_tau`.
- `dt_ms` (float): simulation timestep in milliseconds.
- `order` (int, default `2`): number of cascaded single-pole RC sections.

**Returns:** filtered source-proxy traces, shape `[T, N]`.

**Description:**
Computes, per neuron, the cascaded single-pole transfer function
`H[f, n] = 1 / (1 + 2j*pi*f*tau_s[n]) ** order` and applies it along the time
axis via FFT. A phenomenological proxy for passive dendritic cable
filtering — a Relative-value projection (`field_solver_status="linear_solver"`,
`physical_amplitude_calibrated=False`).

Validated on a 100-neuron canonical V1 column (10 trials x 6000 ms,
`cable_filter_tau` defaults, `order=2`): alpha/beta deep:superficial power
ratio 1.30 (deep-dominant, same direction as the unfiltered baseline) and
gamma deep:superficial power ratio 0.66 (flips to superficial-dominant — a
genuine absolute band-selective effect, absent from the unfiltered
flat-gain baseline). `order=1` gives the same direction but a much weaker
gamma flip (0.93); `order=2` is the sweep-selected default.

**Example:**
```python
tau_s = jtfne.cable_filter_tau(cell_type, depth_z)
filtered = jtfne.cable_filter_sources(sources, tau_s, dt_ms=0.5, order=2)
fo = jtfne.project_laminar_sources(filtered, positions, n_contacts=32)
```

---

### `cable_filter_report(tau_s, order=2) -> dict`

JSON-safe truth-gate report for a `cable_filter_sources` call.

**Parameters:**
- `tau_s` (`jax.Array`): the same per-neuron tau array passed to
  `cable_filter_sources`.
- `order` (int): the same filter order passed to `cable_filter_sources`.

**Returns:** a `dict` with `tau_s_mean/min/max`, `cutoff_hz_mean`,
`field_solver_status="linear_solver"`, `physical_amplitude_calibrated=False`,
`claim_level="computational_scaffold"`, and a `finite_tau` flag.

**Example:**
```python
report = jtfne.cable_filter_report(tau_s, order=2)
assert report["finite_tau"]
```

---

### `csd_tensor(phi_e_proxy, dz) -> csd_proxy`

Spatial second-derivative CSD tensor (readout family, depth-axis stage).

**Parameters:**
- `phi_e_proxy` (`jax.Array`): extracellular-potential proxy, shape `[T, n_contacts]`.
- `dz` (`jax.Array | float`): contact spacing in the same relative-depth units
  as the contact axis.

**Returns:** CSD proxy, shape `[T, n_contacts]`. Returns zeros when `n_contacts < 3`.

**Description:**
`csd_proxy[c] = -(phi[c+1] - 2*phi[c] + phi[c-1]) / dz**2`, edge-padded at the
boundaries. Factored out of `project_laminar_sources` (which still calls this
function internally — confirmed byte-identical via regression test) so CSD
can be recomputed standalone from any `[T, n_contacts]` potential-proxy array
without re-running the full projection. Unlike `cable_filter_sources`, this is
a purely spatial operator, not a frequency-domain one — CSD is the 2nd
spatial derivative of whatever LFP-proxy it is given, nothing more.

**Example:**
```python
fo = jtfne.project_laminar_sources(sources, positions, n_contacts=16)
dz = fo.contact_depths[1] - fo.contact_depths[0]
csd = jtfne.csd_tensor(fo.phi_e_proxy, dz)  # == fo.csd_proxy
```

---

### `jaxfne.fields.experimental_poisson_1d(sources, conductivity, dx, boundary="mean_zero_neumann", gauge="mean_zero") -> (phi, residual, manifest)`

An actual 1D Poisson PDE solve — distinct from the proxy operators above and
from the fenced multi-dimensional placeholder below.

**Where this sits in the field-solver layer:**
- **Proxy layer** (`project_laminar_sources`, `csd_tensor`, etc., above): Gaussian-kernel
  / finite-difference approximations, no PDE assembled or solved. `field_solver_status`
  stays `"linear_solver"`.
- **`experimental_poisson_1d` (this function):** assembles and solves a real
  1D linear system `d/dx (conductivity * d/dx phi) = -sources` via
  `jnp.linalg.lstsq`, declaring `field_solver_status="experimental_pde_solver"`
  in its returned manifest. It is a minimal, single-dimension PDE solve — not
  the calibrated multi-dimensional volume-conductor solver the package's
  longer-term scope describes.
- **`solve_volume_conductor_experimental` / `PhysicalFieldSolverSpec`**
  (`jaxfne/experimental_hpc/`): a loud-fail skeleton for that
  multi-dimensional solver. Construction is inert; `.validate()` /
  `solve_physical_field()` raise `NotImplementedError` pending
  boundary/gauge/calibration validation (see
  [Tensor Electromagnetics Scope](../tensor_electromagnetics_scope.md)).

**Parameters:**
- `sources` (`jax.Array`): 1D array of current/charge sources.
- `conductivity` (float or `jax.Array`): conductivity scalar (uniform medium,
  original behavior, unchanged), or a per-face array of shape `(N-1,)` giving
  the conductivity between each pair of adjacent grid nodes (a piecewise-
  constant "layered" medium, e.g. distinct cortical-layer conductivities —
  added 2026-07-18 toward `plans.json`'s
  `novelty::tfne-differentiable-field-solver`). The layered case discretizes
  the variable-coefficient flux divergence at cell faces, the standard
  finite-difference treatment for `d/dx(sigma(x) dphi/dx)`. Passing a scalar
  is bit-identical to the pre-2026-07-18 implementation (verified in
  `tests/test_experimental_poisson_1d_layered.py`).
- `dx` (float): grid spacing.
- `boundary` (str, default `"mean_zero_neumann"`): boundary condition declaration.
- `gauge` (str, default `"mean_zero"`): gauge choice; the returned `phi` is the
  minimum-norm least-squares solution, which satisfies the mean-zero gauge.

**Analytic validation reference:** the layered case's ground truth is the
classical "two half-space" point-source volume-conductor result (reflection
coefficient `k=(sigma1-sigma2)/(sigma1+sigma2)`, derived from continuity of
potential and normal current density) — see
`tests/test_experimental_poisson_1d_layered.py`'s module docstring for the
full derivation and 3 limiting-case sanity checks (uniform medium, insulating
boundary, grounded boundary) before it's used to validate any solver output.

**Returns:** `(phi, residual, manifest)` —
- `phi` (`jax.Array`): solved potential array.
- `residual` (`jax.Array`): `A @ phi - b` residual of the assembled linear system.
- `manifest` (`dict`): `claim_level="computational_scaffold"`,
  `field_solver_status="experimental_pde_solver"`, `boundary_condition`,
  `gauge_choice`, `residual_norm`, `convergence_status`
  (`"converged"` if `residual_norm < 1e-3` else `"failed"`),
  `physical_amplitude_calibrated=False`.

**Example:**
```python
import jax.numpy as jnp
# Pure-Neumann Poisson is only solvable for zero-net-flux sources;
# a single nonzero entry violates that compatibility condition.
sources = jnp.zeros(32).at[8].set(1.0).at[24].set(-1.0)
phi, residual, manifest = jtfne.fields.experimental_poisson_1d(sources, conductivity=1.0, dx=0.1)
assert manifest["convergence_status"] == "converged"
```

---

## Boundary Conditions & Constraints

### Mean-Zero Constraint

The default boundary condition enforces zero-mean field solutions:

```
∫ φ(x) dx = 0  (for LFP/CSD)
```

This prevents unrealistic DC offsets and ensures conservation of charge.

### Neumann Boundary Condition

Open boundary (zero normal flux at edges):

```
dφ/dn = 0  (at domain boundary)
```

Suitable for isolated laminar columns away from edge effects.

### Dirichlet Boundary Condition

Fixed potential at boundaries (less common in tutorial simulations).

---

## Validation & Diagnostics

### `validate_source_field_status(field_output) -> dict`

Validate field output for numerical consistency.

**Parameters:**
- `field_output` (FieldOutput): Computed field

**Returns:** Dictionary of validation results

**Checks:**
- Finite values (no NaN/Inf)
- Mean-zero properties
- Conservation properties (if applicable)

**Example:**
```python
status = jtfne.validate_source_field_status(field)
assert status["all_finite"]
```

### `validate_projection_invariants(*, sources, positions, kernel, source_proxy, phi_e_proxy, csd_proxy, lfp_proxy, mode="row_normalize") -> dict`

Check structural invariants of the laminar proxy projection (kernel
row-normalization, finiteness, shape consistency). See
[Validation API](validation.md#validate_projection_invariants-sources-positions-kernel-source_proxy-phi_e_proxy-csd_proxy-lfp_proxy-moderow_normalize-dict)
for the full parameter list — all keyword-only, returns a `dict` of
pass/fail diagnostics, not a `bool`.

**Example:**
```python
report = jtfne.validate_projection_invariants(
    sources=sources, positions=positions, kernel=kernel,
    source_proxy=source_proxy, phi_e_proxy=phi_e_proxy,
    csd_proxy=csd_proxy, lfp_proxy=lfp_proxy,
)
```

### `compute_conservation_proxy_diagnostics(sources, field) -> dict`

Compute conservation-inspired diagnostic metrics.

**Parameters:**
- `sources` (jax.Array): Source signals [time, locations]
- `field` (FieldOutput): Field output

**Returns:** Dictionary of diagnostic metrics

**Metrics:**
- `"total_source_power"`: Sum of |source|²
- `"field_energy"`: Sum of |LFP|² + |CSD|²
- `"energy_ratio"`: Field energy / Source energy
- `"source_moments"`: Spatial center of mass over time

**Example:**
```python
diag = jtfne.compute_conservation_proxy_diagnostics(sources, field)
print(f"Energy ratio: {diag['energy_ratio']:.3f}")
```

---

## Mathematical Form

$$\phi_{\mathrm{proxy}}(t,c) = \sum_{n=1}^{N} W_{cn} S_n(t)$$

**Default (`mode="density_preserving"`):** $W_{cn} = \exp(-\|z_c - z_n\|^2 / 2\sigma^2)$ — preserves source density.

**Optional (`mode="row_normalize"`):** row-stochastic $W$ with $\sum_n W_{cn} = 1$ for all contacts $c$.

$$\mathrm{CSD}_{\mathrm{proxy}}(t,c) = \frac{\phi_{\mathrm{proxy}}(t,c+1) - 2\phi_{\mathrm{proxy}}(t,c) + \phi_{\mathrm{proxy}}(t,c-1)}{(\Delta z)^2}$$

---

## Proxy Scope & Limitations

⚠️ **All field computations are proxy approximations:**

- **Not a full PDE solve:** No coupled Maxwell/Poisson equations
- **No 3D conductivity:** Anatomy is declared, not solved
- **No extracellular detail:** Simplified layer-to-layer projections
- **Proxy LFP/CSD:** Spatial convolution, not source localization
- **Sign convention:** Positive CSD = extracellular source (inward current)
- **Field model status:** `"proxy_readout"` — not physical amplitude

**Use for:**
- Tutorial visualization
- Relative amplitude comparison
- Spatial pattern exploration
- Prototype validation

**Not suitable for:**
- Quantitative comparison with real recordings
- Source localization statements
- Biophysical parameter fitting

---

## JSON Serialization

All field outputs must be JSON-safe with `allow_nan=False`:

```python
import json
from jaxfne.io import json_safe

json.dumps(json_safe(field_output.diagnostics), allow_nan=False)
```

Outputs with NaN or Inf values will fail serialization and must be diagnosed.

## See also

- [Probe Operators](probes.md) — Readout operators using field outputs
- [Computation Basis](../computation_basis.md) — TFNE architecture overview
- [API reference](index.md)
