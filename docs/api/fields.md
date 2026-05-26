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

Defines the spatial anatomy of neural populations in laminar columns.

### Attributes

- `x` (jax.Array): X-coordinates (depth dimension) of source locations
- `y` (jax.Array): Y-coordinates (lateral) of source locations
- `z` (jax.Array): Z-coordinates (anterior-posterior) of source locations
- `contact_locations` (jax.Array, optional): Electrode contact positions
- `n_sources` (int): Number of distinct sources
- `n_contacts` (int): Number of recording contacts

### Methods

#### `from_dict(geometry_dict: dict) -> LaminarSourceGeometry`

Create geometry from configuration dictionary.

**Parameters:**
- `geometry_dict` (dict): Dictionary with 'x', 'y', 'z' keys

**Returns:** `LaminarSourceGeometry`

**Example:**
```python
geom = jtfne.LaminarSourceGeometry.from_dict({
    "x": [0, 200, 400],  # depths in µm
    "y": [0, 0, 0],
    "z": [0, 0, 0]
})
```

---

### `project_laminar_sources(currents, geometry) -> source_signals`

Project neural currents into laminar space.

**Parameters:**
- `currents` (jax.Array): Emitter transmembrane currents [time, neurons]
- `geometry` (LaminarSourceGeometry): Spatial geometry

**Returns:** Source signals [time, locations]

**Description:**
Transforms point-neuron currents into distributed source density using anatomical position mapping. Current values are assigned to nearest spatial locations; not a full dipole solve.

**Example:**
```python
source = jtfne.project_laminar_sources(I_mem, geometry)
```

---

### `project_sources_to_laminar_field(sources, geometry, ...) -> field_output`

Project laminar sources to field readouts (LFP, CSD).

**Parameters:**
- `sources` (jax.Array): Source density [time, locations]
- `geometry` (LaminarSourceGeometry): Spatial geometry
- `conductivity_mode` (str, optional): Field approximation mode

**Returns:** `FieldOutput` containing LFP and CSD arrays

**Modes:**
- `"proxy_convolution"`: Spatial convolution approximation (default)
- `"mean_zero_projection"`: Mean-zero constraint

**Example:**
```python
field = jtfne.project_sources_to_laminar_field(
    sources, geometry, conductivity_mode="proxy_convolution"
)
```

---

## FieldOutput

```python
jaxfne.FieldOutput
```

Container for computed field quantities (LFP, CSD, etc.).

### Attributes

- `phi_e_proxy` (jax.Array): Extracellular potential proxy [time, contacts]
- `csd_proxy` (jax.Array): Current source density proxy [time, contacts]
- `lfp_proxy` (jax.Array): Local field potential proxy [time, contacts]
- `source` (jax.Array): Source density [time, locations]
- `diagnostics` (dict): Field solution metadata and validation flags

### Diagnostics Dictionary

All `diagnostics` must include 18 required fields:

| Field | Type | Purpose |
|-------|------|---------|
| `field_solver_status` | str | Solver type: `"laminar_proxy_no_pde"` |
| `solver_name` | str | Human-readable ID: `"laminar_proxy"` |
| `boundary_condition` | str | BC declaration; `"declared_metadata_only"` for proxy |
| `gauge` | str | Gauge convention; `"declared_metadata_only"` for proxy |
| `csd_sign_convention` | str | **Canonical:** `"positive_equals_extracellular_source"` |
| `current_density_layout` | str | J_e status: `"not_applicable"` for proxy |
| `solver_residual_l2_relative` | float \| None | PDE residual; `null` for proxy |
| `n_iterations` | int \| None | Solver iteration count; `null` for proxy |
| `converged` | bool \| None | Convergence flag; `null` for proxy |
| `finite_phi_e` | bool | Is potential array finite? |
| `finite_J_e` | bool | Is current density finite? `false` for proxy |
| `finite_CSD` | bool | Is CSD array finite? |
| `field_claim_level` | str | Claim authority: `"proxy_readout_only"` |
| `physical_amplitude_claim_allowed` | bool | Can claim physical units? Always `false` for proxy |
| `source_projection_mode` | str | How sources map to field: `"proxy_no_field_solve"` |
| `source_current_conservation_status` | str | Conservation test: `"not_applicable_proxy_mode"` |
| `source_conservation_tested` | bool | Was conservation validated? `false` for proxy |
| `source_conservation_claim_allowed` | bool | Can claim conserved sources? `false` for proxy |

### Methods

#### `to_dict() -> dict`

Convert field output to JSON-safe dictionary.

**Example:**
```python
field_dict = field.to_dict()
```

---

## Field Solvers

### `probe_laminar_modes(sources, basis_spec) -> modal_amplitudes`

Decompose source activity into spatial basis functions.

**Parameters:**
- `sources` (jax.Array): Source signals [time, locations]
- `basis_spec` (BasisSpec): Basis function specification

**Returns:** Modal amplitudes [time, n_modes]

**Description:**
Projects spatial source patterns onto a reduced set of basis functions (e.g., Chebyshev polynomials or radial basis functions). Enables dimensionality reduction for visualization and analysis.

**Example:**
```python
modes = jtfne.probe_laminar_modes(sources, cfg.basis_spec)
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

### `validate_projection_invariants(sources, field_output) -> bool`

Check that field respects source-field relationships.

**Parameters:**
- `sources` (jax.Array): Original sources
- `field_output` (FieldOutput): Computed field

**Returns:** Boolean (valid=True)

**Invariants:**
- Field sign convention consistency
- No unphysical amplification
- Conservation properties (relative)

**Example:**
```python
is_valid = jtfne.validate_projection_invariants(sources, field)
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

where $W$ is row-normalized ($\sum_n W_{cn} = 1$).

$$\mathrm{CSD}_{\mathrm{proxy}}(t,c) = \frac{\phi_{\mathrm{proxy}}(t,c+1) - 2\phi_{\mathrm{proxy}}(t,c) + \phi_{\mathrm{proxy}}(t,c-1)}{(\Delta z)^2}$$

---

## Proxy Nature & Limitations

⚠️ **All field computations are proxy approximations:**

- **Not a full PDE solve:** No coupled Maxwell/Poisson equations
- **No 3D conductivity:** Anatomy is declared, not solved
- **No extracellular detail:** Simplified layer-to-layer projections
- **Proxy LFP/CSD:** Spatial convolution, not source localization
- **Sign convention:** Positive CSD = extracellular source (inward current)
- **Field claim level:** `"proxy_readout_only"` — not physical amplitude

**Use for:**
- Tutorial visualization
- Relative amplitude comparison
- Spatial pattern exploration
- Prototype validation

**Not suitable for:**
- Quantitative comparison with real recordings
- Source localization claims
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
