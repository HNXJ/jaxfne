# Conservation-Inspired Proxy Diagnostics

## Purpose

This document describes the v0.2.27 conservation-inspired proxy diagnostics in TFNE
(`compute_conservation_proxy_diagnostics`). These diagnostics operate over the existing
laminar-proxy field outputs — `source_proxy`, `phi_e_proxy`, `csd_proxy`, `lfp_proxy` —
and produce JSON-safe scalar summaries for source/field validation.

**Run boundary (v0.2.27):**
- Proxy diagnostics only — proxy-based computation over existing proxy arrays. Calibration and the reserved physical-field regimes are catalogued in [Limitations and future plans](limitations_and_future_plans.md).
- All values are derived from existing proxy arrays; nothing is fabricated.
- `amplitude_status: false` (immutable).
- `metabolism_status: false` (immutable).
- `j_dot_e_proxy: null` — J_e is not computed in `laminar_proxy_no_pde` mode.
- `poynting_flux_proxy: null` — not implemented in v0.2.x.
- `poisson_solver_status: not_implemented`.
- `maxwell_solver_status: not_implemented`.
- `stress_energy_tensor_status: not_implemented`.

---

## Mathematical Basis

### Source Magnitude Proxy

$$\|q\|_1 = \frac{1}{T \cdot N} \sum_{t,n} |q_{tn}|$$

$$\|q\|_2 = \sqrt{\frac{1}{T \cdot N} \sum_{t,n} q_{tn}^2}$$

**Worded equation:**
`source_norm_l1` = mean absolute value of the declared source proxy array.
`source_norm_l2` = root-mean-square of the declared source proxy array.

**Run boundary:** These are scalar summaries of the proxy source. No physical current density
calibration. The source array is `uncalibrated_izhikevich_native_current` unless separately
declared otherwise.

### Source Conservation Proxy Residual

$$\mathrm{residual}(t) = \left|\frac{1}{N}\sum_n q_{tn}\right|$$

$$\mathrm{source\_conservation\_proxy\_residual} = \frac{1}{T}\sum_t \mathrm{residual}(t)$$

**Worded equation:**
At each timestep, compute the spatial mean of the source. Average the absolute value of this
spatial mean over time. A value near zero suggests the source is approximately spatially balanced.

**Run boundary:** This is a spatial-mean proxy for ∫q dV ≈ 0, not a PDE-enforced conservation
check. The true conservation integral requires a solved field with boundary conditions.

### Potential-Field Gradient Proxy

$$\|\nabla \phi_e\|^2_\mathrm{proxy} = \frac{1}{T \cdot X} \sum_{t,x} \left(\frac{\partial \phi_e}{\partial x}\right)^2$$

**Worded equation:**
`phi_gradient_proxy_norm2` = mean squared spatial variation of the extracellular potential
proxy, computed via finite differences along the laminar depth axis.

**Run boundary:** This is a physical-style field gradient computation on the proxy potential. The potential `phi_e_proxy` is a
laminar row-normalized projection. The gradient magnitude is
a proxy-level diagnostic only.

### Field-Energy Proxy

$$E_\mathrm{proxy} = \|\nabla \phi_e\|^2_\mathrm{proxy}$$

**Worded equation:**
`field_energy_like_proxy` = same as `phi_gradient_proxy_norm2`. It is named separately to
distinguish it as a proxy analog of field energy density; it is not a physical energy.

**Run boundary:** No calibrated conductivity. This is not ½ σ |∇φ_e|² (physical field energy).

### J·E Proxy (Not Computed)

$$\mathbf{J} \cdot \mathbf{E} \approx -\sigma |\nabla \phi_e|^2$$

This product is the local Ohmic power density. In TFNE v0.2.27:

- J_e is not computed in `laminar_proxy_no_pde` mode.
- `j_dot_e_proxy` is always `null`.
- Computing J·E as a physical power density would require a solved field and calibrated σ.

**Reserved regime:** J·E becomes available once a physical-field solver produces current-density arrays. Proxy mode uses the voltage array and reports `null`; see [Limitations and future plans](limitations_and_future_plans.md).

### Poynting Theorem (reserved regime)

$$\frac{\partial u_{em}}{\partial t} + \nabla \cdot \mathbf{S} + \mathbf{J} \cdot \mathbf{E} = 0$$

where $\mathbf{S} = \mathbf{E} \times \mathbf{H}$ is the Poynting vector.

**Run boundary:**
Proxy mode reports `poynting_flux_proxy: null`. Poynting flux, stress-energy tensor, and volumetric field dynamics belong to the reserved regimes in [Limitations and future plans](limitations_and_future_plans.md); the equation is recorded here for reference.

---

## Output Contract

`compute_conservation_proxy_diagnostics()` returns a JSON-safe dict with the following fields:

| Field | Type | Description |
|---|---|---|
| `diagnostic_status` | `"proxy"` | Always `"proxy"` — no physical status |
| `diagnostic_version` | `"v0.2.27"` | Version tag |
| `model_status` | `"computational_scaffold"` | Always scaffold |
| `field_solver_status` | str | Passed through; `"laminar_proxy_no_pde"` by default |
| `field_model_status` | str | Passed through; `"proxy_readout_only"` by default |
| `source_calibration_status` | str | Passed through from run metadata |
| `amplitude_status` | `false` | Always `false` |
| `metabolism_status` | `false` | Always `false` |
| `source_norm_l1` | float or null | Mean absolute source proxy |
| `source_norm_l2` | float or null | RMS source proxy |
| `source_abs_mean` | float or null | Mean absolute source (same as l1) |
| `source_conservation_proxy_residual` | float or null | Spatial-mean conservation proxy |
| `phi_abs_mean` | float or null | Mean absolute phi_e proxy |
| `phi_gradient_proxy_norm2` | float or null | Mean squared spatial gradient of phi_e proxy |
| `csd_abs_mean` | float or null | Mean absolute CSD proxy |
| `csd_norm_l2` | float or null | RMS CSD proxy |
| `lfp_abs_mean` | float or null | Mean absolute LFP proxy |
| `lfp_norm_l2` | float or null | RMS LFP proxy |
| `field_energy_like_proxy` | float or null | Same as phi_gradient_proxy_norm2 |
| `j_dot_e_proxy` | `null` | Not computed: J_e not available in proxy mode |
| `poynting_flux_proxy` | `null` | Not implemented in v0.2.x |
| `stress_energy_tensor_status` | `"not_implemented"` | Explicitly gated |
| `poisson_solver_status` | `"not_implemented"` | Explicitly gated |
| `maxwell_solver_status` | `"not_implemented"` | Explicitly gated |
| `notes` | list[str] | Human-readable statement scope notes |

---

## API

```python
from jaxfne import compute_conservation_proxy_diagnostics

# Option 1: pass arrays directly
diag = compute_conservation_proxy_diagnostics(
    source=source_proxy_array,   # [T, N] or [T, X]
    phi_e=phi_e_proxy_array,     # [T, X]
    csd=csd_proxy_array,         # [T, X]
    lfp=lfp_proxy_array,         # [T, X]
)

# Option 2: pass FieldOutput object
diag = compute_conservation_proxy_diagnostics(
    field_solution=signals.field,
    source_calibration_status=signals.metadata["source_calibration_status"],
)

# Option 3: no arrays (returns all-None norms)
diag = compute_conservation_proxy_diagnostics()

# All outputs are JSON-safe
import json
json.dumps(diag, allow_nan=False)
```

### Manifest integration

When `Model.manifest(signals)` is called and `signals.field` is not `None`, the manifest
automatically includes a `conservation_proxy_diagnostics` block:

```python
manifest = model.manifest(signals)
cpd = manifest["conservation_proxy_diagnostics"]
print(cpd["source_norm_l1"])         # float
print(cpd["phi_gradient_proxy_norm2"])  # float
print(cpd["poisson_solver_status"])  # "not_implemented"
print(cpd["amplitude_status"])  # False
```

---

## Reserved capabilities

Elliptic (Poisson) and volumetric (Maxwell) field solvers, Poynting-flux and stress-energy computation, calibrated conductivity, and physical-amplitude reporting are reserved regimes. Proxy mode holds `amplitude_status: false` and `metabolism_status: false`. The regime catalogue lives in [Limitations and future plans](limitations_and_future_plans.md).

---

## See Also

- [Computation Basis](computation_basis.md) — Field regime gating doctrine
- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — Source/field equations
- [Source/Field Equations](source_field_equations.md) — Source modes and bookkeeping
- [Probe Operators](probe_operators.md) — Readout operator statement boundaries
- [Elliptic Field Equation Specification](poisson_admissibility.md) — admissibility mathematics
- [Limitations and future plans](limitations_and_future_plans.md) — scope boundaries and reserved regimes
