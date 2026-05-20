# Fields API

## FieldOutput

The `FieldOutput` container holds computed field quantities and diagnostic metadata.

### Contract

```python
@dataclass
class FieldOutput:
    phi_e_proxy: jax.Array  # Extracellular potential proxy [time, contacts]
    csd_proxy: jax.Array    # CSD proxy [time, contacts]
    lfp_proxy: jax.Array    # LFP proxy [time, contacts]
    diagnostics: dict       # Field solution metadata and validation flags
```

### Field solution metadata (18 required fields)

All `diagnostics` must include:

| Field | Type | Purpose |
|-------|------|---------|
| `field_solver_status` | str | Solver type: `"laminar_proxy_no_pde"` (v0.2.13) or future physical solvers |
| `solver_name` | str | Human-readable ID: `"laminar_proxy"` |
| `boundary_condition` | str | BC declaration; `"declared_metadata_only"` for proxy |
| `gauge` | str | Gauge convention; `"declared_metadata_only"` for proxy |
| `csd_sign_convention` | str | **Canonical:** `"positive_equals_extracellular_source"` (no `_like` suffix) |
| `current_density_layout` | str | J_e status: `"not_applicable"` for proxy |
| `solver_residual_l2_relative` | float \| None | PDE residual; `null` for proxy |
| `n_iterations` | int \| None | Solver iteration count; `null` for proxy |
| `converged` | bool \| None | Convergence flag; `null` for proxy |
| `finite_phi_e` | bool | Is potential array finite? |
| `finite_J_e` | bool | Is current density finite? `false` for proxy |
| `finite_CSD` | bool | Is CSD array finite? |
| `field_claim_level` | str | Claim authority: `"proxy_readout_only"` (v0.2.13) |
| `physical_amplitude_claim_allowed` | bool | Can claim physical units? Always `false` for proxy |
| `source_projection_mode` | str | How sources map to field: `"proxy_no_field_solve"` |
| `source_current_conservation_status` | str | Conservation test: `"not_applicable_proxy_mode"` for proxy |
| `source_conservation_tested` | bool | Was conservation validated? `false` for proxy |
| `source_conservation_claim_allowed` | bool | Can claim conserved sources? `false` for proxy |

### Mathematical form

$$\phi_{\mathrm{proxy}}(t,c) = \sum_{n=1}^{N} W_{cn} S_n(t)$$

where $W$ is row-normalized ($\sum_n W_{cn} = 1$).

$$\mathrm{CSD}_{\mathrm{proxy}}(t,c) = \frac{\phi_{\mathrm{proxy}}(t,c+1) - 2\phi_{\mathrm{proxy}}(t,c) + \phi_{\mathrm{proxy}}(t,c-1)}{(\Delta z)^2}$$

### Status in v0.2.13+

- Proxy fields: `field_solver_status = "laminar_proxy_no_pde"`, all solver metrics `null`
- Physical fields (future): Will have non-null metrics and higher claim levels
- See [Field solution metadata skill](../skills/skill_field_solution_metadata.md) for full validation guide

## JSON serialization

All field outputs must be JSON-safe with `allow_nan=False`:

```python
import json
from jaxfne.io import json_safe

json.dumps(json_safe(field_output.diagnostics), allow_nan=False)
```

Outputs with NaN or Inf values will fail serialization and must be diagnosed.

## See also

- [Field solution metadata skill](../skills/skill_field_solution_metadata.md) — Contract validation and common mistakes
- [Tensor-field workflows](../tensor_field_workflows.md) — Pipeline overview
- [API reference](index.md)
