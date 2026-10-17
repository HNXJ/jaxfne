# Public API

This page records the **public export contract** of jaxfne 0.5.6: 208 root symbols in
`jaxfne.__all__`, with tier membership defined in `jaxfne/public_surface.py`.

Machine-readable receipt: `artifacts/assurance/api_contracts/public_surface_contract_v0413.json`.

## Tiers

| Tier | Meaning |
|------|---------|
| **CANONICAL** | Root `__all__` — supported TFNE grammar |
| **COMPATIBILITY** | Root `__all__` — deprecated aliases |
| **ADVANCED** | Importable from submodules; not root `__all__` |
| **EXPERIMENTAL_INTERNAL** | Not part of the public contract |

## HDP / H-state

**HDP** names the adaptive-dynamics family. **H-state** is its finite-dimensional
latent representation over biophysical parameter coordinates. H-state ⊂ HDP
formulation; H-state ≠ HDP.

`hdp_params` is a **compatibility transport dict**, grouped semantically:

### H-state

- `h_state_dim`, `h_state_locality` (`node` | `population`)
- `h_state_readout`, `h_state_coupling`

### H-dynamics

- `hdp_rule` (node plasticity families: `signed_linear`, `signed_quadratic`, `hebbian_product`)
- income/spending/barrier coefficients, `K_HDP`, `K_ctrl`, `tau_0_ms`, bounds, diagnostics flags

### Theta-adaptation (adaptive parameter dynamics)

- `controller_*` coefficients, channel masks, bounds
- \(\Theta\) (adaptive coordinates, synaptic and intrinsic) is distinct from synaptic weight storage \(W\)

**Internal dispatch identifiers** (e.g. population restoring controller rule ids)
are not public vocabulary. Population-H semantics are expressed via
`h_state_locality="population"` and the theta-adaptation coefficient group.

Validation: `jaxfne.validate_runtime_config(cfg)` and
`jaxfne.public_surface.validate_hdp_params_semantics(hdp_params)`.
