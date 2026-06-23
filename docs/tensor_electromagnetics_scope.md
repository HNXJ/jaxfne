# Tensor Electromagnetics: Reserved Capability Stages

**Status:** scope statement, not a release plan. Every stage past P0 is
**deferred and not promised** — recorded here so the boundary between "shipped"
and "reserved" stays explicit, matching the tone of
[Tensor-Network Ancestry, Part 6](tensor_network_ancestry.md#part-6-reserved-optional-path-not-implemented).

## Why this page exists

[Limitations and future plans](limitations_and_future_plans.md) already
declares the field-computation regimes in one table (laminar proxy / shipped,
conservation diagnostics / partial, elliptic field solver / reserved, full
electrodynamic solver / reserved). This page expands the two reserved rows into
named sub-stages so a reader can see what "reserved" would require, stage by
stage, without implying any of it is scheduled.

---

## Scope ladder

| Stage | Description | Status |
|---|---|---|
| **P0** | Proxy projection — Gaussian-leadfield source-to-field projection, finite-difference CSD | shipped |
| **P1** | Declared geometry — boundary/gauge/conductivity fields carried as metadata, not solved | shipped (`physical_field_solver_v040.PhysicalFieldSolverSpec`) |
| **P2** | Boundary-aware solve — an elliptic field solve that honors a declared boundary condition and gauge | reserved |
| **P3** | Discrete volume solve — a mesh/volume-conductor solve over declared geometry | reserved |
| **P4** | Differentiable adjoint — a solve path compatible with gradient-based tuning | reserved |
| **P5** | External validation — comparison against reference physical measurements | reserved |

P0 and P1 ship today. P2 through P5 are not implemented, not scheduled, and not
promised; they are recorded so the boundary between proxy and solved field
computation is visible rather than implicit.

---

## What each reserved stage would require

### P2 — Boundary-aware solve

[Elliptic Field Equation Specification](guides/poisson_admissibility.md) already
specifies the five admissibility gates a solve at this stage would need to pass
(conductivity symmetric positive definite, source conservation, gauge condition,
field-array finiteness, solver convergence) before any output could carry
`field_solver_status` other than `"linear_solver"`.

### P3 — Discrete volume solve

Requires a declared mesh/volume geometry beyond the `geometry` field already
present in `PhysicalFieldSolverSpec` (`jaxfne/experimental_hpc/physical_field_solver_v040.py`),
plus a discretization scheme. `solve_physical_field(spec, sources)` is the
reserved entry point; it raises `NotImplementedError(">TBI-not-ready: v0.4 physical
field solver")` by design, so the proxy-to-solved transition cannot happen
silently.

### P4 — Differentiable adjoint

Any solve must compose with `jax.grad` for `Model.tune()` to use it. The hard
spike-reset non-differentiability gate (`gradient_path_safe()`) that already
guards the emitter stage would extend to this stage.

### P5 — External validation

[Conservation-Inspired Proxy Diagnostics](conservation_proxy_diagnostics.md)
already reserves the relevant report fields for this stage —
`poisson_solver_status`, `maxwell_solver_status`, `stress_energy_tensor_status`,
`j_dot_e_proxy`, `poynting_flux_proxy` — all held at `"not_implemented"` or
`null` until a solved field exists to validate against reference measurements.

---

## What does not change

Reaching any reserved stage above does not change `claim_level`,
`physical_amplitude_calibrated`, or any other truth gate by itself — those
remain conservative defaults until separate calibration evidence exists, per
[Limitations and future plans](limitations_and_future_plans.md).

---

## See Also

- [Limitations and future plans](limitations_and_future_plans.md) — the regime table this page expands
- [Conservation-Inspired Proxy Diagnostics](conservation_proxy_diagnostics.md) — reserved report fields
- [Elliptic Field Equation Specification](guides/poisson_admissibility.md) — admissibility gates for P2
- [TFNE Operator Doctrine](operator_doctrine.md) — the solver operator class this page details
- [Tensor-Network Ancestry](tensor_network_ancestry.md) — the reserved-path tone this page follows
