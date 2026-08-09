# Solver acceptance

This page describes the acceptance checklist a field solver must satisfy before
it can join jaxfne's field-solver surface. It applies to the current laminar
proxy projection and to any future solver added to `jaxfne/fields/`.

> **Scope:** Every jaxfne field output is a **Relative** proxy readout inside a
> computational scaffold. Passing the checklist documents that a solver behaves
> correctly as a *computational* operator; it does **not** elevate the output
> beyond the proxy readout boundary.

## Why the checklist is needed

Field solver functions in jaxfne produce arrays plus a metadata contract (see
[Field Schema](../api/field_schema.md)). A solver is usable only if its array
outputs are well-behaved under the operations jaxfne relies on and if the
metadata correctly states what the arrays mean. The checklist below is the
machine-checkable definition of "well-behaved" for a solver's outputs.

## Acceptance checklist

A solver passes acceptance when all five conditions hold, verified by
behavioral tests (not by inspecting source):

1. **Finite output for finite input.** Any finite `sources` array and finite
   `positions` produce finite field output arrays (no NaN, no Inf).
2. **Linear superposition.** The projection operator is additive: projecting
   `a + b` equals the sum of projecting `a` and projecting `b`, component for
   component, within a tolerance appropriate to the array dtype.
3. **JAX execution.** The solver must run inside `jax.jit` and produce finite
   arrays from valid array inputs. This is tested by executing it, not by
   looking for a decorator.
4. **Field claim metadata.** The output exposes `field_claim_level` on its
   metadata surface.
5. **Amplitude truth gate.** The output reports
   `physical_amplitude_calibrated = False`.

A zero-source null control accompanies the checklist: projecting a zero source
yields exact zero proxy outputs (with correct shapes) under the solver's dtype.

## Proxy-readout boundary

All field output in jaxfne sits under the same truth boundary:

- `field_claim_level = "proxy_readout"` — the arrays are structural proxy
  readouts, not measurements.
- `field_solver_status` distinguishes the implementation (`linear_solver`,
  `experimental_pde_solver`); it does not change the claim level.
- `physical_amplitude_calibrated = False` on every solver path.

A solver that passes the checklist is still bound by these three values. The
checklist never grants a physical-amplitude interpretation.

## Metadata contract for new solvers

A new solver function must expose exactly these keys (value vocabulary per
solver type):

| Key | Value | Meaning |
|-----|-------|---------|
| `field_claim_level` | `"proxy_readout"` | output interpretation boundary |
| `field_solver_status` | `"linear_solver"` or `"experimental_pde_solver"` | solver implementation category |
| `physical_amplitude_calibrated` | `False` | amplitude status |

The keys live on the solver's output surface (a `FieldOutput.diagnostics` dict
or a returned `manifest`, depending on the solver's return convention). Keep
each solver's actual surface; do not stage a unified container solely for
metadata.

## Where regression tests live

Solver acceptance tests go in `tests/test_phaseF_solver_acceptance.py` (one
test per checklist item plus the zero-source control). Schema/claim-level
regressions live in `tests/test_phaseE_field_schema.py`; field admissibility
reports are covered by `tests/test_field_admissibility_v020.py` and
`tests/test_field_proxy_admissibility_v024.py`.

## Test tolerance note

JAX CPU arrays default to `float32`. Kernel-matrix products carry rounding on
the order of `1e-7`, while a second-derivative stencil can amplify rounding to
roughly `1e-5`. Acceptance comparisons use `rtol`/`atol` of `1e-4` so the
superposition and zero-source checks are stable across the supported dtypes.

## Summary

The acceptance checklist guards that a solver is computationally sound
(finite, additive, JAX-executable) and honestly labeled (`proxy_readout`,
uncalibrated). Passing it is a precondition for a solver to be part of the
public field surface — and nothing more.