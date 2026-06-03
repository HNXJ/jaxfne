# CLAUDE.md - JAXFNE Agent Contract

## Identity

`jaxfne` is a compact JAX-native computational scaffold for Tensor-Field Neural Equations (TFNE). It supports source/field/probe modeling, tutorial evidence generation, validation reports, and optimizer workflows.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Required posture

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Stronger status needs run-specific geometry, units, calibration, boundary, gauge, solver, convergence, and validation evidence.

## Package use

- Canonical import: `import jaxfne as jtfne`.
- Use package-native flow: configure -> construct -> simulate -> visualize -> optimize.
- JAX is the numerical core. Jaxley is optional emitter-model infrastructure. Prefer bridges over reimplementation when available.
- Do not add local notebook simulators, source operators, objective engines, or field solvers when package APIs exist.
- Placeholder future callables must fail loudly with explicit errors.

## Suite and Etude rules

- Suite: structured release/tutorial unit.
- Etude: full-detail hardcore workflow with explicit config, diagnostics, artifacts, and execution receipts.
- Release notebooks need deterministic seed, `dt_ms=0.1`, `dtype=float32`, full-mode `duration_ms>=1000`, finite outputs, strict JSON, PNG figures, proxy-safe titles, and SMOKE/FULL execution receipts.

## Worker report format

Include:

```text
repo / branch / SHA
changed files
commands run
exact results
runtime facts
truth/evidence status
blockers
next safe action
```

Treat worker test counts as unverified until exact commands, branch/SHA, and receipts are shown.

## Stop conditions

Stop for THETA audit when any appear:

```text
invented public API
hidden local scientific engine
NaN/Inf export
proxy path described as solved field
uncalibrated source described as physical amplitude
silent placeholder success
test changed before failure provenance is known
```