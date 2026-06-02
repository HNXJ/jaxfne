# JAXFNE Agent Contract

## Identity

`jaxfne` is a compact JAX-native computational scaffold for Tensor-Field Neural Equation workflows.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

The 0.3.28+ package architecture is:

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

## Required posture

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Stronger status needs run-specific geometry, units, calibration, boundary, gauge, solver, convergence, residual, and validation evidence.

## Package use

- Canonical import: `import jaxfne as jtfne`.
- Use package-native flow: configure -> construct -> simulate -> visualize -> optimize.
- Do not add local notebook simulators, source operators, objective engines, or field solvers when package APIs exist.
- Placeholder future callables must raise `NotImplementedError(">TBI-not-ready")`.
- Preserve public wrappers when moving APIs.

## Current canonical names

```text
Config, Net, Paradigm, Objective, Trainer, Signals, FlatNet
```

Deprecated aliases for one release line:

```text
Configuration, Model, FlatModel
```

## Worker report format

```text
repo / branch / SHA
changed files
commands run
exact results
runtime facts
status/evidence level
blockers
next safe action
```

Treat worker test counts as unverified until exact commands, branch/SHA, and receipts are shown.

## Stop conditions

Stop and report when any appear:

```text
invented public API
hidden local scientific engine
NaN/Inf export
proxy path described as solved field
uncalibrated source described as physical amplitude
silent placeholder success
test changed before failure provenance is known
non-JSON-safe ndarray stored directly in Config
JIT path includes plotting, JSON, pandas, or file I/O
release tag/upload has stale pyproject version
```
