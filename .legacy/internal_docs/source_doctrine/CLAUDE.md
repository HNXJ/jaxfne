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

## ⚡ Fast-path doctrine (intelligence-per-token · per-watt) — all agents

Fast executive models: act on this block; read the rest only when blocked.
1. **Verify-before-call.** Never name a skill/API/flag/path/branch from memory — grep `__all__` / the contract first. Fabrication→fail→retry is the costliest loop.
2. **Contract-first.** Read the frozen contract (docstrings + `>TBI-not-ready`), not the 1000-line module.
3. **Skill = cache.** Run the skill; don't re-reason its checklist.
4. **Receipt, not justification.** Show `command` + `N passed`. Never report what you didn't run.
5. **Route by altitude.** Opus=architecture/contracts/truth-gates · Sonnet=implement vs contract+skill · Gemini=cross-file synthesis + batch edits.

JAX/jaxfne: `N_compile<=1` (recompilation is the silent watt+token sink) · stable shapes+dtypes · jit/vmap on (`runtime_report()`) · x64 at startup only · sparse>dense · `segment_sum`+`lax.scan` · pytree children=dynamic, aux=static. Depth → jaxfne skills + `~/.claude/CLAUDE.md`.
