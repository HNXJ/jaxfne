---
name: jaxfne-worker-context-router
description: >-
  Route a jaxfne task to the owning module, current public API, and relevant
  validation lane. Use before editing, refactoring, testing, documenting, or
  releasing jaxfne.
---

# jaxfne worker context router

Read `AGENTS.md` and `catalog-glossary-jaxfne` first. This skill routes work; it
does not define TFNE mathematics.

## Canonical software path

```text
CircuitSpec -> construct -> Model -> simulate -> Signals
```

`CircuitSpec` includes `Configuration` and `NeuronalTensor`. Compatibility
aliases such as `Config`/`Configuration` and `Net`/`Model` are API conveniences,
not a scientific grammar. Do not invent `FlatNet`, `FlatModel`, `weld`, or
typed `Configuration.circuit`/`.objective` fields.

## Owning modules

- configuration builders: `jaxfne/_config.py`, `jaxfne/builders.py`
- model lifecycle: `jaxfne/_model*.py`
- dispatch and construction: `jaxfne/_construct*.py`
- runtime policy: `jaxfne/_runtime_config.py`
- signals, simulation, objectives: `jaxfne/_signals.py`
- tensor bridge: `jaxfne/neuronal_tensor.py`
- emitters and kernels: `jaxfne/emitters.py`
- connectivity: `jaxfne/connectivity.py`
- source, field, and probes: `jaxfne/fields/`
- optimization: `jaxfne/optim/`
- visualization: `jaxfne/vis/`
- manifests and validation: `jaxfne/io.py`, `jaxfne/validation.py`
- notebook glue: `jaxfne/tutorial_utils.py`, `scripts/`

`jaxfne/core.py` is a public re-export facade. Verify the defining module
before editing a symbol.

## Routing

- `Configuration` or laminar builder → `jaxfne-config`.
- `NeuronalTensor`, `RuntimeConfiguration`, or HDP dispatch → `jaxfne-neural-tensor`.
- `construct`, `simulate`, `Signals`, probe, objective, or tune → `jaxfne-neural-network`.
- selectors, serialized connection rules, or schema → `jaxfne-modeling-optimization-schema`.
- paradigms and event schedules → `jaxfne-paradigm-design`.
- figures/readouts → `jaxfne-vis-modules` and, for depth-frequency suites,
  `jaxfne-spectrolaminar-suite`.
- notebooks/docs/artifacts → `jaxfne-notebook-release-gate`.
- remote mutation or publication → `jaxfne-release-mutation-guard`.
- content identity → `jaxfne-sha256-artifact-integrity`.
- implementation safeguards → `jaxfne-harden`.

## Preflight

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git ls-remote origin refs/heads/main refs/heads/dev
python3 scripts/repo_state_snapshot.py
```

Verify unfamiliar names with the public import and current tests. Do not use
remembered line numbers, branch state, or test counts.

## Boundaries

- Keep package APIs ahead of notebook-local engines.
- Keep optional dependencies lazy.
- Keep plotting and serialization out of numerical kernels.
- Preserve the current proxy/scaffold evidence boundary.
- Do not change scientific behavior under a context/governance-only task.

## Worker report

Report exact repository state, changed files, commands and results, evidence
status, unresolved risks, and one next safe action. For implementation work,
also report:

```text
API delta:
Mathematical delta:
Numerical delta:
Claim/evidence delta:
Documentation delta:
Compatibility delta:
```
