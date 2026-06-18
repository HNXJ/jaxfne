---
name: jaxfne-worker-context-router
summary: Route any jaxfne repo task to the correct module, current API, branch state, and validation lane before editing.
trigger: Use whenever the task mentions jaxfne, repo, file, module, codebase, refactor, API, branch, SHA, worker, handoff, tests, docs, tutorial, notebook, or release.
---

# jaxfne Worker Context Router

## Purpose

Use this first for most jaxfne work. It prevents the common agent error: editing the wrong layer.

## Canonical package story

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Legacy names remain compatibility aliases for one release line:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Required preflight

```bash
git status --short
git branch --show-current
git rev-parse HEAD
find jaxfne -maxdepth 2 -type f -name '*.py' | sort
```

Before adding or changing an API, inspect existing names:

```bash
grep -R "def <name>\|class <name>" -n jaxfne tests docs examples tutorials scripts
```

## Ownership map

```text
Config/schema/specs          -> jaxfne/config.py or current core compatibility layer
Net/construct/introspection  -> jaxfne/net.py, builders.py, current core wrappers
Paradigm/task/stimulus       -> jaxfne/paradigm.py
Objective/metrics/gates      -> jaxfne/objective.py, objectives.py
Trainer/AGSDR/tuning         -> jaxfne/optim/trainer.py, optim/*
Signals/layout/query         -> jaxfne/signals.py
Source/field/probe/readout   -> jaxfne/fields/*
Visualization                -> jaxfne/vis/* only
Notebook glue                -> jaxfne/tutorial_utils.py or scripts/*
Release/package mutation     -> release skills and scripts only
```

## Routing rules

- Do not add plotting imports to core, config, net, fields, or optim.
- Do not add optimizer/trainer loops to vis.
- Do not add scientific source/field/probe operators to notebooks.
- Do not add notebook-only helper code to stable package APIs unless it has tests.
- Preserve public API with wrappers during 0.3.x migrations.
- Placeholder future APIs must fail loudly with `NotImplementedError(">TBI-not-ready")`.

## Report contract

Every worker report must include:

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

Treat test counts as unverified unless the exact commands and receipts are shown.

## Intelligence-per-token + fleet routing

**Root rule: declare the boundary, don't rediscover it.** Read the contract, not the monolith. Before grepping a 1000-line module (e.g. `core.py`), check for a frozen contract (docstrings + `>TBI-not-ready` guards, e.g. `jaxfne/experimental_hpc/contracts.py`) and route from that.

**Levers:** contract-first not discovery-first · skills are the cache (run the skill, don't re-reason its checklist) · receipts beat justification (command + `N passed`) · verify-before-call (grep `__all__`/the contract before naming any symbol, flag, path, or skill — never invoke by remembered name).

**The 7 real active skills** (never invoke archived/old names): `jaxfne-worker-context-router` (this, use first) · `jaxfne-modeling-optimization-schema` · `jax-jit-pmap-performance-guard` · `jaxfne-notebook-release-gate` · `jaxfne-release-mutation-guard` · `jaxfne-sha256-artifact-integrity` · `jaxfne-visualization-schema`.

**Route by altitude:** Opus → architecture/contract authoring/truth-gate review. Sonnet → implement against a frozen contract + the relevant skill. Gemini → large-context cross-file synthesis + repo-scale batch edits. Frozen contracts + skills + the §11 invariants are what make the cheaper tiers safe.
