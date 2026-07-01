---
name: jaxfne-worker-context-router
description: >-
  Route any jaxfne repo task to the correct module, current API, branch
  state, and validation lane before editing. Use at the start of jaxfne
  work — orienting in the repo, picking which module/skill applies,
  checking branch/SHA state — before diving into a refactor, API change,
  test, doc, tutorial, or release task.
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

## Ownership map (verified on disk — no jaxfne/config.py or net.py)

```text
Config/Model/Signals/Simulation  -> jaxfne/core.py (primary monolith)
Builders / canonical columns     -> jaxfne/builders.py
NeuronalTensor / tensor bridge   -> jaxfne/neuronal_tensor.py
HDP builder / defaults           -> jaxfne/hdp_network.py
Paradigm/task/stimulus           -> jaxfne/paradigm.py, jaxfne/stimulus.py
Objective/metrics                -> jaxfne/objectives.py (+ Objective in core.py)
Trainer/AGSDR/tuning             -> jaxfne/optim/*
Emitters/kernels                 -> jaxfne/emitters.py
Source/field/probe/readout       -> jaxfne/fields/*, jaxfne/bridges.py
Connectivity                     -> jaxfne/connectivity.py
Validation/manifest              -> jaxfne/validation.py, jaxfne/io.py
Visualization                    -> jaxfne/vis/* only
Notebook glue                    -> jaxfne/tutorial_utils.py, scripts/*
Agent skills (reference)         -> skills/* (this folder), NOT jaxfne/skills/
Release/package mutation         -> release skills and scripts only
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

**Repo skills index (consolidated 2026-06-30, 17→13 skills):** see `skills/README.md`.
Use first on most tasks:

`jaxfne-worker-context-router` (this) · `catalog-glossary-jaxfne` · `jaxfne-objective-grammar`
(top-level chain, routes into the 4 below) · `jaxfne-config` (Configuration fluent API +
canonical column template) · `jaxfne-neural-tensor` (NeuronalTensor build path + HDP) ·
`jaxfne-neural-network` (construct/simulate/Signals/probe/objective/tune) ·
`jaxfne-vis-modules` (jaxfne.vis) · `jaxfne-modeling-optimization-schema` (deep
schema/truth-gate reference) · `jaxfne-paradigm-design` · `jaxfne-spectrolaminar-suite` ·
`jaxfne-notebook-release-gate` · `jaxfne-release-mutation-guard` · `jaxfne-sha256-artifact-integrity`.

Merged away 2026-06-30 (content absorbed into the skills above, do not look for these
names): `jaxfne-configuration-fluent-api` → `jaxfne-config`, `jaxfne-cortical-column-default`
→ `jaxfne-config`, `jaxfne-visualization-schema` → `jaxfne-vis-modules`,
`jaxfne-signals-probe-objective-chain` → `jaxfne-neural-network`.

Flat enforcement checklist: `skills/00_INDEX.md` → `01_–11_` markdown files.

Open contradictions: `skills/FRICTIONS_STACK.md` (check before claiming API/science ground truth).

Global-only (not in repo): `jax-jit-pmap-performance-guard`, `jax-neuro-diffsim-guard`, `neuro-biophysics-units-sanity`.

**Route by altitude:** Opus → architecture/contract authoring/truth-gate review. Sonnet → implement against a frozen contract + the relevant skill. Gemini → large-context cross-file synthesis + repo-scale batch edits. Frozen contracts + skills + the truth-gate principle (global `~/.claude/CLAUDE.md`) are what make the cheaper tiers safe.

## Claude Code subagent routing (`Agent` tool)

When spawning subagents, pick `subagent_type` by task shape — do not default everything to `general-purpose`. Confirm current names with `ToolSearch`/the Agent tool's own listing before invoking, since the roster can change:

| Subagent | Use when |
|----------|----------|
| `Explore` | Find files/APIs quickly (`jaxfne/vis/*.py`, grep-style orientation), read-only |
| `general-purpose` | Multi-step implementation or research with unclear file ownership |
| `gemini-worker` | Repo-scale mechanical edits / large-context cross-file synthesis across many files |
| `Plan` | Design an implementation strategy before editing |
| `code-reviewer` (if present) | User explicitly asks for review of a local diff |

Always pass repo path, branch/SHA, and which skill contract applies. Subagents do not inherit chat context — brief them like a colleague who just walked in.
