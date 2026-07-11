# AGENTS.md — jaxfne (agent pointer)

**Clone:** this repo · **Import:** `import jaxfne as jtfne`

Depth lives in **`skills/`**, **`docs/`**, and **`artifacts/developer/`** — not here.
Full AI-agent guide: [docs/for_ai_agents.md](docs/for_ai_agents.md).

## Object grammar

```text
Config → Net → Paradigm → Objective → Trainer → Signals → Vis/Export
```

`construct()` is the single dispatch — extend it, don't bypass.

## Three build paths (pick one per script)

| Path | When | Returns |
|------|------|---------|
| **Config** `laminar_cortex_config` / `build_laminar_column` → `construct` → `simulate` | single run, AGSDR, homeostasis, per-neuron drive, HDP via `RuntimeConfig` | `Model` / `Signals` |
| **tutorial_utils** `make_laminar_column_config` → `build_laminar_column` → `simulate_laminar_trials` | multi-trial spectrolaminar sweeps | plain `dict` |
| **NeuronalTensor** → `construct(tensor, RuntimeConfiguration(...))` → `simulate` | multi-area, JSON tensors, explicit 3D | `Model` / `Signals` |

HDP on tensor path: `simulate(..., runtime=RuntimeConfig(enable_hdp=True, hdp_params={...}))`.

Per-event layer targeting: `target_indices` on **event dict**, not schedule ctor. Build indices from `model.neuron_table()`.

Laminar readout: `jtfne.vis.spectrolaminar_suite(sig)`.

## Truth gates (never escalate)

`claim_level=computational_scaffold` · `field_solver_status=linear_solver` · `field_claim_level=proxy_readout` · `physical_amplitude_calibrated=False`

Language: simulated/proxy/scaffold — not validated/physical/mechanism without receipts.

Plausible Izhikevich sanity: rest ≈ −66 mV, spike peak ≈ +30 mV, mean rate ≈ 8–25 Hz. `|Vm| > 150` or NaN/Inf = blowup.

## Before writing helpers

Read `skills/catalog-glossary-jaxfne/SKILL.md`. Contradictions: `skills/FRICTIONS_STACK.md`.

## PRP backlog

`artifacts/developer/{plans,progress,review}.json` + `AGENT_CHANNEL.md` (stable path). Skill: `progress-review-plan`.
**Done rule:** `status=done` requires `achieved_score >= target_score`. JSON edits ≠ work done.

## Module map

`core.py` re-exports → `_config.py`, `_model.py`, `_signals.py`, `_construct.py`, `_runtime_config.py`, `emitters.py`, `fields/`.
Plotting: **`jaxfne/vis/*` only** (modular-grammar rule 2).

## Root freeze

Repo root frozen 2026-06-17 — modify existing root files/folders only; no new top-level dirs except approved patches (`skills/` kept by design).

## Known stubs

`GLIFEmitter`, `LIFEmitter`, `write_nwb`, `read_nwb` — exported names, `NotImplementedError` on use.

## Known fragilities (track)

1. `jaxfne/__init__.py` runtime wrapper — brittle function/submodule collision.
2. `_CONFIG_RUNTIME_WARNINGS` global in `core.py` — not thread-safe.
3. Hardcoded 20.0 spike gain in source proxy — dense/edge kernels must stay in sync.

## Validation (`python3`)

```bash
git status --short --branch && git rev-parse HEAD
python3 -m compileall -q jaxfne tests scripts
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py -q --tb=short
```

## Branches

`main`, `dev`, `agy`, `cur`, `ops` — verify SHA before mutations; no force-push without approval. Push PRP work to `dev` as autosave.

## Agent channel

Read `artifacts/developer/AGENT_CHANNEL.md` before starting; append before finishing. Treat other agents' claims as hypotheses — verify against source/tests.
