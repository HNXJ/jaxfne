# AGENTS.md — jaxfne (agent pointer)

**Clone:** this repo · **Import:** `import jaxfne as jtfne`

Depth lives in **`skills/`**, **`docs/`**, and (local-only, see below) **`artifacts/developer/`**
— not here. Full AI-agent guide: [docs/for_ai_agents.md](docs/for_ai_agents.md).

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
`Model.simulate()` does not inherit `Configuration.hdp(...)`/`.runtime(...)` the way
top-level `simulate()` does — pass `runtime=RuntimeConfig(enable_hdp=True, ...)` explicitly
when calling the `Model` method directly. For true turn-to-turn state continuity (not just
`Model.with_hdp_initial_state()`'s partial `H`/`w`-only carry), use
`jaxfne._pipeline.compile_step_fn`/`scan_network` with `DynamicState` (all six fields:
`v, u, prev_spikes, syn_state, H, w`). Full detail: `skills/jaxfne-neural-tensor/SKILL.md`,
`skills/FRICTIONS_STACK.md` F-031.

Per-event layer targeting: `target_indices` on **event dict**, not schedule ctor. Build indices from `model.neuron_table()`.

## Config complexity tiers

`Configuration` (flat fluent builder) and `NeuronalTensor` (structured Areas ×
Layers × NeuronTypes) are separate tiers, not a hierarchy — both converge on
the same `Model` via `construct()`'s type-dispatch, neither wraps the other.
There is deliberately no `Configuration -> NeuronalTensor` converter:
`Configuration` is the simpler tier, and promoting it up to the structured
tier adds no information a caller didn't already have. Only the reverse
(`neuronal_tensor_to_configuration`) exists, since going from structured to
flat is a real simplification. `HDPColumnConfig` is a third, narrower tier
(canonical 6-layer laminar column only) — see `jaxfne-neural-tensor` skill.

Laminar readout: `jtfne.vis.spectrolaminar_suite(sig)`.

## Truth gates (never escalate)

`claim_level=computational_scaffold` · `field_solver_status=linear_solver` · `field_claim_level=proxy_readout` · `physical_amplitude_calibrated=False`

Language: simulated/proxy/scaffold — not validated/physical/mechanism without receipts.

Plausible Izhikevich sanity: rest ≈ −66 mV, spike peak ≈ +30 mV, mean rate ≈ 8–25 Hz. `|Vm| > 150` or NaN/Inf = blowup.

**This section is agent-facing, not for public docs.** README/`docs/**` get the *result* of this
rule (every value stated as **Relative** or **Absolute**, nothing else), never the rule itself.
Don't copy "language: prefer X avoid Y" phrasing or "does not claim" framing into `docs/` or
`README.md`. Don't compare jaxfne to other named projects in public docs. Don't build dict keys
via string concatenation to dodge a grep check. Guard: `python3 scripts/audit_public_docs_language.py --check` (wired into CI).

**Same rule applies to non-doc-language leaks.** Nothing describing internal roadmap, competitive
positioning, per-file audit scores, or agent-to-agent working notes gets tracked at any path in
this repo — public or not, since the whole repo is public. See "PRP backlog" below for the
concrete incident this rule exists because of.

## Before writing helpers

Read `skills/catalog-glossary-jaxfne/SKILL.md`. Contradictions: `skills/FRICTIONS_STACK.md`.

**Delegating jaxfne work to a fast/weak model (agy, Haiku-tier, any subagent):** don't just
describe the task — paste the relevant skill's verified names/signatures into the prompt
directly. This repo's own API surface is large and easy to guess wrong (`jtfne.weld()`,
`cfg.circuit`, a fictional `FlatModel`/`FlatNet` alias were all found and struck from the skills
this way — a plausible-sounding invented name a fast model would readily produce unprompted). A
skill's contract handed inline removes that guesswork; an open-ended ask invites it.

## PRP backlog — local-only, not tracked

`artifacts/developer/{plans,progress,review}.json` + `AGENT_CHANNEL.md`. Skill:
`progress-review-plan`. **`git rm --cached` + gitignored 2026-07-14** — an independent audit
confirmed this directory (internal roadmap, per-file scores, competitive citation-strategy notes)
was tracked and pushed to this public repo. Stopped tracking going forward; git history was not
scrubbed (explicit decision — repo stays public). **Do not re-add these paths to git.** Two files
under `artifacts/legacy/internal_docs/` are kept tracked as explicit `.gitignore` exceptions
(load-bearing for `tests/test_agent_context_hygiene.py` / `tests/test_docs_equations_plotly_v0214.py`,
confirmed non-sensitive) — don't add a third without the same content review.

**Done rule:** `status=done` requires `achieved_score >= target_score`. JSON edits ≠ work done.

## Module map

`core.py` re-exports → `_config.py`, `_model.py`, `_signals.py`, `_construct.py`, `_runtime_config.py`, `emitters.py`, `fields/`.
Plotting: **`jaxfne/vis/*` only** (modular-grammar rule 2).

`_model.py` and `_construct.py` are themselves thin re-export aggregators as
of 2026-07-20 (Phase 2 defragmentation, 0.4.8-0.4.48 roadmap) — same pattern
as `core.py`. `_model.py` (Model dataclass + 7 lifecycle methods) re-exports
from `_model_simulate.py`, `_model_readout.py`, `_model_evaluate.py`,
`_model_tune.py`, `_model_manifest.py`. `_construct.py` re-exports from
`_construct_population.py`, `_construct_connectivity.py`,
`_construct_presets.py`, `_construct_core.py`, `_construct_extras.py`.
Import from `jaxfne.core`, not any of these directly, unless working on the
split itself.

## Root freeze

Repo root frozen 2026-06-17 — no new top-level **folders** except approved patches (`skills/`
kept by design). A single new top-level **file** (a GitHub-convention file expected at repo
root) is not a folder-clutter violation on its own — this doesn't reopen the freeze generally.

## Known stubs

`GLIFEmitter`, `LIFEmitter`, `write_nwb`, `read_nwb` — exported names, `NotImplementedError` on use.

## Known fragilities (track)

1. `jaxfne/__init__.py` runtime wrapper — brittle function/submodule collision.
2. Hardcoded 20.0 spike gain in source proxy — dense/edge kernels must stay in sync.
3. `DEFAULT_HDP`'s `K_w_ctrl=0.0` permits unbounded weight drift on long/custom HDP runs outside
   the specific presets already verified — see `plans.json`'s `hdp-k-w-ctrl-default-runaway-gap`
   (local-only, see PRP backlog above).

## Validation (`python3`)

```bash
git status --short --branch && git rev-parse HEAD
python3 -m compileall -q jaxfne tests scripts
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py -q --tb=short
```

## Branches

`main`, `dev`, `agy`, `cur` — verify SHA before mutations; no force-push without approval. Push
PRP work to `dev` as autosave. `ops` was deleted 2026-07-08 — do not reference it; if you see it
in a stale local ref, `git branch -d ops`.

## Agent channel

Read `artifacts/developer/AGENT_CHANNEL.md` (local-only, see PRP backlog above) before starting;
append before finishing. Treat other agents' claims as hypotheses — verify against source/tests.
