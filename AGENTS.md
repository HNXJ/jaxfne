# AGENTS.md — jaxfne

## Read first

This file is the quickref. `internal_docs/` (the old home for `AGENT_QUICKREF.md` /
`CURRENT_PUBLICATION_STATE.md` / `JAXFNE_BIOPHYSICS_GLOSSARY.md`) was deliberately
removed from the public repo at the same commit that froze the root (`ca43c1e`,
2026-06-17) — links into it are dead by design, not staleness; do not recreate it.
What replaced each:

- API catalog → the `catalog-glossary-jaxfne` skill (`skills/catalog-glossary-jaxfne/SKILL.md`) — check it before writing
  any helper or hand-rolling PSD/raster/LFP-proxy/CSD-proxy/spectrolaminar logic.
- Known skill/doc contradictions → `skills/FRICTIONS_STACK.md` (resolve before escalating claims).
- Per-file completeness/status tracker → `artifacts/developer/progress.json` (path, purpose, score/100,
  TBI/TBD/warnings, evidence). **Check it before claiming a file is done, broken, or untouched, and before
  trusting a score/finding recalled from a prior session — verify against current state, then update the
  entry.** Session-scoped, not a repo-wide sweep; grow it incrementally, don't backfill unverified rows.
- Evidence/publication-state snapshot → `python3 scripts/evidence_inventory.py` (run
  it; don't trust a remembered SHA).
- Biophysics deep reference → the global `~/.claude/CLAUDE.md` "COMPUTATIONAL NEURONAL
  BIOPHYSICS" section + the `jax-neuro-diffsim-guard` / `neuro-biophysics-units-sanity`
  skills.

## Identity

jaxfne is a compact JAX-native TFNE scaffold.

```python
import jaxfne as jtfne
```

Public flow (shorthand): `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export` (package computes; notebooks configure/plot/export). For the full typed chain see README § "Object grammar".

## Root freeze policy

**[INVARIANT] Repo root is frozen as of 2026-06-17 (commit ca43c1e).**

Structure is final. All future changes must:
- **Modify existing files** in root (e.g., README.md, LICENSE, .gitignore)
- **Add content inside existing folders** (jaxfne/, docs/, examples/, tutorials/, tests/, scripts/, .github/, skills/)
- **NOT create new top-level folders or files** except in exceptional patches (e.g., critical security fix requiring new root artifact)

**Approved exceptions:**
- `skills/` (added 2026-06-18, v0.4.0): versioned jaxfne agent skills, sign-off recorded for this release.

This ensures:
- Stable project surface (no structural surprises)
- Predictable navigation (familiar structure)
- Professional package state (no root clutter)

Breaking this rule requires explicit approval. Violations: flag and revert immediately.

## jaxfne-modular-grammar

**[INVARIANT, added 2026-07-01] The standing architecture rule set for this repo** — not
enforced by tooling yet, but every future change should be checked against it before landing:

1. **Maximum modularity** — 5 main modules, each ~5 object rule (desired target, not a hard cap).
2. **All visualization lives under `jaxfne/vis/*`** — not one plotting call anywhere else.
   **Known violations as of 2026-07-01** (not yet fixed, tracked in `progress.json`):
   `jaxfne/export.py`, `jaxfne/tutorial_utils.py`, and all of `scripts/evidence_figures/*.py`
   contain direct `matplotlib`/`plotly` calls outside `jaxfne/vis/`.
3. **Computation is jax-maximal, jax-parallel, `float32` by default.** Device order is always
   GPU → jax-metal → CPU. `float32` is already the practical default in `core.py`'s dtype
   handling (verified 2026-07-01); the GPU→metal→CPU fallback order is **not yet an enforced
   utility** — `jax.devices()` is called in several places (`sharding_utils.py`, `runtime.py`,
   `core.py`) but none encode this specific priority order.
4. **Every observed flaw goes straight into `progress.json`** on the file's existing row
   (`warnings`/`tbd`) or a new row if the file isn't tracked yet — immediately, not batched into
   a separate report. This is the same discipline the `progress-review-plan` skill already
   enforces; this rule just says "don't wait for a formal Review pass to log something you
   noticed in passing."
5. **Maximum borrowing** — prefer `jax`/`jaxlib`/`jaxley`/`plotly`/`scipy`/`sklearn`/`torch`/
   `torchvision` primitives over hand-rolled logic; jax gets priority when multiple libraries
   offer the same primitive.
6. **`NeuronalTensor` and the network/`Model` object must stay one identical format for any
   config** — no new bespoke construction function per config variant; new configs merge into
   the existing objects (`Configuration`, `NeuronalTensor`, `Model`) rather than growing parallel
   one-off helpers. `construct(cfg_or_tensor, runtime=None)` in `core.py` is the existing
   dispatch point for this — extend it, don't bypass it.

## Gates

```text
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

**[INVARIANT] = non-negotiable.** Rules, exact code locations, and what each prevents (moved
here 2026-06-30 from the global `~/.claude/CLAUDE.md`, which now only states the generic
truth-gate pattern):

| Rule | Code location | What it prevents |
|------|---|---|
| **Never escalate claim gates upward** | `core.py`, all dataclasses | `claim_level`, `field_solver_status`, `physical_amplitude_calibrated` stay at conservative defaults; code may read, never flip |
| **Proxy ≠ PDE** | `fields.py` | Laminar proxy (Gaussian kernel + FD CSD) is not a field solve; carry the `*_proxy` suffix and `field_solver_status` at its conservative default (`linear_solver`); never synthesize J_e |
| **Receipts write-once** | `core.py` `save_receipt()` | refuses overwrite without explicit flag; never hand-edit truth files |
| **x64 before arrays** | session start | `jax.config.update('jax_enable_x64', True)` before array construction; verify `runtime_report()["actual_dtype"]` |
| **Explicit PRNG only** | all stochastic paths | every SDR/GSDR/AGSDR call takes an explicit `jax.random.PRNGKey`; raises if `key=None`; no numpy.random in reproducible code |
| **Hard spike reset non-differentiable** | `core.py`, `Model.tune()` (guard = `gradient_path_safe()`; confirm `grep -n gradient_path_safe jaxfne/core.py`) | `Model.tune()` blocks Optax unless `gradient_path_safe()` is true; do not remove the guard. (The "+30" is the +30 mV spike threshold, not a version.) |

**JAX execution defaults:** time = `jax.lax.scan` · edges = `jax.ops.segment_sum` (double-count
guard) · batch = `jax.vmap` over PRNG keys · compile = `jit=True` in RuntimeConfig only on
repeat runs. Perf debug order (cheapest first): confirm jit/vmap via `runtime_report()` → check
`recurrent_backend` (sparse > dense for large W) → reduce `n_steps` before profiling → keep
shapes/dtypes stable to avoid recompilation. Finiteness gates: reuse `_finite_or_none`,
`_finite_bool`, `validate_*`; JSON uses `allow_nan=False`.

**Claim language:** use "simulated"/"proxy"/"scaffold"/"computational diagnostic"; avoid
"validated"/"physical"/"proved"/"mechanism" without a manifest+hashes receipt. No real
EEG/MEG/LFP/CSD or calibrated-amplitude language for proxy readouts unless
`physical_amplitude_calibrated=True` with evidence. TFNE claim source-of-truth: `hnyxj/rules/`
(in `/Users/hamednejat/workspace/main/hnyxj/rules/`).

## Known fragilities (track, don't just warn)

1. **`jaxfne/__init__.py` runtime wrapper** — CustomModuleType intercepts function/submodule collision; brittle. FIX: refactor public `runtime()` surface + regression tests.
2. **`_CONFIG_RUNTIME_WARNINGS` global in `core.py`** — not thread-safe; couples `config_to_simulation`/`config_to_configuration`. FIX: return warnings instead of a global.
3. **Hardcoded 20.0 spike gain in source proxy** — dense + edge kernels must stay in sync or the double-count guard breaks. FIX: parameterize + automated sync check.

## Docs-migration status (snapshot, not a standing fact — re-check before citing)

As of 2026-06-25: `docs/api/neuronal_tensor.md` + `docs/api/index.md` described NeuronalTensor
correctly; `README.md`, `docs/quickstart.md`, `docs/guides/hdp.md`, and `tutorials/` still
taught Configuration-only. **Standing rule:** when a code change introduces or fixes a new
top-level API path, update this file in the same pass — don't let docs drift the way
README/quickstart did.

**Schema-state note:** the v0.4.0 gate schema (`linear_solver` / `proxy_readout` /
`physical_amplitude_calibrated` / `migrate_schema`, api 195) is canonical; releases since
(v0.4.1+) build on it without changing it. Legacy JSON/manifests upgrade via
`jtfne.migrate_schema(old_dict)`. A clone with `git grep -I laminar_proxy_no_pde | wc -l > 0`
predates the v0.4.0 migration and is stale.

## Backlog protocol

`artifacts/developer/{plans,progress,review}.json` is run via the global `progress-review-plan`
skill (Plan/Progress/Review/Brainstorm) — see `~/.claude/skills/progress-review-plan/SKILL.md`
for the mechanism. This repo's specifics: `review_command` for a `.py` file entry is usually
`python3 -m py_compile <path>` plus the nearest test in `tests/`; prioritize entries touching
`core.py`/`fields.py`/`emitters.py` first (truth-gate-adjacent code) unless told otherwise.

## Branch policy

Five permanent branches: `main`, `dev`, `agy`, `cur`, `ops` (kept aligned at the same SHA after integration; main is the release source-of-truth, dev the integration branch). Do not mutate any permanent branch without approval. No force-push, tag, release, or publish without approval.

## Validation

```bash
python3 scripts/evidence_inventory.py
# scripts/evidence_figures_inventory.py is kept as a compatibility wrapper.
python3 -m compileall -q scripts/evidence_figures jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py -q --tb=short
```

## Report format

Status, repo state, changed files, commands run, exact results, evidence/truth status, blockers, next safe action.

## API catalog (read before writing helpers)

Before writing any jaxfne helper or hand-rolling PSD/raster/LFP-proxy/CSD-proxy/
EEG-proxy/MEG-proxy/spectrolaminar/AGSDR/manifest logic, consult the
`catalog-glossary-jaxfne` skill (global, `~/.claude/skills/`) — it lists the
package-native functions (incl. the exact spectrolaminar pipeline) so existing
APIs are reused, not rediscovered. Canonical import: `import jaxfne as jtfne`.

**Stub warning:** `GLIFEmitter` and `LIFEmitter` are exported in `__all__` but raise `NotImplementedError` on use — do not reference them in examples or tutorials. `write_nwb` / `read_nwb` are similarly exported but not implemented.

## Verified laminar pipeline (reuse; do not rediscover)

```python
cfg = jtfne.laminar_cortex_config(areas=["V1"], layers=["L1","L2/3","L4","L5","L6"],
                                  n=10000, duration_ms=1000.0, dt_ms=0.1, seed=0)
cfg = cfg.layer_fractions(layer_fractions={L: (z_lo, z_hi), ...})   # per-layer DEPTH band; count ∝ thickness
cfg = cfg.area_layer_cell_types("V1", {L: {"E":.., "PV":.., "SST":.., "VIP":..}, ...})
model = jtfne.construct(cfg)
sig   = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)   # sig.field auto-computed (lfp_proxy, csd_proxy)
```

**Canonical column E:I gradient (ground truth, verified 2026-06-17; skill `jaxfne-config`):**
E peaks DEEP — E-fraction rises with depth to L6 ≈ 90% E. I peaks SUPERFICIAL — L1 50% I,
I-fraction falls monotonically 50→30→25→20→12→10% (L1→L6); the largest inhibitory neuron
COUNT sits in the dense superficial L2. Overall ≈ 77E:23I. PV concentrates in L4
(feedforward), absent in L1 (VIP/SST only). Set this per layer via
`.area_layer_cell_types(area, {...})` — the global `cell_types=` weight produces the WRONG
(over-inhibitory, 41:59) gradient. Verify with `construct(cfg).neuron_table()` (list of dict rows).

- Per-neuron/per-layer DC drive: `model.with_emitter_parameters(drive_per_neuron=array)`
  (build a deep-layer mask from `model.neuron_table()`; recurrent `W` is dense, so deep drive reaches superficial).
  Per-cell-type DC: `baseline_drive_by_cell_type={"E":..}` in the config.
- Explicit 128-contact projection: `jtfne.project_laminar_sources(source, positions_(N,3), n_contacts=128)`.
  **Default `mode="density_preserving"`** (`jaxfne/fields/proxy.py`) — SUM weights, no row-normalize.
  Opt-in **`mode="row_normalize"`** only for pedagogy or when every contact lies inside the
  modeled population; row-normalizing erases attenuation for off-population contacts (e.g. a probe
  beyond the depth band): weights sum to 1 regardless of Gaussian falloff, so outside contacts can
  read *louder*, not weaker. Verified 2026-06-21 on a 300-neuron V1 column with 2 contacts at
  z=-0.15/1.15: `row_normalize` gave outside RMS *higher* than the inside mean;
  `density_preserving` gave ~9× attenuation. Skill: `jaxfne-config` § Projection.
- **Layer naming (F-002):** 5-layer (`L1,L2/3,L4,L5,L6`) in `laminar_cortex_config` examples vs
  **6-layer** (`L1…L6`) required by `build_laminar_column(..., ei_profile="canonical")` /
  `CANONICAL_LAYERS_6L`. Pick one set per script; canonical E:I fractions in
  `jaxfne-config` are verified on **6-layer** names.
- `spectrolaminar_psd_jax` wants `(n_trials, n_steps, n_contacts)`.
- All `jtfne.vis.*` (lfp/csd/eeg/meg/emm/raster/rate/psd/spectrolaminar_suite/layer_celltype_counts)
  take `sig` and return matplotlib Figures; **`vis.spectrolaminar_suite(sig)` is the preferred laminar readout**.
  3D circuit: `vis.visualize_network_3d(model.neuron_table(), output_html=...)`.

## Two (now three) paths to a laminar run — pick before you start, don't mix mid-task

There are three independent ways to build+run a laminar column/circuit. All
are real and supported; they are not interchangeable mid-script because they
hand back different object types (`Model`/`Signals` vs a plain trial `dict`).

| | **Config-path** (`laminar_cortex_config` / `build_laminar_column` → `construct` → `simulate`) | **tutorial_utils-path** (`make_laminar_column_config` → `build_laminar_column` → `simulate_laminar_trials`) | **NeuronalTensor-path** (`NeuronalTensor` → `construct(tensor, RuntimeConfiguration(...))` → `simulate`) |
|---|---|---|---|
| Use when | single run, AGSDR tuning, homeostasis/plasticity, custom per-neuron drive, HDP via `RuntimeConfig` | multi-trial spectrolaminar sweeps, `summarize_spectrolaminar_similarity` | declarative Areas×Layers×Types, canonical JSON tensors, multi-area merge |
| Returns | `Model` / `Signals` | plain `dict` of trial arrays | `Model` / `Signals` |
| Per-event drive targeting | per-event `target_indices` key — see below | not exposed; drive is column-wide | same as Config-path after bridge |
| Noise control | kernel-dependent — see caveat below | `cell_type_izh_params[ct]["noise"]`, swept directly | same as Config-path after bridge |
| Homeostasis/plasticity | wired (`Configuration.homeostasis(...)`, see below) | not wired | HDP: pass `runtime=RuntimeConfig(enable_hdp=True, ...)` to `simulate()` |
| Docs | this file, `catalog-glossary-jaxfne` §1 | `catalog-glossary-jaxfne` §2 | `docs/guides/hdp.md`, `docs/api/neuronal_tensor.md` |

**NeuronalTensor-path detail (0.4.7):** `RuntimeConfiguration` (tensor path, frozen) has **no**
HDP field; `RuntimeConfig` (Config-path) does. To enable **HDP homeostatic plasticity**
(synaptic + H-factor adaptation, cube-law `tau_i = tau_0_ms * size_i**3`) on a tensor-built
`Model`, pass an explicit `runtime=RuntimeConfig(enable_hdp=True, hdp_params={...})`
to `simulate()` — it overrides any `Configuration`-derived metadata. Full pattern:
[`docs/guides/hdp.md`](docs/guides/hdp.md) § "Tensor-first" and
[`docs/api/neuronal_tensor.md`](docs/api/neuronal_tensor.md).

**HDP v2 sign orientation (`hdp_rule=`, `jaxfne/emitters.py` `simulate_receptor_exponential_izhikevich`):**
`signed_linear` and `signed_quadratic` compute their weight-update basis as
`H_post - H_pre` (edge-indexed `H_next[post]` minus `H_next[pre]`), not the naively-expected
`H_pre - H_post` that the top-level docstring's rule-family summary states. This flip is
deliberate: it preserves the postsynaptic-indexing invariant used everywhere else in this
kernel (`W_i` from outgoing edges, `I_syn_i` from incoming edges, the E/I sign split via
`exc_mask`) so that a resource-starved postsynaptic neuron (`H_post < H_pre`) still weakens its
incoming excitatory weights and strengthens its incoming inhibitory weights — the correct
restoring direction — under either expression once the sign convention is applied consistently.
See the inline comment at `jaxfne/emitters.py:1347` for the exact basis per rule.

**Noise-scale caveat (verified, not uniform across kernels):** `simulate_eig_izhikevich`,
`simulate_edge_recurrent_izhikevich`, and `simulate_edge_recurrent_izhikevich_homeostatic`
all accept `noise_scale=` (`None` keeps the historical `0.5` scalar; pass a scalar or
`(n_neurons,)` array to override). `simulate_receptor_exponential_izhikevich`
(`jaxfne/emitters.py:1088`) has **no** `noise_scale` parameter at all — its noise
coefficient is hardcoded to `0.5` inline (lines 1159/1186). Check which kernel
`construct()`/`simulate()` actually dispatches to for your config before assuming
noise is or isn't sweepable.

**Per-event, per-neuron-subset drive (`target_indices`):** `StimulusSchedule`
(`jaxfne/core.py:3185`) does **not** take a `target_indices=` constructor kwarg —
`target_indices` is a key on each **event dict**, read by
`StimulusSchedule.to_array`/`to_array_jax` (`ev.get("target_indices", None)`) to
restrict that event's amplitude to a specific neuron subset (e.g. L4-E-only)
rather than the whole column. Build the index list from `model.neuron_table()`
(filter by `layer`/`cell_type`) and put it on the event, e.g.:

```python
nt = model.neuron_table()
l4e_idx = [i for i, row in enumerate(nt) if row["layer"] == "L4" and row["cell_type"] == "E"]
event = {"onset_ms": 0.0, "duration_ms": 50.0, "amplitude": 5.0,
         "label": "p1", "is_drive_event": True, "target_indices": l4e_idx}
sched = jtfne.StimulusSchedule(events=(event,), n_neurons=model.n_neurons)
```

This is the real mechanism for "stimulate only this cell-type/layer subset on this
event" — don't hand-roll a per-neuron drive mask for that case. Worked end-to-end
example (4-slot sequential paradigm, L4-E-only targeting per slot, laminar readout):
[`tutorials/jaxfne_v040_continuous_omission_oddball.ipynb`](tutorials/jaxfne_v040_continuous_omission_oddball.ipynb)
([doc](docs/tutorials/index.md#tutorial-stack)).

## Self-check before claiming a result is real

Reduced Izhikevich, native units: Vm rest ≈ −66 mV, spike peak ≈ +30 mV (hard reset),
plausible mean rate ≈ 8–25 Hz. `|Vm| > 150` or NaN/Inf = a dt/solver/unit blowup, **not** a finding.
10k-neuron / 1000 ms run on CPU: `construct` ≈ 40 s, `simulate` ≈ 90 s — if 5–10× slower the
machine is thermally throttled (cool, run as a fresh subprocess), it is not hung. A real run with
plausible numbers is the receipt; never report a simulated value you did not sanity-check.

## Homeostasis & plasticity (verified 2026-06-21; reuse, do not rediscover)

Three distinct mechanisms share the word "plasticity" here — do not conflate them:

1. `Configuration.plasticity()` — **declarative only**, `status="declared_not_wired_to_simulate"`.
   No kernel runs; it only records intent in `metadata["plasticity"]` for `manifest()`.
2. `Configuration.homeostasis(eta=...)` — **real and wired**. `eta != 0` engages homeostatic
   synaptic plasticity inside `simulate_edge_recurrent_izhikevich_homeostatic`
   (`jaxfne/emitters.py`): `dw = eta*(r_star - r_post)*x_pre`, clipped to `[w_min, w_max]`.
   Verified: most edges' weights measurably change and stay clipped/finite. As of `86e19e0`,
   `Model.last_homeostasis_diagnostics()` and `Signals.metadata["homeostasis"]` surface
   `w_final`/`w_trace` when `eta != 0` (earlier commits silently dropped them after computing
   them — check you're past `86e19e0` before relying on this).
3. `run_stdp_stream` / `make_ei_cloud_network` — a separate STDP path, **not connected** to
   `Model.simulate()` at all.

`Configuration.homeostasis(k_gain=...)` (the `g_bias` excitability term, independent of `eta`)
is a **one-sided damper on this canonical-column prior, not a bidirectional rate-setpoint
controller** — verified by sweep, not assumed: at a ~10.8 Hz natural baseline, the activity
trace `r` settles at 0.65–0.73 regardless of `r_star` in the small range you'd naively pick
(spikes/step units), so `g=clip(k_gain*(r_star-r))` stays negative; pushing `r_star` to its
ceiling (1.0) only ever recovers baseline, never exceeds it, and raising `g_max` past the
default 8 does nothing (confirms it isn't a clipping-ceiling issue — `r` itself saturates near
`r_max` within the run). If you need a rate *above* baseline, this mechanism cannot do it
without a kernel change (none made; would need sign-off). For suppression-toward-a-lower-target
demos it works smoothly up to about `k_gain≈1.5–2.0` (default `tau_r_ms=300`); past
`k_gain≈2.5` the population enters a bursty bang-bang relaxation oscillation (full-silence
windows every ~`tau_r_ms`-scale period) rather than settling — check a 20-100ms-windowed rate
trace, not just the run mean, before calling a homeostasis result "stable."

## PR review

Automated review runs via **Macroscope** (comment `@macroscope-app review` on a GitHub PR;
code review is enabled by default). Reviewers — human or bot — obey the repo skills under
`skills/` and apply this rubric:

- **Posture:** `jaxfne` is a **computational scaffold / proxy-readout** codebase, NOT a
  physical-solver or biological-mechanism codebase. Review accordingly.
- **Review for:** correctness, JAX efficiency (jit/vmap/scan, `N_compile ≤ 1`, stable shapes/dtypes),
  API stability, docs/tutorial alignment (notebook grammar), and truth-gate compliance (§ Gates).
- **Block (revise/reject):** silent fallbacks; fabricated/synthetic analysis output presented as real;
  dense O(N²) where a sparse path is possible (all-to-all is inherently dense → warn, don't block;
  sparse `p_connect<1` at scale should use the direct edge builder); ambiguous projection semantics
  (normalization mode must be explicit); public stubs masquerading as finished APIs; any
  biological/mechanistic overclaim.
- **Homeostasis/plasticity:** jaxfne is the mathematical backend — biophysical fidelity follows
  the config you provide, not a fixed ceiling (decided 2026-06-20, see memory
  `jaxfne-math-backend-framing`). Do **not** require disclaimer-triple stamping
  (`physical_amplitude_calibrated=False`/`biological_learning_claim=False`/
  `mechanism_claim_status=not_claimed`) in reports, changelogs, or docs prose. Block only on
  actual overclaims — e.g. asserting a validated biological mechanism without nulls/ablations/
  repeated-seed evidence — not on missing disclaimer language.
- **Output per PR:** accept / revise / reject · exact blockers · files involved · score /100.
  Prefer concrete file/line references and decisive recommendations.
