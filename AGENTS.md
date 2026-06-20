# AGENTS.md — jaxfne

## Read first

[`internal_docs/loop_context/AGENT_QUICKREF.md`](internal_docs/loop_context/AGENT_QUICKREF.md)

Evidence snapshot (refresh SHA each session): [`internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md`](internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md)

Deep reference (on demand): [`internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md`](internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md)

## Identity

jaxfne is a compact JAX-native TFNE scaffold.

```python
import jaxfne as jtfne
```

Public flow: `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export` (package computes; notebooks configure/plot/export).

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

## Gates

```text
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

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
EEG-proxy/MEG-proxy/spectrolaminar/AGSDR/manifest logic, consult the curated
lookup table: [`internal_docs/JAXFNE_AGENT_API_CATALOG.md`](internal_docs/JAXFNE_AGENT_API_CATALOG.md).
It lists the package-native functions (incl. the exact spectrolaminar pipeline)
so existing APIs are reused, not rediscovered. Canonical import: `import jaxfne as jtfne`.

## Verified laminar pipeline (reuse; do not rediscover)

```python
cfg = jtfne.laminar_cortex_config(areas=["V1"], layers=["L1","L2/3","L4","L5","L6"],
                                  n=10000, duration_ms=1000.0, dt_ms=0.1, seed=0)
cfg = cfg.layer_fractions(layer_fractions={L: (z_lo, z_hi), ...})   # per-layer DEPTH band; count ∝ thickness
cfg = cfg.area_layer_cell_types("V1", {L: {"E":.., "PV":.., "SST":.., "VIP":..}, ...})
model = jtfne.construct(cfg)
sig   = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)   # sig.field auto-computed (lfp_proxy, csd_proxy)
```

**Canonical column E:I gradient (ground truth, verified 2026-06-17; skill `jaxfne-cortical-column-default`):**
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
- `spectrolaminar_psd_jax` wants `(n_trials, n_steps, n_contacts)`.
- All `jtfne.vis.*` (lfp/csd/eeg/meg/emm/raster/rate/psd/spectrolaminar_suite/layer_celltype_counts)
  take `sig` and return matplotlib Figures; **`vis.spectrolaminar_suite(sig)` is the preferred laminar readout**.
  3D circuit: `vis.visualize_network_3d(model.neuron_table(), output_html=...)`.

## Self-check before claiming a result is real

Reduced Izhikevich, native units: Vm rest ≈ −66 mV, spike peak ≈ +30 mV (hard reset),
plausible mean rate ≈ 8–25 Hz. `|Vm| > 150` or NaN/Inf = a dt/solver/unit blowup, **not** a finding.
10k-neuron / 1000 ms run on CPU: `construct` ≈ 40 s, `simulate` ≈ 90 s — if 5–10× slower the
machine is thermally throttled (cool, run as a fresh subprocess), it is not hung. A real run with
plausible numbers is the receipt; never report a simulated value you did not sanity-check.

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
- **Homeostasis/plasticity:** keep the framing explicit —
  `claim_status = "computational_control_proxy_not_biological_mechanism"`,
  `biological_learning_claim = False`, `mechanism_claim_status = "not_claimed"`.
- **Output per PR:** accept / revise / reject · exact blockers · files involved · score /100.
  Prefer concrete file/line references and decisive recommendations.
