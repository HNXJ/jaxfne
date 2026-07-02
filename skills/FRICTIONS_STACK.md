# jaxfne skills — friction & contradiction stack

**Purpose:** Durable ledger of known mismatches between skills, docs, code, and
external doctrine. Resolve items here before claiming a skill is authoritative.
**Do not delete rows** — mark `status: resolved` with SHA/date when fixed.

Last audited: 2026-07-01 (F-020 added — checkpoint/reload treedef + cfg.metadata mutation caught and fixed).

---

## Open

| ID | Severity | Topic | Contradiction / friction | Authoritative source (today) | Suggested resolution |
|----|----------|-------|--------------------------|----------------------------|----------------------|
| F-007 | low | Config fluent API surface | `Configuration()` bare constructor vs builder-first | Both exist; builders preferred | Documented in `jaxfne-config` (merged 2026-06-30 from `jaxfne-configuration-fluent-api`) — keep builder-first in tutorials |
| F-008 | low | Objective composition | No top-level `jtfne.band_power` / `phase_locking` | `Objective`, `Model.tune` | Use verified objective skill APIs only |
| F-009 | low | Signals trial axis | `Signals.get(..., trial=)` raises `NotImplementedError` | `core.py` Signals.get | Use `run_trials` / tutorial_utils for multi-trial |
| F-016 | low | `export.save_figure` deprecation | `export.save_figure` has no `DeprecationWarning`; audit recommends routing to `vis.export_figure` | `jaxfne/export.py` | Emit `warnings.warn(..., DeprecationWarning)` in `save_figure`; update any tutorial that calls it |
| F-018 | low | HDP v2 sign orientation undocumented at API level | `signed_linear` and `signed_quadratic` use `H_post - H_pre` (flipped from naive spec `H_pre - H_post`) to preserve postsynaptic-indexing invariant; flip is noted in inline comments but not in any skill or top-level API doc | `jaxfne/emitters.py` line 1332 | Add one-paragraph note to the HDP section of `AGENTS.md` explaining the postsynaptic-indexing convention and the direction of each rule |
| F-020 | medium | `IzhikevichParams`/pytree treedef is NOT N-independent | `labels`/`layer_labels` are pytree aux data (length N, not leaves) — a treedef built from a differently-sized dummy `construct()` silently produces a corrupted model on `jax.tree_util.tree_unflatten` (no error, just wrong `simulate()` output, confirmed 2026-07-01 by a real bit-identical-output check that failed). Also: `construct()` mutates `Configuration.metadata` in place (`recurrent_backend` key, `circuit.connections[*].status`) — reusing a pre-construct `cfg`'s metadata after a checkpoint reload reproduces a different, still-finite V_m trace, not an error. | `jaxfne/emitters.py` (`IzhikevichParams`/`EdgeList` pytree registration), `scripts/cortical_column_localized_workflow.py::save_column`/`load_column` | Resolved in `save_column`/`load_column`: persist labels/layer_labels/calibration strings + `cfg.metadata` explicitly and reconstruct dataclasses directly, never via a mismatched-N treedef. Any future construct-once/save/reload code must do the same — do not reintroduce the dummy-treedef shortcut. |
| F-022 | low | `stimulus_schedule()`'s drive heuristic treats ALL non-omission `ParadigmEvent`s as drive events | `is_drive = not e.is_omission and e.onset_ms is not None` (`jaxfne/core.py::stimulus_schedule`) does not distinguish an actual labeled stimulus event (e.g. `standard`/`deviant`) from a pure timing marker (e.g. `trial_start`/`post_stimulus`/`post_omission`) — both get the default `drive_amplitude` (5.0) injected unless the event's `metadata` explicitly overrides it. `omission_oddball_paradigm`'s built-in conditions do not set that override, so their marker events are NOT truly silent padding. Confirmed by direct inspection + a real run (Étude 11) — does not corrupt a paired same-marker-structure contrast (both conditions share the markers, so it cancels), but would corrupt any comparison across conditions with differently-timed markers. | `jaxfne/core.py::stimulus_schedule`, `jaxfne/paradigm.py::omission_oddball_paradigm` | Not fixed upstream this pass (would change existing `omission_oddball_paradigm` behavior for any other caller). Documented in `tutorials/etudes/jaxfne_etude_no_11_omission_local.ipynb`'s markdown. Future fix: either give `ParadigmEvent` an explicit `is_stimulus`/drive-amplitude-zero default for marker-only events, or have `omission_oddball_paradigm` set `metadata={"drive_amplitude": 0.0}` on its `trial_start`/`post_*` events. |
| F-023 | medium | `HDPColumnConfig.base_drive_by_cell_type` has no "use emitter preset, do not override" sentinel | `hdp_network.build_model(cfg)` always passes an explicit `baseline_drive_by_cell_type=dict(cfg.base_drive_by_cell_type)` to `laminar_cortex_config` (default `BASE_DRIVE_BY_CELL_TYPE_DEFAULT` = 4.0-for-all-cell-types). But `laminar_cortex_config(baseline_drive_by_cell_type=None)` (its actual default) instead falls through to the izhikevich emitter's own per-cell-type PRESET drive (confirmed empirically 2026-07-01: E=5.0, PV=3.0, SST=3.5, VIP=3.0 — NOT 4.0-for-all). Any script that relied on the `None`-passthrough preset behavior cannot be blindly delegated to `hdp_network.build_model` without silently changing every neuron's baseline drive. Same root cause makes `hdp_network.apply_drive_correction`'s MULTIPLICATIVE correction (`corrected = existing_drive * factor`) unsafe to substitute for a script that does an ABSOLUTE override (`drive = explicit_base[ct] * explicit_correction[ct]`, ignoring the model's existing drive) — confirmed for `scripts/hdp_1000_neuronal_tensor_column.py`. | `jaxfne/hdp_network.py::HDPColumnConfig`/`build_model`/`apply_drive_correction`, `jaxfne/builders.py::laminar_cortex_config` | Blocks `plans.json` item `merge-build_model-apply_drive_correction` for 2 of its 3 target scripts (the 3rd, `hdp_suite2_visualizations.py`, was already correctly consolidated and is not affected). Real fix needs a design change: `Optional[...] = None` passthrough on `HDPColumnConfig.base_drive_by_cell_type` + a multiplicative-vs-absolute mode flag on `apply_drive_correction` — not done this pass, deliberately, to avoid forcing scripts onto a default that silently changes their science. |

---

## Resolved (this pass, 2026-07-01 continued)

| ID | Resolution | Date |
|----|------------|------|
| F-021 | `jaxfne/_pipeline.py::checkpoint_state`/`restore_state` did NOT actually round-trip bfloat16 correctly, contradicting an earlier same-day assumption ("already dtype-transparent, just needs re-verifying with a real bf16 array") in plans.json item `bf16-quantized-tfne-izhikevich-mode` (point 8). Confirmed the failure directly: plain `np.savez`/`np.load` silently mangles ml_dtypes bfloat16 arrays into raw void bytes (`dtype('|V2')`) on read-back — no exception, `restore_state` then crashes downstream when JAX tries to interpret the void array. Fixed: `checkpoint_state` now upcasts bf16 leaves to float32 before saving (exact, no precision loss — bf16 occupies the top 16 bits of float32) and records each leaf's original dtype name in the JSON sidecar (`leaf_dtypes`); `restore_state` casts back down using that name. Verified with a real bf16 `IzhikevichParams` leaf round-trip (dtype match + value match) and a new regression test `tests/test_pipeline_pure_functions.py::test_checkpoint_restore_roundtrip_preserves_bfloat16` (15/15 pass in that file, no regression). | 2026-07-01 |

## Resolved (this pass)

| ID | Resolution | Date |
|----|------------|------|
| F-017 | RESOLVED 2026-07-01: root-caused and fixed. rho_passive/H^2 fails at all 75 swept candidates because it is >=0 everywhere (H_min<H<H_max) -- with DEFAULT_HDP's own gamma=delta=0, no term in dH/dt could ever go negative above H*=1, so H was only ever stopped by the H_max hard clip, not a real restoring force (confirmed: H_max_obs pinned near 10.0 across nearly the whole sweep). Fix: revived `K_ctrl*(1-H_i)` in jaxfne/emitters.py (was dead code -- computed, never used) as a genuine two-sided restoring term. Re-verified at the full 20s/5-seed gate with K_ctrl=1.0 and 5.0: 10/10 rows pass (H_std 0.0002-0.0030, well under the 0.05 bar), and got TIGHTER at 20s than a 5s smoke check (genuine convergence, not slow drift -- the exact failure mode that fooled rho_passive). DEFAULT_HDP's existing K_ctrl=5.0 value is now actually active and validated. F-019 (formula redesign) is resolved by this fix; not opened as a separate row. | 2026-06-29, resolved 2026-07-01 |
| F-001 | HDP E size: code + docs aligned to `DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE` | 2026-06-27 |
| F-002 | Layer naming table in `AGENTS.md` + cortical-column skill | 2026-06-27 |
| F-005 | Global `AGENTS.md` JAX §11 gate strings updated | 2026-06-27 |
| F-006 | `jaxfne/skills/` removed | 2026-06-27 |
| F-010 | Worker router skill list aligned | 2026-06-27 |
| F-011 | `layer_celltype_count_table` / `column_density_table` implemented in `builders.py` | 2026-06-27 |
| F-012 | Homeostasis disambiguation in modeling skill | 2026-06-27 |
| F-013 | Plasticity overload in catalog | 2026-06-27 |
| F-014 | v038 projection pedagogy: row-normalize label added to notebook; default confirmed `density_preserving` | 2026-06-27 |
| F-015 | Objective evaluation path in skills | 2026-06-27 |
| — | Extensionless Python "skills" replaced with verified `SKILL.md` files | 2026-06-27 |
| — | `jaxfne-vis-modules` (renamed 2026-06-30 from `jaxfne-visualization-schema`) module map matches `jaxfne/vis/*.py` | 2026-06-27 |
| — | F-001 doc/script sweep (hdp.md, neuronal_tensor.md, hdp script, tutorial 07 equations) | 2026-06-27 |
| — | F-004 doc sweep (étude7, showcases, tutorial 07, changelog) | 2026-06-27 |
| — | F-003 doc sweep (fields.md, probes.md, tensor_field_workflows, v038 tutorial) | 2026-06-27 |
| — | Global skill sync script `skills/SYNC_GLOBAL.sh` + doctrine F-005 | 2026-06-27 |

---

## How to use

1. Before editing a skill, grep this file for the topic.
2. If you fix a friction in code or docs, update the row to `resolved` — do not erase history.
3. New frictions: append with next `F-0XX` ID.
