# jaxfne skills — friction & contradiction stack

**Purpose:** Durable ledger of known mismatches between skills, docs, code, and
external doctrine. Resolve items here before claiming a skill is authoritative.
**Do not delete rows** — mark `status: resolved` with SHA/date when fixed.

Last audited: 2026-06-29 (HDP v2 patch blocking — rho_passive creates regime bifurcation, K_ctrl restored).

---

## Open

| ID | Severity | Topic | Contradiction / friction | Authoritative source (today) | Suggested resolution |
|----|----------|-------|--------------------------|----------------------------|----------------------|
| F-007 | low | Config fluent API surface | `Configuration()` bare constructor vs builder-first | Both exist; builders preferred | Documented in `jaxfne-config` (merged 2026-06-30 from `jaxfne-configuration-fluent-api`) — keep builder-first in tutorials |
| F-008 | low | Objective composition | No top-level `jtfne.band_power` / `phase_locking` | `Objective`, `Model.tune` | Use verified objective skill APIs only |
| F-009 | low | Signals trial axis | `Signals.get(..., trial=)` raises `NotImplementedError` | `core.py` Signals.get | Use `run_trials` / tutorial_utils for multi-trial |
| F-016 | low | `export.save_figure` deprecation | `export.save_figure` has no `DeprecationWarning`; audit recommends routing to `vis.export_figure` | `jaxfne/export.py` | Emit `warnings.warn(..., DeprecationWarning)` in `save_figure`; update any tutorial that calls it |
| F-018 | low | HDP v2 sign orientation undocumented at API level | `signed_linear` and `signed_quadratic` use `H_post - H_pre` (flipped from naive spec `H_pre - H_post`) to preserve postsynaptic-indexing invariant; flip is noted in inline comments but not in any skill or top-level API doc | `jaxfne/emitters.py` line 1332 | Add one-paragraph note to the HDP section of `AGENTS.md` explaining the postsynaptic-indexing convention and the direction of each rule |

---

## Resolved (this pass)

| ID | Resolution | Date |
|----|------------|------|
| F-017 | STILL BLOCKED, corrected 2026-07-01: rho_passive/H^2 mechanism fails a full 20s/5-seed geomspace(0.005,2.0) sweep at every candidate (`scripts/hdp_v2_rho_sweep.py --preset default`, 75/75 rows fail H_std<=0.05; script's own verdict: "NO candidate passed all seeds"). Prior text here ("K_ctrl=5.0/0.15 restored as canonical restoring term") does NOT match the code: `K_ctrl_arr` (jaxfne/emitters.py:1247) is computed and never used in dH/dt -- confirmed dead code, not a restoring term. DEFAULT_HDP/DEFAULT_HDP_DESYNC currently have NO working restoring mechanism (K_ctrl inert; rho_passive=0.0 inert-by-default and unfixable within the swept range). Requires formula redesign (F-019, still not its own ledger row) | 2026-06-29, corrected 2026-07-01 |
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
