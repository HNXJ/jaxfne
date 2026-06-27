# jaxfne skills — friction & contradiction stack

**Purpose:** Durable ledger of known mismatches between skills, docs, code, and
external doctrine. Resolve items here before claiming a skill is authoritative.
**Do not delete rows** — mark `status: resolved` with SHA/date when fixed.

Last audited: 2026-06-27 (skills realignment pass).

---

## Open

| ID | Severity | Topic | Contradiction / friction | Authoritative source (today) | Suggested resolution |
|----|----------|-------|--------------------------|----------------------------|----------------------|
| F-001 | high | HDP E size default | `NeuronalTensor.DEFAULT_RELATIVE_SIZE` E=2.0 vs `emitters.py` `DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE` E=5.0 | Each module's own default until reconciled (`catalog-glossary-jaxfne` §1b) | Single public constant + migration note in `neuronal_tensor.py` / `emitters.py` |
| F-002 | high | Layer naming | 5-layer `L2/3` merged (`DEFAULT_LAYERS`, `laminar_cortex_config` examples) vs 6-layer `L1…L6` (`CANONICAL_LAYERS_6L`, `ei_profile="canonical"`) | `builders.py`: canonical E:I requires 6-layer set; 5-layer has merged fractions | Skills/docs must state **which layer set** per builder; add cross-link table in `jaxfne-cortical-column-default` |
| F-003 | medium | Projection semantics history | Older docs/skills said row-normalize was default; code default is `density_preserving` since proxy fix | `jaxfne/fields/proxy.py:120`, `tests/test_fields_projection_finite_and_normalization.py` | Resolved in skills 2026-06-27; grep repo for stale "row-normalize default" prose |
| F-004 | medium | Spectrolaminar crossover | Old claim: crossover "scale-emergent at 10k". Verified: crossover needs **oscillatory regime**, not N alone | `jaxfne-spectrolaminar-suite/SKILL.md`, etude sweeps | Resolved in `jaxfne-cortical-column-default` 2026-06-27; verify notebooks don't repeat scale-emergent |
| F-005 | medium | Global vs repo doctrine gates | Workspace `AGENTS.md` (Gamma) JAX §11 still cites `field_solver_status = "laminar_proxy_no_pde"` | Repo + catalog: `linear_solver`; `laminar_proxy_no_pde` RETIRED | Update global `AGENTS.md` JAX section outside this repo |
| F-006 | medium | Dual skill trees | `skills/` (canonical) vs `jaxfne/skills/` (older numbering) | `AGENTS.md` → `skills/` only | Deprecate `jaxfne/skills/` or delete after one release; grep for imports |
| F-007 | low | Config fluent API surface | `Configuration()` bare constructor vs builder-first (`laminar_cortex_config`, `build_laminar_column`) | Both exist; builders preferred in tutorials | `jaxfne-configuration-fluent-api` documents **verified** methods only |
| F-008 | low | Objective composition | `Objective.compose()` exists; no top-level `jtfne.band_power` / `phase_locking` | `core.py` Objective, `objectives.py`, `Model.tune` | Keep objective skill on verified `rate_targets`, `readout_spec`, `Model.tune` |
| F-009 | low | Signals trial axis | `Signals.get(..., trial=)` raises `NotImplementedError` | `core.py` Signals.get docstring | Document in signals skill; use `run_trials` / tutorial_utils for multi-trial |
| F-010 | low | Worker skill list cardinality | Router once listed "7 active skills" vs README 12+ | `skills/README.md` | Resolved in worker-router 2026-06-27 |
| F-011 | low | `column_density_table` / `layer_celltype_count_table` | Catalog lists them; may raise on raw `Configuration` | Use `construct(cfg).neuron_table()` for counts (cortical-column skill) | Implement tables or fence with explicit error in docs |
| F-012 | low | Homeostasis `k_gain` vs `eta` | `k_gain` is one-sided damper, not bidirectional rate setpoint | `AGENTS.md` homeostasis section | Cross-link from modeling-optimization skill |
| F-013 | low | Plasticity word overload | Three mechanisms share "plasticity" (declarative, homeostasis eta, STDP stream) | `AGENTS.md` | Add one-line disambiguation to `catalog-glossary-jaxfne` §5 if agents confuse |
| F-015 | low | Objective evaluation | Evaluation is `model.evaluate(signals, objective)`, not `objective.evaluate(signals)` | `core.py` Model.evaluate | Fixed in objective/signals skills 2026-06-27; move to Resolved when verified in tutorials |

---

## Resolved (this pass)

| ID | Resolution | Date |
|----|------------|------|
| F-003 | Skills updated: default `density_preserving`; row_normalize opt-in | 2026-06-27 |
| F-004 | `jaxfne-cortical-column-default/SKILL.md` points to regime-based crossover | 2026-06-27 |
| F-010 | Worker router skill list aligned with `skills/README.md` | 2026-06-27 |
| — | Extensionless Python "skills" replaced with verified `SKILL.md` files | 2026-06-27 |
| — | `jaxfne-visualization-schema` module map matches `jaxfne/vis/*.py` | 2026-06-27 |
| — | Repo `catalog-glossary-jaxfne` synced (NeuronalTensor, HDP, projection default) | 2026-06-27 |
| — | `AGENTS.md` laminar-path table expanded to three columns | 2026-06-27 |

---

## How to use

1. Before editing a skill, grep this file for the topic.
2. If you fix a friction in code or docs, update the row to `resolved` — do not erase history.
3. New frictions: append with next `F-0XX` ID.
