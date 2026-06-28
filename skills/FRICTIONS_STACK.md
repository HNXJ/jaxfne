# jaxfne skills — friction & contradiction stack

**Purpose:** Durable ledger of known mismatches between skills, docs, code, and
external doctrine. Resolve items here before claiming a skill is authoritative.
**Do not delete rows** — mark `status: resolved` with SHA/date when fixed.

Last audited: 2026-06-27 (skills realignment pass).

---

## Open

| ID | Severity | Topic | Contradiction / friction | Authoritative source (today) | Suggested resolution |
|----|----------|-------|--------------------------|----------------------------|----------------------|
| F-007 | low | Config fluent API surface | `Configuration()` bare constructor vs builder-first | Both exist; builders preferred | Documented in `jaxfne-configuration-fluent-api` — keep builder-first in tutorials |
| F-008 | low | Objective composition | No top-level `jtfne.band_power` / `phase_locking` | `Objective`, `Model.tune` | Use verified objective skill APIs only |
| F-009 | low | Signals trial axis | `Signals.get(..., trial=)` raises `NotImplementedError` | `core.py` Signals.get | Use `run_trials` / tutorial_utils for multi-trial |
| F-014 | low | v038 projection pedagogy | ~~Notebook equation text describes row-normalized kernel without label~~ | Default API: `density_preserving` | **Resolved 2026-06-27** — manifest notes pedagogy vs package default |

---

## Resolved (this pass)

| ID | Resolution | Date |
|----|------------|------|
| F-001 | HDP E size: code + docs aligned to `DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE` | 2026-06-27 |
| F-002 | Layer naming table in `AGENTS.md` + cortical-column skill | 2026-06-27 |
| F-005 | Global `AGENTS.md` JAX §11 gate strings updated | 2026-06-27 |
| F-006 | `jaxfne/skills/` removed | 2026-06-27 |
| F-010 | Worker router skill list aligned | 2026-06-27 |
| F-011 | `layer_celltype_count_table` / `column_density_table` implemented in `builders.py` | 2026-06-27 |
| F-012 | Homeostasis disambiguation in modeling skill | 2026-06-27 |
| F-013 | Plasticity overload in catalog | 2026-06-27 |
| F-015 | Objective evaluation path in skills | 2026-06-27 |
| — | Extensionless Python "skills" replaced with verified `SKILL.md` files | 2026-06-27 |
| — | `jaxfne-visualization-schema` module map matches `jaxfne/vis/*.py` | 2026-06-27 |
| — | F-001 doc/script sweep (hdp.md, neuronal_tensor.md, hdp script, tutorial 07 equations) | 2026-06-27 |
| — | F-004 doc sweep (étude7, showcases, tutorial 07, changelog) | 2026-06-27 |
| — | F-003 doc sweep (fields.md, probes.md, tensor_field_workflows, v038 tutorial) | 2026-06-27 |
| — | Global skill sync script `skills/SYNC_GLOBAL.sh` + doctrine F-005 | 2026-06-27 |

---

## How to use

1. Before editing a skill, grep this file for the topic.
2. If you fix a friction in code or docs, update the row to `resolved` — do not erase history.
3. New frictions: append with next `F-0XX` ID.
