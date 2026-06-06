# jaxfne docs audit & restructure plan (v0.3.30)

Read-only audit of `docs/` against live git (`main == dev == 33f99db`, `jaxfne 0.3.29`).
Goal: make the API docs **less verbose, more structured, table-driven**, and fix
what is orphaned, duplicated, stale, or undocumented.

Generated: 2026-06-04. Branch: `docs/docs-audit-v0330`.

---

## 0. Summary of findings

| # | Issue | Severity | Class |
|---|---|---|---|
| 1 | 22 docs not referenced in `mkdocs.yml` nav (orphaned) | high | PATCH |
| 2 | 7 root pages duplicated by `docs/guides/` copies (with inbound links) | high | PATCH |
| 3 | 98/146 public names undocumented in per-module API pages | high | PLAN |
| 4 | 11 public callables have no docstring at all | medium | PATCH (code) |
| 5 | `_KNOWN_METRICS` (private) leaks into `__all__` | medium | PATCH (code) |
| 6 | Old `api/index.md` had stale API refs + verbose prose | medium | ✅ DONE |
| 7 | No dedicated API page for optim/io/bridges/paradigm/sharding (43 names) | medium | PLAN |
| 8 | Tutorial files use colliding numeric prefixes (06/07/08 ×2-3) | low | FOLLOW-UP |
| 9 | Per-module API pages verbose (261–345 lines each) | medium | PLAN |

---

## 1. Orphaned docs (not in `mkdocs.yml` nav) — 22

These render to the site only via incoming links (or not at all). Decide per file:
**add to nav**, **fold into a nav'd page**, or **move to `internal_docs/`**.

| File | Likely action |
|---|---|
| `RELEASE_CHECKLIST.md` | move to `internal_docs/` (dev-facing) |
| `ci_policy.md` | move to `internal_docs/` |
| `colab.md` | add to nav (Getting started) |
| `computation_basis.md` | add to nav (Guides) |
| `conservation_proxy_diagnostics.md` | add to nav (Guides) |
| `interactive_visualizations.md` | merge into `guides/plotly_visualization.md` |
| `manuscript_alignment.md` | add to nav (About) |
| `mathematical_glossary_flow.md` | add to nav (Guides) |
| `performance_baseline.md` | add to nav (About) |
| `source_field_equations.md` | add to nav (Guides) |
| `tensor_network_ancestry.md` | add to nav (About) |
| `tutorial_figures.md` | merge into `tutorials/index.md` |
| `calibration.md`, `jaxley_interop.md`, `output_bundles.md`, `plotly_visualization.md`, `poisson_admissibility.md`, `probe_operators.md`, `tensor_field_workflows.md` | **duplicates — see §2** |
| `tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md` | add to Tutorials nav |
| `tutorials/10_v0313_omission_oddball.md` | add to Tutorials nav |
| `tutorials/tutorial_outputs.md` | merge into `tutorials/index.md` |

## 2. Duplicate root ↔ `guides/` pages — 7

`mkdocs.yml` nav points at the **`guides/`** copies; the root copies are stale
orphans that nonetheless differ slightly and **have inbound links**, so they
cannot just be deleted.

| Root (stale) | Canonical (nav) | Δ lines | Inbound links to root copy |
|---|---|---:|---|
| `docs/calibration.md` | `guides/calibration.md` | 16 | output_bundles, tensor_field_workflows, faq, notebook_standard |
| `docs/jaxley_interop.md` | `guides/jaxley_interop.md` | 14 | tensor_field_workflows, faq, tutorials/index |
| `docs/output_bundles.md` | `guides/output_bundles.md` | 2 | source_field_equations, faq, 01_single_neuron |
| `docs/plotly_visualization.md` | `guides/plotly_visualization.md` | 42 | (none) |
| `docs/poisson_admissibility.md` | `guides/poisson_admissibility.md` | 46 | manuscript_alignment, conservation_proxy_diagnostics |
| `docs/probe_operators.md` (53 L stub) | `guides/probe_operators.md` (336 L) | 359 | index, faq, scope, api/probes, api/index, many |
| `docs/tensor_field_workflows.md` | `guides/tensor_field_workflows.md` | 6 | plotly_visualization, jaxley_interop |

**Plan (PATCH `docs/dedupe-root-guides`):** for each pair, (a) confirm `guides/`
is the superset, (b) replace the root copy with a one-line stub linking to the
guide _or_ repoint inbound links to `guides/...` and delete the root copy, (c)
`mkdocs build --strict`. Do **not** bulk-delete — repoint links first.

## 3. API coverage gap

- `jaxfne.__all__` = **146** names; only **48** appear in `docs/api/*.md` → **98 undocumented** at the per-module level.
- The new `api/index.md` now lists **all 146** in tables (✅), but per-module pages still omit most.
- **11 public callables have no docstring** (so no summary anywhere):
  `configuration`, `construct`, `objective`, `runtime`, `runtime_report`,
  `simulation`, `paradigm`, `probe_laminar_modes`, `construct_source_tensor`,
  `project_sources_to_laminar_field`, `validate_projection_invariants`.
  → PATCH (code): add one-line docstrings; these are top-level public API.
- **`_KNOWN_METRICS`** (private, underscore) is exported in `__all__` → PATCH
  (code): drop from `__all__` (API-honesty; matches the loop-context backlog).

## 4. Missing module pages (43 names)

No dedicated `api/*.md` for: **Optimizers (`optim`, 15)**, **IO/receipts (7)**,
**Bridges/Jaxley (7)**, **Paradigms (4)**, **Sharding (4)**, **Tutorial utils (4)**,
**Experimental HPC (2)**. Plan: add `api/optim.md`, `api/io.md`, `api/bridges.md`,
`api/paradigms.md` (sharding/tutorial-utils/experimental can live in a single
`api/advanced.md`), each as a compact table page; add to nav.

## 5. Tutorial numbering collisions

`06_*`, `07_*`, `08_*` prefixes are each used by 2–3 unrelated files
(`06_jaxfne_suite_no_1…` vs `06_v036_100_neuron…`, etc.). Nav disambiguates by
title but the filenames are confusing. FOLLOW-UP: renumber into a single
monotonic sequence, or split "Suites" vs "Versioned walkthroughs" into
subdirectories.

## 6. Verbosity (the stated goal)

Per-module API pages: `core.md` 345, `probes.md` 344, `validation.md` 336,
`fields.md` 321, `runtime.md` 313, `emitters.md` 261, `objectives.md` 50.
**Target structure for each page:**
1. One-paragraph intro + truth-gate note.
2. **Symbol table** (Name · Kind · Signature · Summary) — the primary content.
3. A short "Detailed reference" section only for the 3–5 most-used symbols
   (full signature + one example), not every symbol.
4. Drop repeated scope boilerplate (link to `scope_and_limitations.md` once).

---

## Done in this pass (easy edits)

- ✅ **`docs/api/index.md` rewritten**: replaced 269 lines of verbose prose +
  stale API references (`config_status_boundary`, `model.compute_readout`,
  `gsdr().optimize()`) and links to orphaned root docs, with a compact
  module-map table, a **verified** minimal workflow, and a **complete 146-symbol
  index** grouped by module (auto-generated from `__all__`, undocumented names
  flagged). LIF/GLIF summaries corrected to "loud stub". `_KNOWN_METRICS` flagged.

## Proposed patch sequence (ranked, plan-before-each)

| Rank | Task | Class | Risk | Notes |
|---:|---|---|---|---|
| 1 | Per-module API pages → table format (§6) | PLAN→PATCH | low | docs only; one page per PR |
| 2 | Add missing module pages optim/io/bridges/paradigms (§4) | PLAN→PATCH | low | docs only |
| 3 | Dedupe root↔guides, repoint links (§2) | PATCH | med | link integrity; `--strict` gate |
| 4 | Nav reconciliation for orphans (§1) | PATCH | low | docs only |
| 5 | Add docstrings to 11 public callables (§3) | PATCH | low | **code** + docs |
| 6 | Drop `_KNOWN_METRICS` from `__all__` (§3) | PATCH | low | **code**; API-honesty |
| 7 | Tutorial renumbering (§5) | FOLLOW-UP | med | many inbound links |

Validation gate for every docs PR:
```bash
mkdocs build --strict
python scripts/audit_notebooks_and_assets.py --check   # if assets touched
```
Stop rules: stop if `--strict` reports a broken link; stop if a root-page delete
orphans an inbound link without a repoint; stop if any code docstring change
alters behavior (it shouldn't).
