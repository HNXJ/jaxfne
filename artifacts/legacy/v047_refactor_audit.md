# jaxfne v0.4.7 Repository Refactor Audit

**Generated:** 2026-06-28  
**Branch:** `cur`  
**SHA:** `a16b0ea` (pre-refactor baseline; see git log for refactor commits)  
**Package version:** 0.4.4  

This document is the canonical deliverable for the pre-v0.4.7 architectural refactor mission.

---

## 1. Repository Audit Summary

### Scale

| Area | Files | LOC (approx.) | Notes |
|------|------:|--------------:|-------|
| `jaxfne/` package | 79 `.py` | ~34,600 | `core.py` alone is 8,370 lines (25%) |
| `jaxfne/vis/` | 37 `.py` | ~10,470 | Matplotlib + Plotly + canonical dispatcher |
| `tests/` | 200+ | — | Primary contract surface |
| `docs/` | 80+ `.md` | — | mkdocs strict build |
| `tutorials/` | 51 | — | Notebooks + études |
| `examples/` | 61 | — | Headless CI runners |
| `.legacy/` | 110 tracked | ~1.9 MB | Former `internal_docs/`, old notebooks |
| `scripts/` | 56 → 43 | — | 13 orphan scripts removed in this pass |
| `skills/` | 13 | — | Repo source-of-truth for agent skills |

### Classification of Issues Found

| Category | Count | Severity |
|----------|------:|----------|
| Monolithic modules (`core.py`, `emitters.py`, `tutorial_utils.py`) | 3 | High — maintenance cost |
| Duplicate spectrolaminar implementations (NumPy vs JAX) | 2 | Medium |
| Stale `.legacy/notebooks/` doc links | 6 pages | Medium — fixed in this pass |
| Orphan / broken scripts | 13 | Low — deleted in this pass |
| Tracked generated artifacts (`scripts/out/`) | 4 files, 5.6 MB | Low — untracked in this pass |
| mkdocs nav gaps | 7 pages | Medium — partially fixed |
| Superseded examples (`03_*`, `04_*`) still in CI | 2 | Medium — removed from CI lane |
| Internal reports in `docs/` (`HDP_REPORT`, `STDP_*`) | 7 | Low — relocate next |
| `releases/` vs `changelog.md` duplication | 6 pages | Low — merge next |
| `_RuntimeModuleWrapper` fragility | 1 | High — deferred (needs regression suite) |
| `experimental_hpc/` shadow API (TBI) | 716 LOC | Low — keep fenced |

---

## 2. Architecture Improvements (Recommended & Applied)

### Applied in this refactor pass

1. **Script hygiene** — Removed 13 orphan one-shot / broken scripts (v0.3 smoke, notebook builders, ad-hoc probes).
2. **Artifact hygiene** — `scripts/out/` gitignored; 5.6 MB generated HTML/PNG untracked.
3. **Documentation discoverability** — Fixed Colab links from `.legacy/notebooks/` → `tutorials/` canonical paths.
4. **mkdocs nav** — Added Suite 2 evoked L4, v0.3.13 omission, and missing API pages (`neuronal_tensor`, `plasticity`, `solvers`, `sharding`).
5. **CI deduplication** — Removed superseded `examples/03_single_neuron_multimodal_probe.py` and `examples/04_two_neuron_ei_multimodal.py` from example smoke loop (`v031`/`v033` remain).
6. **Tutorial index** — Updated quick-start snippet to current `suite2_four_celltype_config` + `simulate()` API.

### Recommended for v0.4.7 (not yet applied — needs sign-off)

| Change | Rationale | Risk |
|--------|-----------|------|
| Split `core.py` into `config.py`, `model.py`, `signals.py`, `simulate.py` | 8.8k LOC monolith | High — import graph + test churn |
| Unify spectrolaminar PSD (`fields/proxy.py` NumPy vs `analysis/spectral.py` JAX) | Two shapes, two backends | Medium — API deprecation path |
| Remove `export.py` shim → `vis.exporters` only | Deprecated since v0.3.x | Low — keep one release compat |
| Move `docs/HDP_REPORT.md`, `STDP_*_REPORT.md` to `scripts/evidence_figures/` or `.legacy/` | Not user docs | Low |
| Merge `docs/releases/v0.2.*.md` into `changelog.md` anchors | Single release history | Low — need redirect stubs for strict build |
| Rename `docs/tutorials_v030/` → `docs/assets/v030_figures/` | Misleading name (0 md files) | Medium — update v031–v033 script paths |
| Delete `.legacy/` after test migration | 110 tracked files, 1.9 MB | High — 6+ tests still anchor `.legacy/notebooks/` |
| Refactor `_RuntimeModuleWrapper` | Known DELTA fragility | High — needs dedicated regression tests |
| Archive `examples/03_*`, `examples/04_*` | Superseded by v031/v033 | Medium — version-specific tests remain |

---

## 3. Files Removed

### Scripts (13)

- `scripts/run_v030_tutorial_smoke.py` — broken paths (`examples/tutorials/v030_*` missing)
- `scripts/collect_v030_tutorial_manifests.py`
- `scripts/audit_v030_docs_links.py`
- `scripts/inject_agsdr_notebook_cells.py`
- `scripts/build_etude3.py`, `build_etude1_v2.py`, `build_tcm_v1.py`
- `scripts/run_etude_no1_colab_snapshot.py`
- `scripts/generate_v034_plasticity_reports.py`, `generate_v035_coop_reports.py`
- `scripts/benchmark_v0341_kernels.py`
- `scripts/import_cost_report.py`, `score_all_functions.py`
- `benchmarks/bench_vectorized_paths.py`

### Git-untracked artifacts (4)

- `scripts/out/cortex_1000_cylinder_3d.html` (~5.0 MB)
- `scripts/out/spectrolaminar_drive_sweep_boost_{1,3,5}p0.png`

---

## 4. Files Merged / Consolidated

No file merges in this pass (documentation content merges deferred). Planned merges:

| Source | Target |
|--------|--------|
| `docs/releases/v0.2.*.md`, `v0.3.4.md` | `docs/changelog.md` anchored sections |
| `docs/CORTEX_CALIBRATION_CHECKLIST.md`, `BASELINE_DRIVE_REFERENCE.md` | `docs/guides/calibration.md` appendices |
| `fields/proxy.py:spectrolaminar_psd` | Delegate to `analysis/spectral.py` with shape adapter |

---

## 5. APIs Simplified

### Current public surface

- **~120 symbols** in `jaxfne.__all__`
- **218 entries** in generated `docs/_generated/operator_inventory.md`

### Simplification opportunities

| API | Action |
|-----|--------|
| `export.save_figure` | Deprecate → `vis.export_figure` (already warned) |
| `fields/proxy.spectrolaminar_objective` | Legacy alias → `objectives.spectrolaminar_objective` |
| `GLIFEmitter`, `LIFEmitter` | Stub raises — consider removing from `__all__` or `experimental` namespace |
| `write_nwb` / `read_nwb` | Stub — keep lazy, document as not implemented |
| `experimental_hpc.*` | Fence behind explicit import; remove from root `__all__` if not wired |
| `tutorial_utils.build_laminar_column` | Keep (tutorial-path); document vs `builders.build_laminar_column` |
| `Configuration.*` backward-compat aliases | Keep through v0.4.x; audit for v0.5 removal |

### Dual laminar paths (intentional — do not merge)

| Path | Returns | Use when |
|------|---------|----------|
| Config-path | `Model` / `Signals` | AGSDR, homeostasis, per-neuron drive |
| tutorial_utils-path | `dict` | Multi-trial spectrolaminar sweeps |
| NeuronalTensor-path | `Model` / `Signals` | Declarative JSON tensors, HDP via `RuntimeConfig` |

---

## 6. Documentation Consolidation Summary

### Fixed

- Colab links: `01`–`03` tutorials, `notebook_standard.md`, `index.md` running section
- Étude 11 Colab badge → `jaxfne_etude_no_1_base.ipynb`
- Mislabeled `10_v0313_omission_oddball.md` title (was "Suite No. 3")
- mkdocs nav gaps for API and tutorial pages

### Remaining duplication (one canonical location each)

| Concept | Canonical | Demote / merge from |
|---------|-----------|---------------------|
| Scope / truth gates | `limitations_and_future_plans.md` | Tutorial "Coverage Boundary" blocks, `showcases.md` opener |
| Probe operators | `guides/probe_operators.md` | Per-tutorial operator prose |
| Pipeline grammar | `guides/configuration_grammar.md` + `objective_grammar.md` | `tensor_field_workflows.md` duplicate diagrams |
| Homeostasis vs HDP | `guides/homeostasis.md`, `guides/hdp.md` | `HDP_REPORT.md` |
| NeuronalTensor | `api/neuronal_tensor.md` | `migration_guide.md` code blocks |
| Release history | `changelog.md` | `docs/releases/v0.2.*.md` |

### `docs/tutorials_v030/`

Artifact store only (12 PNGs, JSON manifests). Not in nav. Rename to `docs/assets/v030_figures/` in a follow-up (updates v031–v033 example output paths).

---

## 7. Performance and Maintainability Improvements

| Improvement | Impact |
|-------------|--------|
| Remove 5.6 MB tracked HTML/PNG | Faster clones, smaller repo |
| Remove 13 orphan scripts | Less navigation noise |
| CI drops duplicate example runs | ~6 min saved per matrix cell (03+04) |
| Lazy `vis` via `_RuntimeModuleWrapper` | Preserved — `import jaxfne` avoids matplotlib |
| `core.py` split (deferred) | Would improve import-time for subset users |
| `experimental_hpc` not in hot path | Only `NodeIdentity`/`SelectorSpec` imported by `core` |

### Import weight (current)

`import jaxfne` eagerly loads: `core`, `emitters`, `builders`, `bridges`, `analysis`, `optim`, `neuronal_tensor`. Matplotlib/plotly deferred via wrapper.

---

## 8. Remaining Technical Debt

### P0 (blocks clean v0.4.7 architecture)

1. **`core.py` monolith** (8,370 LOC) — split behind compatibility re-exports.
2. **`_RuntimeModuleWrapper`** — replace with explicit `jaxfne.runtime` submodule + `from jaxfne import runtime` function alias pattern.
3. **`.legacy/` migration** — 6 test files + doc links still anchor `.legacy/notebooks/`.

### P1 (quality / consistency)

4. Dual spectrolaminar PSD implementations.
5. `docs/releases/` merge into changelog.
6. Internal reports (`HDP_REPORT`, `STDP_*`) out of user docs tree.
7. Superseded `examples/03_*`, `04_*` archive + test migration to v031/v033.
8. `docs/tutorials_v030/` rename + path updates in v031–v033 scripts.

### P2 (nice-to-have)

9. `vis/` consolidation — reduce matplotlib/plotly/canonical triple layering where plots are identical.
10. `tutorial_utils.py` (2,023 LOC) — extract spectrolaminar trial pipeline to `analysis/` or dedicated module.
11. Global agent skill installs (`jaxfne-harden-*`) drift vs repo `skills/` flat rules.
12. Version bump 0.4.4 → 0.4.7 across `pyproject.toml`, `__init__.py`, `mkdocs.yml`.

### Known fragilities (from AGENTS.md)

- `_CONFIG_RUNTIME_WARNINGS` global in `core.py` (not thread-safe)
- Hardcoded 20.0 spike gain in source proxy (dense + edge sync)

---

## 9. Validation Report

Run after applying this refactor:

```bash
python3 scripts/evidence_inventory.py
python3 -m compileall -q jaxfne tests examples scripts
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/ -q --tb=line
```

### Acceptance criteria for v0.4.7 release

- [ ] All gates above pass
- [ ] No tracked generated artifacts under `scripts/out/`
- [ ] Zero doc links to `.legacy/notebooks/` in published tutorials (tests may retain until migrated)
- [ ] `operator_inventory.md` regenerated
- [x] Version aligned: `pyproject.toml` == `jaxfne.__version__` == `mkdocs extra.jaxfne_version` (verified 2026-07-03: all three are `0.4.4`)
- [ ] Public API changes documented in `changelog.md`

---

## Guiding Principles Applied

- **Preserve behavior** — No package API changes in this pass; CI example list trimmed only for duplicates.
- **Prefer delete over move** — Orphan scripts deleted, not relocated.
- **One canonical location** — Doc links point to `tutorials/`, not `.legacy/`.
- **Root freeze respected** — No new top-level folders; changes inside existing trees only.

---

*Next safe action:* Run full pytest + mkdocs strict on `cur`, then open a PR with this audit as the refactor receipt.
