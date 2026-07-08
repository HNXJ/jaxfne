# Documentation Quality Report

**Branch:** `docs/auto-docs-20260706`  
**Base SHA:** `c9464ed6902cfe303275ebd238f88336dc9246e7`  
**Date:** 2026-07-06  
**Automation:** cron docs maintenance (`docs/auto-docs-YYYYMMDD`)

## Score

| Metric | Before | After |
|--------|--------|-------|
| **Overall docs quality** | 88/100 | **94/100** |
| mkdocs `--strict` | pass (with anchor INFO) | pass |
| Showcase figure coverage | 0/8 PNG | **8/8 PNG** |
| Broken internal anchors | 3 | **0** |

Deductions remaining (−6): orphan internal reports not in nav (−2), evidence
main figures still absent from `docs/evidence/` (−2), version string drift
mkdocs `0.4.5` vs package `0.4.4` (−1), some theory pages still use
machine-readable `truth` field names in API cross-refs (−1, acceptable in API docs).

## Commands run

```bash
git fetch --all --prune
git switch cur && git pull --ff-only origin cur
git switch -c docs/auto-docs-20260706
pip install -q -r docs/requirements.txt
pip install -q -e ".[dev]"
python3 -m compileall -q scripts
python3 scripts/evidence_inventory.py
python3 scripts/generate_showcase_figures.py
python3 -m mkdocs build --strict
```

## Broken links found and fixed

| File | Issue | Fix |
|------|-------|-----|
| `STDP_GLOBAL_SCALE_REPORT.md` | Anchor `#q2--reasonable-plasticity-scale-global_stdp` | `#q2-reasonable-plasticity-scale-global_stdp` |
| `STDP_REAL_TEST_REPORT.md` | Same Q2 anchor | Same fix |
| `api/fields.md` | Stale `validate_projection_invariants` anchor | Updated to mkdocs-generated heading slug |

## Stale API references found and fixed

- `api/fields.md` cross-link to validation API now matches current
  `validate_projection_invariants(*, sources, …)` signature slug.

## Missing docs addressed

- Added `docs/publication/docs_quality_report.md` (this file).
- Added **Proof and practice flows** section to `mathematical_glossary_flow.md`
  (linearity/superposition, source projection, CSD FD, conservation diagnostic,
  proxy-vs-solver ladder).
- Expanded `contributing.md` with validation command block and link here.
- Wired quality report into `mkdocs.yml` nav under About jaxfne.

## Figure issues

| Issue | Resolution |
|-------|------------|
| 8 showcase PNGs referenced by `guides/showcases.md` but absent | Created `scripts/generate_showcase_figures.py`; regenerated all 8 PNGs under `docs/assets/showcases/` |
| Doc-regeneration uses reduced network sizes vs prose (e.g. 300 n vs 10k for slow-deep homeostasis) | Documented here; titles remain proxy-safe; figures are real `construct`/`simulate` outputs |

**Figures added/updated (2026-07-06):**

- `homeostasis_rate_change_10s.png`
- `homeostasis_full_raster_10s.png`
- `plasticity_random_stim_stability.png`
- `plasticity_weight_distribution.png`
- `spectrolaminar_slow_homeostasis_suite.png`
- `spectrolaminar_depth_distribution_crossings.png`
- `spectrolaminar_suite_corrected.png`
- `spectrolaminar_absolute_power_1f_check.png`

## Wording issues fixed

| File | Change |
|------|--------|
| `api/index.md` | Heading `Scope & truth gates` → `Scope` |
| `guides/showcases.md` | `truth gates` → `scope defaults` in opener |
| `tensor_electromagnetics_scope.md` | `truth gate` → `scope metadata` in P-stage section |

## Unresolved blockers

1. **Evidence main figures** (`fig01`–`fig08`) still missing from `docs/evidence/`
   — tracked by `scripts/evidence_inventory.py`; separate from mkdocs nav.
2. **Orphan pages** (STDP/HDP reports, `colab.md`, `ci_policy.md`, etc.) remain
   outside nav by design; consider a "Internal reports" nav section or move to
   `.legacy/` in a future pass.
3. **mkdocs `extra.jaxfne_version`** (`0.4.5`) ≠ `pyproject.toml` / package
   (`0.4.4`) — docs-only version sync deferred (package change out of scope).
4. **`scripts/publication_inventory.py`** referenced in automation prompt but
   not present; use `scripts/evidence_inventory.py`.

## Next safe actions

1. Merge `docs/auto-docs-20260706` → `cur` after review.
2. Run full docs hygiene suite:
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_public_docs_hygiene.py -q`
3. Schedule evidence main-figure generation into `docs/evidence/` when release-facing.
4. Sync `mkdocs.yml` `jaxfne_version` with `pyproject.toml` in a version-bump PR.

## Acceptance checklist

- [x] `mkdocs build --strict` passes
- [x] No scientific overclaims added
- [x] No package API changes
- [x] Showcase figures exist and are referenced
- [x] No absolute local machine paths in tracked docs
- [x] `docs_quality_report.md` updated
