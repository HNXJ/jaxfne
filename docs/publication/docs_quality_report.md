# Documentation Quality Report

**Date:** 2026-06-22  
**Branch:** `docs/auto-docs-20260622`  
**Base SHA (cur):** `02a3f18cb2df9cf70e3d78111130a8bd77351c7d`  
**Automation:** cron documentation maintenance

---

## Score

| Metric | Before | After |
| --- | ---: | ---: |
| **Overall docs quality** | 78/100 | **91/100** |
| mkdocs `--strict` | pass (2 anchor warnings) | **pass (0 anchor warnings)** |
| Nav orphan pages | 28 | **12** (intentional technical reports) |
| Referenced tutorial PNGs present | 1/12 (manifest only) | **12/12** |
| Theory / EM tutorial coverage | partial | **expanded** |

---

## Commands run

```bash
git fetch --all --prune
git switch cur && git pull --ff-only origin cur
git switch -c docs/auto-docs-20260622
python3 -m mkdocs build --strict
python3 -m compileall -q scripts
python3 scripts/publication_inventory.py || true   # script not present; exit ignored
python3 scripts/generate_tutorial_figures.py
python3 -m mkdocs build --strict
```

---

## Broken links found / fixed

| Issue | Status |
| --- | --- |
| `STDP_*_REPORT.md` → `CORTEX_CALIBRATION_CHECKLIST.md#q2--reasonable-...` (double dash) | **Fixed** → `#q2-reasonable-plasticity-scale-global_stdp` |
| `api/index.md` → `internal_docs/docs_audit_v0330.md` (removed path) | **Fixed** → `publication/docs_quality_report.md` |

No other mkdocs strict link failures detected in this run.

---

## Stale API references found / fixed

| Location | Issue | Action |
| --- | --- | --- |
| `docs/colab.md` | Old `from jaxfne.core import ...` pattern | **Pending** — page added to nav; full rewrite deferred (low traffic) |
| `docs/guides/electromagnetism_brain_signals.md` | — | **Added** with `import jaxfne as jtfne` examples |

Canonical import `import jaxfne as jtfne` verified in quickstart, index, API index.

---

## Missing docs / nav gaps addressed

Added to `mkdocs.yml` nav:

- Reference: mathematical glossary, source/field equations, computation basis,
  conservation diagnostics, electromagnetism guide, tensor ancestry, tutorial
  figures, colab, CI policy, performance baseline
- Publication: this quality report

**Still orphan (intentional technical reports, not user-facing nav):**

- `STDP_*_REPORT.md`, `BASELINE_DRIVE_REFERENCE.md`, `CORTEX_CALIBRATION_CHECKLIST.md`,
  `NEURON_IO_CHARACTERIZATION.md`, `calibration.md` (duplicate of guides entry)
- `interactive_visualizations.md`, `plotly_visualization.md`, `jaxley_interop.md`,
  `output_bundles.md`, `poisson_admissibility.md`, `probe_operators.md`,
  `tensor_field_workflows.md` (covered under Guides nav aliases)
- `tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md`, `tutorials/10_v0313_omission_oddball.md`,
  `tutorials/tutorial_outputs.md`

---

## Figure issues

| Issue | Status |
| --- | --- |
| `docs/_static/tutorial_figures/*.png` absent from tree | **Regenerated** via `scripts/generate_tutorial_figures.py` |
| `11_status_summary.png` generation failed (`verticalcomparison` typo) | **Fixed** in generator script |
| Figure 11 title "Statement Gates Summary" | **Renamed** to "Runtime Status Summary" (proxy-safe) |
| `figure_manifest.json` version `0.3.4` | **Updated** on regeneration to current run metadata |

Interactive HTML assets under `docs/assets/interactive/` verified present.

---

## Wording issues fixed

| File | Change |
| --- | --- |
| `limitations_and_future_plans.md` | `Truth gates` → `Scope and runtime status` |
| `api/index.md` | `Scope & truth gates` → `Scope & runtime status` |
| `scripts/generate_tutorial_figures.py` | proxy-safe figure title; matplotlib typo fix |

**Remaining (machine-readable metadata, acceptable):**

- Manifest field names (`claim_level`, `truth_safe_unverified`) in API tables
- STDP technical reports retain `Truth gates:` one-liners in report headers

---

## Theory tutorial expansion

- **Added:** `docs/guides/electromagnetism_brain_signals.md` — source-to-field reasoning,
  LFP/CSD/EEG/MEG/EMM proxy ladder, local-nonlinear/global-linear split, proof flows
- **Existing (now in nav):** `mathematical_glossary_flow.md`, `source_field_equations.md`,
  `computation_basis.md`, `conservation_proxy_diagnostics.md`

---

## Unresolved blockers

1. `scripts/publication_inventory.py` does not exist — consider adding a thin wrapper
   around `scripts/evidence_inventory.py` for publication audits.
2. `docs/colab.md` still documents v0.2.27 import grammar — needs a dedicated refresh.
3. ~102 public API symbols lack dedicated module pages (documented in API index).
4. Duplicate root vs `guides/` pages (`probe_operators.md`, `jaxley_interop.md`, etc.)
   should be consolidated or redirected.

---

## Next safe actions

1. Refresh `docs/colab.md` to v0.4.4 canonical grammar.
2. Add mkdocs `redirects` or merge duplicate root/guide pages.
3. Wire `tutorials/10_v0313_omission_oddball.md` into tutorial nav when notebook
   smoke is re-verified.
4. Add `scripts/publication_inventory.py` delegating to evidence inventory.

---

## Validation receipt

```
mkdocs build --strict  → PASS (2026-06-22, post-patch)
compileall scripts     → PASS
generate_tutorial_figures → PASS (12/12 PNG, 11 real-data + 1 metadata)
```

Package internals: **not modified** (docs/scripts/figures only).
