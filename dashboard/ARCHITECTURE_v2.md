# Replication dashboard v2: architecture and model inventory (approved by Hamm, 2026-10-12)

Extends the live `/dashboard/` (rules in `DESIGN.md` stay: no verdict computed in the page, failures
as prominent as passes, every number linked to a committed source at a SHA, one generated data
file with a `--check` gate, static files, Plotly pinned). Hamm approved the status definitions,
the demonstrations 3–6 and the new layout on 2026-10-12.

## Status classes (approved; the classification is Hamm's)

| Class | Definition | Evidence required |
|---|---|---|
| Verified replication | every primary criterion of a signed preregistration passes, run through the jaxfne library (B1) | scorer JSON at a SHA, ledger row, library-path receipt |
| Partial replication | a signed preregistration is scored and some criteria pass, or all pass on a standalone port only | scorer JSON, ledger |
| Not reproduced | a signed preregistration is scored and its verdict is a fail (shown with the same prominence, B2) | scorer JSON, ledger |
| Demonstration | a published model runs in jaxfne without a scored numerical comparison to the paper | code path, run command |

The class sits beside the preregistered verdict, which always stays visible.

## Inventory, ranked by readiness

| Rank | Model | Class (proposed) | What exists | Missing for the page |
|---|---|---|---|---|
| 1 | CAMINOS Fig. 1 (Tahvili, Vinck, di Volo 2026, PLoS Comput Biol) | partial: PASS under v7 on the standalone port (`l2-caminos` 1e768f9d); library path (P6) in build | prereg v6 + v7, scores, ledger with paper values beside ours (≈35 Hz vs 33.8 Hz; ≈8.5 ms vs 7.04 ms), sensitivity S1–S5, seeds 1–5, command | equations block, P6 result, a static figure |
| 2 | PCL Fig. 3 (N'dri et al. 2025, Nat Commun) | not reproduced: T1, T3 pass; T2, T4 fail (`pcl-port` 5640ce18) | prereg v4, scores, ledger, unscored diagnostic | equations block, static figure; paper criteria are qualitative, so no paper numbers |
| 3 | PCL Fig. 2 (same paper) | demonstration: P0 PASS and K0b PASS against our own gates; the paper's Fig. 2d values were not available for a numeric comparison | `artifacts/etudes/pcl/README.md`, `fig2*.json`, library kernel route for K0b | a scored comparison would need the paper's numbers |
| 4 | Hodgkin-Huxley reference vs reduced neuron (AT-01) | demonstration (reference check): rate passes; v_peak, v_rest, first-spike time fail | `artifacts/atlas/results/at01_055.json`, `at_reduction_055.json`, library bridge | an overall verdict is not recorded; no seed |
| 5 | Q-learning on chain5/corridor7 vs analytic Q* | demonstration | `artifacts/etudes/qlearning/`, tests, seeds 0–4, command | no paper citation; no verdict word recorded |
| 6 | Izhikevich 2003, Figs. 2 and 3 | in progress (P3) | `artifacts/etudes/izhikevich2003/` P1 source, P2 port spec; library presets in `jaxfne/presets.py` | signed prereg, P4 build, P5 score |

Planned and unimplemented (roadmap tab only): Brunel 2000 and Vogels et al. 2011 (0.5.7), PCL Fig. 4
(0.5.7), Potjans and Diesmann 2014 and Mejias et al. 2016 (0.5.8), PCL Figs. 5–6 (0.5.9), Mazzoni
et al. 2015 (0.5.10); held P0 rows listed in `artifacts/etudes/literature_p0_candidates.md`.

Not shown: études with no published target (heterogeneous emitters, multiscale observation,
experiment A, HDP balance and controllability, theory F1/F2/F4). They test our own claims.

## Pages (hash-routed tabs, JNWB look: dark theme, build-SHA stamp, KPI cards)

1. **Overview**: KPI cards (models per class, criteria passed / failed / pending, last scored run,
   dev SHA); one card per model with class chip and verdict chip.
2. **Model** (`#model/<id>`), sections in this order: publication and scientific question;
   formulation (equations in KaTeX, parameters table with units and source table numbers);
   implementation (route: library or standalone port; code links at the lane SHA); original vs
   replicated (table with paper value, our value, criterion, verdict; Plotly charts, each with a
   static PNG and a data table fallback); agreement, deviations and limitations (verbatim from the
   ledger, never paraphrased into a stronger claim); reproduce (command, seeds, environment,
   receipts path).
3. **Validation**: every criterion across models in one sortable table, filter by verdict.
4. **Roadmap**: planned models by release, with their stage P0–P7.
5. **Method**: the class definitions, the rules above, links to preregistrations.

## Data and build

- `dashboard/sources.json` gains schema `jaxfne-replications/2`: per model `class`, `question`,
  `formulation` (markdown + TeX, written once per model and reviewed), `implementation`
  (`route`, `paths`), `reproduce` (`command`, `seeds`, `environment`), `original_values`. Scores
  are still copied from lane JSONs at a SHA by `scripts/build_dashboard_data.py`; `--check`
  stays in the Dashboard workflow.
- Static figures: `scripts/build_dashboard_figures.py` renders each chart spec to PNG from the
  same data (matplotlib), committed under `dashboard/figures/`, with its own `--check`.
- Libraries: Plotly 2.35.2 (already pinned) and KaTeX, pinned, from jsdelivr. No framework, no
  build step. Deploy: the existing `dashboard.yml` workflow on dev push.

## Order of work after approval

1. Schema v2 and generator, with tests (worker), CAMINOS and PCL Fig. 3 filled first.
2. Page shell with tabs and the model page (opencode or worker), QA at 1440 px and 390 px.
3. Static figures script.
4. Demonstrations 3–6 added as sources entries.
5. Deploy by the workflow; Hamm approved the first publish of the new layout (2026-10-12).
