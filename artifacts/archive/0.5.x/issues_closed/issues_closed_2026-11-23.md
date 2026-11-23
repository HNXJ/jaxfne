# jaxfne closed issues, batch of 2026-11-23 (0.5.10)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`) after its resolution line.
Earlier closed entries: `issues_closed_2026-10-26.md`, `issues_closed_2026-10-27.md`,
`issues_closed_2026-11-03.md`, `issues_closed_2026-11-21.md` and `issues_closed_2026-11-22.md` beside this file,
and `../issues_closed.md`, `../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`, pinned write-once.

### P-068

- Date: 2026-11-23. Type: BUG (cause inferred). Area: etude rerun scripts, Dashboard figures job. Severity: major
  for the dashboard (G-4); none for the package.
- Observation: Dashboard run 45118760753 (dev 4773b4d9) fails in `scripts/rerun_etude_figures.py`: "r90 lfp_ref
  mismatch (geometry should be hash-exact)" (multiscale_observation) and "izh P_LFP centroid outside tolerance"
  (heterogeneous_emitters). Last green Dashboard: 022da7a0 (2026-11-19), before P-066.
- Inferred cause: P-066 made realized z mm; these scripts call `project_laminar_sources` without
  `column_height_mm` (`rerun_etude_figures.py:104,218`, `run_multiscale_observation_etude.py:206`,
  `run_heterogeneous_emitters_etude.py:219-220`, and other script sites).
- Repair (2026-11-23, `_temp_p068`): 9 script projection sites pass the model's `column_height_mm`; z used beside
  relative contact depths is divided by it (`observe` leadfield, `mean_r90`, hdp_suite2 rescale and panel-A z).
  Cause confirmed: with the height the heterogeneous izh P_LFP centroid misses by rel 6.4e-7 (tol 1e-3), without it
  by 1.46e-2. The multiscale r90 exact check misses by rel 1.9e-8 (float32 mm round trip); Hamm 2026-11-23: rel
  1e-6 (without the height rel 1.9e-2, still caught). Site build of both etudes: 0 errors. Review ACCEPT (minors
  fixed). Open: no unit test drives `_verify_multiscale_claims` (todo T-DASH-P068). Closes when the Dashboard
  workflow is green on dev.
- Resolution (2026-11-23): `tests/test_rerun_etude_claims_p068.py` drives `_verify_multiscale_claims` (pass with
  the height and at rel 5e-7; abort at height 1.0, rel 2e-6 and a NaN reference; each mutant fails it). Integrated
  to dev 10e2f85f (broad gate at fd470064: 8431 passed, 0 failed; review ACCEPT). Dashboard run 45224171653
  (workflow_dispatch, Hamm's go 2026-11-23) on dev 10e2f85f: publish and figures jobs success. Closed.
