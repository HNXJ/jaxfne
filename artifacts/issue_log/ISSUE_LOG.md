# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

### P-032
- **date:** 2026-10-04
- **type:** BUG (nondeterministic tests)
- **area:** `tests/test_analysis_metrics.py:28,48,62,75-82,94+`
- **observation:** unseeded `np.random.rand` with threshold asserts (`>0.9`, tight bands); only some tests set `np.random.seed`
- **severity:** MAJOR (flake-prone; a passing run proves little)
- **minimal reproduction:** read the cited lines; run the module twice and compare
- **expected behavior:** seeded `Generator` per stochastic test, or distribution-free asserts
- **actual behavior:** unseeded randomness under tight asserts
- **evidence:** code sweep 2026-10-04 (`artifacts/audit/sweep_2026-10-04_code.md`)
- **possible future change:** seed + re-run; open

### P-033
- **date:** 2026-10-04
- **type:** DOC (overclaims, stale versions, unit leaks)
- **area:** `docs/` (batch from prose sweep)
- **observation:** `tutorial_figures.md:3` "current development tree: 0.4.8" (stale); `05_v1_pfc_dual_column.md:131` "shows stable, long-term homeostatic adaptation" (overclaim vs evidence); `plotly_visualization.md:24` 0.4.8 test label; `04_simulate_tensor.md:16` untraced ~215k; `api/probes.md:316` "400 µm depth" (absolute units without calibration); `ci_policy.md:57` "proves exhaustiveness"; plus slop stack (S1-S3)
- **severity:** MAJOR (P1/P2) + minors (rest)
- **minimal reproduction:** read the cited lines
- **expected behavior:** versions current, claims at evidence strength, relative units unless calibrated
- **actual behavior:** as observed
- **evidence:** prose sweep 2026-10-04 (`artifacts/audit/sweep_2026-10-04_prose_visual_hygiene.md`)
- **possible future change:** D0-style batch fix + doc gates; open

### P-034
- **date:** 2026-10-04
- **type:** DOC (figure rendering defects, generator-side)
- **area:** `artifacts/figures/publication/` + figure generators
- **observation:** fig01 panel-A title/annotation collision + legend/x-axis overlap; fig06 legend-over-data, overprinted subtitles, truncated inset text; fig03 y-labels missing "(rel. units)"; e2_fig08 rep-label stacking, inset occlusion, callouts crossing bars; sub-6pt micro-type in three figures
- **severity:** MAJOR (V1-V4) + minor (V5)
- **minimal reproduction:** read the PNGs at full size (worker-observed)
- **expected behavior:** generator-side fixes; CONSTRAINT: fig01/fig03/fig06 are the frozen Figure 1–7 snapshot — files stay untouched, fixes land in generators + future figures only
- **actual behavior:** as observed
- **evidence:** visual sweep 2026-10-04 (`artifacts/audit/sweep_2026-10-04_prose_visual_hygiene.md`)
- **possible future change:** generator fixes + re-render of non-frozen figures; open
