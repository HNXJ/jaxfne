# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

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
