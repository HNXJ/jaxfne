# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

### P-035

- Date: 2026-10-18. Type: BUG. Area: docs build (Read the Docs). Severity: high (published docs
  stale since 2026-10-13).
- Observation: every Read the Docs build since 2026-10-14 01:46Z fails (builds 35131933 on
  413fb4a2 through 35176753 on b24e09ef, the v0.5.6 tag, on both `latest` and `stable`); the last
  success is 35110238 on ff89b24b, 2026-10-13.
- Reproduction: `python scripts/build_docs_assets.py --build` on the RTD image (Python 3.13.14,
  `pip install ".[viz]"`, CPU).
- Expected: rc 0. Actual: rc 1 at `docs/_static/etudes/hdp_controllability_reachability`:
  `condition-B population_rate_hz drift: 10.561121940612793 vs 10.62124252319336`, raised by the
  exact float equality at `scripts/rerun_etude_figures.py:548-552` against the frozen checkpoint.
  The tuned theta check just above it passes. Atlases 25 ok, the other 3 études ok.
- Evidence: RTD API v2 build logs (`/api/v2/build/<id>/`). Reproduced in WSL on CPU
  (`rerun_etude_figures.py --etude hdp_mcc3`: ff89b24b passes, 413fb4a2 gives the same 10.5611).
  `git bisect` over the 154 commits: first bad commit 5090af24 (D1, Hamm 2026-10-13: Izhikevich
  noise made dt-invariant, `noise_coef = noise_scale * sqrt(0.5 / dt_ms)`), an intended model
  change. The frozen digest `artifacts/runs/mcc3_10s_checkpoint/mcc3_10s_metrics_digest.json`
  holds the pre-D1 firing; the tuned theta still matches.
- Possible change: keep the pre-D1 digest as the historical record and pin a post-D1 digest
  beside it as a declared regeneration (reason: D1), which the generator checks; the docs panels
  then show the post-D1 run. The docs gate (`run_test_gate.py docs`) does not run the generators, so CI
  stays green while RTD fails.
