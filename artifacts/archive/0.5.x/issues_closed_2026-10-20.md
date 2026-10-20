# jaxfne closed issues, batch of 2026-10-20

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`). Earlier closed
entries: `issues_closed.md`, pinned write-once beside this file.

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
- Fix 2026-10-18: `mcc3_10s_post_d1_digest.json` (condition B after D1: population 10.5611 Hz,
  1063 spikes; theta unchanged) beside the pre-D1 digest; `mode_hdp_mcc3` checks theta against
  the frozen digest and condition B against the post-D1 one. Moves to closed when a Read the
  Docs build succeeds.
- Closed 2026-10-19: on main by PR 110 (merge cf46d6d3, fix-only, Hamm 2026-10-18); Read the
  Docs `latest` built green on cf46d6d3; `stable` follows the next tag. Stays here until the
  hash-pinned `archive/0.5.x/issues_closed.md` is re-pinned with its next batch.
- **closed:** 2026-10-20 (agent): entry moved from the open log.

### P-036

- Date: 2026-10-18. Type: BUG. Area: core public API. Severity: high.
- `Simulation.plasticity` (`jaxfne/_signals.py:87`, `with_plasticity` :147) is accepted, never
  refused, and only copied into metadata (`_model_simulate.py:1271` `plasticity_gain`): a
  stored-but-unconsumed parameter (P-014 class).
- `Model.tune` fall-through (`jaxfne/_model_tune.py:404-422`) returns
  `acceptance_decision: "ACCEPT_CANDIDATE"` with `best_parameters={}`, `best_score=inf`,
  `history=[]`; `tests/test_optim_tune.py:229` accepts either verdict.
- `generations or 8`, `population_size or 6` (`_model_tune.py:228-229`) turn an explicit 0 into
  the default; seed 0 likewise is replaced (`:220-222`).
- `RuntimeConfig.backend="gpu"/"tpu"` on a host without it runs on the default device
  (`_runtime_config.py:310-319`); only `runtime_report` says `backend_enforced=False`.
- Expected: each refused or consumed. Possible change: refuse non-zero `plasticity` (or wire
  it), make the fall-through REVISE, `is not None` defaults, refuse an unavailable backend.
- Fixed 2026-10-18: the fall-through reports REVISE; explicit `generations`/`population_size`
  below 1 raise (`tests/test_tune_explicit_values_p036.py`, 4 tests fail on the old code).
  Plasticity label and backend downgrade are documented designs: Hamm's call (O-47).
- Closed 2026-10-19 by Hamm's ruling of 2026-10-18: non-zero `Simulation(plasticity=)` and an
  absent gpu/tpu backend are refused (fragments `simulation_plasticity_refused.fixed.md`,
  `absent_backend_refused.fixed.md`). Moves to the closed archive at its next re-pin.
- **closed:** 2026-10-20 (agent): entry moved from the open log.
