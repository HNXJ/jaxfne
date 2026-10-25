# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

Entries P-037 and P-040 come from the whole-project scan of 2026-10-18 (six Haiku workers,
read-only; each line was re-opened and confirmed by the dispatcher unless marked inferred).

### P-037

- Date: 2026-10-18. Type: BUG. Area: tests and gates. Severity: high (tests pass vacuously).
- `tests/conftest.py:22-31` turns x64 off before every test (except `test_protocol_w_*`), so
  `test_closure_hp_reconciliation.py:49-52` always skips and the float64 asserts in
  `test_optim_jax_native_audit.py:94-97,118-121` (inside `except Exception: pass`) never run.
- Skips on gitignored inputs: `test_agent_api_catalog.py:36` (6 tests, catalog gitignored),
  `test_notebook_structure_v0330.py:57`; warehouse-dependent skips in CI
  (`test_pcl_stage_c.py:57`, `test_gsdr2026_labels.py:167`).
- Exact float pins on optimizer output: `test_param_handles.py:496-497`.
- cwd-dependent relative paths: `test_experiment_a_b3_receipt.py:13,26`,
  `test_experiment_a_b1_canonical.py:18,52` (from another cwd: one error, two silent skips).
- The 11 lane test files under `artifacts/etudes/papers/**` are in no gate and no CI workflow
  (`pyproject` `testpaths=["tests"]`); they pass today (226 passed, 1 xfailed).
- `scripts/harness/tests/test_harness_adversarial.py` is reached by no gate; `networkx`
  (`test_network3d_visualize.py:175`) is declared nowhere.
- 2026-10-20 (release plan A1): optimizer pins in `test_param_handles.py` use `assert_allclose`;
  the float64 asserts, the `__file__` paths and the `networkx` importorskip already held at
  81b602bd. Open: lane tests in a gate need one process per lane (todo queue item 5); the
  harness tests moved to `scripts/archive/`; skips on gitignored inputs stay (listed in A1).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- `destexhe2009/PREREG_P3.md:1` reads "draft, UNSIGNED — no scored run may use it";
  `CALIBRATION.md:3-4` scores E1-E4 against it (header "Verdict: OFF"), and its final section
  says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12` says "no verdict
  yet" and "28 passed" (30 pass today).
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.
