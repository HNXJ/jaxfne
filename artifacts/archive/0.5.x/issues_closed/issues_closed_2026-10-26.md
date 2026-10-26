# jaxfne closed issues, batch of 2026-10-26 (0.5.8)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`). Earlier closed
entries: `../issues_closed.md`, `../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`,
pinned write-once beside this folder.

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
- 2026-10-25: the lane tests run in `scripts/run_test_gate.py` family `pytest_lanes`, one process per
  lane (seven lanes pass in the broad gate; `.github/workflows/ci.yml` runs broad); the skip on the
  gitignored catalog names its input. Closes for 0.5.8; the entry moves to a closed batch at release.
- Resolution 2026-10-26: closed with the 0.5.8 release candidate; the broad gate on that candidate
  runs `pytest_lanes`.
