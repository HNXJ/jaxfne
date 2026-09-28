# Closed issues after 2026-09-27

Earlier closed issues: `artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md`.

### P-011
- **date:** 2026-09-25
- **type:** FRICTION
- **area:** Kaleido static export under xdist load
- **observation:** `test_vis_smoke[exporters.export_figures]` failed once in the
  broad gate with `RuntimeError: Couldn't close or kill browser subprocess`
  (choreographer/Kaleido browser teardown), on a tree that changed neither
  `jaxfne/vis/exporters.py` nor the test.
- **severity:** MINOR (teardown of the headless browser; no output semantics)
- **minimal reproduction:** broad gate (`-n auto`); alone it passes 3/3
- **expected behavior:** stable PASS
- **actual behavior:** intermittent FAIL under parallel load
- **evidence:** broad gate on `d0d7360` 2026-09-25 (1 failed / 4430 passed); isolated reruns 3/3 PASS. Recurred 2026-09-26 (broad gate before 99482f7) and 2026-09-27 (before the AT-10-N20 commit), each isolated rerun PASS: a gate fix is now due
- **possible future change:** retry Kaleido teardown once, or serialize Kaleido tests (xdist group); open
- **resolution (2026-09-27, agent):** `jaxfne/vis/exporters.py` retries `write_image` once on exactly this teardown error; other errors propagate (`tests/test_vis_export_teardown_retry.py`).

### P-018
- **date:** 2026-09-27
- **type:** DEFECT (configuration stored, not consumed)
- **area:** `jaxfne/_construct_core.py` plain `network(n=)` route
- **observation:** only the population route (`column()`, `uniform3d()`,
  `layer_fractions()`) reads `drive(baseline_drive_by_cell_type=)`; on the
  plain route the declared drive is dropped. P-014 rewrote the per-type
  drives of 3 scripts, 7 tests and one doc-page atlas spec as `drive()`;
  they build on this route, so those drives still never run (among them
  `scripts/mcc3_10s_scientific_checkpoint.py` and the
  `benchmark_050_baseline` models that `benchmark_051_matrix` reuses).
- **severity:** MAJOR (same class as P-014)
- **minimal reproduction:** `network(n=2, cell_types={"E": 0.5, "PV": 0.5})`
  with drives from default to `{"E": 15.0, "PV": 10.0}`: E fires 12 Hz and
  PV 0 Hz at every setting; with `.uniform3d()` the rates follow the drive
- **expected behavior:** consumed or refused
- **actual behavior:** accepted, recorded, ignored
- **evidence:** scratch probes 2026-09-27; `git grep baseline_drive_by_cell_type`
  finds one reader, `_construct_population.py`; `mcc3_config()` declares
  E 8.0, PV 8.0 and builds the defaults 5.0, 3.0, with an emitter drive
  identical to the configuration without the declaration
- **possible future change:** human decision: wire the per-type drive on the
  plain route (outputs of those callers change) or refuse it (callers drop
  it; outputs unchanged); open
- **resolution (2026-09-28, agent; human decision: wire it):** the plain route
  applies `drive(baseline_drive_by_cell_type=)` and `cell_params()` as the
  population route does (`_apply_baseline_drive`, `_apply_cell_params` in
  `jaxfne/_construct_population.py`; `tests/test_consumption_gate_055.py::test_drive_and_cell_params_reach_the_emitter`)
  and refuses a `cell_params()` layer selector. Regenerated the `hdp_10` doc
  atlas. Receipts made while the drive was dropped stay as made, and their
  configurations declare what ran: `mcc3_config()` in
  `scripts/mcc3_10s_scientific_checkpoint.py` keeps the default drives (as
  `_mcc3_model` did under P-014; the etude rerun reproduces the checkpoint,
  P-020 records a condition-C drift) and `tests/test_equiv01_table.py` the
  drives its bounds were measured at (P-019). `artifacts/perf/matrix_051.json`
  stays the 0.5.0 receipt; its output checksums came from the default drives.

