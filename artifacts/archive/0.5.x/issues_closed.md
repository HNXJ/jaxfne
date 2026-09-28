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

### P-017
- **date:** 2026-09-27
- **type:** DEFECT (paradigm declared, not realized as declared)
- **area:** `jaxfne/_model_simulate.py` `_resolve_stimulus_schedule`,
  `jaxfne/_model.py` `stimulus_schedule`, `jaxfne/paradigm.py`
  `evoked_l4_drive_paradigm`
- **observation:** (1) `simulate(paradigm=<Paradigm>)` resolves no schedule
  for a multi-condition `Paradigm`; it is recorded in metadata only. (2) For
  a `ParadigmCondition`, `stimulus_schedule` injects `drive_amplitude` 5.0
  into every neuron for `event_duration_ms` 50 ms at the onset of each
  event that is not an omission, including `trial_start`, `stim_absent` and
  `post_stim`; it does not read the event's `duration_ms` or `stimulus`.
  The evoked-L4 "baseline" and "evoked" conditions therefore inject the same
  schedule (0, 200 and 400 ms). (3) `evoked_l4_drive_paradigm` does not read
  `l4_onset_ms` or `l4_amplitude`; its drive event starts at
  `pre_stimulus_buffer_ms`.
- **severity:** MAJOR (tutorial 08's evoked run equals its baseline;
  condition runs inject drive at events that carry no stimulus)
- **minimal reproduction:** evoked-L4 notebook configuration, 1000 ms, seed 7:
  spikes with `paradigm=paradigm` equal spikes without a paradigm;
  `stimulus_schedule(c.events, 4)` for both conditions lists the same three
  5.0 x 50 ms events on all neurons
- **expected behavior:** a stimulus event is injected at its onset for its
  duration with its amplitude and targets; events without a stimulus inject
  nothing; a paradigm type simulate cannot run is refused
- **actual behavior:** as observed
- **evidence:** scratch probes 2026-09-27 on bdf56cb; code read. Atlas runs
  (AT-08/09, AT-10-N20) build dict events whose duration, amplitude and
  target_indices are honoured, and are unaffected.
- **possible future change:** human decision: fix the semantics (outputs of
  ParadigmCondition runs change) or refuse what is not realized
- **resolution (2026-09-28, agent; human decision: fix the semantics):**
  `stimulus_schedule` injects only events with a stimulus, for their own
  duration, at their amplitude and targets (new `target_layer`); simulate
  refuses a multi-condition `Paradigm` and unknown types;
  `evoked_l4_drive_paradigm` reads its onset and amplitude and targets L4;
  COOP pulses carry a stimulus (`tests/test_paradigm_semantics_p017.py`).
  Page 08 still passes a full `Paradigm`; D1b rebuilds it.

### P-021
- **date:** 2026-09-28
- **type:** BUG
- **area:** release notebooks (`tests/test_notebook_execution_suite.py`),
  outside the broad gate
- **observation:** three release notebooks stop at a refusal the 0.5.5
  consumption gate added: `jaxfne_suite_no_2_evoked_l4_drive.ipynb` passes a
  two-condition `Paradigm` (P-017 refusal); `jaxfne_v033_two_neuron_ei.ipynb`
  and `jaxfne_v035_small_recurrent_ei.ipynb` declare
  `emitter(preset='cortical_eig_e_plus_pv')`, which the emitter does not realize
- **severity:** MAJOR (shipped notebooks do not run)
- **minimal reproduction:** `pytest tests/test_notebook_execution_suite.py -k
  "evoked_l4 or v033_two or v035_small"`
- **expected behavior:** each notebook runs, declaring only what is realized
- **actual behavior:** 3 failed (`CellExecutionError`, ValueError as above)
- **evidence:** run at 42e0e2f, 2026-09-28; the broad gate does not execute
  these notebooks, so it stayed green
- **possible future change:** migrate the notebooks (evoked: one
  `paradigm.condition(...)` per run, with D1b; v033/v035: drop the preset or
  declare what the E+PV setup realizes), and put one fast notebook-execution
  check in the broad gate
- **resolution (2026-09-28, agent):** c34876c. v033/v035 declare the
  realized `cortical_eig` preset (the other name was metadata only) and
  the v035 exercise varies `drive()`; the evoked-L4 notebook passes
  `paradigm.condition("evoked")`. The three notebook tests pass. The
  broad-gate notebook check moved to the todo stack.
