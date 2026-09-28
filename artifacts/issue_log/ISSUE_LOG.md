# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-014, P-015) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md`. A new issue takes the next
free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

### P-012
- **date:** 2026-09-25
- **type:** DEFECT (H11: tests depend on untracked local state)
- **area:** `tests/test_memory_brief.py::test_brief_paths_exist`,
  `tests/test_v033_two_neuron_ei.py::test_v033_all_json_files_parseable`
- **observation:** broad gate in a fresh `git worktree` at `b110493` failed both;
  the main tree at the same code passes both. memory.md names
  `artifacts/release_candidate/`, `artifacts/developer/` (gitignored) and
  `jaxfne/publication/` (untracked); the v033 test skips only when
  `outputs/v030_03_two_neuron_ei_multimodal` is absent, and in the fresh tree
  it existed but held no JSON.
- **severity:** MINOR (gate false-fails outside the author's checkout)
- **minimal reproduction:** `git worktree add <dir> b110493`, run the broad gate there
- **expected behavior:** PASS or explicit skip from a fresh clone
- **actual behavior:** 2 FAIL
- **evidence:** broad gate log 2026-09-25 (2 failed / 4422 passed in worktree); main tree rerun 2/2 PASS
- **possible future change:** brief-path test treats gitignored/untracked paths as local-only
  (or memory.md stops naming them); v033 test skips on an empty output dir; open

### P-013
- **date:** 2026-09-26
- **type:** DEFECT (benchmark task under-specified)
- **area:** `artifacts/benchmark/agent_tasks_055.json` / `agent_bench.packet`
- **observation:** tasks name their arms but never define them. Arm
  semantics live in the runners and in the spec's `cited` rows, and the
  packet drops both because they point at answer files. AT-07 `repro` (same
  noisy spec and seed on a fresh model, so identical to `noisy`) failed
  `mean_vm` in 5/6 runs across both arms; `fixed` failed in 4/6.
- **severity:** MINOR (hits both arms equally; lowers absolute scores on
  multi-arm tasks, not the arm comparison's direction)
- **minimal reproduction:** `python artifacts/benchmark/agent_bench.py packet AT-07 skills`
  names `repro` with no definition
- **expected behavior:** each task declares per-arm definitions (what differs
  from the base spec) without pointing at runner code
- **actual behavior:** workers guess arm meaning from names
- **evidence:** `artifacts/benchmark/results_055.json` (AT-07 runs)
- **possible future change:** add declarative `arm_definitions` to each task
  before the next benchmark cycle; open

### P-016
- **date:** 2026-09-26
- **type:** DEFECT (slow-tier regression pin fails at HEAD)
- **area:** `tests/test_hdp_population_restoring.py::test_population_restoring_etude_regression_metrics`
- **observation:** vector-HDP terminal error 0.0028 vs frozen 0.0296
  (tol abs 0.02); fails identically on 409f125 without any change. The
  test is `slow`, outside the broad gate. The etude runner
  `scripts/hdp_mvc_etude.py` named in the manifest no longer exists.
- **severity:** MINOR (smaller terminal error than frozen; pin, not physics)
- **minimal reproduction:** `pytest tests/test_hdp_population_restoring.py -q`
- **expected behavior:** PASS against the frozen etude
- **actual behavior:** 1 FAIL
- **evidence:** run 2026-09-26 on 409f125 and on the P-014 tree, same value
- **possible future change:** find the commit that moved it (bisect over
  HDP changes); re-freeze only with a stated cause; open

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
  ParadigmCondition runs change) or refuse what is not realized; open

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

---
