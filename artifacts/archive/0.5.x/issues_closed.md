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

### P-019
- **date:** 2026-09-28
- **type:** DEFECT (equivalence bound stated wider than measured)
- **area:** `tests/test_equiv01_table.py` (24-EQUIV-01, jit vs eager)
- **observation:** the table's bounds were measured at the default drives
  (E 5.0, PV 3.0): the test declared E/PV drive 10.0, which the plain route
  dropped until P-018. At drive 10.0, spikes stay equal but jit and eager
  V_m differ by up to 2.5e-3 (legacy and eligibility HDP rules) and 1.4e-3
  (no HDP), against the 1e-4 bound; with the registered rule V_m and H hold
  and the W trace, stated exact, differs.
- **severity:** MINOR (float32 reassociation; the contract names no regime)
- **minimal reproduction:** restore `.drive(baseline_drive_by_cell_type={"E": 10.0, "PV": 10.0})`
  in `_model()` of `tests/test_equiv01_table.py` at b0817dc and run the file
- **expected behavior:** the bounds hold, or the table states the regime they cover
- **actual behavior:** 4 FAIL (3 HDP regimes and the no-HDP baseline)
- **evidence:** broad gate in the P-018 clone, 2026-09-28
- **possible future change:** measure the jit-vs-eager deviation over a drive
  sweep and state a bound per regime, or find the fusion that moves it; open
- **resolution (2026-09-29, agent):** drive-10.0 corner carries its own
  empirical bounds in `tests/test_equiv01_table.py` (V_m 1e-2, H 1e-5,
  W 1e-6, spikes exact; margin >= 4x over the seed-3 60 ms measurement:
  dV 2.50e-03, dH 7.15e-07, dW 8.94e-08); default-drive bounds unchanged.
  New drive-10 tests fail under the old strictness (4 failed) and pass
  under the corner bounds (10 passed with the file).

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
- **resolution (2026-09-29, agent):** fixed in `6478966` (brief-path test
  exempts paths `git check-ignore` reports; v033 test skips on an empty
  output dir). Verified in a fresh `git worktree` at `26662db` with an empty
  `outputs/v030_03_two_neuron_ei_multimodal`, run from the worktree root:
  1 passed, 1 skipped ("No JSON outputs generated").

### P-024
- **date:** 2026-09-30
- **type:** SCIENCE (stored is not consumed)
- **area:** `jaxfne/paradigm.py` (`omission_oddball_paradigm`)
- **observation:** the `unexpected` condition differs from `expected` only by
  `stimulus="deviant_tone"` and label; no numeric consumer reads either, so
  at equal onsets the two conditions simulate identically. Page 10 now says so.
- **severity:** MINOR (scaffold says "no empirical validation"; an oddball
  result built on it would compare equal runs)
- **expected behavior:** a deviant carries a distinct declared drive, or the
  factory refuses equal standard and deviant drive
- **evidence:** D1b opencode run 2026-09-30 (expected == unexpected spikes at
  seed 42); `jaxfne/paradigm.py:675-697`
- **possible future change:** owner decision: deviant gain parameter or
  refusal; open
- **resolution (2026-09-30, owner ruling "refuse equal drives"):** `omission_oddball_paradigm` takes `standard_drive_amplitude` and
  `deviant_drive_amplitude`; None realizes the simulator default read from
  `stimulus_schedule`'s signature; the factory refuses realized-equal or
  non-finite drives. Callers declare a deviant amplitude. Tests in
  `tests/test_paradigm_semantics_p017.py` (critic-found bypasses None vs 5.0,
  "10" vs 10.0, NaN each shown to fail on the first version).

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
  before the next benchmark cycle; definitions added 2026-09-30 (packet
  prints them, refuses on arm mismatch); open until the human reviews them
- **resolution (2026-09-30, owner approved as written):** every task in
  `agent_tasks_055.json` declares `arm_definitions` (inputs only; critic removed
  outcome clauses); `packet()` prints them and refuses a mismatch;
  `freeze_task` carries them on re-freeze (`carry_arm_definitions`, tested).
