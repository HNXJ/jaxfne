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
- **resolution (2026-09-30, owner ruling: re-freeze with cause):** cause `865e74b`
  (bisect `artifacts/programme/bisect_p016_receipt.md`); frozen value
  `0.029602456101573416` -> `0.00280088258438386`; receipt
  `artifacts/programme/refreeze_p016_p020_2026-09-30.md`; old file archived.

### P-020
- **date:** 2026-09-28
- **type:** SCIENCE (frozen receipt reproduces in part)
- **area:** `artifacts/mcc3_10s_checkpoint/` (receipt of c9e12f8, v0.4.8),
  `scripts/mcc3_10s_scientific_checkpoint.py`
- **observation:** rerun at b0817dc with the repaired script: conditions A
  and B match the receipt to float32 (κ within 1e-14, PSD within ~1e-7
  relative). Condition C (θ̂, HDP off) keeps every rate and spike count, but
  its spike timing moved: population ISI CV 0.01270 → 0.01228, κ 0.0229 →
  0.0183, source |max| 29.88 → 30.86. The etude rerun
  (`scripts/rerun_etude_figures.py --etude hdp_mcc3`) checks θ̂ and
  condition B only, and passes.
- **severity:** MINOR (rates and counts equal; timing statistics of one condition)
- **minimal reproduction:** run `scripts/mcc3_10s_scientific_checkpoint.py`
  and compare `mcc3_10s_metrics.json` leaf by leaf with the committed file
  (132 of 1,800,946 leaves differ)
- **expected behavior:** the receipt reproduces, or its drift has a stated cause
- **actual behavior:** as observed
- **evidence:** run 2026-09-28 in a clone at b0817dc; the receipt was
  restored byte for byte
- **possible future change:** bisect the HDP-off simulation path since
  c9e12f8 for the commit that moved condition C, then re-freeze with the
  stated cause or extend the etude check to C; open
- **resolution (2026-09-30, owner ruling: re-freeze with cause):** cause `865e74b`
  (bisect `artifacts/programme/bisect_p020_receipt.md`); condition C re-frozen
  (kappa 0.0229 -> 0.0183, ISI CV 0.01270 -> 0.01228), A/B unchanged beyond float
  dust; regenerated from a clean tree at 1a7998cd; receipt as for P-016.

### P-022
- **date:** 2026-09-30
- **type:** SCIENCE (fact evidence)
- **area:** `jaxfne/neuronal_tensor.py` (`neuronal_tensor_to_configuration`,
  `_construct_neuronal_tensor_impl`), fact "entrances, single lowering"
  (`artifacts/fact_stack.md:88-89`)
- **observation:** no test compares the Signals of two independent entrances
  for one circuit. `construct(tensor)` lowers through the bridge and then
  `construct(cfg)`, so Configuration vs NeuronalTensor is one lowering plus
  overlays. JDNA reaches construct through NeuronalTensor too, with no
  output-equality test. The bridge alone drops `Layer.geometry`, so field
  proxies differ by entrance (source_proxy max abs diff ~145) while spikes,
  V_m and sources are bit-identical.
- **severity:** MINOR (activity equal; field output depends on which call
  the user makes, as the bridge docstring states)
- **expected behavior:** the fact holds with evidence, i.e. a native
  Configuration spelling the same circuit without the bridge gives equal
  Signals, and field geometry does not depend on the entrance
- **evidence:** opencode sweep and critic 2026-09-30; test
  `test_tensor_entrance_adds_only_overlays_to_the_configuration_lowering`
  pins the overlays and the field gap (killed by removing the position overlay)
- **possible future change:** carry `Layer.geometry` through the bridge;
  add a hand-built-Configuration equality test; open
- **resolution (2026-09-30, owner rulings):** the bridge carries each
  `Layer.geometry` as column-relative ranges; positions use the Configuration
  frame and a non-default `Pose3D` applies about the area's own column origin;
  tensor, bridged and hand-spelled entrances give bit-identical positions and
  fields. Pushed 41762619 + bae3c41c (canonical atlas re-pinned, cause P-022).

### P-023
- **date:** 2026-09-30
- **type:** SCIENCE (default and sign semantics)
- **area:** `jaxfne/neuronal_tensor.py` (`StaticParams.dT_ms`, `_wire_connection`)
- **observation:** `StaticParams.dT_ms` defaults to 0.1 ms and becomes the
  mechanism `tau_ms`, so a JDNA- or tensor-declared GABA_A synapse with no
  explicit time constant decays in 0.1 ms. The edge sign follows the source
  cell type (E -> excitatory), so an E->PV connection declared `GABA_A`
  with reversal -80 mV runs as excitatory; the declared reversal has no
  numeric consumer (stated in the bridge docstring).
- **severity:** MINOR until a result depends on an undeclared tau or on a
  mechanism whose name contradicts its sign
- **expected behavior:** a default time constant per mechanism with a cited
  source, or a required value; sign from the mechanism's reversal or a
  refusal when name and source type disagree (fact: apply or refuse)
- **evidence:** JDNA single-lowering test 2026-09-30 (developed tensor:
  `GABA_A__dt0.1__0`, `tau_ms=0.1`, sign excitatory); `neuronal_tensor.py:140`
- **possible future change:** owner decision on defaults and refusal; open
- **resolution (2026-09-30, owner rulings):** time constants are required
  (finite > 0, no default); JDNA genomes declare `mechanism_tau_ms` (canonical:
  `presets.RECEPTOR_KINETICS`, approval pending; G20: the 0.1 ms it ran with);
  sign by mechanism name, clashes and unknown names refused except
  `monotonic_cable_synapse`; TFNE-minted tensors keep polarity sign through a
  private registry. Commits 83848805 + 3d75c12d.

### P-025
- **date:** 2026-09-30
- **type:** BUG (a script can run against the wrong package)
- **area:** `scripts/*.py`
- **observation:** `python scripts/x.py` puts `scripts/` first on `sys.path`, so
  an installed jaxfne (site-packages 0.5.0 on the owner's machine) shadows the
  working tree. The first P-016/P-020 re-freeze runs used 0.5.0 (values
  discarded). 47 of the 61 scripts that import jaxfne have no repo-root guard.
- **severity:** MAJOR for any frozen output made by a script; MINOR otherwise
- **minimal reproduction:** with jaxfne 0.5.0 installed, run a script and print
  `jaxfne.__file__`
- **expected behavior:** every script imports the tree it lives in
- **actual behavior:** imported site-packages when executed directly without PYTHONPATH
- **evidence:** refreeze 2026-09-30; guard added to
  `scripts/mcc3_10s_scientific_checkpoint.py` (rerun loads the tree, values
  unchanged); `scripts/regenerate_hdp_population_restoring_metrics.py` has it
- **possible future change:** add the guard to the remaining scripts with a
  test that each script importing jaxfne inserts the repo root; open
- **resolution (2026-10-01, agent):** commit baf43d49 added repo-root guards
  (`sys.path.insert(0, str(Path(__file__).resolve().parents[depth]))`) and
  `assert Path(jtfne.__file__).resolve().is_relative_to(REPO_ROOT)` across all
  125 scripts, artifacts, and examples importing jaxfne; AST gate test
  `tests/test_script_repo_root_guard.py` mechanically prevents regressions (2/2
  passed, verified in p-jaxfne #53).

### P-026
- **date:** 2026-09-30
- **type:** SCIENCE (estimator differs from its name or docstring)
- **area:** `jaxfne/analysis/metrics.py`
- **observation:** `fano_factor` says "Computed per neuron, then averaged"
  but computes one variance over mean of population-summed binned counts;
  `burst_index` takes `bin_ms` and never uses it (per-step active fraction);
  `mean_pairwise_spike_correlation` correlates raw per-step spike rows with
  no bin width.
- **severity:** MINOR (no computational caller in jaxfne, scripts, examples
  or docs; no committed or frozen artifact depends on them; only
  `tests/test_analysis_metrics.py` pins current values, including
  `test_variable_dt_scaling`, which pins bin-insensitivity)
- **minimal reproduction:** `pytest tests/test_analysis_metrics.py -k test_variable_dt_scaling`
- **expected behavior:** per-unit Fano across trials, network bursts with a
  used bin width, count correlation at a stated bin width; jnwb 0.2.8 lane D
  provides these (`fano_factor`, `network_burst_index`,
  `spike_count_correlation`)
- **actual behavior:** estimators compute population-summed Fano and unbinned correlations
- **evidence:** opencode sweep 2026-09-30, lines re-read: `metrics.py:197`
  vs `:255-275`, `:124-126`, `:87-88`
- **possible future change:** delegate to jnwb when 0.2.8 ships (todo 0d,
  short-list goal 3), rewriting the pinning tests
- **resolution (2026-10-01, owner ruling):** deprecated with corrected docstrings, values unchanged for one release (commit 2ad5d7a5); replacement jnwb spike_count_correlation (window_s, bin_ms), fano_factor (onsets_s, window_s, summary), network_burst_index (window_s, bin_ms, threshold_hz, min_duration_ms) via jaxfne.jnwb_view.to_jnwb / to_jnwb_trials
- **closed:** 2026-10-04 (agent): entry moved from the open log; deprecation commit 2ad5d7a5 is an ancestor of origin/dev.

### P-027
- **date:** 2026-10-04
- **type:** BUG (public parameter accepted but never consumed)
- **area:** `jaxfne/paradigm.py` (`omission_oddball_paradigm`)
- **observation:** `standard_onset_ms`/`deviant_onset_ms` were accepted and documented, but events were built at buffer offsets only
- **severity:** MAJOR (silent timing error in a public paradigm builder)
- **resolution (2026-10-04, agent, owner-delegated):** refuse path — non-default onsets raise `ValueError` (P-027) with field/value/fix; docstrings state buffer placement; the one live non-default caller (`generate_doc_page_atlases.py:560`, onset 50.0 silently ignored while stimulus sat at the 200 ms buffer edge of a 200 ms run) now passes `pre_stimulus_buffer_ms=50.0`; `objective_60` pages regenerated and re-promoted to validated. Failing-first test `test_omission_oddball_nondefault_onsets_refuse_p027` failed pre-fix (3 cases), 24 paradigm tests green post-fix. Commit 0a75a05f.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-028
- **date:** 2026-10-04
- **type:** BUG (fail-open solver selection + ignored tolerances)
- **area:** `jaxfne/solvers.py`
- **observation:** unknown `solver_type` fell through to `Tsit5`; `SolverConfig(rtol, atol)` silently ignored on the Euler path; non-divisible grids silently rounded
- **severity:** MAJOR (public params that did not mean what they said)
- **resolution (2026-10-04, agent, owner-delegated):** refusals — unknown `solver_type` raises (None/dopri5/tsit5 accepted, checked before the diffrax import); non-default `rtol/atol` with `method="euler"` raises; shared `_checked_steps` helper (same 1e-9 rule as `agent._checked_time`) guards both Euler and Diffrax save grids. Failing-first `tests/test_solvers_refusal_p028.py`: 3 fail pre-fix (named-stash proof), 4 pass post-fix. No in-tree callers affected (`streaming` uses only `.dt`). Commit 948cfec9.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-029
- **date:** 2026-10-04
- **type:** BUG (three competing canonical compositions)
- **area:** `jaxfne/hdp_network.py`, `jaxfne/_construct_population.py`, `jaxfne/builders.py`
- **observation:** three composition tables with different numbers presented as competing canonicals
- **severity:** MAJOR
- **resolution (2026-10-04, agent, owner-delegated):** rename, not renumber — numeric unification would void the HDP stationarity receipt (N=250) and suite2 frozen outputs. `LAYER_CELL_TYPE_FRAC_DEFAULT` → `HDP_LAYER_CELL_TYPE_FRAC_DEFAULT` (sole internal use; no external importers) with ownership comment; `_SUITE2_*` annotated as versioned suite presets; source of truth declared as `builders.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`. Ownership test `test_hdp_default_table_distinct_from_canonical_p029` pins distinctness; 11 HDP/canonical tests green. Commit 1ced2661.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-030
- **date:** 2026-10-04
- **type:** BUG (fail-open validators + broad exception handlers)
- **area:** `jaxfne/validation.py`, `jaxfne/_model.py`, `jaxfne/_pipeline.py`, `jaxfne/util.py`
- **observation:** dead fail-open `_is_finite_value`; over-broad handlers on `z` extraction and dtype fallback; ~30 broad probes in `util.py` summaries
- **severity:** MAJOR
- **resolution (2026-10-04, agent, owner-delegated):** `_is_finite_value` deleted (zero callers repo-wide; slop, not a contract); `z` extraction narrowed to `(IndexError, TypeError, ValueError)`; dtype fallback narrowed to `(TypeError, ValueError)`; `_pipeline:990` metadata fallback and the 31 `util.py` probes deliberately left broad — best-effort inspection over duck-typed/JAX-tracer objects where failure modes genuinely vary; narrowing risks breaking valid inspection for zero behavior gain. 73 tests green (signal/continuation/tensor). Commit db996476.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-031
- **date:** 2026-10-04
- **type:** BUG (stored-but-unconsumed parameter)
- **area:** `jaxfne/_runtime_config.py` (`SurrogateConfig`)
- **observation:** declaration-only object with no consuming kernel; typo'd `method` silently reported `required_but_missing`
- **severity:** MAJOR
- **resolution (2026-10-04, agent, owner-delegated):** declaration validated, not wired — the object is EXPERIMENTAL_INTERNAL and explicitly declaration-only by design (status + test pin it), so inventing a gradient path would be scope fabrication. `__post_init__` now refuses unknown `method` (P-031) and non-finite `beta`. Failing-first test failed pre-fix, 14 module tests green post-fix. Commit 05c66d61.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-032
- **date:** 2026-10-04
- **type:** BUG (nondeterministic tests)
- **area:** `tests/test_analysis_metrics.py`
- **observation:** unseeded draws under tight asserts risked flakes
- **severity:** MAJOR
- **resolution (2026-10-04, agent, owner-delegated):** scoped by evidence — all band asserts were already seeded or deterministic; only `test_perfectly_correlated` (>0.9) and `test_variable_dt_scaling` (<0.2) had draw-dependent tight asserts, now `default_rng(7)`/`default_rng(11)`. Loose asserts (isfinite/range/monotonic) are draw-independent by construction and untouched. 55 passed twice. Commit 2a623d24.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-033
- **date:** 2026-10-04
- **type:** DOC (overclaims, stale versions, unit leaks)
- **area:** `docs/` (batch from prose sweep)
- **observation:** stale 0.4.8 labels, "shows homeostatic adaptation" overclaim, untraced ~215k, µm example, "proves exhaustiveness"
- **severity:** MAJOR + minors
- **resolution (2026-10-04, agent, owner-delegated):** stale parenthetical deleted (not re-asserted untested); overclaim softened to settling-consistent with the listed numbers; ~215k labeled observed-2026-10; µm example rewritten in relative fractions; "proves" → "checks" with test name. S3 kept (defined margin-policy term). Doc gates green (language, vocabulary, integrity). Commit 31bae896.
- **closed:** 2026-10-04 (agent): entry moved from the open log.

### P-034
- **date:** 2026-10-04
- **type:** DOC (figure rendering defects)
- **area:** `artifacts/figures/publication/` + generators
- **observation:** V1-V4 layout defects (collisions, overplotting, missing units, stacked labels) + V5 micro-type, worker-observed at full size
- **severity:** MAJOR + minor
- **resolution (2026-10-04, agent, owner-delegated):** NO PIXELS MOVED — all four targets are hash-pinned (fig01/03/06 in `frozen_manifest.json` under write-once; e2_fig08 under G5 pixel-identity receipt). Any generator edit creates script↔output drift against pinned bytes with no authorized regen path. Defects are per-figure coordinates (no shared-helper fix exists), so the findings are logged as manuscript input: `artifacts/audit/sweep_2026-10-04_prose_visual_hygiene.md` (V1-V5 table) constrains Atlas item 10's new write-once figures (annotation/legend clearance, unit-labeled axes, jittered rep labels, ≥6pt type). Revisit only with a re-freeze receipt.
- **closed:** 2026-10-04 (agent): entry moved from the open log.
