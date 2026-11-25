# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`; from 0.5.8 the batches sit in `artifacts/archive/0.5.x/issues_closed/`,
P-037: `issues_closed_2026-10-26.md`; P-045, P-046, P-049, P-050: `issues_closed_2026-10-27.md`; P-048, P-051, P-054:
`issues_closed_2026-11-03.md`; P-040, P-057 to P-065: `issues_closed_2026-11-21.md`; P-066:
`issues_closed_2026-11-22.md`; P-068: `issues_closed_2026-11-23.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

### P-072

- Date: 2026-11-25. Type: DOC / FRICTION. Area: field probes and receipts. Severity: low. Field-operator audit
  residuals, recorded for separate priority (Hamm 2026-11-25: no incidental API, label or numeric change).
- D2: `meg_proxy_probe(orientation_convention=...)` is accepted and discarded (`jaxfne/fields/probes.py:729`, documented
  as a compatibility argument): stored-but-unconsumed; refuse or keep is Hamm's call.
- D3: `csd_proxy_probe` docstring says it estimates CSD; it passes its input through (`probes.py:612-658`).
- D6: `run_observation_suite` labels `meg_relative` the EEG leadfield with random sign flips
  (`jaxfne/experiment_a/receipt.py:101-104`); labels are not changed without a ruling.
- D7: `spectrolaminar_readout` contact depth = `pos_from_l4 * 0.5`, an undeclared constant (`fields/proxy.py:1186`).
- D8: `experimental_poisson_1d` docstring claims the scalar path is bit-identical to a constant layered σ; the test
  checks `atol=1e-4` (`tests/test_experimental_poisson_1d_layered.py`, `test_layered_reduces_to_scalar_path_when_uniform`).

### P-071

- Date: 2026-11-25. Type: BUG (checks weaker than advertised). Area: E4 linearity gate, Poisson gauge gate.
  Severity: medium (both gates can pass without establishing their property).
- D4: `verify_field_linearity` (`jaxfne/protocol_e_integration/e4_execution.py:331-357`) combines 0.7 Q and 0.3 Q, so
  it tests homogeneity only; a positively homogeneous nonlinearity F(|q|) passes it.
- D5: `validate_poisson_gauge_condition(None)` returns True (`jaxfne/validation.py:915-958`), so the report's gauge
  pass flag reads the same for missing evidence and a satisfied gauge; `test_gauge_none_data` pins that.
- Evidence (dev since 8018e191, test-only): additivity on independent sources at the E4 `lfp_ref`
  parameters holds within a rigorous float32 bound (excess 2.8e-2); F(|q|) exceeds it 7.0e4x; strict xfails
  `test_e4_linearity_gate_detects_a_homogeneous_nonlinearity` and
  `test_gauge_gate_flag_differs_for_missing_and_satisfied_mean`.
- Expected: closes when a ruled production change turns both xfails into passing tests.

### P-070

- Date: 2026-11-25. Type: BUG. Area: `experimental_poisson_1d` (`jaxfne/fields/solvers.py`). Severity: medium (an
  unsupported value silently implies a model that was not solved).
- Observation: `boundary` and `gauge` are written to the manifest only (`solvers.py:148-149`); the assembly is always
  Neumann and `lstsq` gives the min-norm (mean-zero) gauge, so `boundary="dirichlet"` returns the Neumann solution
  labelled Dirichlet. `simulate` passes the defaults only (`_model_simulate.py:1249`); direct callers reach it.
  `build_poisson_admissibility_report` defaults `boundary_condition="dirichlet"` (`validation.py:1018`), the
  opposite of the solver.
- Reproduction: strict xfails `test_unsupported_boundary_is_refused`, `test_unsupported_gauge_is_refused`
  (`tests/test_experimental_poisson_1d_layered.py`, on dev since 8018e191).
- Expected: a `ValueError` for any value other than `mean_zero_neumann` / `mean_zero`; closes when the xfails pass
  (production change awaits Hamm's ruling).

### P-069

- Date: 2026-11-23. Type: BUG. Area: simulation routes that do not realize HDP. Severity: medium (a requested rule
  is dropped without an error; the run's output carries no trace of it).
- Observation (Solmind R12, `artifacts/programme/design/plans/solmind/ephaptic_routes_r12_2026-11-23.md`): three routes run and drop
  an HDP request: `simulate_adex_seed_batch` with `enable_hdp=True`; homeostatic E/I with a simulate-time runtime
  (`Configuration.runtime(enable_hdp=True)` already refuses at construct); Izhikevich with `enable_hdp=False` and
  `hdp_params` given. Sibling routes (AdEx `Model`, HH, rate, LIF event scan) refuse with a `ValueError`.
- Reproduction: `pytest --runxfail tests/test_hdp_rules_ephaptic.py -k "seed_batch or homeostatic or without_enable"`:
  3 failed, "DID NOT RAISE".
- Expected: a `ValueError` naming the route, as on the sibling routes. Closes when the three strict xfails are
  plain tests that pass (todo T-P069).
- 2026-11-23: `simulate_adex_seed_batch` and homeostatic E/I now refuse (two xfails are plain tests, red on the
  old code). Left: the Izhikevich case, on Hamm's call (O-P069).
- Corrections to the pinned R12 table (outside review, 2026-11-23): row 1's `v_rest` refusal is at
  `hdp_rules_ephaptic.py:106-107` (tree after the R13 docstring change), not 103-104; row 5's `drive_bias` update
  (`emitters_lif.py:3217-3223`) is drift and jump clipped to `b_min`, `b_max` (unbounded for this rule), not "jump
  add"; the "Run-time evidence" paragraph splits the second table, and rows 18-19 belong to it. The fix shifts
  `_model_simulate.py` lines after 1691 by 6 and after 3797 by 11 relative to the table's da42b095.

### P-067

- Date: 2026-11-22. Type: BUG (inferred, unmeasured). Area: `NeuronalTensor` pose, field proxy. Severity: unknown.
  Split from P-066 at its closure.
- Observation: `Pose3D.plane` "xz" and "yz" map the local depth axis to global y or x (`_PLANE_AXIS_MAP`,
  `jaxfne/neuronal_tensor.py:143-148`), while `project_laminar_sources` reads depth from `positions[:, 2]`
  (`jaxfne/fields/proxy.py`). An area posed in xz or yz would project its depth from the wrong axis.
- Expected: the proxy reads depth along the area's posed depth axis, or refuses a non-"xy" pose.
- Next: a red test that projects an "xz"-posed laminar area and compares against the same area posed "xy".

## Moved to 0.5.11 (Hamm 2026-11-23, plan §10.3 item 2)

Not open for 0.5.10. Each entry keeps its closing check; it returns to Open when 0.5.11 starts.

### P-047

- 2026-11-23 (Hamm): kept as designed, moved to 0.5.11. Closes when: the selector docs state that `"all"` is
  not a wildcard and `test_connections_no_match_marked_compiled_no_edges` still passes.

- Date: 2026-10-26. Type: FRICTION. Area: `Configuration.connections` selectors. Severity: low.
- A rule with `target={"cell_type": "all"}` compiles 0 edges on both compilers (`{}` selects every
  cell). It is recorded, not silent: the rule's status reads `compiled_no_matching_edges`, a design
  pinned by `test_connections_no_match_marked_compiled_no_edges`. A refusal would reverse it and break
  that test (9C-E1 core packet, 2026-10-27); left as is. An `"all"` value is not a wildcard.

### P-052

- Status 2026-11-03: the implementation part is closed (sigmoid_beta, 2026-10-27, below); open is only
  whether the surrogate gradient is fit for training (Hamm, deferred to 0.5.10). Proposed closing check
  (external audit 2026-11-03): surrogate-based optimization improves a prespecified forward-model loss on
  held-out tasks across seeds and horizons; agreement with finite differences is neither necessary nor
  sufficient.
- Date: 2026-10-27. Type: SCIENCE. Area: gradient path of the Izhikevich dense route
  (`jaxfne/optim/handles.py`, `SurrogateConfig` in `_runtime_config.py`). Severity: high for any
  training or credit-assignment claim; none for forward simulation.
- The hard spike threshold carries no derivative, and the surrogate exists only at the loss
  (sigmoid soft rate on `V_m`); `SurrogateConfig` is a declaration only. So the gradient never
  crosses from one neuron to another through `W . spikes`. Reproduction
  (`scripts/perf/recurrent_grad_check.py`, N=20 E/PV, 2000 steps, float64, CPU): Jacobian of
  per-neuron mean `V_m` w.r.t. per-neuron `a`, off-diagonal max 0 (0 of 380 nonzero), while a
  central finite difference on `a[0]` moves all 19 other neurons' mean `V_m` (max 3.8 mV per unit).
  The compiled gradient at N=3200 has one dot (forward `W @ spikes`) and no transpose in the
  backward loop, so gradient timings (precision profile 2026-10-27) are a lower bound on training.
- Possible change: a kernel-level surrogate (`custom_jvp` on the spike threshold, e.g. the declared
  `straight_through` or `sigmoid_beta`), then the same check shows nonzero off-diagonal entries
  matching the surrogate derivative. Which surrogate is Hamm's call (it defines the training model).
- Resolved 2026-10-27 (Hamm chose sigmoid_beta): `RuntimeConfig(surrogate=SurrogateConfig("sigmoid_beta",
  beta))` emits spikes through `emitters.spike_sigmoid_beta` on the dense default-order route; forward
  outputs are bit-identical, other routes refuse it (`tests/test_surrogate_gradient.py`). The check now
  shows Wᵀ in the backward graph and nonzero off-diagonal entries, whose direction disagrees with the
  finite difference (corr -0.2 to -0.4) and whose norm grows with T; record in
  `artifacts/runs/perf/profiles/precision_058.md`. The slope and whether a surrogate that disagrees with
  the finite difference is fit for training stay open (Hamm).
- 2026-11-03: deferred to 0.5.10 (Hamm); the 0.5.9 changelog entry states the surrogate is not validated for
  training.
- 2026-11-08 (Hamm, O-85): the finite-difference disagreement and the growth with T stand as recorded
  properties; fitness is decided by the held-out training-loss check above, preregistered and run in 0.5.11
  (todo 24).
- 2026-11-23 (Hamm): moved to 0.5.11. Closes when: the preregistered held-out training-loss check (O-85, R58) is
  run and scored.

### P-053

- Date: 2026-11-02. Type: SCIENCE. Area: PCL event scan (`run_tuning.run_core_events`) against the event
  oracle (`artifacts/etudes/papers/pcl/port/pcl_event_ref.py`). Severity: medium for PCL R3 to R5
  exactness; none for spikes. Status: NOT_REPRODUCED on the declared sparse fixture under criteria Amendment 1 (a diagnostic validation of
  run 1; N = 170 exploratory; result `artifacts/etudes/papers/pcl/port/P053_sparse_fixture_result.md`): spikes equal
  and `w_s`, `wi_s` within rtol 1e-12 for V0 to V3 on seeds 0 to 2, with P5 engagement and the `w_s` control met. Not a
  refutation of P-053. Review note: V1 (one entry per row) never consumes its LAT half and equals V3 bitwise in seeds
  1 and 2, so the sparse content that tested the core arm is V2 and V1-TD in seed 0. R3 (2026-11-03): the core arm is exact against C++ on training file 0
  (max relative weight difference 5.3e-15, `CALIBRATION.md` R3). Stays open for fixture forms not tried.
- Minimal reproduction: none on the declared fixture. `artifacts/etudes/papers/pcl/port/test_p053_sparse_fixture.py`:
  a synthetic lateral and top-down network, learning on, LAT and TD hand-zeroed (V1 to V3), 170 events at one pixel,
  seeds 0 to 2. The worker's 60-event fixture was not kept; 60 events give no spike.
- Expected: spikes exact and weights within rtol 1e-12, as with dense seeded init weights
  (`test_p_core_arm_equals_oracle_arm_on_chain`).
- Actual (reported): identical spikes; `w_s` differs by about 0.4%. Suspected cause: the order of STDP
  updates within an event step.
- Possible change: reproduce on a minimal fixture, then attribute in PCL R3 (`CALIBRATION.md` R3
  section), where the training files run both arms against the C++ chain. Discriminator plan (static, not run):
  `p053_p056_source_audit_2026-11-13.md`. Correction (review 2026-11-17): that plan says the oracle
  L1-normalizes the whole simple weight tensor; `pcl_event_ref.py:549,561` normalizes one depth slice `w_s[k]`
  (init likewise, `:459-460`). The plan is a pinned dated record and stays as written.
- 2026-11-23 (Hamm): moved to 0.5.11. Closes when: a fixture form not yet tried (event order, repeated addresses,
  ties, zero vs epsilon weights) reproduces the `w_s` difference and it is attributed, or each form is recorded
  NOT_REPRODUCED.

### P-055

- Date: 2026-11-03, attributed 2026-11-12. Type: SCIENCE. Area: PCL event scan (`run_tuning.py --engine core`)
  against the numba oracle on `test7`, learning off, `--inhibition no-local`. Severity: high for any event-exact
  claim of the core on `test7`; R5-exact scored FAIL under signed R4b (`pcl/prereg/P3_fig3_r5_exact_result.md`).
- Observation: 4 of 16 files equal; 6 equal in counts with complex spike times or order differing; 3 off by 1-2
  complex spikes; 3 diverge in about 300 simple and 600 complex cells. The simple layer is exact on 13 of 16;
  the first divergence is in the complex layer in every file that diverges. Receipts `F:/warehouse/pcl/r5/`
  (`diff_exact.py`, `diff_exact.log`, `approx_diff.json`).
- Expected: event-exact equality, as on the non-scored pilot file and in R3 (learning on, training file 0).
- Cause (attributed 2026-11-12): complex cells have long integration constants (`tau_m = 50 ms` vs simple `18 ms`),
  64-fold simple-to-complex fan-in, and low spike threshold (`VTHRESH = 3.0 mV`). Evaluation differences (standard
  libm `math.exp` in Numba vs XLA vector intrinsics in JAX float64) accumulate in complex membrane potentials at
  near-threshold events: on `135/0/0/0.npz`, complex cell 797 at 153986 us receives identical simple inputs and
  weights, but reaches V = 2.9788 mV in core vs 3.1950 mV in numba against threshold 3.0 mV. This delays the complex
  spike by ~1 ms (to 154989 us). On low-drive files the simple layer stays exact and complex counts stay identical;
  on high-drive files matching cell receptive fields (`0/9`, `135/9`, `23/0`), the delayed complex spike delays
  top-down inhibition (`td_on=True`), allowing a simple cell to cross threshold and triggering an avalanche.
  Scoring both arms under signed R4b yields identical test verdicts (`P3_fig3_r5_approx_result.md`). Closed.
- 2026-11-23 (Hamm): moved to 0.5.11. Closes when: the attribution is reviewed by a non-writer and the entry moves
  to a closed batch, or a complex-layer exp matched between arms makes the 16 files event-exact.

### P-056

- Date: 2026-10-20, recorded 2026-11-06 (was O-56 on the `_temp_kernel-lfp` lane). Type: SCIENCE. Area: hybrid
  kernel LFP (`jaxfne.fields.kernel_lfp`) of a column on the AdEx route (Billeh 2020 wiring, Hagen 2016 cells,
  iSTDP settle). Severity: medium for any laminar-spectrum claim from that column.
- Observation: in the lane pilot, deep layers carry under 0.5% of the L2/3 LFP power; no deep alpha-beta peak.
- Expected: deep-layer alpha-beta power, as in laminar recordings.
- Possible change: settle the column longer, then attribute the deficit (inferred candidates: deep-layer rates,
  kernel amplitude of deep pyramids, coherence radius). Closes when a settled column shows deep alpha-beta power
  or the cause is named. Discriminator plan (static, not run): `p053_p056_source_audit_2026-11-13.md`.
- 2026-11-23 (Hamm): moved to 0.5.11 while the S08 inputs (population map, per-class N, weights, drive, deep-power
  threshold) are unset. Closes when: as above (T-90).
