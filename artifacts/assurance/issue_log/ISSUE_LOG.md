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

### P-074

- Date: 2026-11-25. Type: BUG. Area: `with_rule_local_inhibition` (`jaxfne/connectivity.py:1746`) under
  `resolution="event_scan"`. Severity: medium (a valid declaration fails with `KeyError`, not its refusal).
- Observation: `validate_local_inhibition` returns `{"order", "resolution"}` for `event_scan` with no `rounds`
  (its docstring says so, `jaxfne/pcl_topology.py`). The patch read `normalized["rounds"]` unconditionally, so every
  `event_scan` declaration raised `KeyError: 'rounds'`. Found in passing by the F5 worker; reproduced in the lead tree.
- Reproduction: `pytest tests/test_pcl_topology.py -k event_scan_local_inhibition_patch_does_not_key_error` before the
  fix: 1 failed, `KeyError: 'rounds'` at `connectivity.py:1747`.
- Expected: no `KeyError`; the sign-only route refuses the declared spec with `ValueError` naming `local_inhibition`,
  as its sibling tests assert.
- Applied: `local_inhibition=dict(normalized)` in `449d59ab`; its test moved with the test folds in `898137c0`;
  review nits in `3918fd79`. Read from the validator's return values, the old dict and the new one carry the same
  keys in the same order for every non-`event_scan` spec (`rounds`, `order`, `resolution` if declared); not
  byte-compared. Broad gate at `3918fd79`: rc 0 (8505 passed, 54 skipped, 3 xfailed, 0 failed). Independent review:
  ACCEPT on `fef81762..898137c0` and on `3918fd79`. Resolved 2026-11-25; closed batch at the next seal.

### P-075

- Date: 2026-11-25. Type: SCIENCE. Area: AT-10-N20 atlas (`artifacts/atlas/at10_n20_055.py`; tracked result
  `artifacts/atlas/results/at10_n20_055.json`; receipt `artifacts/publication/atlas/AT-10-N20/atlas_run.json`, bound to
  474c4065). Severity: medium (the 0.5.10 atlas numbers and F2 would come from a superseded default).
- Observation: at HEAD the CPU run reproduces itself (two runs equal except `wall_s`) but not the tracked file:
  baseline 9.1736 vs 9.2374 Hz, hebbian_hdp 8.9989 vs 9.0911, noisy_hdp 9.1730 vs 9.2393; `n_significant` 11, 8, 11
  unchanged. The CPU run at the tracked commit `cd4453bd` reproduces the tracked rates exactly. The tracked JSON
  predates the `h_space` block (`ab2cd208`) and lacks three `method` labels that the run writes. The GPU is not
  reproducible: at HEAD, run 1 equals the CPU rates, run 2 does not (hebbian 9.0190; noisy 9.1800, `n_significant` 12).
- Cause (bisect, 19 CPU runs on baseline rate, both boundaries checked): the first commit that moves the rates is
  `b1d9dfe1` (T-R28 wave 2, Izhikevich: default `u0 = b * v0` per cell, the P-048 ruling of 2026-10-28, and an
  instantaneous receptor). The runner does not set `u0`, so the default reaches the result. Which of the two changes
  moves the rates is not separated. Monotonicity between the boundaries is assumed, not shown.
- Also: the verdict objects under `assay.arms.*.h_space.stability` differ between `cd4453bd` and HEAD (PERSISTING 7 to
  4, RETURNING 11 to 9, DIVERGING 0 to 5). The row-level SUPPORTED criterion is in neither the runner, the results file,
  nor the atlas, plan, fact or todo files searched, so whether the row verdict changes is unknown.
- Expected: AT-10-N20 regenerated at the frozen candidate SHA into a new bundle path (the generator refuses an existing
  AT directory), with the default change and the row criterion stated, and F2 rendered to a new file. Or Hamm rules
  that the atlas pins the pre-P-048 default, which is a runner change. Closes when one is recorded with its numbers.
- Evidence (scratch, not durable): `scratchpad/at_probe/` (`cpu_a.json`, `cpu_b.json`, `cpu_base_cd4453bd.json`,
  `bisect_r1..r3`), driver `scratchpad/_temp_bisect_round.sh`, runner `scratchpad/_temp_at_det.py`.

### P-074

### P-073

- Date: 2026-11-25. Type: BUG / SCIENCE. Area: `experimental_poisson_1d` (O-GPU (3)). Severity: medium (blocks P2/FNO
  admissibility). Layered fixture of `tests/test_experimental_poisson_1d_convergence.py`, N 11-321; paths CPU and GPU
  float32 (GPU default and highest+deterministic equal), float64; GPU run twice, same values.
- Status gate: float32 residual / 1e-3 at N=81 is 0.49 (CPU), 1.21 (GPU); fails at N=61, 81 and >= 161 (GPU), >= 161 (CPU).
  Normwise backward error 0.2-8.9 eps32 for N <= 201 (backward stable); the rounded exact solution leaves <= 8% of tol
  for N <= 161, so the excess is the solver's, not the residual evaluation's. Flux criteria pass for N <= 161 on every path (GPU N=81: mean 4.1e-4 of the
  1e-3 rtol). The status is not scale invariant: at N=81 Q -> 2^k Q scales the residual by 2^k at fixed backward error
  (CPU flips at Q=8, GPU at Q=2).
- Discretization: the fixture's discrete solution is exactly the continuum one at the nodes (analytic recursion vs
  float64 lstsq <= 4.7e-12), so it cannot see discretization error. Smooth oracle cos(k pi x): first order with a
  full-weight boundary source, second order with the half-cell weight (float64, N 21-321, ratio 2.0 vs 4.0); cos(2 pi x)
  with full weight is incompatible with the Neumann rows (node sum 4 pi^2) and fails at every precision.
- Rank cut: default `lstsq` rcond = eps32 N truncates the lowest physical mode once cond (0.86 N^2) > 1/(eps32 N):
  rank N-2 at N=241 and 321 on both backends, forward error 0.998, status "failed" (reported).
- Controls: interface sigma shift, 1/dx scaling, rcond 1e-3 and a 1e-3 smooth perturbation fail the status; a Dirichlet
  row at a zero-source node shifts only the gauge (residual and flux blind; forward oracle catches it). Explicit
  rcond = eps keeps the null vector on CPU (gauge error 0.2): rejected.
- Ruling (Hamm 2026-11-25, conditional): one float32 iterative-refinement step on the same min-norm `lstsq`, float32
  path only (float64 unchanged); frozen criterion kept; no rank-cutoff, tolerance, boundary or label change.
- Applied: `jaxfne/fields/solvers.py` (pending gate and review). Residual / tol at N=81: CPU 0.49 -> 0.073, GPU 1.21 -> 0.067.
  Forward error vs the exact discrete solution at N=81: GPU 3.5e-4 -> 4.3e-6, CPU 3.0e-5 -> 1.4e-5. Layered fixture:
  forward error and residual are not increased at any tested N <= 201 on either backend (all N checked, scratch
  script `/mnt/e/repos/_wt/jaxfne/_temp_diag_poisson/_temp_check_claims.py`, outside the repo); the changes at N=41, 161 and 201 are GPU 2.877e-5 -> 2.861e-5, CPU 5.497e-4 -> 5.403e-4,
  CPU 3.136e-4 -> 3.064e-4. Uniform branch: GPU N=41 forward error rises 3.931e-6 -> 4.160e-6; N=241 status flips to
  converged (residual/tol CPU 2.15 -> 0.17, GPU 6.26 -> 0.28).
- Caveat for the status: the residual gate now reads "converged" at CPU N=161 (5.4e-4 forward error) and CPU N=201
  (3.1e-4), where it read "failed" before. Forward error did not improve there. Whether the status may pass at those N
  is Hamm's call.
- Rank cut (N >= 241, layered): still "failed" at N=241 (304 x tol) and N=321 (264 x tol) on both backends. The uniform
  branch converges at N=241 with forward error 2.6e-5 (GPU) and 8.1e-5 (CPU).
- Tests: new oracle test `test_phi_matches_exact_discrete_solution_at_frozen_resolutions`; on the pre-change solver on GPU
  it and both frozen flux tests fail; on the refined solver all 7 tests in the convergence file pass on GPU and CPU.
  The new test is GPU-discriminating only: the pre-change CPU forward error at these N is below its 1e-4 threshold.
  Solver-dependent files on CPU: 252 passed, 2 xfailed. One transient failure of
  `test_precision_float64_converges_at_n300_when_x64_enabled` was seen in an independent review run; not reproduced in
  six later runs (three GPU, three CPU). Cause unknown; its subprocess timeout is 120 s (unverified hypothesis).
- Receipts: `artifacts/etudes/core_checks/fields/score.json` regenerated on CPU. `poisson_1d_flux_magnitude` error
  3.425e-05 -> 5.09e-07; `poisson_1d_flux_span` 2.570e-06 -> 1.118e-07. Pre-change values are at eb5cc9cd.
  The dashboard reads the fields lane at its pinned SHA, so it shows the old values until that lane advances (Hamm).
- Evidence: scratch scripts and jsonl in `/mnt/e/repos/_wt/jaxfne/_temp_diag_poisson/` (`_temp_oracle.py`,
  `_temp_candidates.py`, `_temp_uniform.py`; `oracle/*.jsonl`).
- Status ruling (Hamm 2026-11-25): `convergence_status` is `"converged"` exactly when the existing residual criterion
  (residual_norm < 1e-3, unchanged) holds. Forward error is a separate metric: it stays in the oracle test and in this
  entry, and is not part of the status. The uniform N=241 status change is disclosed (solvers.py, fields.md, changelog).
  No threshold, criterion or rank-cutoff change.
- Independent review: ACCEPT on 93e0257c (2026-11-25; two rounds, findings fixed in the docs commit).
- Gate: broad gate at `93e0257c` rc=0 (2026-11-25): pytest 8467 passed, 54 skipped, 5 xfailed, 0 failed; every
  lane rc=0; audit checks passed. Candidate `efc1c2d4` stopped at the public-docs language audit
  (`docs/start/faq.md`, "not validated"; reworded). Candidate `daa17386`: pytest 8467 passed, 54 skipped, 3 xfailed
  (5 at 93e0257c: two P-070 xfails became plain tests in `5f00da8d`), 3 failed: the errors-index check (2 tests) and
  the "X, not Y" docs check at `runtime.md:180`. `d5ae605f` regenerates `docs/reference/errors.md` and rewords that
  sentence; the 3 tests pass there (293 and 6 passed), and the docs-reading audits pass. The code is identical to
  `daa17386`, so no full pytest run is repeated; the dev push follows review and the state record.
- Expected: closes when the candidate gate passes and the dev push lands; then a resolution line and a move to a dated
  closed batch (the budget decision is pending, see the Solmind plan §10.8).

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
- Applied (pending broad gate and independent review). D2 in `16dc1b96`: `meg_proxy_probe` raises `ValueError` for any
  `orientation_convention` other than `"declared"`; this follows plan §10.8 and is to be confirmed. D3 and D7 docstrings
  corrected in `16dc1b96` (D7's constant is named, not declared as a parameter). D8 docstring in `5f00da8d`. D6: deferred
  2026-11-25 with labels unchanged; its release note is not yet written. Closes when the gate passes and that note exists.
- Residual, not changed (no label change in this batch): the CSD report still carries `stencil`
  `negative_second_difference_edge_padded` and `identity` `laminar_second_derivative` (`probes.py`, observation block
  ~649-655), which describe a second difference the function does not apply. `contact_depths_m` keeps a `_m` suffix
  while its docstring says the values are not metres.

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
- Expected: a `ValueError` for any value other than `mean_zero_neumann` / `mean_zero`.
- Applied: `5f00da8d` (pending broad gate and independent review). The refusal follows plan §10.8 (a public parameter is
  refused, not deferred). Hamm's ruling text does not name P-070; the reading is to be confirmed. Closes when the
  candidate gate passes and the review accepts `5f00da8d`.

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
