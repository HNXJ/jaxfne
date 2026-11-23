# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`; from 0.5.8 the batches sit in `artifacts/archive/0.5.x/issues_closed/`,
P-037: `issues_closed_2026-10-26.md`; P-045, P-046, P-049, P-050: `issues_closed_2026-10-27.md`; P-048, P-051, P-054:
`issues_closed_2026-11-03.md`; P-040, P-057 to P-065: `issues_closed_2026-11-21.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

### P-047

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

### P-056

- Date: 2026-10-20, recorded 2026-11-06 (was O-56 on the `_temp_kernel-lfp` lane). Type: SCIENCE. Area: hybrid
  kernel LFP (`jaxfne.fields.kernel_lfp`) of a column on the AdEx route (Billeh 2020 wiring, Hagen 2016 cells,
  iSTDP settle). Severity: medium for any laminar-spectrum claim from that column.
- Observation: in the lane pilot, deep layers carry under 0.5% of the L2/3 LFP power; no deep alpha-beta peak.
- Expected: deep-layer alpha-beta power, as in laminar recordings.
- Possible change: settle the column longer, then attribute the deficit (inferred candidates: deep-layer rates,
  kernel amplitude of deep pyramids, coherence radius). Closes when a settled column shows deep alpha-beta power
  or the cause is named. Discriminator plan (static, not run): `p053_p056_source_audit_2026-11-13.md`.

### P-066

- Date: 2026-11-19. Type: BUG. Area: population construction, neuron positions, distance-based connectivity.
  Found by task-units-csd. Severity: medium for any run whose rules use Euclidean distance (localized sampling,
  `spatial_sigma`, radius gates) on the laminar construct path; affected runs not enumerated.
- Observation: x and y are sampled in mm (`[-column_radius_mm, column_radius_mm]` plus the area offset), z in
  the layer's `layer_fractions` range inside [0,1], never scaled by `column_height_mm`
  (`jaxfne/_construct_population.py:77-89,505-529`). `column_height_mm` is used only for layer volume
  (`jaxfne/builders.py:716-731`). `compile_connection_rules` takes Euclidean distance on x, y, z in "the same
  units as x/y/z" (`jaxfne/connectivity.py:886,1110-1115`), so depth counts 1/1.6 of its mm length at the default
  height. `NeuronalTensor` documents positions as mm (`neuronal_tensor.py:96-103`).
- Reproduction (`F:/warehouse/jaxfne/P-066_2026-11-19/probe_zextent.py`, `SHA256SUMS`; `build_laminar_column("V1", 60, geometry="laminar")`):
  known positive radius 0.25 -> 0.5 mm moves x from ±0.247 to ±0.495; height 1.6, 2.0, 8.0 mm leaves z at
  [0.0045, 0.9787] in all three. `tests/test_laminar_geometry_consumed.py` checks that height reaches metadata,
  not positions.
- Expected: one length unit for positions (mm, z scaled by `column_height_mm`), or z declared relative and
  distances refused or computed per axis. The field path reads relative depth (contacts in the same [0,1] axis,
  `fields/proxy.py:266,283-285`) and must move with the choice.
- Ruling (Hamm 2026-11-19): z in mm, scaled by `column_height_mm`; field proxy divides depth by the height.
- Repair (implemented 2026-11-19, checkpoint `31ca6c39` on `_temp_tfne-manuscript`; not merged):
  - `_construct_population.py`: layer z = fraction x `column_height_mm` (default 1.60, as in the uniform path).
  - `fields/proxy.py`: `project_laminar_sources(..., column_height_mm=None)` divides depth by the height when given;
    `project_model_sources` passes `model.params["column_height_mm"]`.
  - `_construct_core.py`: population models carry `params["column_height_mm"]` (not with a geometry override).
  - `_construct_connectivity.py connect()`: carries the height; refuses mixed laminar/non-laminar members and
    mismatched heights.
  - Tests: `tests/test_laminar_z_mm.py` (3), golden key set in `tests/test_construct_golden_snapshot.py`.
- Measured (before = `0a0d5ab5`, after = `31ca6c39`; script `_temp_p066_compare.py`, diff `_temp_p066_diff.py`,
  scratch only, seed-fixed sources, `project_model_sources` on both trees):
  | scenario | positions | field phi | note |
  |---|---|---|---|
  | laminar, height 1.0 | identical | identical | identity check |
  | laminar, height 1.6 (default) | z x1.6 (max 0.979 -> 1.566) | max rel. diff 1.8e-7 | float32 rounding |
  | laminar, height 2.0 / 8.0 | z x2 / x8 | identical | depth = fraction again |
  | laminar, radius 0.5, height 2.0 | z x2; x,y as before | identical | |
  | merge of two height-2.0 laminar models | z x2 | identical | |
  | merge of height 2.0 with height 8.0 | baseline merged silently | now refused | intended |
  | uniform3d, default 1.6 and 2.0 | identical (already mm) | **changed**, max rel. 0.71 and 1.14 | see below |
  - Unexpected: uniform3d positions were already mm before the change, so the baseline projection read them as
    fraction. The change divides by the height, which alters the uniform3d field output. Intended by the same
    logic; not listed in the original observation; needs review.
  - Not measured: realized connectivity of distance rules (`spatial_sigma`, `spatial_radius`) changes, since z
    now spans the column height. Probability-only rules do not read positions.
- Full suite on checkpoint `31ca6c39` (clone): 14 failed, 8391 passed, 49 skipped.
  Baseline `0a0d5ab5` on the same 8 files: 2 failed (`test_scaling_benchmark`, subprocess cannot import
  jaxfne; environment, pre-existing). So 12 failures are from this change, classified:
  - Two real regressions, found and fixed (commit on top of `31ca6c39`):
    1. `_model_simulate.py` (`simulate(record_fields=True)`, two sites) projected mm positions with no height
       (canary `field_lfp_csd`, csd_rms 53.9 vs 201.7). Now passes `column_height_mm`.
    2. `_construct_core.py` canonical PV<->E gate (`exp(-(dz/0.15)^2)`) read raw z; it is a relative-depth
       width. Now divides by the height for layered models; uniform_3d keeps its pre-change units.
       Canary `hdp_plastic_w` raster hash matches the register again.
  - Generated index: `docs/reference/errors.md` stale; regenerated with `scripts/generate_errors_index.py --write`
    (`test_generate_errors_index` 6 passed).
  - Intended unit pins, updated (approved by Hamm 2026-11-19), each to the new z units:
    `test_fields_positions_g8` x2 (delegate reference now passes the column height),
    `test_neuronal_tensor` x1 (z in [0.2h, 0.6h]), `test_neuronal_tensor_connectivity` x1 (z in 3 + [0.5h, 0.6h]),
    `test_tfne_parameter_transfer` x2 (executed z divided by h against the fractional domain),
    `canary_register.json` jdna `positions_mean/std` (only these changed; checked before and after),
    `test_neuronal_tensor_lane6b` (digest split: edges digest unchanged `0bc6b793...`, measured equal on both trees;
    positions digest now `94cb0202...`).
  - Touched files: 8 test/register pins above, plus no other test edits.
- Affected subset after the pin updates: the 8 pinned tests pass (34 in the 5 files; canary 7/7).
- Call-site review (4 Muse lenses, 2026-11-20, read-only; claims checked by hand). Fixed:
  `project_sources_to_laminar_field` forwards `column_height_mm`; `CanonicalDataset` and `FrozenE4Trajectory`
  carry the height (default None = legacy fraction; new builds record it; npz round-trips it when present);
  `materialize_field`, `zero_field`, the receipt leadfield and the E4 field calls use it. Tests:
  `tests/test_height_callsites.py` (4). Not a defect: geometry override (`LaminarSourceGeometry` validates
  depth to [0,1], so positions are already relative).
- Remaining: full suite rerun after all changes (not yet run); connectivity before/after; uniform3d ruling;
  independent review; merge decision. Pre-existing, separate: `Pose3D` plane xz/yz remaps the depth axis
  (`neuronal_tensor.py:144-148`) while the proxy reads `positions[:, 2]`.
