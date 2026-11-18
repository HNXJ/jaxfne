# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`; from 0.5.8 the batches sit in `artifacts/archive/0.5.x/issues_closed/`,
P-037: `issues_closed_2026-10-26.md`; P-045, P-046, P-049, P-050: `issues_closed_2026-10-27.md`; P-048, P-051, P-054:
`issues_closed_2026-11-03.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

Entry P-040 comes from the whole-project scan of 2026-10-18 (six Haiku workers,
read-only; each line was re-opened and confirmed by the dispatcher unless marked inferred).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- Re-checked 2026-11-12: `destexhe2009/PREREG_P3.md` no longer exists in the lane, so the row's original
  contradiction (a header reading "draft, UNSIGNED" while `CALIBRATION.md` scores E1-E4 against it) cannot be
  re-read at its cited path. `CALIBRATION.md:3-5` still states the verdict is OFF and unscored, against "the
  unsigned draft `PREREG_P3.md`". `docs/start/replications.md` now reads "pending". The lane is closed by the
  0.5.10 scope freeze (Hamm 2026-10-25) and its P3 scored FAIL, so no scored claim rests on the unsigned prereg.
- Resolution 2026-11-12: closed as superseded, not as fixed. The unsigned-prereg scoring it reports was already
  marked unscored by the 2026-10-18 ruling and the lane's own calibration header; the file it names has been
  removed, so no reader can now take it for a signed protocol. Any future Destexhe scoring needs a new signed
  prereg, which is T-32's 0.5.11 lane work.
- Original observation, kept as history:
  `CALIBRATION.md:3-4` scores E1-E4 against a prereg whose header reads "draft, UNSIGNED — no scored run may use
  it" (`:1`), and its final section says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12`
  said "no verdict yet" and "28 passed".
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.
- 2026-10-26: the Destexhe prereg moves to 0.5.9 (scope freeze, Hamm 2026-10-25); the entry stays
  open into 0.5.9.
- 2026-11-03: P3 v3 was signed and scored FAIL 0 of 5 seeds (RESULT_P3.md); lane closed by Hamm 2026-11-03. Destexhe status reads pending/closed. Closed.

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
  exactness; none for spikes. Status: reported by the PCL-R3-HARNESS worker; not reproduced. R3 (2026-11-03):
  the core arm is exact against C++ on training file 0 (max relative weight difference 5.3e-15, `CALIBRATION.md`
  R3), so the reported gap needs the sparse fixture to show; it stays open until that fixture runs.
- Minimal reproduction: [MISSING: the worker's fixture; it was not kept]. A synthetic lateral and top-down
  network with hand-zeroed sparse weights, 60 events, learning on.
- Expected: spikes exact and weights within rtol 1e-12, as with dense seeded init weights
  (`test_p_core_arm_equals_oracle_arm_on_chain`).
- Actual (reported): identical spikes; `w_s` differs by about 0.4%. Suspected cause: the order of STDP
  updates within an event step.
- Possible change: reproduce on a minimal fixture, then attribute in PCL R3 (`CALIBRATION.md` R3
  section), where the training files run both arms against the C++ chain.

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
  or the cause is named.

### P-057

- Date: 2026-11-17. Type: BUG. Area: public `simulate()` on the Izhikevich route with `enable_hdp=True` and a
  registered rule. Severity: medium for any claim that a zero-parameter rule run equals the rule-free run.
- Observation: a registered rule with every gain at zero does not reproduce the rule-free run through `simulate()`.
- Minimal reproduction: 100-cell `NeuronalTensor` (E 0.9, PV 0.1, one L4 layer), `construct(..., seed=0,
  duration_ms=200, dt_ms=0.5)`, then `simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)` with and without
  `runtime=RuntimeConfig(enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})` (defaults all 0).
  Reproduced twice (author and independent review, 2026-11-17).
- Expected: identical spikes (`N_system` null under matched PRNG and inputs).
- Actual: 211 spikes rule-free, 214 with the zero rule; `ephaptic_field` at `alpha=0` also gives 214. At kernel
  level with `noise_scale=0.0` the identity holds bit-exactly (`tests/test_hdp_reduction_identity.py`).
- Cause: unexplained. Membrane noise is excluded: both kernels draw it as `split(key)` then `normal(noise_key)`
  (`emitters.py:725`, `_hdp_registrable_kernel.py:189-195`) from the same `noise_kw` (`_pipeline.py:597-600`).
  Next: diff the two kernels' per-step `v` on this model and find the first step that differs. Closes when the
  zero-rule public run equals the rule-free run, or the difference is declared with its cause.

### P-059

- Date: 2026-11-17. Type: BUG. Area: `jaxfne/optim/core.py` `sdr_transform`, `gsdr_transform`, `agsdr_transform`
  (public Optax transforms; no `jaxfne/` caller, `Model.tune` does not reach them). Severity: medium for any use
  of these transforms on a pytree with equal-shape leaves.
- Observation: found by the 2026-11-17 architecture review (packet A, defect D1). One `key_delta` served every leaf
  of the noise `tree_map`, so equal-shape leaves drew identical noise. Direct `update()` check with zero gradients:
  two (4,) leaves starting at 0 and 1 got identical nonzero updates at steps 0 and 1 in all three transforms.
- Fix (Hamm 2026-11-17, scoped): `_leafwise_normal` draws each leaf from its own split key. Fixed-seed outputs
  change for multi-leaf parameter trees; a single-leaf tree keeps its draws (see P-060).
  `tests/test_optim_transform_leaf_noise.py` failed on the old code for all three transforms and passes;
  `scripts/score_objective_optimizer_checks.py --check` still matches its `score.json` (stochastic term off there).
- Left: the transforms' `update()` is not jittable (state dataclasses are not registered pytrees); recorded, not
  changed. Closed by the fix commit for the three transforms named; the sibling is P-060.

### P-060

- Date: 2026-11-17. Type: BUG (port fidelity). Area: `jaxfne/optim/gsdr.py:305-310` `gsdr_mcdp_transform`. Found
  by the outside review of the P-059 fix.
- Observation: the same pattern as P-059: one `key_delta` for every leaf of the noise `tree_map`, so equal-shape
  leaves drew identical noise.
- Source (Hamm 2026-11-17: check the source first): `HNXJ/GSDR` b4a96fae `gsdr/optimizers.py` splits one key per
  parameter leaf in SDR (:69-71), GSDR (:194-196) and AGSDR (:303-305). The shared draw was a port deviation.
- Effect on results: the signed P3 fit (`RESULT_P3_fit.md`, FAIL) passes one parameter vector
  (`fit/fit_gsdr_500.py:206`), a single leaf, so the defect could not correlate anything there; its verdict stands.
  `_leafwise_normal` draws a single-leaf tree from the key itself, so that fit replays its draws bit for bit.
  Tested: float32 only (`tests/test_optim_transform_leaf_noise.py::test_single_leaf_draw_is_unchanged`; outside
  review probe, 6 steps, `a_dynamic` on and off, byte-identical to 24a95963). Other dtypes hold by construction
  (same key, shape, dtype), untested. No caller passes a multi-leaf tree to this transform (`scripts/`,
  `artifacts/`, `jaxfne/` grepped 2026-11-17).
- Fix: `gsdr_mcdp_transform` uses `_leafwise_normal`; `test_gsdr_mcdp_equal_leaves_get_independent_noise` failed on
  the old code and passes; `scripts/score_port_parity_checks.py --check` still matches. Closed by the fix commit.
