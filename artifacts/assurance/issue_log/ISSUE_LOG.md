# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`; from 0.5.8 the batches sit in `artifacts/archive/0.5.x/issues_closed/`,
P-037: `issues_closed_2026-10-26.md`; P-045, P-046, P-049, P-050: `issues_closed_2026-10-27.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

Entry P-040 comes from the whole-project scan of 2026-10-18 (six Haiku workers,
read-only; each line was re-opened and confirmed by the dispatcher unless marked inferred).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- `destexhe2009/PREREG_P3.md:1` reads "draft, UNSIGNED — no scored run may use it";
  `CALIBRATION.md:3-4` scores E1-E4 against it (header "Verdict: OFF"), and its final section
  says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12` says "no verdict
  yet" and "28 passed" (30 pass today).
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.
- 2026-10-26: the Destexhe prereg moves to 0.5.9 (scope freeze, Hamm 2026-10-25); the entry stays
  open into 0.5.9.

### P-047

- Date: 2026-10-26. Type: FRICTION. Area: `Configuration.connections` selectors. Severity: low.
- A rule with `target={"cell_type": "all"}` compiles 0 edges on both compilers (`{}` selects every
  cell). It is recorded, not silent: the rule's status reads `compiled_no_matching_edges`, a design
  pinned by `test_connections_no_match_marked_compiled_no_edges`. A refusal would reverse it and break
  that test (9C-E1 core packet, 2026-10-27); left as is. An `"all"` value is not a wildcard.

### P-048

- Date: 2026-10-26. Type: SCIENCE. Area: Izhikevich initial state. Severity: low.
- `u0` keeps the preset value (E -13, I +6.5) when `cell_params` overrides `b` and when `random_v0`
  draws `v0`, so `u0 != b * v0` (the paper's `u = b.*v`); `cell_params` takes no `v0` or `u0`.
  The E1 library path needs a declared per-cell initial state; whether the default should follow
  `b` is Hamm's call (it moves existing realizations).
- 2026-10-27: `cell_params` takes `v0` and `u0` (a declared `v0` needs `runtime(random_v0=False)`);
  the default question stays open.
- 2026-10-28: ruled and fixed (Hamm: the default is `u0 = b * v0`). `_apply_default_u0`
  (`jaxfne/_construct_population.py`) runs in `construct()` after the canonical block, on the realized
  `b` and `v0`; a declared `u0` still wins. Tests: `tests/test_izh_default_u0.py`. The default route
  moves (`random_v0` is on by default); the three izhi pins in `tests/test_inputs_xsource_t24.py` were
  re-recorded.

### P-051

- Date: 2026-10-27. Type: FRICTION. Area: Izhikevich `edge_list` route. Severity: low.
- The edge route filters every edge with its mechanism tau (`g <- g exp(-dt/tau) + spikes`), and
  `weight_range` compiles only with a mechanism, so instantaneous coupling (Izhikevich 2003,
  the dense route's `W @ prev_spikes`) has no named form on it. At tau 5 ms the Fig. 3 network
  saturates (2000 Hz); the E1 pilot declares tau 1e-3 ms, where the decay is exactly 0 in float32
  at dt 0.5 ms (`izhikevich2003/core/run_izh_core.py`). Possible change: a named instantaneous
  receptor on the edge route, equal to the tau -> 0 limit.
- 2026-10-28: ruled and fixed (Hamm: a named instantaneous receptor). `instantaneous` on the
  `edge_list` route: the current at step t is the weighted sum of presynaptic spikes at t-1, with no
  carry-over. It compiles with `weight_range` and must declare `tau_ms=0.0`
  (`emitters.INSTANTANEOUS_MECHANISM`). Tests: `tests/test_izh_instantaneous_receptor.py`; dense and
  edge rasters are identical and V agrees at rtol 1e-10.

### P-052

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

### P-054

- Date: 2026-11-02. Type: SCIENCE. Area: Izhikevich presets `VIP` and `I` (`jaxfne/emitters.py:124,130`).
  Severity: medium; every realization with a VIP or I population. Found while answering jomission MIN-07.
- Observation: the presets carry Izhikevich 2004's class-1 excitable a 0.02, b -0.1, c -55, d 6
  (`izhikevich.org/publications/figure1.m`, panel G), but that panel integrates V' = 0.04V^2 + 4.1V + 108 - u
  + I, while jaxfne integrates 0.04V^2 + 5V + 140 - u + I. Rheobase (5 - b)^2/0.16 - 140 = 22.56 in jaxfne
  against (4.1 - b)^2/0.16 - 108 = 2.25 in the source, so a VIP cell at its preset drive 3.0 is silent
  (documented in `docs/reference/api/model/emitters.md`, "Default drives and rheobase").
- Minimal reproduction: one cell, Euler dt 0.1 ms, 1000 ms, u0 = b*v0, v0 in {-70, -65, -60}: 0 spikes at
  I = 4 and 22.5; 4 spikes in [200, 1000) ms at I = 23.
- Possible change (Hamm's call: it moves every VIP and I realization): keep the preset and record its
  provenance as "class-1 a,b,c,d in the standard equation", or give VIP its own source-faithful v equation
  coefficients, or take a VIP set from another source.
- 2026-11-02: ruled (Hamm: class-1 equation) and fixed. `VIP` and `I` carry `v_lin` 4.1 and `v_const` 108;
  `IzhikevichParams.v_lin`/`v_const` (None when all cells are standard) reach every Izhikevich integrator,
  `connect()`, checkpoints and the sanity runtime's VIP cells. Tests: `tests/test_izh_class1_presets.py`
  (fail on the pre-fix code). Not changed: `w3_stability_analysis` (analytic, standard cells by its
  arguments) and `streaming.py` (RS only).
