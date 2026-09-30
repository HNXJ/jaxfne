# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-014, P-015) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md`. A new issue takes the next
free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

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

---

### P-021
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
- **expected behavior:** per-unit Fano across trials, network bursts with a
  used bin width, count correlation at a stated bin width; jnwb 0.2.8 lane D
  provides these (`fano_factor`, `network_burst_index`,
  `spike_count_correlation`)
- **evidence:** opencode sweep 2026-09-30, lines re-read: `metrics.py:197`
  vs `:255-275`, `:124-126`, `:87-88`
- **possible future change:** delegate to jnwb when 0.2.8 ships (todo 0d,
  short-list goal 3), rewriting the pinning tests; or rename and fix the
  docstrings. Owner decision on deprecation; open

---

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

---

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

