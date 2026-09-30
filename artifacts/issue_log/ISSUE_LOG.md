# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-014, P-015) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md`. A new issue takes the next
free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

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
