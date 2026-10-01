# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-024) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to `artifacts/archive/0.5.x/issues_closed.md`.

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

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
- **evidence:** refreeze 2026-09-30; guard added to
  `scripts/mcc3_10s_scientific_checkpoint.py` (rerun loads the tree, values
  unchanged); `scripts/regenerate_hdp_population_restoring_metrics.py` has it
- **possible future change:** add the guard to the remaining scripts with a
  test that each script importing jaxfne inserts the repo root; open

---

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
  short-list goal 3), rewriting the pinning tests; or rename and fix the
  docstrings. Owner decision on deprecation; open
- **resolution:** owner ruling 2026-10-01: deprecated with corrected docstrings, values unchanged for one release; replacement jnwb spike_count_correlation (window_s, bin_ms), fano_factor (onsets_s, window_s, summary), network_burst_index (window_s, bin_ms, threshold_hz, min_duration_ms) via jaxfne.jnwb_view.to_jnwb / to_jnwb_trials
