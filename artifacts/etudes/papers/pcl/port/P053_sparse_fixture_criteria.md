# P-053 sparse-weight fixture: criteria (declared before any run)

Declared 2026-11-15 by the orchestrator under Hamm's chat authorization of 2026-11-15 (todo T-83). Baseline jaxfne 24a95963 (origin/dev). Mac, numba 0.68.1, CPU. No source under `jaxfne/` and no pinned record is edited.

Question: does the core arm (`run_tuning.run_core_events(..., learning=True)`) differ from the oracle (`pcl_event_ref.PCLEventRef.run`, learning on) when lateral and top-down weights are hand-zeroed to sparse patterns? P-053 reported identical spikes and `w_s` differing by about 0.4% on a 60-event fixture; the worker's fixture was not kept.
Estimand: per variant, spike equality (counts, times, cells) and the maximum relative difference of `w_s` and `wi_s` between the arms.

## Fixture

`net = ref.PCLEventRef(seed=s, learning=True, layer_inhibitions=[["local","lateral"],["local","topdown"]])`, seeds 0, 1, 2. Oracle on `copy.deepcopy(net)` with `run(t, x, y, p)`; core arm on `net` with `learning=True`. Same pattern as `_check_simple_stdp_against_oracle` in `test_run_tuning.py`. The event burst starts at 60 events at one pixel; if 60 events produce no spike or move no weight, the worker raises the count to the smallest value that does and records both numbers. The test asserts these preconditions inside itself: at least one simple-cell spike, `w_s` moved, `wi_s` moved, LAT and TD nonzero in the dense variant.

Variants of the lateral (`LAT`) and top-down (`TD`) weight arrays, applied to both arms identically before either runs:
- V0 dense control: no zeroing. Known positive: existing tests show this matches.
- V1 one nonzero per row: zero every entry except one per row (the largest magnitude).
- V2 random half: zero a seeded random 50% of the entries.
- V3 all zero: LAT and TD set to zero.

## Tolerance (fixed here)

Spike counts, times and cells: exact. `w_s`, `wi_s`: `assert_allclose(rtol=1e-12, atol=0)`.

## Verdicts

- REPRODUCED: V0 matches in all seeds and at least one of V1 to V3 differs in at least one seed. Report per variant and seed: max relative difference of `w_s` and `wi_s`, the index of the maximum, and whether spikes are equal.
- NOT_REPRODUCED: all variants match in all seeds. P-053 stays open only for fixture forms not tried; the row records exactly the variants and seeds that passed.
- INVALID: V0 differs, or any precondition fails in V0, or the negative control detects nothing.
- Negative control (must detect): the same comparison with one core-arm weight perturbed by +0.4% (one `LAT` nonzero entry) must report a difference; otherwise the comparison is vacuous.
- A REPRODUCED outcome is kept: the equality test is committed `xfail(strict=True)` with the measured numbers in its reason. No fix of the core arm or oracle is attempted in this packet.

## Amendment 1 (2026-11-15, after run 1 stopped INVALID; declared before any rerun)

Run 1 (worker, event count 170 because 60 events give 0 simple spikes in all seeds): all 12 equality tests (V0 to V3, seeds 0 to 2) passed at the declared tolerance, and the declared negative control (+0.4% on the largest `LAT` entry, core arm only) changed neither spikes nor `w_s` (differences at 1e-15), so the verdict was INVALID. Run 1 is kept as a record. Cause of the vacuous control is unknown; it suggests the lateral and top-down path may not engage at this fixture, which would make V1 to V3 vacuous too. New declaration:
- P5 engagement precondition, per seed, oracle only: V0 and V3 (all-zero `LAT` and `TD`) differ in spike times or in `w_s`/`wi_s` beyond rtol 1e-12. If P5 fails for a seed, that seed's V1 to V3 results are VACUOUS: reported, not counted toward NOT_REPRODUCED. If P5 fails for every seed, the verdict is VACUOUS (the fixture cannot test P-053; stop and report, do not enlarge the fixture without a new declaration).
- Negative control replaced by a control on the compared quantity: perturb by +0.4% one element of `net.w_s` that the oracle's run moved, in the core arm's input only; the comparison must report a `w_s` difference. The run-1 `LAT` perturbation is kept as a reported engagement measurement, not as a control.
- The event count is the smallest value that passes the preconditions among those the worker probed (170 in run 1); report it and the probes.
- V1 ties pick the first index; V2 zeroes a seeded random 50% of the nonzero entries.
- The result file reports, for every variant and seed, the maximum relative difference of `w_s` and `wi_s` and its index (also when they match), and P5's per-seed outcome.
- Verdict set gains VACUOUS; REPRODUCED, NOT_REPRODUCED and INVALID are as declared above, with NOT_REPRODUCED counting only non-vacuous seeds and requiring at least one.

## Outputs (the worker writes only these)

- `artifacts/etudes/papers/pcl/port/test_p053_sparse_fixture.py`
- `artifacts/etudes/papers/pcl/port/P053_sparse_fixture_result.md` (variants, seeds, numbers, verdict, command, SHA)
- The P-053 row of `artifacts/assurance/issue_log/ISSUE_LOG.md`: Status line and Minimal reproduction line only.
- Nothing under `scripts/tree_size_allowlist.json`, pinned prereg files, `jaxfne/`, or `artifacts/state.md`.
