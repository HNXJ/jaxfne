# P-053 sparse-weight fixture: result

Verdict: **NOT_REPRODUCED** under criteria `P053_sparse_fixture_criteria.md`, Amendment 1 (run 2). Scope: this means only "under Amendment 1, on the declared variants, seeds and N, arms matched within rtol 1e-12 with engagement and control met". It is not a refutation of P-053.

Status of the run: Amendment 1 is a diagnostic validation linked to run 1, not a confirmation under the original criteria. N = 170 is exploratory and was chosen by probing (see Event count).

Baseline: jaxfne worktree HEAD `4f5a8bbe55f40f56a2fde9968352b645147a1026` (criteria commit `3cc29785`). The test file is new and uncommitted. Mac, numba 0.68.1, venv `artifacts/_temp/venv`.

Test: `artifacts/etudes/papers/pcl/port/test_p053_sparse_fixture.py`. Core arm: `run_tuning.run_core_events(..., learning=True)`. Oracle: `pcl_event_ref.PCLEventRef.run` on a deep copy, learning on. Network: `PCLEventRef(seed=s, learning=True, layer_inhibitions=[["local","lateral"],["local","topdown"]])`, default STDP modes. Events: N = 170 at one pixel, 20 us spacing.

## Run record

- Run 1 (criteria before Amendment 1, N = 170): all 12 equality tests (V0 to V3, seeds 0 to 2) passed; the declared negative control (+0.4% on the largest LAT entry, core arm only) changed neither spikes nor `w_s` beyond roundoff. Verdict INVALID under the original criteria. Kept as a record.
- Run 2 (Amendment 1, this file): P5 engaged in every seed; all 12 equality tests pass; the `w_s` control detects in every seed (threshold 1e-6, owner ruling); pathway isolation reported (REPORTING ONLY, no gate). Verdict NOT_REPRODUCED under Amendment 1, with the scope stated above.

## Event count (exploratory)

| N | V0 result (seeds 0 to 2) |
|---|---|
| 60 | 0 simple spikes in every seed (run 1) |
| 120, 160, 165, 166, 167 | no spike failure; `wi_s` not moved in 2 seeds |
| 168, 169 | `wi_s` not moved in 1 seed |
| 170 | all preconditions pass in every seed |
| 180, 200 | all preconditions pass in every seed |

170 is the smallest value probed that passes. Values 61 to 119 and 121 to 159 were not probed; monotonicity is inferred.

## P5 engagement (oracle only, V0 vs V3, Amendment 1)

| seed | spikes equal | w_s max rel (index) | wi_s max rel (index) | engaged |
|---|---|---|---|---|
| 0 | False | 4.678e+00 (12,0,6,2) | 5.826e-01 (20,47) | yes |
| 1 | False | 5.174e+00 (62,0,6,2) | 1.268e+00 (33,11) | yes |
| 2 | False | 5.086e+00 (37,0,6,9) | 5.942e-01 (15,21) | yes |

No seed is vacuous.

## Core arm against oracle, every variant and seed

Tolerance: spikes exact; `w_s`, `wi_s` at rtol 1e-12, atol 0 (all 12 passed). Relative difference is |a-b|/|b|, absolute where b = 0.

| variant | seed | spikes s/c | spikes equal | w_s max rel (index) | wi_s max rel (index) |
|---|---|---|---|---|---|
| V0 | 0 | 3/14 | True | 1.094e-15 (8,1,8,8) | 3.563e-16 (9,49) |
| V0 | 1 | 3/13 | True | 8.818e-16 (54,1,1,3) | 4.547e-16 (16,39) |
| V0 | 2 | 3/11 | True | 8.629e-16 (37,1,5,3) | 3.614e-16 (15,23) |
| V1 | 0 | 7/26 | True | 1.094e-15 (8,1,8,8) | 4.384e-16 (47,53) |
| V1 | 1 | 7/24 | True | 9.787e-16 (54,1,5,3) | 4.433e-16 (54,58) |
| V1 | 2 | 8/28 | True | 7.372e-16 (40,1,9,8) | 4.628e-16 (15,38) |
| V2 | 0 | 4/21 | True | 1.094e-15 (8,1,8,8) | 3.796e-16 (51,45) |
| V2 | 1 | 4/20 | True | 7.364e-16 (43,1,2,9) | 0.000e+00 (0,0) |
| V2 | 2 | 4/21 | True | 8.847e-16 (7,1,7,7) | 4.357e-16 (43,29) |
| V3 | 0 | 7/26 | True | 1.094e-15 (8,1,8,8) | 4.384e-16 (47,53) |
| V3 | 1 | 7/24 | True | 9.787e-16 (54,1,5,3) | 4.433e-16 (54,58) |
| V3 | 2 | 8/28 | True | 7.372e-16 (40,1,9,8) | 4.628e-16 (15,38) |

Observation (worker): V1 and V3 print identical counts and max-relative values and indices in every seed.
Independent review (oracle-only probes, 2026-11-15, not in the committed test): that identity is a coincidence of the core-vs-oracle comparison in seed 0, where V1 differs from V3 (spike times by up to 280 us, `w_s` by up to 0.115 absolute); in seeds 1 and 2 V1 equals V3 bitwise. None of the 5184 retained V1 LAT entries has a spiking source in any seed, so V1's LAT half is never consumed. V1's TD half is consumed in seed 0 (576 retained entries with a spiking complex source), marginally in seed 1 (64, output equal to V3), not in seed 2. V3 is not silent: STDP LTP lifts zero weights (441 nonzero-weight `inhibit_simple` calls), so "hand-zeroed" holds at t = 0 only. Consequence for inference: V1 adds information beyond V3 only in seed 0, only through TD. The sparse content that exercised the core arm is V2 (all seeds) and V1-TD in seed 0.

## Negative control (Amendment 1): +0.4% on one `w_s` element the oracle moved, core arm input only

Detection threshold: measured `w_s` max relative difference >= 1e-6 (owner ruling).

| seed | element | oracle w_s final (init) | spikes equal | w_s max rel (index) | detects (>= 1e-6) |
|---|---|---|---|---|---|
| 0 | (9,0,6,9) | 2.45241825195 (0.499211767059) | False | 4.137e+00 (51,0,6,2) | yes |
| 1 | (16,0,6,2) | 2.44476144695 (0.493708345886) | True | 1.340e-04 (16,0,6,2) | yes |
| 2 | (15,0,6,9) | 2.44431985988 (0.507072240855) | False | 4.033e-01 (43,0,6,9) | yes |

The control detects in 3 of 3 seeds.

## Pathway isolation (REPORTING ONLY: no gate, no verdict change)

Oracle only, one pathway removed, each compared with V0 of the same seed. Engaged alone = spikes differ, or `w_s` or `wi_s` differ beyond rtol 1e-12.

| pathway | seed | spikes s/c | spikes equal | spike times vs V0 | w_s max rel (index) | wi_s max rel (index) | engaged alone |
|---|---|---|---|---|---|---|---|
| LAT=0, TD intact | 0 | 3/14 | False | max rel 4.412e-02 at (1,) | 1.908e-02 (7,0,6,2) | 8.131e-04 (9,8) | yes |
| LAT=0, TD intact | 1 | 3/13 | False | max rel 6.154e-02 at (1,) | 2.674e-02 (54,0,6,2) | 0.000e+00 (0,0) | yes |
| LAT=0, TD intact | 2 | 3/13 | False | lengths differ (14, 16) | 2.542e+00 (51,0,6,2) | 8.920e-05 (15,43) | yes |
| TD=0, LAT intact | 0 | 7/26 | False | lengths differ (17, 33) | 4.831e+00 (12,0,6,2) | 5.777e-01 (20,47) | yes |
| TD=0, LAT intact | 1 | 7/24 | False | lengths differ (16, 31) | 5.332e+00 (62,0,6,2) | 1.232e+00 (33,11) | yes |
| TD=0, LAT intact | 2 | 7/28 | False | lengths differ (14, 35) | 3.860e+00 (9,0,6,9) | 1.556e-01 (4,37) | yes |

Each pathway engages separately in every seed at this fixture. These are single-pathway oracle removals, not the V1 to V3 sparse patterns; conclusions about the arms under V1 to V3 are limited to the combined and sparse forms tested.

## Run-1 LAT perturbation, reported as an engagement measurement only (not a control)

+0.4% on the largest-magnitude nonzero LAT entry, core arm only, V0.

| seed | spikes equal | w_s max rel (index) | wi_s max rel (index) |
|---|---|---|---|
| 0 | True | 1.094e-15 (8,1,8,8) | 3.563e-16 (9,49) |
| 1 | True | 8.818e-16 (54,1,1,3) | 4.547e-16 (16,39) |
| 2 | True | 8.629e-16 (37,1,5,3) | 3.614e-16 (15,23) |

This perturbation reaches no compared quantity in 170 events, even though removing LAT alone engages (pathway table). Cause (independent review, observed in oracle-only probes): the perturbed entry is `argmax(|LAT|)` on the flat array, `LAT[0,64]` in every seed; source 64 never spikes (spiking simple cells are about 1352 to 1470) and `LAT[m,n]` is read only when source n spikes, so the entry is never consumed. It is not evidence of a non-engaged lateral path. A control on a consumed entry (`LAT[m, n_spk]`) was proposed and not run.

## Verdict logic

- V0 matches in all seeds: yes. Preconditions in V0: pass in all seeds. Negative control detects: yes, 3 of 3 seeds (threshold 1e-6).
- P5 engaged: 3 of 3 seeds, so no seed is VACUOUS.
- V1 to V3 match in all 9 seed pairs.
- NOT_REPRODUCED (Amendment 1): all non-vacuous variants match in all non-vacuous seeds, with 3 non-vacuous seeds. Scope as stated at the top; P-053 stays open for fixture forms not tried.

## Commands

- Run 1 (criteria as at `3cc29785`, N = 170): `python -m pytest -q -p no:cacheprovider <new file>` gave `3 failed, 15 passed in 178.71s (0:02:58)`. The 3 failures were the negative control.
- Probes, `-k preconditions`: N = 200 `3 passed, 15 deselected in 32.97s`; N = 180 `3 passed, 15 deselected in 32.82s`; N = 170 `3 passed, 15 deselected in 32.69s`; N = 120 `2 failed, 1 passed, 15 deselected in 30.68s`; N = 160 `2 failed, 1 passed, 15 deselected in 32.03s`; N = 165 `2 failed, 1 passed, 15 deselected in 32.37s`; N = 166 `2 failed, 1 passed, 15 deselected in 32.66s`; N = 167 `2 failed, 1 passed, 15 deselected in 32.23s`; N = 168 `1 failed, 2 passed, 15 deselected in 33.12s`; N = 169 `1 failed, 2 passed, 15 deselected in 32.51s`.
- Run 2 (Amendment 1 file, 24 tests, `-s`, numbers): `24 passed in 174.93s (0:02:54)`.
- Run 2, full Accept, before the threshold and pathway edits (24-test file): `66 passed, 3 skipped in 509.13s (0:08:29)`.
- Run 2 (final file, 30 tests, `-s`, numbers): `30 passed in 176.43s (0:02:56)`.
- Final full Accept on the final file (rc 0): `72 passed, 3 skipped in 508.28s (0:08:28)` (30 new, 42 baseline passed, 3 baseline skipped).

Command form: `PYTHONDONTWRITEBYTECODE=1 artifacts/_temp/venv/bin/python -m pytest -q -p no:cacheprovider <files>`.

## Siblings

- Reused from `test_run_tuning.py`: `INH` and `_counts`, imported. `rt._gather` and `run_core_events` come from `run_tuning.py`.
- Sites inspected, not reused (other fixtures): `test_run_tuning.py:475` (inline literal), `test_pcl_trained_load.py:176` (`_scaled_all_branches`), `test_run_r2_reference.py:409` (chain base).
