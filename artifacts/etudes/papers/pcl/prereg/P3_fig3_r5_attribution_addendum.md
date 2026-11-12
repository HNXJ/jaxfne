# PCL Fig. 3: attribution addendum to the R5-approximate result

Addendum to `P3_fig3_r5_approx_result.md` (pinned, unchanged). A pinned record's bytes may not change, so the
attribution question that result left open is answered here as a new record. Scored 2026-11-12 at dev 4e601897;
nothing in the R5 verdict changes.

## Question

`P3_fig3_r5_approx_result.md` reported a complex-cell orientation collapse (T1 complex R_ori 5.30 under v4's
inhibition alias against 1.19 under the signed protocol's `no-local`) and named two candidates: the inhibition
setting, or tree drift between port 5d101eb3 (v4) and dev 07a90e70 (R5). It did not separate them.

## Runs compared

| Run | Files | `baseline_sha` | `inhibition` |
|---|---|---|---|
| `F:/warehouse/pcl/runs/fig3_cSTDP3` (v4) | 3168 | 5d101eb3 | absent (predates the flag; local on) |
| `F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal` | 3168 | 98261a3e | `no-local` |
| `F:/warehouse/pcl/r5/approx_core` (R5) | 3168 | 07a90e70 | `no-local` |
| `F:/warehouse/pcl/r5/approx_numba` (R5 oracle) | 3168 | 07a90e70 | `no-local` |

98261a3e is the parent of the commit that added the `no-local` flag, and 5d101eb3 is its ancestor. Between them
`pcl_event_ref.py`, `pcl_event_ref_numba.py` and `pcl_binned_jax.py` are the same blobs.

## Scored comparison (scorer `port/fig3_analysis.py --signed`, seed 0, 1000 null draws)

| Statistic | v4 (local on) | v4 tree, `no-local` | R5 core | R5 oracle |
|---|---|---|---|---|
| T1 simple R_ori | 3.569 | 37.099 | 38.109 | 37.099 |
| T1 complex R_ori | 5.297 | 1.188 | 1.188 | 1.188 |
| T2 simple R_sf | 1.450 | 1.634 | 1.649 | 1.634 |
| T2 complex R_sf | 1.461 | 1.125 | 1.125 | 1.125 |
| T3 median X simple | 0.530 | 0.465 | 0.473 | 0.465 |
| T3 median X complex | −0.122 | −0.171 | −0.172 | −0.171 |
| T4 median L simple | −1.017 | −0.829 | −0.816 | −0.829 |
| T4 median L complex | −0.189 | −0.947 | −0.981 | −0.947 |

Every pass/fail flag matches the R5 arms; `all_pass` false in all four.

## Controls

- The v4-tree `no-local` run's seven scientific arrays are array-identical to the R5 oracle arm in all 3168 files;
  only `wall_s` differs.
- Its pilot control (`diag_runs/full_pilot`, same tree, `--inhibition full`) reproduces all 16 v4 pilot outputs
  array-for-array, so the driver is v4-equivalent.
- The v4-tree `no-local` score was computed twice, independently, at dev 4e601897; every statistic agrees to 1e-12.
- `artifacts/etudes/papers/pcl/port/test_fig3_analysis.py`: 42 passed, exit 0.

## Attribution

Turning local inhibition off causes the complex-cell orientation collapse; tree drift does not. Simple-cell
orientation tuning moves the other way, from 3.57 to 37.1, so the two cell sets respond oppositely to the same
setting. Tree drift is excluded by event-exact reproduction over all 3168 tasks, not by decimal agreement.

Scope: this names the cause of the scored difference for this network and stimulus set. It does not identify the
responsible line in the authors' C++, it does not revisit the R5 verdict, and it does not make v4's alias correct —
the signed protocol's `no-local` is the C++ setting.

## Receipts

`F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal/manifest.json` (baseline_sha, inhibition) and its 3168 outputs;
`artifacts/etudes/papers/pcl/prereg/scores/fig3_v4tree_nolocal.json` (this score, pinned);
`artifacts/etudes/papers/pcl/prereg/scores/fig3_r5_approx.json` and `_oracle.json` (the arms compared).
