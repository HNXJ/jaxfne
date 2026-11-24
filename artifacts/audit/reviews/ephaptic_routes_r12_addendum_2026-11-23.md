# Addendum to `ephaptic_routes_r12_2026-11-23.md`, 2026-11-23

The R12 route table is pinned write-once; these corrections come from the outside review of the batch through
0d0c0582 and from the P-069 fix that followed.

| Row | Correction |
|---|---|
| header | Line numbers are at da42b095, except `hdp_rules_ephaptic.py` (the tree carries the R13 docstring change). The P-069 fix shifts `_model_simulate.py` lines after 1691 by 6 and after 3797 by 11. |
| 1 | The `None` refusal for `v_rest` is at `hdp_rules_ephaptic.py:106-107`, not 103-104. |
| 5 | The `drive_bias` update at `emitters_lif.py:3217-3223` is drift and jump, clipped to the descriptor's `b_min`, `b_max` (`bias = jnp.clip(b_drift, b_min, b_max)`); for this rule the bounds are unbounded. "jump add" was wrong. |
| 11 | `simulate_adex_seed_batch` refuses `enable_hdp=True` since the P-069 fix. |
| 16 | A simulate-time `RuntimeConfig(enable_hdp=True)` on the homeostatic E/I route refuses since the P-069 fix. |
| 17 | Unchanged: Izh with `enable_hdp=False` and HDP params still runs silently; open as O-P069. |
| layout | The "Run-time evidence" paragraph sits between rows 17 and 18 and splits the table; rows 18 and 19 belong to the "Routes that refuse or ignore" table. |

Run-time evidence after the P-069 fix (`tests/test_hdp_rules_ephaptic.py`): the refusal tests for rows 11 and 16
are plain passing tests; row 17 stays a strict xfail pinned to P-069.
