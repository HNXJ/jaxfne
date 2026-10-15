# 24-EQUIV-01 receipt — per-observable equivalence table (jit-vs-eager)

**Branch:** `dev`. No product code changed (one gate test only).

## Table (transformation: eager → jit; small nets, float32 CPU)
| observable | standard regimes | saturated corner |
| --- | --- | --- |
| spikes | exact (all regimes incl. boundary) | exact |
| V_m | exact on baseline; d≤1e-4 on HDP (worst observed 4.6e-05) | d≤1e-2 (observed 2.2e-03) |
| H | exact on legacy HDP; d≤1e-6 on registered (observed 2.4e-07) | d≤1e-4 (measured in-corner) |
| W/Theta | exact (all regimes incl. clips) | exact |
| sources | d≤1e-4 (mechanism-consistent; observed exact) | (same bound; corner V-driven) |
| fields/LFP | analytic d ≤ ‖K‖_F · d_sources, checked numerically | inherits sources bound |
| aux | d≤1e-6 (same reassociation class as H) | — |
| continuation leaves/keys/delay_state, disabled identity, strided frames,
  record toggles | exact (HDP-01/STOCH-01/REC-01/LAW-01 receipts; not re-proven) | — |

## Adversarial findings (kept, not smoothed)
- Saturation amplifies absolute reassociation ~50× (2.2e-03 vs 4.6e-05)
  while spikes stay exact: epsilon must be regime-aware, not global.
  Corner bound is empirical (margin ~5×), stated as such.
- Baseline V/H/W exactness was NOT loosened to accommodate HDP corners.

## Evidence
`tests/test_equiv01_table.py` (6): three HDP regimes, baseline exactness,
boundary-clip corner, LFP derived bound. JAX-01's ε_V=1e-4 confirmed as
the standard-regime bound (worst case 4.6e-05); H/W/aux bounds new here.
