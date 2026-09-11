# 23-SIMP-04 receipt — NO_CHANGE

**Branch:** `dev` (on top of `7066e38`)
**Verdict:** no compact-tau specialization. No code changed.

## Method
Isolated micro-op `resolve tau(E) -> exp(E)` vs class-native
`exp(K) -> gather(E)` under `jax.jit`, steady-exec median-of-10,
E ∈ {1k, 10k, 100k}. Script: `probe_simp04.py` (temp; numbers below).

## Results (steady ms; ratio = current / class-native)
| mode | E=1k | E=10k | E=100k |
| --- | --- | --- | --- |
| per_edge random | 0.014 (no class form; ~E unique per R01) | 0.010 | 0.024 |
| sign_from_receptor K=2 | 0.009 vs 0.014 (0.62×, native slower) | 0.010 vs 0.015 (0.67×) | 0.032 vs 0.029 (1.10×, noise) |
| mechanism_table K=4 | 0.009 vs 0.010 (0.91×) | 0.017 vs 0.017 (1.05×) | 0.026 vs 0.024 (1.06×) |

- Numerical equivalence: exact (max abs diff 0.00e+00, all modes/scales).
- Absolute scale ~0.01–0.03 ms vs full-kernel wall ~10² ms: the decay op
  is <0.01% of step cost; transient memory delta is one (E,) array.
- Compile time not separately decisive: both forms are single fused ops.

## Why NO_CHANGE
No mode shows a gain above noise, let alone one covering a new
`resolve_edge_decay` helper plus edits at ~8 kernel call sites.
per_edge has no class form by construction (R01 already showed gather
slower for near-unique taus). Specialization rejected per mode, as
instructed — no universal winner required or found.
