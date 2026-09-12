# 24-JAX-01 receipt — execution profile baseline (measurement programme)

**Branch:** `dev`. No product code changed (runner + gate test only).

## Measured table (N=64, T=500, float32 CPU; `scripts/perf/jax_profile_v0424.py`)
| path | first call | steady | output |
| --- | --- | --- | --- |
| baseline eager | 0.87 s | 145 ms | 0.26 MB |
| baseline jit | 0.37 s | 14 ms (~10x) | 0.26 MB |
| HDP eager | 1.47 s | 335 ms | 0.26 MB |
| HDP jit | 0.50 s | 46 ms (~7x) | 0.26 MB |
| batch vmap (3 seeds) | 1.12 s | — | 0.15 MB |
| batch loop (3 seeds) | 0.79 s | — | — |

## Findings for future JIT work
- JIT buys ~7–10x steady-state; first call pays compile (0.4–1.5 s here).
- **JIT is not free bits on HDP paths:** spikes bit-exact everywhere;
  V bit-exact on baseline, but HDP V differs by ≤3.1e-05 (59/480 entries,
  XLA reassociation). Pre-declared bound for 24-EQUIV-01: d ≤ 1e-4 on
  jit-vs-eager V for HDP paths; exactness retained for spikes and all
  baseline outputs.
- vmap == python-loop bit-exact (batch); vmap compile costs more upfront.
- Single-compile holds under the exception guard (pinned in test).
- `memory_report` byte agreement pinned (recording.V_m exact).

## Evidence
`tests/test_jax01_profile.py` (4): jit/eager (exact + ε-bounded),
vmap/loop exact, single-compile guard, byte agreement.
