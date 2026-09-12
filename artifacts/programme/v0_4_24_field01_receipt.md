# 24-FIELD-01 receipt — computational audit (NO_CHANGE)

**Branch:** `dev`. No product code changed.

## Cost decomposition (measured, float32 CPU)
Forward path `project_laminar_sources`: Gaussian kernel build O(C·N),
projection matmul O(T·N·C), CSD stencil O(T·C), invariant validation +
solution report (linear scans + host-sync assembly).

Steady-state at canonical scale (T=1000, N=64): full 3.05 ms =
kernel 0.74 + matmul 0.17 + validation 0.74 + CSD/report/rest ~1.4.
The essential math is ~6% of the call; dispatch + report assembly
dominate. Cold first call ~500 ms (dispatch compilation).

## Scaling (independent T/N/C)
- 4× T → 1.1× time; 16× C → 0.84× time (jitter): dispatch-bound
  regime, no superlinear blowup anywhere. FLOP-dominated knee not
  reached at canonical scale.
- End-to-end (N=64, T=1000): sim 0.8 s vs projection sub-second —
  source generation dominates; projection is not the bottleneck.

## Exact vs bounded
- Exact: chunked projection is bit-exact (linearity) — streaming field
  computation needs zero error (proven in test). Kernel depends only on
  (positions, C, width): reusable in principle.
- Bounded: none proposed. Kernel reuse across segments would save
  ~0.7 ms/call against cache-invalidation semantics — fails
  benefit-vs-concepts at canonical scale. Validation/report overhead
  stays: it is the S→F→P evidence, not removable cost.

## Verdict
NO_CHANGE. Geometry-sensitive quantities preserved by not approximating.
Revisit only with a hot-path profile showing projection dominant
(24-JAX-01 owns that measurement).

## Evidence
`tests/test_field01_audit.py` (5): T/C scaling guards, chunked
bit-exactness (all four proxies + kernel identity), CSD bound,
sim-vs-projection split with byte counts.
