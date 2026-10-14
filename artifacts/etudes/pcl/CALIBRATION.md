# Lane 2 CALIBRATION.md — PCL Fig. 3 on jaxfne core grammar (residual protocol)

Lane: reproduce PCL Fig. 3 (N'dri et al. 2025) with z from jaxfne core only.
Baseline commit: 3dd02ed060390ecf711e04860a1d6d46e854ef83. All changes uncommitted.
Prereg: `prereg/P3_fig3_v5.md` (signed v5); scored result: `prereg/P3_fig3_v5_result.md`
(verdict FAIL — T2 fails both sets; T1, T3, T4 pass; nothing tuned after scoring).

## Estimand table

| Estimand | y (published) | z at θ₀ (oracle, v4 runs) | z at θ* (core) | e (band) |
|---|---|---|---|---|
| T1 R_ori, simple / complex | qualitative (C1) | 3.57 / 5.30, PASS | missing (no core network run) | as v4 |
| T2 SF curve, simple (64 cells) | digitized Fig. 3c | 10/16 points, SSE 0.349 = 1.50 × flat, FAIL | missing | \|ours−fig\| ≤ 0.04 + 2·√(se_fig² + se_ours²), ≥ 80 %, SSE ≤ 0.5 × flat |
| T2 SF curve, complex (128 cells) | digitized Fig. 3d | 12/15 points, SSE 0.204 = 0.52 × flat, FAIL (guard) | missing | same |
| T3 median X, simple / complex | qualitative (C2) | 0.53 / −0.12, PASS | missing | as v4 |
| T4 median L complex − simple | qualitative (C3) | 0.83 vs null 99th 0.42, PASS | missing | complex above simple, above null |

θ₀ = trained cSTDP3 weights (`F:/warehouse/pcl/.../add_simulations_cSTDP/cSTDP3`),
port 5d101eb3, `runs/fig3_cSTDP3` (3168 tasks, camera-0 feed, learning off).

## Core modules added (exact oracle math at edge scale)

- `jaxfne/emitters_lif.py`: current-based LIF with PCL refractory terms, clock-driven
  `lax.scan` translation; function-level only, no construct route (OPEN).
- `jaxfne/hdp_rules_pcl.py`: registered core rule `pcl_stdp`, update copied exactly
  from `artifacts/etudes/pcl/pcl_hdp_rule.py` (whose `register()` now skips it).
- Exports: re-export in `jaxfne/emitters.py`, +5 ADVANCED in `jaxfne/public_surface.py`,
  contract artifact regenerated (288 → 293). No existing public function, default or
  numeric touched.
- `--engine {numba,core}` in `port/run_tuning.py` (default `numba`, behaviour unchanged);
  `core` refuses the tuning run with CORE_GAP; manifest records `engine`.

## Simplifications and fidelity fixes (with file)

- Clock translation (`emitters_lif.py` docstring): decay every step; the k arrivals of a
  step share one subtraction of k·(R+A); per-event C++ order inside a step not modelled.
- Simple threshold 10 mV from paper Table 2, not the default config's 30 mV; per-network
  values ride overrides. Simple adaptation increment defaults to 0 (default ETA_SRA).
- LTD test expectation corrected to exp(−0.5/7) (trace enters the arrival step at 1.0);
  rule math unchanged.
- Strict V > thr; floor −20 mV; t_last NEVER=−1e30 (never −1e30).

## D, θ*, a/X

- D = {} — no network-scale calibration attempted: there is no core mapping to tune yet.
- θ* = θ₀.
- X (the finding): the depth-sequential local-inhibition cascade (SpikingNetwork.cpp:41-67,
  visit order with R_ROUNDS rounds), the cover-table feed and the complex-layer pooling
  (`port/pcl_event_ref.py:41-179, :474-489`) have no mapping onto the edge-list core kernel.
  |a| unquantified — no partial mapping exists to score. This is the protocol's step-6
  outcome: the paper's model needs something the core grammar does not yet state.
- Held-out result: none (no fit, nothing to hold out). Band: the oracle T2 band above stands.

## Claim

Needs X: full-network core cascade mapping (depth-sequential local inhibition, cover
tables, complex pooling as core-expressible dynamics). Core z at Fig. 3 scale is missing;
edge-scale core z (single cell, STDP pairs, delayed drive) is proven by the tests below.
Oracle v5 verdict (FAIL on T2) stands unchanged.

## Files and tests

- `jaxfne/emitters_lif.py`, `jaxfne/hdp_rules_pcl.py` (new); `tests/test_emitters_lif.py`,
  `tests/test_hdp_rules_pcl.py` (new, 13 tests); `port/run_tuning.py` +3 engine tests.
- Baseline demo: new tests fail on baseline without the modules
  (`ModuleNotFoundError: No module named 'jaxfne.hdp_rules_pcl'`, 2 collection errors).
- `tests/test_emitters_lif.py tests/test_hdp_rules_pcl.py`: 13 passed in 25.50s (WSL gate venv, CPU).
- With exports: 32 passed in 49.81s (both new files + contract + snapshot).
- `port/test_run_tuning.py`: 14 passed in 17.63s (Windows, oracle REF counts intact).
- GPU time: 0 — every run in this lane was CPU (JAX_PLATFORMS=cpu / Windows).
