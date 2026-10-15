# 24-W16-6 receipt — BOUNDED CLASSIFICATION EXPERIMENT (verdict recorded)

**Branch:** `dev`. No product code changed (experiment-local remap helper
in the test file only). No augmentation implemented or proposed.

## Design (human-authorized)
- Operator: spatial decimation, keep every k-th neuron in x order, edge
  remap through EDGE-01 selector semantics (`model.select(ids=…)` parity
  asserted); k ∈ {1, 2, 4} ordered; N=48 laminar column, 800 ms, dt 1.0,
  4 deterministic seeds, noise_scale=0.
- Primary: `spike_rate_hz_mean`; error |Δ|; bias/variance/absolute across
  seeds. Criterion (experiment-only): ε_r = max(0.5 Hz, 0.1·s_full).
- Negative control: laminar LFP proxy, probe geometry preserved
  (fixed linspace contacts); normalized waveform error + amplitude error;
  no ε gate.

## Results
- Full rates [9.167, 9.167, 9.167, 9.141] Hz; s_r = 0.011 Hz → ε_r = 0.5 Hz
  (floor binds; deterministic-noise condition recorded).
- k=1: identity remap, rate exact.
- k=2: bias −0.046, abs 0.046 ≤ ε ✓; LFP NWE 0.347, amplitude error 0.503.
- k=4: bias +0.319, abs 0.319 ≤ ε ✓; LFP NWE 0.606, amplitude error 0.746.
- Cost: recording bytes scale exactly with stride (153600 → 38400);
  wall time flat at this scale (JIT-dominated, 0.13 → 0.12 s).

## Verdict
**SAFE_FOR_TESTED_SELF_AVERAGING_OBSERVABLE** — and the classification
holds: the geometry-sensitive observable diverges structurally (waveform)
and in scale (amplitude) where the self-averaging rate stays within
criterion. Scope: one config, N=48, short horizon, deterministic drive.
No universal factor; no augmentation proposed (no correctable error
demonstrated — the LFP gap is structural absence, not a removable bias).

## Evidence
`tests/test_w166_decimation_classification.py` (4 tests: validity,
stride-1 identity, error-curve reporting, cost reduction).
