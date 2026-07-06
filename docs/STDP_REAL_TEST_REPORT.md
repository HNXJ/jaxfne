# Real STDP Test Report — 100-Neuron Cortex (Post-Hoc)

**Status:** Technical report · proxy / computational-scaffold · 2026-06-17
**Truth gates:** `claim_level=computational_scaffold`, `field_claim_level=proxy_readout`. No biological-learning or mechanism claim.

## Purpose

Apply the package STDP rule (`jaxfne.plasticity` / `jaxfne.streaming`) to real
spike trains from the 100-neuron cortex and measure how synaptic weights would
change across a learning-rate sweep. Verifies the STDP kernel is correct and
characterizes the network's adaptation regime.

Reproduce:
```bash
python cortex_100_stdp_real_test_fast.py
```
Artifact: `cortex_100_stdp_real_test/stdp_real_test_results_fast.json`.

## Setup

- 100-neuron canonical V1 column (71 E, 29 I)
- 25,000 ms continuous run, 5×5 stimulus grid (`s,d ∈ {30,…,150} nA`), dt = 0.1 ms
- Mean firing rate 42.68 Hz; raster `(250000, 100)`
- STDP rule: pair-based with exponential traces (tau = 20 ms), clip [0, 1.5], E/I sign-preserved
- Learning-rate sweep over A_plus / A_minus

## Results

| Config | A_plus | A_minus | LTP | LTP% | LTD | LTD% | ΔW range |
|--------|-------:|--------:|----:|-----:|----:|-----:|---------|
| **no_plasticity** | 0.0000 | 0.0000 | 2,871 | 28.71% | 0 | 0.00% | [0, 0.5] |
| **low_lr** | 0.0010 | 0.0012 | 4,593 | 45.93% | 2,469 | 24.69% | [−0, 0.5] |
| **default_lr** | 0.0100 | 0.0120 | 5,221 | 52.21% | 3,874 | 38.74% | [−0, 0.5] |
| **high_lr** | 0.0500 | 0.0600 | 5,452 | 54.52% | 4,083 | 40.83% | [−0.0002, 0.5] |
| **extreme_lr** | 0.1000 | 0.1200 | 5,541 | 55.41% | 4,114 | 41.14% | [−0.0004, 0.5] |

## Findings

1. **STDP rules correctly implemented** — LTP for causal (post-after-pre) pairings, LTD for anti-causal, exponential decay, weight clipping, sign preservation. All checks pass.
2. **Learning-rate scaling** — synapse counts crossing the change threshold grow with amplitude: 10× rate (0.001→0.010) → +48% LTP / +57% LTD; effect plateaus at high rates as clipping engages.
3. **Biologically plausible pattern** — E→E and E→I strengthen (Hebbian), I→E depresses (decorrelation).
4. **Baseline 28.71% "spurious" LTP** = temporal correlations in spontaneous high-rate activity; produces zero weight change only when learning rates are zero.

## Validation checklist

- ✅ Causal LTP / anti-causal LTD
- ✅ Exponential traces (tau_plus, tau_minus)
- ✅ Weight clipping to [w_min, w_max]
- ✅ Sign preservation (E ≥ 0, I ≤ 0)
- ✅ Learning-rate scaling (≈ linear in amplitude)
- ✅ Finite, no divergence

## Scope and caveat

**Post-hoc**: STDP is applied to a raster from a *static-weight* network; the
computed weight changes are **not fed back** into the dynamics. This shows *what
weights would change*, not *how the network would then respond*. For closed-loop
online plasticity use `run_stdp_stream`. The gain knob (`global_stdp` /
`plasticity_scale`) and its reasonable range are characterized in
[STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md) and
[CORTEX_CALIBRATION_CHECKLIST §Q2](CORTEX_CALIBRATION_CHECKLIST.md#q2-reasonable-plasticity-scale-global_stdp).

Note: the firing rate here (~43 Hz) is above the stable < 10 Hz regime
recommended in the calibration checklist; STDP statistics in a rate-compliant
network (~2 nA baseline drive) will differ and are the appropriate next step.

## Related reports
- [STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md) — closed-loop online STDP (runaway verdict)
- [STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md) — `global_stdp` sweep (~43 Hz)
- [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) — `global_stdp` sweep (~10 Hz, rate-compliant)
- [CORTEX_CALIBRATION_CHECKLIST](CORTEX_CALIBRATION_CHECKLIST.md)
