# STDP `global_stdp` Scale Report — 100-Neuron Cortex

**Status:** Technical report · proxy / computational-scaffold · 2026-06-17
**Scope:** `claim_level=computational_scaffold`, `field_claim_level=proxy_readout`. No biological-learning or mechanism claim.

## What `global_stdp` is

`global_stdp` is the **plasticity gain** multiplying the STDP weight update:

```
dW_effective = global_stdp · (dW_LTP − dW_LTD)
```

It maps directly onto `jaxfne.streaming.run_stdp_stream`'s `plasticity_scale`
argument (the kernel computes `dW = plasticity_scale * (dW_ltp - dW_ltd)`). The
post-hoc sweep here uses the identical kernel math (exponential pre/post traces,
A_plus/A_minus, w_min/w_max clip, E/I sign preservation), computing the unit
update once and rescaling by `global_stdp`.

| `global_stdp` | Meaning | Effect on dW |
|---:|---|---|
| `0.0` | Plasticity **OFF** | `dW · 0 = 0` → weights frozen |
| `1.0` | **Default baseline** STDP | standard Hebbian update |
| `−1.0` | **Inverse / anti-Hebbian** | LTP ↔ LTD swapped |
| `2.0`, `10.0`, `100.0` | **Amplified** | 2× / 10× / 100× the update |

> **Regime note:** this sweep runs at **42.68 Hz mean** (10 nA baseline + 5×5
> stimulus), which the calibration shows is ~5× over the < 10 Hz stability
> budget. For the rate-compliant version (2 nA, 9.63 Hz) see
> [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) — the scale
> behavior is identical but the per-step signal is ~10× smaller.

## Setup

- 100-neuron canonical V1 column (71 E, 29 I), `tutorial_utils.build_laminar_column`
- Existing 5×5 `(s, d)` stimulus grid, `s,d ∈ {30,60,90,120,150} nA`, 10 nA E baseline
- One continuous 25,000 ms run, dt = 0.1 ms → raster `(250000, 100)`, mean rate 42.68 Hz
- Fixed rule: A_plus=0.01, A_minus=0.012, tau_plus=tau_minus=20 ms, w ∈ [0, 1.5]
- Only `global_stdp` is swept.

Reproduce:
```bash
python cortex_100_stdp_global_sweep.py    # sim + sweep → JSON
python cortex_100_stdp_global_figure.py   # → PNG
```
Artifacts: `cortex_100_stdp_global_sweep/{stdp_global_sweep.json, stdp_global_sweep.png}`.

## Results

| `global_stdp` | LTP | LTP% | LTD | LTD% | ΔW min | ΔW max | W̄ after | sign✓ | finite✓ |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **0.00** | 0 | 0.00% | 0 | 0.00% | +0.0000 | +0.0000 | 0.2241 | ✅ | ✅ |
| **−1.00** | 5895 | 58.95% | 2440 | 24.40% | −0.0000 | +0.0001 | 0.2241 | ✅ | ✅ |
| **0.50** | 2292 | 22.92% | 5549 | 55.49% | −0.0000 | +0.0000 | 0.2241 | ✅ | ✅ |
| **1.00** | 2440 | 24.40% | 5895 | 58.95% | −0.0001 | +0.0000 | 0.2241 | ✅ | ✅ |
| **2.00** | 2600 | 26.00% | 6069 | 60.69% | −0.0001 | +0.0001 | 0.2241 | ✅ | ✅ |
| **10.00** | 2996 | 29.96% | 6262 | 62.62% | −0.0006 | +0.0003 | 0.2241 | ✅ | ✅ |
| **100.00** | 3307 | 33.07% | 6566 | 65.66% | −0.0058 | +0.0031 | 0.2234 | ✅ | ✅ |

## Invariant checks (verified)

| Check | Result |
|---|:--:|
| `global_stdp = 0` → dW = 0 and weights frozen | ✅ |
| `global_stdp = 0` → W_after == W_before | ✅ |
| `global_stdp = −1` flips dW sign vs `+1` | ✅ |
| `global_stdp = 100` scales dW ≈ 100× vs `+1` | ✅ |

**Cleanest proof — inversion swaps LTP↔LTD exactly:**
```
global_stdp = +1.0 :  LTP = 2440 , LTD = 5895
global_stdp = −1.0 :  LTP = 5895 , LTD = 2440   ← exact mirror
```

## Interpretation

1. **OFF (0.0)** is a true null: zero change, identical to the static network.
2. **Baseline (1.0)** is **LTD-dominant** (59% LTD vs 24% LTP). At ~43 Hz with A_minus > A_plus, pre-after-post pairings outweigh post-after-pre — the rule depresses more than it potentiates, consistent with a high-rate, weakly-correlated regime.
3. **Inverse (−1.0)** mirrors to LTP-dominant.
4. **Amplification (2 → 100)** grows ΔW magnitude **linearly** (|ΔW mean| vs |global_stdp| has log-log slope ≈ 1 — a clean gain, no nonlinearity). Mean weight holds ≈ 0.2241 until 100× begins clipping at w_max (W̄ → 0.2234), the expected saturation.

## Numerical safety

All weights finite at every scale (incl. 100×); sign preservation holds (E presyn ≥ 0, I presyn ≤ 0); clipping engages only at 100×.

## Scope

Post-hoc: weight changes are computed from a static-network raster and not fed
back into dynamics. For closed-loop online plasticity (weights influencing
subsequent spikes), run `run_stdp_stream` with `plasticity_scale = global_stdp`
— the per-step update is identical, so these numbers are the per-step driving
signal of that loop. For a reasonable scale band see
[CORTEX_CALIBRATION_CHECKLIST §Q2](CORTEX_CALIBRATION_CHECKLIST.md#q2-reasonable-plasticity-scale-global_stdp).

## Related reports
- [STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md) — closed-loop online STDP (runaway verdict)
- [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) — same sweep in the rate-compliant ~10 Hz regime
- [STDP_REAL_TEST_REPORT](STDP_REAL_TEST_REPORT.md) — post-hoc STDP weight test (learning-rate sweep)
- [CORTEX_CALIBRATION_CHECKLIST](CORTEX_CALIBRATION_CHECKLIST.md) — operating-point calibration
