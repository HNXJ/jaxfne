# STDP in the Rate-Compliant ~10 Hz Regime — 100-Neuron Cortex

**Status:** Technical report · proxy / computational-scaffold · 2026-06-17
Scope: proxy / computational scaffold — see [Scope & status](scope_and_status.md).

## Purpose

Re-run the `global_stdp` scale sweep with the cortex held in the **physiological
< 10 Hz regime** (the [calibration checklist](CORTEX_CALIBRATION_CHECKLIST.md)
showed the earlier 43 Hz sweep ran ~5× over the stability budget). This confirms
the STDP scale behavior at a rate-compliant operating point and quantifies the
rate↔scale coupling the checklist predicted.

Reproduce:
```bash
python cortex_100_stdp_lowrate_sweep.py    # sim @ 2 nA + global_stdp sweep → JSON
python cortex_100_stdp_lowrate_figure.py   # → PNG (vs the 43 Hz regime)
```
Artifacts: `cortex_100_stdp_lowrate_sweep/{stdp_lowrate_sweep.json, stdp_lowrate_sweep.png}`.

> **Not reproducible as described:** these scripts and artifact files are not
> preserved in this repo (verified absent from working tree and full git
> history 2026-07-03). This report documents a historical result, not a
> currently-reproducible one.

## Regime (rate-compliant ✅)

Baseline E-drive **2 nA**, no (s,d) stimulus, 25,000 ms, dt = 0.1 ms:

| metric | value | bound | pass |
|---|---:|---|:--:|
| mean rate | 9.63 Hz | ≤ 10 Hz (stable cortex) | ✅ |
| E rate | 7.42 Hz | — | — |
| I rate | 15.05 Hz | — | — |
| max single neuron | 29.7 Hz | < 100 Hz (per-neuron cap) | ✅ |

Contrast: the [43 Hz sweep](STDP_GLOBAL_SCALE_REPORT.md) used 10 nA baseline +
the 5×5 (s,d) grid → 42.68 Hz mean, over budget.

## Results — `global_stdp` sweep @ 9.63 Hz

| `global_stdp` | LTP | LTP% | LTD | LTD% | ΔW min | ΔW max | W̄ after | sign✓ | finite✓ |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **0.00** | 0 | 0.00% | 0 | 0.00% | +0.00000 | +0.00000 | 0.2241 | ✅ | ✅ |
| **−1.00** | 3660 | 36.60% | 1929 | 19.29% | −0.00001 | +0.00002 | 0.2241 | ✅ | ✅ |
| **0.10** | 105 | 1.05% | 407 | 4.07% | −0.00000 | +0.00000 | 0.2241 | ✅ | ✅ |
| **0.50** | 1466 | 14.66% | 2347 | 23.47% | −0.00001 | +0.00001 | 0.2241 | ✅ | ✅ |
| **1.00** | 1929 | 19.29% | 3660 | 36.60% | −0.00002 | +0.00001 | 0.2241 | ✅ | ✅ |
| **2.00** | 2264 | 22.64% | 4878 | 48.78% | −0.00003 | +0.00002 | 0.2241 | ✅ | ✅ |
| **10.00** | 2979 | 29.79% | 5642 | 56.42% | −0.00017 | +0.00012 | 0.2241 | ✅ | ✅ |
| **100.00** | 3344 | 33.44% | 5927 | 59.27% | −0.00174 | +0.00120 | 0.2241 | ✅ | ✅ |

Invariant checks (same spec): `off → dW=0` ✅, `inverse flips sign` ✅
(g+1.0: LTP=1929/LTD=3660 ↔ g−1.0: LTP=3660/LTD=1929, exact swap),
`100× scales ~100×` ✅.

## Key result — rate ↔ scale coupling (quantified)

Per-step unit weight change (`global_stdp = 1.0`):

| regime | unit dW mean / step | ratio |
|---|---:|---:|
| ~43 Hz | −7.96e−6 | 1.0× |
| **~10 Hz** | **−7.79e−7** | **~0.10×** |

**The per-step STDP signal is ~10× smaller at 10 Hz than at 43 Hz** — fewer
spikes → fewer coincidences → slower accumulation. The figure's middle panel
shows two parallel slope-1 lines (clean linear gain in both regimes), the 10 Hz
curve sitting ~10× lower. W̄ never moves off 0.2241 even at 100× — no clipping at
this rate (vs the 43 Hz sweep where 100× nudged W̄ down).

### Recommended `global_stdp` band at ~10 Hz

Criterion (hottest synapse reaches ~10% of w_max ≈ 0.15 over a useful window;
unit dW_max ≈ 1.2e−5/step here):

- Evolve over the **full multi-second run**: per-step ≈ 6e−7 → **scale ≈ 0.05**
- Evolve over **~hundreds of ms**: per-step ≈ 6e−5 → **scale ≈ 5**

**Recommended band at ~10 Hz: `global_stdp ≈ 0.05 – 1.0`** — about **10× higher**
than the 43 Hz band (0.01–0.1), exactly as the rate↔scale coupling predicts. The
lower rate means you turn the gain up to keep the same learning timescale.

## Interpretation

- Baseline STDP stays **LTD-dominant** even at 10 Hz (36.6% LTD vs 19.3% LTP at scale 1.0), consistent with A_minus > A_plus; inverse mirrors it.
- All numerics safe: finite, sign-preserved, no clipping at any scale (the 10 Hz regime is far from saturation).
- This is the operating point where "is the rate physiological?" (yes, ≤ 10 Hz mean, < 100 Hz per neuron) and "is STDP well-scaled?" (yes, band 0.05–1.0) both hold.

## Scope

Post-hoc on a static-network raster (weights not fed back). The closed-loop
verification has now been run — see
[STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md): with feedback, scales
above ~0.1 **run away** (LTP positive feedback), so the post-hoc band 0.05–1.0
is an upper bound on the *driving signal*, not the safe closed-loop range.

## Related reports
- [STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md) — closed-loop online STDP (runaway verdict)
- [CORTEX_CALIBRATION_CHECKLIST](CORTEX_CALIBRATION_CHECKLIST.md) — operating-point calibration (Q1–Q4)
- [STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md) — the ~43 Hz scale sweep
- [STDP_REAL_TEST_REPORT](STDP_REAL_TEST_REPORT.md) — post-hoc STDP weight test
