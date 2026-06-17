# Homeostatic STDP — Canonical Model (100-Neuron Cortex)

**Status:** Technical report · **canonical STDP model** · proxy / computational-scaffold · 2026-06-17
**Truth gates:** `truth_mode=truth_safe_unverified`, `claim_level=computational_scaffold`. No biological-learning or mechanism claim.

## Summary

This is the **single STDP model kept going forward**: closed-loop online STDP
(`jaxfne.streaming.run_stdp_stream`) plus the **simplest homeostatic regulator —
multiplicative synaptic scaling**. It makes the default `global_stdp = 1.0`
stable, where unregulated STDP ran away
([prior closed-loop report](STDP_CLOSED_LOOP_REPORT.md)).

## Model (simplest form)

1. **Neurons:** Izhikevich (E, PV, SST, VIP), per-type params from the cortex builder.
2. **Synapses:** plastic weight matrix `W` (the substrate). Recurrent signed
   connectivity, only **excitatory** columns plastic, sign-preserved, clipped `[0, 1.5]`.
3. **Plasticity (online):** pair-based STDP, `dW = global_stdp · (dW_LTP − dW_LTD)`,
   applied every timestep, fed back into dynamics.
   - `global_stdp = 1.0` → **default-on**
   - `global_stdp = 0.0` → **off** (W frozen)
4. **Homeostasis (the addition):** every 1 s, rescale each postsynaptic neuron's
   incoming excitatory weights back to their **initial sum** `S0`:
   ```
   W[i, E] *= S0[i] / sum(W[i, E])
   ```
   STDP sets the *pattern* of weights (competition); homeostasis holds the
   *total* excitatory input per neuron constant. This is standard synaptic
   scaling (Turrigiano-style), the minimal mechanism that prevents runaway.

Reproduce:
```bash
python cortex_100_homeostatic_stdp.py          # → JSON
python cortex_100_homeostatic_stdp_figure.py   # → PNG
```
Artifacts: `cortex_100_homeostatic_stdp/{homeostatic_stdp.json, homeostatic_stdp.png}`.

## Setup

- 100-neuron canonical V1 column (71 E, 29 I), 25,000 ms, dt = 0.1 ms
- Drive: −1 nA extra-E → ~8.5 Hz regime (rate-compliant, see [calibration](CORTEX_CALIBRATION_CHECKLIST.md))
- STDP rule: A_plus=0.01, A_minus=0.012, τ=20 ms, w∈[0, 1.5]
- Homeostatic interval: 1 s (25 rescales over the run)

> **Kernel note:** `run_stdp_stream` uses a simpler synapse than
> `simulate_laminar_trials` (instantaneous `W·spike`, τ=5 ms, no intrinsic
> drive). The homeostatic stability result is the transferable finding.

## Results

| run | rate trajectory (Hz) | exc-W mean (0→25 s) | growth | LTP | LTD | stable? |
|---|---|---|---:|---:|---:|:--:|
| **off** (`global_stdp=0.0`) | 8.6 → 8.5 (flat) | 0.0495 → 0.0495 (frozen) | 0% | 0 | 0 | ✅ |
| **STDP 1.0, no homeostasis** | 8.6 → **14.8** (rising) | 0.051 → **0.209** | **+308%** | 3190 | 3342 | ❌ runaway |
| **homeostatic STDP 1.0** | 8.6 → **8.5** (flat) | 0.0495 → 0.0495 | **0%** | 2295 | 4237 | ✅ |

Figure: [`homeostatic_stdp.png`](../cortex_100_homeostatic_stdp/homeostatic_stdp.png) —
rate, excitatory-weight, and LTP/LTD-count panels.

## Verdict (all checks pass)

| Check | Result |
|---|:--:|
| homeostasis prevents runaway (vs no-homeostasis) | ✅ |
| homeostatic STDP stable at `global_stdp = 1.0` | ✅ |
| `global_stdp = 0.0` freezes W | ✅ |
| homeostatic STDP still learns (weights redistributed) | ✅ |

## Interpretation

The decisive result is the third row vs the second:

- **Without homeostasis**, `global_stdp = 1.0` drives LTP positive feedback —
  excitatory weights grow +308%, firing climbs 8.6 → 14.8 Hz, leaving the < 10 Hz band.
- **With homeostasis**, the *total* excitatory input per neuron is pinned, so
  firing stays flat at 8.5 Hz and weights stay bounded — **yet STDP still acts**:
  LTP=2295 / LTD=4237 means synapses are continuously re-weighted relative to
  each other. Homeostasis removes the runaway *gain* without removing the
  *learning*.

This is exactly the intended division of labor: **STDP = what pattern, synaptic
scaling = how much total.**

## Recommendation

- **Canonical model:** closed-loop STDP + synaptic scaling, as above.
- **Default `global_stdp = 1.0`** is now safe (plasticity-on by default; `0.0` = off).
- Homeostatic interval ~1 s and "rescale incoming-E weights to initial sum" is
  the minimal regulator — no rate set-point or extra state required.

## Scope

Proxy/computational-scaffold; weights are not a calibrated biological learning
claim. Uses the streaming kernel's simplified synapse model. The earlier
exploratory reports (post-hoc sweeps, unregulated closed loop) remain as the
analysis record that motivated this model; this report is the one to build on.

## Related reports
- [STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md) — unregulated closed loop (the runaway this fixes)
- [CORTEX_CALIBRATION_CHECKLIST](CORTEX_CALIBRATION_CHECKLIST.md) — operating-point calibration
- [STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md) · [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) · [STDP_REAL_TEST_REPORT](STDP_REAL_TEST_REPORT.md) — post-hoc analysis history
