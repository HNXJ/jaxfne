# Cortex Calibration Checklist — 100-Neuron V1 Column

**Status:** Technical report · proxy / computational-scaffold · 2026-06-17
**Truth gates:** `truth_mode=truth_safe_unverified`, `claim_level=computational_scaffold`. No biological-calibration or mechanism claim.

## Purpose

Answer four operating-point questions for the canonical 100-neuron laminar
column with measured Izhikevich simulations (dt = 0.1 ms):

1. How does DC drive map to firing rate, per cell type? (F-I curves)
2. What plasticity scale (`global_stdp`) is reasonable?
3. Physiological hard bounds: per-neuron < 100 Hz; stable cortex mean < 10 Hz.
4. Drive sweetpoints for target rates (1 / 5 / 10 / 20 Hz) per cell type.

Reproduce:
```bash
python cortex_calibration_fi_curves.py      # Q1, Q4, Q3-per-neuron
python cortex_calibration_network_rate.py   # Q3-network
python cortex_100_stdp_global_sweep.py      # Q2 (global_stdp scale sweep)
```
Data artifacts: `cortex_calibration/{fi_curves.json, network_rate.json}`,
`cortex_100_stdp_global_sweep/stdp_global_sweep.json`.

---

## Q1 — Drive → firing rate, per cell type (single isolated neuron)

Firing rate (Hz) of one neuron at constant injected drive, no synapses, no
noise. Intrinsic baseline drive (built into each type) is added on top:

| cell | intrinsic | −100 | −50 | −20 | −10 | 0 | 10 | 20 | 50 | 100 |
|------|----------:|----:|----:|----:|----:|----:|-----:|-----:|------:|------:|
| **E**   | 5.0 | 0 | 0 | 0 | 0 | 10.6 | 32.8 | 53.9 | 119.4 | 232.2 |
| **PV**  | 3.0 | 0 | 0 | 0 | 0 | 0 | **185** | **345** | 714 | 1111 |
| **SST** | 3.5 | 0 | 0 | 0 | 0 | 28.9 | 97.8 | 172.8 | 376 | 668 |
| **VIP** | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.0 | 104 | 272 |

- **Negative drive (≤ −10 nA) silences every type** — a cliff, not graded suppression.
- **E** is the only type with a usable graded curve (10 → 54 Hz over drive 0 → 20).
- **PV is fast-spiking**: 0 → 185 Hz between drive 0 and 10; no low-rate DC regime.
- **SST is tonically active** (28.9 Hz at injected 0).
- **VIP is reluctant**: needs ~+20 nA before firing (its `b = −0.1` resists recruitment).

---

## Q2 — Reasonable plasticity scale (`global_stdp`)

`global_stdp` is the `plasticity_scale` gain on the STDP update
`dW = global_stdp · (dW_LTP − dW_LTD)`. Per-step weight change vs scale
(rule A_plus=0.01, A_minus=0.012; weights O(0.5), w_max=1.5; 250k-step raster):

| global_stdp | dW_mean/step | dW_max/step | accumulated over run (hot synapse) |
|---:|---:|---:|---|
| 0.0 | 0 | 0 | frozen |
| 0.5 | −4.0e−6 | 2e−5 | ~5 (saturates) |
| 1.0 | −8.0e−6 | 6e−5 | ~15 (**saturates hard**) |
| 2.0 | −1.6e−5 | 1.2e−4 | ~30 |
| 10  | −8.0e−5 | 5.8e−4 | ~145 |
| 100 | −8.0e−4 | 5.8e−3 | ~1450 (pure clipping) |

Criterion — a change is "visible but gradual" if the hottest synapse reaches
~10% of w_max (≈ 0.15) over a useful window, not in one step and not never:

- At **scale 1.0**, hot synapse hits 0.15 in **~250 ms** → over a 25 s closed loop that is **massive saturation**. (The post-hoc bulk mean looked flat only because post-hoc does not accumulate.)
- To evolve weights over the **whole multi-second run**: per-step ≈ 6e−7 → **scale ≈ 0.01**.
- To evolve over **~hundreds of ms**: **scale ≈ 0.1–1.0**.

**Recommended band: `global_stdp ≈ 0.01 – 0.1`.**
`< 0.001` is effectively frozen; `> 1` saturates/clips (10–100 is clipping, not
learning); negative values are anti-Hebbian contrast only.

> **Closed-loop + homeostasis (resolved):** post-hoc bands are an upper bound on
> the *driving signal*. With feedback ([STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md)),
> unregulated scales **≥ 0.5 run away**. The fix — and the **canonical STDP
> model** — is closed-loop STDP + the simplest homeostatic regulator (synaptic
> scaling): it keeps `global_stdp = 1.0` stable *and* learning. See
> [STDP_HOMEOSTATIC_REPORT](STDP_HOMEOSTATIC_REPORT.md). Use that going forward.

> **Coupling to Q3 (now verified):** scale and rate trade off. Re-running the
> sweep at the stable 9.63 Hz regime (2 nA baseline) gives a per-step STDP
> signal **~10× smaller** than at 43 Hz, so the reasonable band shifts up to
> **`global_stdp ≈ 0.05 – 1.0`** at ~10 Hz (vs 0.01–0.1 at 43 Hz) — exactly the
> predicted ~10× scaling. See
> [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md).

Full sweeps + invariant checks: [STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md)
(~43 Hz) and [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) (~10 Hz).

---

## Q3 — Physiological hard bounds

### (a) Single neuron must never exceed 100 Hz — drive at which it does

| cell | drive where rate > 100 Hz (isolated) |
|------|:---:|
| **E**   | +42 nA |
| **PV**  | **+6 nA** |
| **SST** | **+11 nA** |
| **VIP** | +50 nA |

Isolated PV/SST exceed 100 Hz at tiny drives — **directly driving interneurons
hard is the failure mode.**

### (b) In-network, per-neuron rate is self-limited by recurrent inhibition

Baseline E-drive sweep (100-neuron cortex, 2 s, noise on):

| E-drive nA | mean | E | I | PV | SST | VIP | max neuron | mean ≤ 10 Hz? |
|---:|----:|----:|----:|----:|----:|----:|----:|:---:|
| 0  | 3.08 | 0.00 | 10.6 | 0.0 | 25.4 | 11.4 | 25.6 | ✅ |
| **2** | **9.46** | 7.22 | 14.9 | 6.7 | 27.5 | 14.1 | 29.4 | ✅ |
| 4  | 11.63 | 9.68 | 16.4 | 9.1 | 27.5 | 15.8 | 30.6 | ❌ |
| 6  | 14.67 | 12.7 | 19.5 | 12.0 | 30.4 | 19.2 | 36.7 | ❌ |
| 8  | 16.12 | 14.4 | 20.3 | 13.2 | 30.1 | 20.9 | 35.0 | ❌ |
| 10 | 18.37 | 17.3 | 21.0 | 15.5 | 30.1 | 19.4 | 36.7 | ❌ |
| 15 | 22.71 | 21.8 | 24.9 | 21.7 | 33.1 | 20.1 | 42.2 | ❌ |
| 20 | 27.74 | 27.4 | 28.5 | 27.1 | 36.8 | 20.5 | 53.3 | ❌ |
| 30 | 37.43 | 38.5 | 34.8 | 37.9 | 37.4 | 25.7 | 75.0 | ❌ |
| 50 | 53.74 | 60.7 | 36.8 | 38.6 | 41.9 | 26.7 | 79.4 | ❌ |

1. **Per-neuron < 100 Hz holds in-network at every drive** (max 79 Hz even at 50 nA): recurrent inhibition tames interneurons that hit 1000 Hz in isolation. The hard per-neuron bound is **structurally respected**.
2. **Mean < 10 Hz requires baseline E-drive ≤ ~2 nA.** Using 10 nA gives ~18 Hz mean — about 2× over budget.

> **Action:** cap baseline E-drive at ~2–3 nA; never inject DC directly into PV/SST.

---

## Q4 — Drive for target firing rates (sweetpoints, per type)

Single-neuron drive needed to reach each target rate:

| target | E | PV | SST | VIP |
|-------:|----:|----:|----:|----:|
| 1 Hz  | ≈ −1 | (none) | ≈ −2 | +20 |
| 5 Hz  | ≈ −1 | (none) | ≈ −2 | +20 |
| 10 Hz | 0 | (none) | ≈ −2 | +22 |
| 20 Hz | +4 | +1* | −1 | +25 |

\* PV/SST collapse to a single threshold — between drive 0 and first firing they
jump past all four targets (185 Hz at +10). **They have no graded DC sweetpoints**;
their rate is set by network inhibition, not DC.

- **Only E has clean, separable sweetpoints**: 1–5 Hz at slightly negative drive, 10 Hz at ~0, 20 Hz at +4.
- **PV/SST**: control rate via the I↔E loop, not DC.
- **VIP**: high-threshold knob, usable from +20 nA upward.

---

## Synthesis — the stable operating point

| knob | recommended | why |
|---|---|---|
| Baseline E-drive | **~2 nA** | mean ≤ 10 Hz (Q3 network) |
| Direct PV/SST drive | **none** | hit 100 Hz at 6–11 nA (Q3 single) |
| E target rate | DC ~0 nA → 10 Hz, +4 → 20 Hz | only E has graded sweetpoints (Q4) |
| I rates | via E-drive + inhibition | no DC sweetpoints (Q4) |
| `global_stdp` | **0.01–0.1** (low rate → up to ~1.0) | gradual learning, no saturation (Q2) |
| per-neuron cap | structurally safe (< 80 Hz in-net) | recurrent inhibition self-limits (Q3) |

**Bottom line:** the earlier 43 Hz cortex ran at ~5× the stable drive. Cap
baseline E-drive at ~2 nA, keep DC off the interneurons, and run STDP at scale
~0.01–0.1 — the regime where both "is the rate physiological?" and "is STDP
well-scaled?" answer yes.

## Related reports
- [NEURON_IO_CHARACTERIZATION](NEURON_IO_CHARACTERIZATION.md) — prior F-I mapping
- [BASELINE_DRIVE_REFERENCE](BASELINE_DRIVE_REFERENCE.md) — baseline drive reference
- [STDP_GLOBAL_SCALE_REPORT](STDP_GLOBAL_SCALE_REPORT.md) — `global_stdp` sweep (~43 Hz)
- [STDP_LOWRATE_REGIME_REPORT](STDP_LOWRATE_REGIME_REPORT.md) — `global_stdp` sweep (~10 Hz, rate-compliant)
- [STDP_HOMEOSTATIC_REPORT](STDP_HOMEOSTATIC_REPORT.md) — **canonical STDP model** (homeostatic, stable at 1.0)
- [STDP_CLOSED_LOOP_REPORT](STDP_CLOSED_LOOP_REPORT.md) — closed-loop online STDP (runaway verdict)
- [STDP_REAL_TEST_REPORT](STDP_REAL_TEST_REPORT.md) — post-hoc STDP weight test
