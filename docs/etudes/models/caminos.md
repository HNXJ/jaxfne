# CAMINOS Reference Model

## Purpose

CAMINOS is the E–PV–SOM spiking network of Tahvili, Vinck and di Volo (PLoS Computational Biology 2026, DOI 10.1371/journal.pcbi.1014378). This page records the stage-A port in jaxfne: the model, the estimands compared with the paper, the verdict bands, and the critical-p result.

## Model

Three populations of AdEx neurons: E (8000), PV (750), SOM (750). Each neuron follows

```
C dV/dt = gL (EL − V) + gL Δ exp((V − VT)/Δ) − w + I
τw dw/dt = a (V − EL) − w,   w += b at each spike
```

with the cell parameters of the paper's Table 1. Synapses are conductance based, with τ_E = 2 ms, τ_I = 8 ms, reversal potentials 0 mV (E) and −80 mV (I), and a 2 ms delay on all synapses.

- Connectivity: every ordered pair connects independently with probability p. Autapses are removed.
- Blocks: E→E, E→PV, E→SOM, PV→E, PV→PV, SOM→E, SOM→PV. PV→SOM and SOM→SOM are absent, as in the reference model.
- External drive: round(p·NE) private 4 Hz Poisson sources per E and PV cell (560 at p = 0.07). SOM receives no drive.
- Integration: Euler, dt = 0.1 ms, float32.

## Estimands and bands

The bands are exact (e = 0), close (every e ≤ 0.05), off (any e > 0.05), with e = |z − y| / |y|. Here z comes from the jaxfne core route: the P6 v2 run (p = 0.07, seeds 1–5) through `caminos_api` into the core kernel `simulate_adex` with `record_v=False`. The analysis is `caminos_analysis.py`. y is the paper's value. The étude-local kernel `caminos_jax.py` is an oracle only and provides no z value.

| Estimand | Paper (y) | Port (z) | e | Band |
|---|---|---|---|---|
| Gamma frequency (Hz) | ≈ 35 | 33.80 | 0.034 | close |
| PV→SOM lag (ms), port's cross-correlation estimator | ≈ 8.5 (spiking network) | 7.22 | 0.151 | off |
| PV→SOM lag (ms), ref. 42 phase method, seeds 1–3 | ≈ 8.5 (spiking network) | 6.47 | 0.239 | off |
| PV→SOM lag (ms), ref. 42 phase method, seeds 4–8 | ≈ 8.5 (spiking network) | 6.32 | 0.257 | off |
| PV→SOM lag (ms), P3 estimator, preprint value (context) | ≈ 7 (preprint) | 7.22 | 0.031 | not banded |
| E rate (Hz) | 1–3 | 1.079 | n/a | inside the interval |
| PV rate (Hz) | higher than SOM and E | 11.25 | n/a | order holds |
| SOM rate (Hz) | higher than E | 6.78 | n/a | order holds |
| Oscillation onset in p | qualitative (Fig. 1E) | 0.04 | n/a | n/a |

The lag is measured two ways. The port's cross-correlation estimator (`caminos_analysis.py`) gives 7.22 ms. The phase method of the paper's ref. 42 gives 6.47 ms on seeds 1–3 and 6.32 ms on seeds 4–8. The phase method works as follows: the E population activity, smoothed by a 1 ms Gaussian, is the LFP proxy; the first 2 s are dropped and 10 s analysed; a zero-phase 4th-order Butterworth band-pass at the peak frequency ± 5 Hz is applied (the paper leaves the filter unstated, so this is declared); each spike's phase is read from the Hilbert transform; each population's mean phase is a circular mean over its spikes. The ref. 42 lag is off against 8.5 ms at every band tested (± 3, ± 5, ± 8 Hz). The P3 estimator has no band.

The port has a 2 ms synaptic delay on all synapses (Table 2 of the paper). The preprint's model has no synaptic delay, and its spiking lag is about 7 ms. This difference is an open fidelity item and is not changed here. A 0.1 ms synaptic delay (one step) and true zero delay are recorded sensitivities, below.

## Critical-p result (R4)

With a fixed drive of 560 sources at every p (v6), every p from 0.02 to 0.10 oscillates. With the drive round(p·NE) (v7), no seed oscillates at p = 0.02, where E fires at 0.61–0.62 Hz. All seeds oscillate from p = 0.04 upward. The drive rule is the Methods wording as read. The paper's mean-field input uses f_ext = ν_ext · p · N_E.

The R4 result is reproduced after a fidelity fix (v7): the drive count follows the Methods rule p·NE. No residual term was fitted to R4. Spike counts are piecewise constant in the drive gain, so any future residual fit would use a derivative-free optimizer.

## Verdict (2026-10-14)

Ruling (Hamm, 2026-10-14, relayed by the coordinator): PLoS CB is primary. The 2 ms model and y = 8.5 ms stay. The lag miss is a finding, "needs X", with no residual fit, since both candidate X move stated parameters. The preprint no-delay model is a recorded sensitivity.

| Estimand | y | z (core route) | e | Band | Verdict |
|---|---|---|---|---|---|
| Gamma f_pk (Hz) | ≈ 35 | 33.80 | 0.034 | close | reproduced |
| E, PV and SOM rates (Hz) | 1–3 (interval); PV > SOM > E | 1.08, 11.25, 6.78 | n/a | inside; order holds | reproduced |
| R4 onset p | qualitative (Fig. 1E) | 0.04 | n/a | reproduced after fidelity fix (v7) | reproduced |
| PV→SOM lag, ref. 42 (seeds 1–3; 4–8) | ≈ 8.5 | 6.47; 6.32 | 0.239; 0.257 | off | finding: needs X |
| PV→SOM lag, P3 estimator (seeds 1–5) | ≈ 8.5 | 7.22 | 0.151 | off | finding: needs X |

Sensitivity: a 0.1 ms synaptic and drive delay (one step) gives a ref. 42 lag of 6.97 ms (seeds 1–3) and 7.07 ms (seeds 4–8), and a P3 lag of 7.10 and 6.96 ms. Every e against the preprint's about 7 ms is at most 0.014. The oscillation moves to 43–46 Hz (primary 34 Hz), so this is a different network state from the primary model. True zero delay runs through the core's declared intra-step fixed point (`intra_step_rounds` on the AdEx route, run to quiescence with the Poisson `"same_step"` drive flag; at most 3 rounds per step on every seed). It gives a ref. 42 lag of 7.18 ms (seeds 1–3) and 7.11 ms (seeds 4–8), a P3 lag of 7.00 and 6.92 ms, and f_pk 43–46 Hz: e against about 7 ms is at most 0.026, and the network state matches the 0.1 ms run.

## Receipts

- Prereg and results: `artifacts/etudes/papers/caminos/PREREG_P3.md`, `PREREG_P3_v7_R4.md`, `PREREG_P6.md`, `RESULT_P3.md`, `RESULT_P6.md`.
- Scores: `artifacts/etudes/papers/caminos/scores/` (`score.json`, `score_v7.json`, `p6_v2_score.json`).
- Calibration record: `artifacts/etudes/papers/caminos/CALIBRATION.md`.
- Raw counts and manifests: `F:\warehouse\caminos\p6_scored_v2\`, `F:\warehouse\caminos\scored\v7_20261012\`, and the held-out runs `F:\warehouse\caminos\calib\heldout_p007\`.
- Lag re-score by the phase method: `F:\warehouse\caminos\calib\lag_ref42_p007.json`, with the estimator in `artifacts/etudes/papers/caminos/caminos_api.py` (`ref42_phase_delay`).
- Sensitivity (0.1 ms delay): runs in `F:\warehouse\caminos\calib\sens_delay0p1_p007\`, scored to `F:\warehouse\caminos\calib\sens_delay0p1_score.json` by `artifacts/etudes/papers/caminos/score_delay_sensitivity.py`.
