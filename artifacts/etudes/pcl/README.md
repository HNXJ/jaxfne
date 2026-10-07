# Predictive Coding Light (PCL) étude

Target: N'dri, Barbier, Teulière, Triesch, "Predictive Coding Light",
*Nat Commun* 16:8880 (2025), doi:10.1038/s41467-025-64234-z. Reference code:
github.com/comsee-research/Predictive-Coding-Light (C++, event-driven).

Standalone first (Hamm 2026-10-06): faithful LIF + PCL rule outside the jaxfne
kernels; port to the registrable HDP kernel after the gates pass. No third
factor until the published model is reproduced.

## Model (paper Methods)

- LIF, event-driven: on each input event
  $V \leftarrow \max(V_{min},\ V e^{-\Delta t/\tau_m} + w_i - \eta_{RP} e^{-(t-t_s)/\tau_{RP}})$;
  spike at $V \ge V_\theta$, reset to $V_{reset}$.
- Causal STDP on every connection, applied at each postsynaptic spike $t_s$ to
  every presynaptic event $t_i$ since the previous update:
  $\Delta w = (w_{max}-w)\eta_+\,\eta_{LTP} e^{-(t_s-t_i)/\tau_{LTP}} + w\,\eta_-\,\eta_{LTD} e^{-(t_i-t_{s-1})/\tau_{LTD}}$, $w \ge 0$.
- L1 normalization of incoming weights per connection type to $\lambda$.
- Prediction is not sent as an error signal: learned inhibition (local
  lateral, distant lateral, top-down) cancels the predictable spikes.

## Gate P0: Fig. 2 (inhibitory STDP removes the most predictable spikes) — PASS

`pcl_fig2.py`: neuron 0 inhibits neurons 1–3, whose inputs copy 90 / 50 / 10 %
of neuron 0's spikes 1 ms later; 140 training samples × 10 epochs, 35 test
samples, 3 networks. Acceptance: learned weights and suppression both ordered
neuron 1 > 2 > 3 in every network.

| neuron (predictable fraction) | 1 (90 %) | 2 (50 %) | 3 (10 %) |
|---|---|---|---|
| final inhibitory weight (mV, mean of 3) | 38.08 | 23.87 | 15.15 |
| spike suppression (mean of 3) | 0.693 | 0.482 | 0.315 |

Per network suppression: [0.677, 0.489, 0.293], [0.700, 0.473, 0.327],
[0.701, 0.483, 0.323]. The fixed point does not depend on the initial weight
(w0 = 0.1, 1, 5, 20 give identical final weights; w0 is consumed: weight
trajectories differ after the first sample).

Observed, not in the paper's text: neuron 3 loses 32 % of its spikes though
only 10 % are predictable, so the learned inhibition also removes some
unpredictable spikes. The paper's Fig. 2d values were not available here for
a numeric comparison.

## Gate C1: small column (protocol declared before the pilot)

`pcl_column.py`. Input 16×16×2 synthetic events: a bright bar (width 2 px)
at one of 8 orientations drifting along its normal at 0.13 px/ms for 200 ms,
ON/OFF events (Poisson, mean 3) where pixels enter/leave the bar, 2 Hz noise.
Simple layer 4×4 positions × 16 features (7×7×2 receptive fields, stride 3);
complex layer 2×2 × 8 features (3×3 simple positions each). Connections:
input→simple, simple→complex (excitatory, shared kernels), local lateral
inhibition in both layers (shared), distant lateral inhibition (simple cells
at other positions) and top-down inhibition (complex→simple cells they cover),
both per neuron. Clock-driven, dt = 0.5 ms.

Training: phase 1 learns excitatory + local inhibition with distant/top-down
inhibition off; phase 2 learns distant + top-down inhibition with everything
else frozen (reference code `STDP_LEARNING` = excitatory / inhibitory).
Test (learning off, 20 sequences per orientation): (a) PCL, (b) distant and
top-down inhibition removed, (c) as (b) with simple-cell spikes removed at
random to match (a)'s count.

Acceptance, each in 3/3 confirmatory seeds (10, 11, 12):
- **A1** orientation tuning: median OSI of responsive simple cells (≥ 0.5
  spikes/sequence) after phase 1 ≥ 0.3 and above the untrained network's.
- **A2** energy: simple-cell spikes (a) ≤ 0.9 × (b).
- **A3** information: 5-fold cross-validated orientation decoding (logistic
  regression on log spike counts, simple + complex) (a) > (c).

Pilot seed 0 may set stimulus parameters and phase lengths only; model
parameters stay at paper values.

Pilot (seed 0):

| stimulus, phase-1 length | median OSI (untrained) | responsive cells | simple spikes (a)/(b) | decoding (a) / (b) / (c) |
|---|---|---|---|---|
| bars, 600 | 0.107 (0.045) | 251 | 0.53 | 0.888 / 0.931 / 0.356 |
| bars, 2400 | 0.122 (0.045) | 170 | 0.54 | 0.806 / 0.794 / 0.500 |
| gratings (period 6 px), 600 | 0.042 (0.019) | 256 | 0.57 | 0.681 / 0.719 / 0.463 |

**Frozen for the confirmatory run: bars, 600 + 300 sequences.** Longer
training and gratings both lowered tuning or decoding. Kernels drift toward
centre blobs (`pilot/*_kernels.png`): every bar crosses the centre of every
receptive field, so a centre detector is driven at all orientations. A1 is
expected to fail; the confirmatory run records it rather than tuning further.

### C1 result (seeds 10, 11, 12): A2 and A3 PASS, A1 FAIL

| seed | median OSI (untrained) | simple spikes (a)/(b) | decoding (a) PCL | (b) no inh. | (c) random, matched |
|---|---|---|---|---|---|
| 10 | 0.123 (0.053) | 0.519 | 0.919 | 0.900 | 0.481 |
| 11 | 0.121 (0.045) | 0.519 | 0.913 | 0.888 | 0.419 |
| 12 | 0.100 (0.049) | 0.511 | 0.850 | 0.869 | 0.538 |

- **A1 FAIL** (3/3): tuning roughly doubles over the untrained network but
  stays near 0.1, below 0.3.
- **A2 PASS** (3/3): learned distant + top-down inhibition removes ~48 % of
  simple-cell spikes.
- **A3 PASS** (3/3): at matched spike removal, PCL decodes orientation at
  0.85–0.92 against 0.42–0.54 for random removal, and stays within 0.02 of
  the uninhibited network. Random removal matches the simple-cell count only
  approximately (≈17 000 vs ≈13 600 spikes, more left in the control), so the
  comparison favours the control.

Reading: the paper's central claim, that learned inhibition removes about half
the spikes while keeping the information, holds in this small column. The
V1-like receptive fields do not form with this stimulus set.

![C1 kernels, seed 10](confirm/c1_seed10_kernels.png)

## Gate K0: Fig. 2 on the jaxfne HDP kernel (Izhikevich) — FAIL (weights, 1/3 seeds)

Hamm 2026-10-07: port PCL to the existing registrable HDP kernel with
Izhikevich neurons, no new neuron model. `pcl_hdp_rule.py` registers rule
`pcl_stdp` (per-edge aux: LTP trace since the last post spike, deferred LTD
sum, post spike trace; soft-bound update and per-group L1 normalization at
post spikes). `pcl_fig2_hdp.py` runs Fig. 2 through
`simulate_edge_recurrent_izhikevich_hdp_registered`, dt = 0.1 ms, RS
Izhikevich (a 0.02, b 0.2, c −65, d 8), GABA_A-indexed inhibitory edges with
τ = 2 ms, inputs as 1 ms current pulses at 1.5× firing threshold.

Unit calibration (declared): the smallest inhibitory weight that blocks a
pulse 1 ms after neuron 0's pulse is 25.7 (native units); the paper's
equivalent is 5.3 mV, so 1 mV ↔ 4.85 native units (w_max = 50 mV ↔ 243).
Blocking is not monotonic above ~10³: very strong inhibition drives v so low
that the quadratic Izhikevich term fires the neuron.

Same acceptance as P0 (weights and suppression ordered 1 > 2 > 3 in every network):

| seed | final w (mV-equivalent) | suppression | weights ordered | suppression ordered |
|---|---|---|---|---|
| 0 | 24.0 / 18.3 / 15.7 | 0.406 / 0.226 / 0.060 | yes | yes |
| 1 | 22.6 / 16.5 / 13.8 | 0.398 / 0.181 / 0.081 | yes | yes |
| 2 | 17.7 / 20.8 / 16.4 | 0.363 / 0.197 / 0.085 | **no** | yes |

Reading: suppression is ordered by predictability in every network, on the
kernel as standalone, at about half the standalone magnitude (0.39 / 0.20 /
0.08 vs 0.69 / 0.48 / 0.32). The weight criterion reads a snapshot: the
end-of-epoch weights are identical across all 10 epochs (deterministic run,
same data each epoch), so they settle within one epoch and the final value
reflects the last training samples. A time-averaged weight is the better
measure; it needs a new declared protocol.

## Deviations from the paper

| item | paper | here | reason |
|---|---|---|---|
| Fig. 2 normalization | outgoing L1 | none | λ = 6500 over 3 synapses is incompatible with $w_{max}$ = 50; the reference code skips normalization for a single lateral synapse and its outgoing normalization line is commented out |
| learning-rate scale | Table 2 η | Table 2 η used directly | Table values equal the code's η × λ (0.000408 × 6500 = 2.652) |
| spike-rate adaptation | not described | omitted | present in the reference code (`DELTA_SRA`), absent from the paper |
| initial inhibitory weight | "same value" | 1 mV | not given; result shown independent of it |

## Reproduce

```bash
python artifacts/etudes/pcl/pcl_fig2.py --out artifacts/etudes/pcl/fig2.json
python artifacts/etudes/pcl/pcl_column.py --seed 10 --out artifacts/etudes/pcl/confirm/c1_seed10.json
python artifacts/etudes/pcl/figure_column.py artifacts/etudes/pcl/confirm/c1_seed10.npz
python artifacts/etudes/pcl/pcl_fig2_hdp.py --out artifacts/etudes/pcl/fig2_hdp.json
```
