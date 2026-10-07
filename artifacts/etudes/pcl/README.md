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
```
