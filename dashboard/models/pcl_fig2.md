## Question

Does inhibitory spike-timing-dependent plasticity remove the most predictable spikes (N'dri,
Barbier, Teulière and Triesch 2025, Nat Commun 16:8880, Fig. 2)? Neuron 0 inhibits neurons 1–3,
whose inputs copy 90, 50 and 10 % of neuron 0's spikes 1 ms later. The claim: the learned
inhibitory weights and the spike suppression are ordered by predictability, neuron 1 > 2 > 3.

This page is a demonstration. The paper's Fig. 2d values were not available here, so both gates
below test the ordering against our own declared criteria, with no numerical comparison to the
paper.

## Formulation

Event-driven leaky neurons (paper Methods). On each input event of weight $w_i$, with the last
spike at $t_s$:

$$V \leftarrow \max\!\left(V_{min},\ V e^{-\Delta t/\tau_m} + w_i - \eta_{RP}\, e^{-(t-t_s)/\tau_{RP}}\right)$$

A spike occurs at $V \ge V_\theta$, then $V \leftarrow V_{reset}$. Causal STDP on every
connection, applied at each postsynaptic spike $t_s$ to every presynaptic event $t_i$ since the
previous update:

$$\Delta w = (w_{max} - w)\,\eta_+\,\eta_{LTP}\, e^{-(t_s - t_i)/\tau_{LTP}} + w\,\eta_-\,\eta_{LTD}\, e^{-(t_i - t_{s-1})/\tau_{LTD}}, \qquad w \ge 0$$

Protocol: 140 training samples × 10 epochs, 35 test samples, 3 networks. Suppression is the
fraction of a neuron's test spikes removed by the learned inhibition.

## Implementation

Two routes, both in `artifacts/etudes/pcl/`:

- **P0, standalone** (`pcl_fig2.py`): the LIF model and rule above, outside the jaxfne kernels.
- **K0 and K0b, jaxfne library** (`pcl_fig2_hdp.py`): the rule registered as `pcl_stdp`
  (`pcl_hdp_rule.py`) and run through `simulate_edge_recurrent_izhikevich_hdp_registered`,
  dt = 0.1 ms, RS Izhikevich cells (a 0.02, b 0.2, c −65, d 8), GABA_A-indexed inhibitory edges
  with τ = 2 ms, inputs as 1 ms current pulses at 1.5× firing threshold. Weights convert at
  1 mV ↔ 4.85 native units, calibrated on the smallest weight that blocks a pulse 1 ms after
  neuron 0's (25.7 native units against 5.3 mV, which we derive from the paper's parameters as
  (15 − 10)/e^(−1/18); `artifacts/etudes/pcl/README.md`).

## Results against our gates

| Gate | Route | Seeds | Weights ordered 1 > 2 > 3 | Suppression ordered | Verdict |
|---|---|---|---|---|---|
| P0 | standalone | 0, 1, 2 | 3 of 3 | 3 of 3 | PASS |
| K0 | library (end-of-epoch weight) | 0, 1, 2 | 2 of 3 | 3 of 3 | FAIL |
| K0b | library (weight averaged over the last epoch, declared before the run) | 10, 11, 12 | 3 of 3 | 3 of 3 | PASS |

P0 means of 3 networks: final weights 38.08 / 23.87 / 15.15 mV; suppression 0.693 / 0.482 /
0.315. K0b suppression (0.38–0.41 / 0.18–0.20 / 0.08–0.12) is 0.55–0.59 / 0.37–0.41 /
0.25–0.38 of the standalone mean per neuron, and the averaged weights are compressed (≈ 22 / 19 / 16 mV).

## Deviations and limitations

- No paper values: the comparison is to our ordering criterion only.
- Neuron 3 loses 32 % of its spikes though only 10 % are predictable, so the learned inhibition
  also removes some unpredictable spikes. The paper's text does not report this.
- Normalization is off for Fig. 2: λ = 6500 over 3 synapses is incompatible with
  $w_{max}$ = 50; the reference code skips normalization for a single lateral synapse.
- Spike-rate adaptation, present in the reference code and absent from the paper, is omitted.
- The initial inhibitory weight is not given; 1 mV is used, and w0 = 0.1, 1, 5 and 20 reach
  identical final weights.
- K0's end-of-epoch weight is a snapshot of the last training samples; K0b's time average is a
  later, declared protocol.

## Reproduce

```
python artifacts/etudes/pcl/pcl_fig2.py --out artifacts/etudes/pcl/fig2.json
python artifacts/etudes/pcl/pcl_fig2_hdp.py --out artifacts/etudes/pcl/fig2_hdp.json
python artifacts/etudes/pcl/pcl_fig2_hdp.py --seeds 10 11 12 --out artifacts/etudes/pcl/fig2_hdp_avg.json
```

Seeds 0–2 (P0, K0) and 10–12 (K0b); `pcl_fig2_hdp.py` records both the end-of-epoch and the
time-averaged weight. The receipts do not record the JAX platform. Results: `artifacts/etudes/pcl/fig2.json`,
`fig2_hdp.json`, `fig2_hdp_avg.json`, `fig2_hdp_avg_s012.json`.
