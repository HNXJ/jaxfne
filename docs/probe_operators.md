# Probe operators

Probe operators map simulation state into named readouts.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Operators

| Operator | Output | Typical shape |
|---|---|---|
| SPK | spike matrix or events | `[time, neuron]` |
| Vm | voltage/state trace | `[time, neuron]` |
| Source | source/current proxy | `[time, source]` |
| LFP-proxy | laminar field proxy | `[time, contact]` |
| CSD-proxy | spatial source/divergence proxy | `[time, contact]` |
| EEG-proxy | linear channel projection | `[time, channel]` |
| MEG-proxy | linear magnetic-channel projection | `[time, channel]` |
| EMM-proxy | normalized activity-cost proxy | `[time]` |

## Mathematical Forms

Computational scaffolds define the following proxy readout operations:

- SPK: $y_{spk}[t, k] = \sum_i \delta(t - t_{i,k})$
- Vm: $y_{vm}[t, k] = V_k[t]$
- Source: $y_{src}[t, k] = J_k[t]$
- LFP-proxy: $y_{lfp}[t, k] = \sum_i \alpha_{ik} V_i[t]$
- CSD-proxy: $y_{csd}[t, k] = \nabla^2 y_{lfp}[t, k]$
- EEG-proxy: $y_{eeg}[t, k] = \sum_i \beta_{ik} y_{lfp}[t, i]$
- MEG-proxy: $y_{meg}[t, k] = \sum_i \gamma_{ik} y_{src}[t, i]$
- EMM-proxy: $y_{emm}[t] = \frac{1}{N} \sum_k |y_{src}[t, k]|


```yaml
name: string
kind: spk | vm | source | lfp_proxy | csd_proxy | eeg_proxy | meg_proxy | emm_proxy
method: string
data_shape: list[int]
units_or_status: string
operator_status: simulated_proxy
calibration_status: string
```

## Example

```python
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
readout = model.probe(signals)
print(readout.to_dict())
```
