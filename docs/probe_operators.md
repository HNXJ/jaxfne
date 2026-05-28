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

## Minimal report fields

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
