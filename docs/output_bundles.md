# Output bundles

jaxfne outputs workflows as JSON-safe bundles containing signals, readouts, and metadata.

## Structure

A typical output bundle includes:

```python
{
  "receipt_id": "uuid-...",
  "version": "0.2.3",
  "signals": {
    "spikes": [[...], [...], ...],
    "V_m": [[...], [...], ...],
    "sources": [[...], [...], ...] or null,
    "field": {
      "lfp": [[...], [...], ...],
      "csd": [[...], [...], ...]
    }
  },
  "readouts": [
    {
      "name": "spike_rate",
      "metric": "spike_rate_hz",
      "value": 12.5,
      "status": "simulated_proxy"
    },
    ...
  ],
  "metadata": {
    "simulation_duration_ms": 100.0,
    "dt_ms": 0.1,
    "n_neurons": 100,
    "seed": 0,
    "execution_time_ms": 245.3
  },
  "operator_reports": [
    {
      "kind": "spk",
      "operator_status": "simulated_proxy",
      "method": "threshold_or_emitter_spike_array",
      "units_or_status": "binary_spike_indicator"
    },
    ...
  ]
}
```

## Validation and serialization

All bundles are JSON-safe:

```python
import json
manifest = model.manifest(signals, readouts)
json.dumps(manifest, allow_nan=False)  # Enforces strict serialization
```

This ensures:

- No NaN or Inf values (numerical errors are caught)
- All arrays converted to lists for JSON portability
- Metadata is human-readable and auditable

## Output readout specs

Common readout specifications:

| Spec | Metric | Description |
|------|--------|-------------|
| `rate` | `spike_rate_hz` | Mean spike rate across neurons |
| `source` | `source_abs_mean` | Mean absolute source current |
| `lfp` | `lfp_abs_mean` | Mean absolute LFP magnitude |
| `csd` | `csd_abs_mean` | Mean absolute CSD magnitude |
| `eeg` | `eeg_power` | EEG-proxy power (if defined) |

See API reference for full list.

## Metadata preservation

Each operator returns a report declaring:

- **operator_status:** `simulated_proxy`, `physical_forward_model`, or `calibrated_empirical`
- **method:** How the operator computes its output
- **units_or_status:** Units (if physical) or proxy status
- **assumptions:** List of assumptions (geometry, solver, etc.)

This metadata supports future validation and calibration workflows.

## Example: Saving and loading bundles

```python
import jaxfne as jtfne
import json

# Simulate
model = jtfne.construct(cfg)
signals = model.simulate(sim)
readouts = model.compute_readout(signals, specs)

# Serialize
manifest = model.manifest(signals, readouts)
with open("output_bundle.json", "w") as f:
    json.dump(manifest, f, allow_nan=False, indent=2)

# Load
with open("output_bundle.json") as f:
    loaded_manifest = json.load(f)

# Inspect
for readout in loaded_manifest["readouts"]:
    print(f"{readout['name']}: {readout['value']}")
```

## Next steps

- [Calibration](calibration.md) for preparing outputs for empirical validation
- [Scope and limitations](scope_and_limitations.md) for understanding operator status
