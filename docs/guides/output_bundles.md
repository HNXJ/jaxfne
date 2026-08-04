# Output bundles

**Export and audit** a simulation run: spikes, voltages, field proxies, readout
metrics, and provenance metadata in one strict JSON-safe package.

## Structure

A typical output bundle includes:

```python
{
  "receipt_id": "uuid-...",
  "version": "0.4.8",
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

## The canonical path: `run_receipt` and `evaluate_report`

`Model.manifest()` (shown above) is a **compatibility method retained from
v0.0.4–v0.0.14**. It still works and is not scheduled for removal, but for
new code the canonical workflow is two typed, immutable, JSON-safe
alternatives:

- **`Model.run_receipt(signals, *, tags=None) -> RunReceipt`** — captures a
  completed simulation run for audit/reproducibility. Produces a
  deterministic `receipt_id` derived from `(config_hash, seed,
  jaxfne_version)`. Prefer this over `manifest()` for recording runs.
- **`Model.evaluate_report(signals, objective, *, readout_specs=None,
  readout=None) -> ObjectiveReport`** — evaluates an `Objective` and returns
  a frozen, JSON-safe report (losses, regularizers, gates,
  `all_gates_pass`, embedded readout results, and a `truth` dict with the
  standard truth-gate fields: `claim_level`, `physical_amplitude_calibrated`,
  `empirical_validation_status`, etc.). Prefer this over `evaluate()` when a
  typed, auditable result is needed.

A module-level convenience wrapper also exists: `jtfne.run_receipt(model,
signals, tags=None)`, equivalent to `model.run_receipt(signals, tags=tags)`.

```python
import jaxfne as jtfne

model = jtfne.construct(cfg)
signals = model.simulate(sim)

# Canonical v0.1 receipt (replaces model.manifest() for run auditing)
receipt = model.run_receipt(signals, tags={"paper": "etude_8"})

# Canonical v0.1 objective evaluation (replaces model.evaluate())
report = model.evaluate_report(signals, objective, readout_specs=specs)
```

Both `RunReceipt` and `ObjectiveReport` are dataclasses with JSON-safe
contents; serialize them the same way as `manifest()`'s dict output (e.g.
`json.dumps(jtfne.io.json_safe(receipt.__dict__), allow_nan=False)`), subject
to the same NaN/Inf checks described above. The `manifest`/`compute_readout`
path documented in this guide remains valid — it is simply not the newest
recommended entry point. See `jaxfne/core.py` (`Model.run_receipt`,
`Model.evaluate_report`) for the full signatures and docstrings; there is no
separate `docs/api/` page for these two methods as of this writing.

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

This metadata supports calibration validation path and calibration workflows.

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
- [Scope and limitations](../limitations_and_future_plans.md) for understanding operator status
