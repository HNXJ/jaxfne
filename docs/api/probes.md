# Probes API

Probe declarations and multimodal readout channels for neural simulation output.

## Overview

A probe is a declarative metadata entry attached to a `Configuration` via
`cfg.probes([...])` (or `Configuration.probe(**kwargs)`). It does not itself
compute anything — it records intent (which readout modes to expect, how many
laminar contacts, etc.) as JSON-safe metadata on the config. The actual
readout arrays are produced at simulation time as fields on the real
`Signals` dataclass (`jaxfne/core.py:2716-2725`):

```python
@dataclass(frozen=True)
class Signals:
    time_ms: jax.Array
    V_m: jax.Array
    spikes: jax.Array
    sources: Optional[jax.Array]
    field: Optional[FieldOutput]
    metadata: dict[str, Any]
```

There is **no** `signals.LFP`, `signals.CSD`, `signals.EEG`, `signals.MEG`,
`signals.EMM`, or `signals.source` (singular) attribute. Laminar
field-derived readouts (LFP-proxy, CSD-proxy, phi_e-proxy, source-proxy) live
on `signals.field`, a `FieldOutput` (`jaxfne/fields/proxy.py:35-60`):

```python
@dataclass(frozen=True)
class FieldOutput:
    source_proxy: jax.Array
    phi_e_proxy: jax.Array
    csd_proxy: jax.Array
    lfp_proxy: jax.Array
    kernel: jax.Array
    contact_depths: jax.Array
    diagnostics: dict[str, Any]

    @property
    def phi_e(self) -> jax.Array: ...
    @property
    def csd(self) -> jax.Array: ...   # alias for csd_proxy
    @property
    def lfp(self) -> jax.Array: ...   # alias for lfp_proxy
```

EEG-proxy / MEG-proxy / EMM-proxy are **not** stored fields at all — they are
produced on demand by explicit transform functions in
`jaxfne/fields/probes.py` (`eeg_proxy_transform`, `meg_proxy_transform`,
`emm_proxy_transform`), which the caller invokes with a leadfield/weighting
matrix it supplies. They are not automatically computed during `simulate()`.

All readouts here are **proxy readouts** — uncalibrated computational
scaffold, not physical measurements (`physical_amplitude_calibrated=False`
throughout).

---

## Getting a signal: `Signals.get()`

The supported, alias-checked way to pull a named array out of `Signals` is
`signals.get(key, ...)` (`jaxfne/core.py:2743`), which raises `KeyError`
on an unknown key rather than silently returning `None`. The real key
aliases (`_SIGNALS_GET_KEY_ALIASES`, `jaxfne/core.py:2694-2708`):

| Alias(es) | Resolves to |
|---|---|
| `vm`, `v_m`, `voltage`, `V_m` | `V_m` |
| `spk`, `spike`, `spikes`, `raster` | `spikes` |
| `src`, `source`, `sources` | `sources` |
| `lfp`, `lfp_proxy` | `field.lfp_proxy` |
| `csd`, `csd_proxy` | `field.csd_proxy` |
| `phi_e`, `phi`, `phi_e_proxy` | `field.phi_e_proxy` |
| `field_source`, `source_proxy` | `field.source_proxy` |
| `eeg`, `eeg_proxy` | `eeg_proxy` (probe-only; not on `FieldOutput`) |
| `meg`, `meg_proxy` | `meg_proxy` (probe-only) |
| `emm`, `emm_proxy` | `emm_proxy` (probe-only) |

Neuron-indexed keys (`V_m`, `spikes`, `sources`) accept a
`selector=`/`area=`/`layer=`/`cell_type=`/`ids=` filter along the trailing
neuron axis. Field-derived keys (`lfp_proxy`, `csd_proxy`, `phi_e_proxy`,
`source_proxy`) are laminar/contact-indexed, not neuron-indexed, and raise
`ValueError` if you pass a selector. `signals.get(..., trial=...)` raises
`NotImplementedError` — core `Signals` has no declared trial axis; use
`jtfne.run_trials(...)` for multi-trial data.

**Example:**
```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

spikes = signals.get("spikes")           # [time, neurons]
V_m = signals.get("V_m")                 # [time, neurons]
lfp = signals.get("lfp")                 # [time, contacts], from signals.field.lfp_proxy
csd = signals.get("csd")                 # [time, contacts], from signals.field.csd_proxy

spike_rate_hz = spikes.mean(axis=0) * 1000.0 / 0.1
```

Direct attribute access also works for the fields that actually exist:
`signals.spikes`, `signals.V_m`, `signals.sources`, `signals.field.lfp`,
`signals.field.csd`, `signals.field.phi_e`.

---

## `Model.probe()`: extracting arrays by mode

`Model.probe(signals, modes=[...])` (`jaxfne/core.py:5120-5137`) is a
compatibility alias that returns a plain dict of the requested modes:

```python
def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
    ...
```

- `"spikes"` in `modes` -> `out["spikes"] = signals.spikes`
- `"V_m"` in `modes` -> `out["V_m"] = signals.V_m`
- `"source"`/`"sources"` in `modes` -> `out["sources"] = signals.sources`
- if `signals.field is not None`, the laminar modes are merged in via
  `probe_laminar_modes(signals.field, modes)` (`jaxfne/fields/proxy.py:450-479`):
  - `"source"`/`"sources"` -> `out["source_proxy"]`
  - `"phi_e"` -> `out["phi_e_proxy"]`
  - `"CSD"` -> `out["CSD"]` and `out["csd_proxy"]`
  - `"LFP"` -> `out["LFP"]` and `out["lfp_proxy"]`
  - `"J_e"` -> `out["J_e_status"] = "not_computed_without_real_field_solver"` (no real field solver is wired in)
  - any of the above also adds `out["readout_metadata"]` (truth-gate status from `validate_source_field_status`)

The **canonical v0.1 workflow** prefers `Model.compute_readout(signals, specs)`
over `Model.probe()` for typed, declarative scalar feature extraction (next
section). `Model.record()` is a user-facing alias for `Model.probe()`.

**Example:**
```python
report = model.probe(signals, modes=["spikes", "V_m", "source", "LFP", "CSD"])
lfp = report["LFP"]          # [time, n_contacts]
csd = report["csd_proxy"]    # [time, n_contacts]
```

---

## Declaring probes on a Configuration

`cfg.probes([...])` is a real callable-list facade
(`_ProbeDeclarations.__call__`, `jaxfne/core.py:1068-1101`) that attaches
probe metadata to the config and, if `ensure_defaults=True` (default), backs
in a default Izhikevich emitter and laminar proxy field declaration when
none are present yet:

```python
cfg = cfg.probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
```

`modes` here are declarative labels stored in config metadata — they are
**not** validated against an enum at declaration time (only later, when a
`Signals.get()`/`Model.probe()` call actually resolves a key, does an unknown
key raise). The set actually used by the shipped suite-2 preset
(`_SUITE2_PROXY_MODES`, `jaxfne/core.py:256-258`) is:

```python
_SUITE2_PROXY_MODES = (
    "spikes", "V_m", "source", "LFP", "CSD", "EEG-proxy", "MEG-proxy", "EMM-proxy"
)
```

There is no `"SPK"`, `"Vm"` (capitalized-short form), or `"MUA-proxy"` mode
anywhere in the codebase — those names do not appear in `_SUITE2_PROXY_MODES`,
`_SIGNALS_GET_KEY_ALIASES`, or `probe_laminar_modes`. Use the real names
above.

```python
cfg = cfg.probes(_SUITE2_PROXY_MODES, n_contacts=16)
```

or selectively:

```python
cfg = cfg.probes(["spikes", "LFP", "CSD"], n_contacts=16)
```

Lower-level equivalent: `Configuration.probe(**kwargs)` appends a raw probe
declaration dict directly (`jaxfne/core.py:1179-1193`); it explicitly rejects
the retired `kind`/`mode`/`modes`-as-legacy-alias keys via
`_reject_retired_like`.

---

## Readout Metrics (`ReadoutSpec` / `compute_readout`)

`jtfne.readout_spec(name, metric, ...)` builds a `ReadoutSpec`
(`jaxfne/core.py:8131-8156`); `model.compute_readout(signals, specs)`
(`jaxfne/core.py:5015-5057` area) evaluates a list of specs against a
`Signals` object and returns a list of `ReadoutResult`.

The **real, complete** set of valid `metric` values is
`_KNOWN_READOUT_METRICS` (`jaxfne/core.py:8287-8294`) — six entries, not
twelve:

| Metric | Description |
|--------|-------------|
| `spike_rate_hz` | Mean firing rate (Hz), from `signals.spikes` |
| `spike_count` | Total spike count |
| `mean_V_m` | Mean membrane voltage (mV) |
| `csd_abs_mean` | Mean absolute CSD-proxy magnitude, from `signals.field.csd` |
| `lfp_abs_mean` | Mean absolute LFP-proxy magnitude, from `signals.field.lfp` |
| `source_abs_mean` | Mean absolute source-proxy magnitude |

Any other `metric` string (e.g. `"mean_LFP"`, `"mean_EEG"`,
`"burst_frequency_hz"`) is **not** in `_KNOWN_READOUT_METRICS`; passing one
does not raise — `compute_readout` returns a `ReadoutResult` with
`status="unknown_metric"` and `value=None` (`jaxfne/core.py:5039-5046`).
There is no burst-frequency, peak-rate, min/max-voltage, or EEG/MEG/EMM
metric wired into `compute_readout` today.

`ReadoutSpec` also supports optional `time_window_ms=(start_ms, end_ms)` and,
for the two field metrics, `n_contacts_slice=(start, end)`.

**Example (only real metric names):**
```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("firing_rate", "spike_rate_hz"),
    jtfne.readout_spec("voltage", "mean_V_m"),
    jtfne.readout_spec("field_strength", "lfp_abs_mean"),
])
for r in readouts:
    print(r.spec_name, r.metric, r.value, r.status)
```

`ReadoutResult` fields (`jaxfne/core.py:3137-3155`): `spec_name`, `metric`,
`value` (float or `None`), `status` (`"computed"` / `"no_field"` /
`"empty_time_window"` / `"unknown_metric"`), `claim_level`
(`"computational_scaffold"`), `physical_amplitude_calibrated` (always
`False`), `metadata`.

---

## No `ProbeReport` dataclass

There is **no** `ProbeReport` class anywhere in the package (confirmed:
`grep -rn "class ProbeReport" jaxfne/` returns nothing). Two real, differently
shaped things exist under similar names — don't confuse them with the
removed `ProbeReport`:

1. **`jaxfne.io.probe_report(n_probes, probe_types=None, metadata=None)`**
   (`jaxfne/io.py:218-234`) — a plain JSON-bundle builder, not a per-probe
   report object. It returns:
   ```python
   {
       "n_probes": int(n_probes),
       "probe_types": probe_types or {},
       "metadata": metadata or {},
   }
   ```
   **Example:**
   ```python
   from jaxfne.io import probe_report
   report = probe_report(n_probes=2, probe_types={"V_m": 1, "spikes": 1})
   ```

2. **`jaxfne.operator_status()`** (`jaxfne/core.py:7726-7738`) — returns a
   dict mapping operator symbol names to readiness strings (e.g.
   `"prototype_api"`, `"not_implemented"`), a registry-level status snapshot,
   not a per-call probe result.

If you need a per-probe-call structured result, use `ReadoutResult` (above)
or the plain dict returned by `Model.probe()`.

---

## Statement Boundaries

All probe/readout paths here are computational proxies:

- **No empirical validation:** results are simulated, not measured.
- **No physical amplitude:** `physical_amplitude_calibrated=False` throughout; do not state mV/µV without separate calibration evidence.
- **Relative metrics only:** use for comparative analysis, not absolute scaling.
- **Sign convention:** CSD-proxy positive = extracellular source = inward current (declared, not independently verified here).
- **No PDE field solver:** laminar field readouts are convolution-based proxies (`field_solver_status="linear_solver"`), not a solved Poisson/volume-conductor equation. EEG-proxy/MEG-proxy require a caller-supplied leadfield matrix and are "toy" projections, not a realistic head model.

**Safe statements:**
- "Spike rate increased by 20%."
- "LFP-proxy magnitude varies with depth."

**Unsafe statements:**
- "LFP amplitude is 50 µV." (uncalibrated)
- "CSD source is located at 400 µm depth." (localization not solved)
- "EEG-proxy matches real recordings." (no empirical validation)

---

## JSON Serialization

Probe/readout outputs built from `jaxfne.io` helpers (`probe_report`,
`ReadoutResult.to_dict()`, `Signals.summary()`) are JSON-safe by construction
(routed through `jaxfne.io.json_safe`):

```python
import json
from jaxfne.io import json_safe, probe_report

report = probe_report(n_probes=2, probe_types={"V_m": 1, "spikes": 1})
json.dumps(json_safe(report), allow_nan=False)  # must not raise

summary = json_safe(signals.summary())
json.dumps(summary, allow_nan=False)
```

NaN or Inf values will fail `json.dumps(..., allow_nan=False)`.

---

## See also

- [Probe Operators Guide](../guides/probe_operators.md) — worked walkthroughs
- [Fields API](fields.md) — Source projection and field computation (`FieldOutput`, `project_laminar_sources`)
- [Core API](core.md) — `Signals`, `Model`, `Configuration`, `ReadoutSpec`/`ReadoutResult`
- [API reference](index.md)
