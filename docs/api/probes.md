# Probes API

Probe operators and multimodal readout channels for neural signals.

## Overview

Probe operators extract various readout modalities from neural simulation:
- **Spike-based:** Spike detection (SPK) from membrane voltage
- **Voltage-based:** Direct voltage readouts (Vm)
- **Field-based:** lfp_proxy and csd_proxy from spatial current distributions
- **Sensor-based:** eeg_proxy and meg_proxy from head/field models
- **Activity cost:** emm_proxy signaling-energy estimation

All operators return **proxy readouts** suitable for tutorial simulations, designed for exploratory workflows.

---

## Probe Operators (Eight Kinds)

### 1. SPK: Spike Detection

```python
cfg = cfg.probes(["SPK"])
```

**Math form:** $\mathrm{SPK}_n = \mathbb{1}[V_n(t) \geq \theta_n]$

**Output:** Binary spike raster [time, neurons]

**Parameters:**
- `threshold` (float): Spike threshold (default: 30 mV for Izhikevich)

**Status:** Simulated proxy; threshold-based detection

**Units:** Binary (0/1)

**Example:**
```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
spikes = signals.spikes  # [time, neurons]
spike_rate_hz = spikes.mean(axis=0) * 1000.0 / dt_ms
```

---

### 2. Vm: Membrane Voltage

```python
cfg = cfg.probes(["Vm"])
```

**Math form:** $\mathrm{Vm}_n(t) = V_n(t)$

**Output:** Voltage trace [time, neurons]

**Status:** Direct state readout from Izhikevich emitter

**Units:** mV (millivolts)

**Range:** Typically -90 to 30 mV

**Example:**
```python
V_m = signals.V_m  # [time, neurons]
mean_voltage = V_m.mean()
voltage_std = V_m.std()
```

---

### 3. source: Transmembrane Current

```python
cfg = cfg.probes(["source"])
```

**Math form:** $S_n(t) = f_{\mathrm{source}}(I_{\mathrm{mem}}, \text{anatomy})$

**Output:** Source density [time, spatial locations]

**Description:**
Projects Izhikevich transmembrane currents into space using anatomical mapping. This is the input to field solvers (LFP, CSD).

**Status:** Spatial projection proxy

**Units:** Current density (relative Izhikevich units, not µA/mm³)

**Example:**
```python
source = signals.source  # [time, locations]
```

---

### 4. lfp_proxy: Local Field Potential

```python
cfg = cfg.probes(["lfp_proxy"])
```

**Math form:** $\phi_{\mathrm{proxy}}(t,c) = \sum_{n=1}^{N} W_{cn} S_n(t)$

where $W_{cn}$ is row-normalized source-to-contact mapping.

**Output:** Potential at recording contacts [time, contacts]

**Status:** Proxy convolution; no PDE solve

**Units:** Proxy voltage units (or mV if calibrated; see documentation)

**Description:**
Extracellular potential sampled at electrode contacts. Computed via weighted summation of sources with spatial weighting. Approximates field without solving Poisson equation.

**Example:**
```python
LFP = signals.LFP  # [time, n_contacts]
LFP_mean = LFP.mean(axis=0)  # Mean per contact
```

---

### 5. csd_proxy: Current Source Density

```python
cfg = cfg.probes(["csd_proxy"])
```

**Math form:** $\mathrm{CSD}_{\mathrm{proxy}}(t,c) = \frac{\phi(t,c+1) - 2\phi(t,c) + \phi(t,c-1)}{(\Delta z)^2}$

**Output:** Current source density [time, locations]

**Status:** Proxy second spatial derivative

**Units:** Proxy current density units (relative)

**Description:**
Estimate of inward/outward transmembrane current at each depth. Computed as second spatial derivative of LFP. Sign convention: **positive CSD = extracellular source = inward current**.

**Example:**
```python
CSD = signals.CSD  # [time, n_locations]
CSD_positive = (CSD > 0).sum()  # Count source voxels
```

---

### 6. eeg_proxy: Electroencephalogram

```python
cfg = cfg.probes(["eeg_proxy"])
```

**Math form:** $Y_{\mathrm{EEG}}(t, e) = \sum_{c=1}^{C} L_{ec} \phi_{\mathrm{proxy}}(t,c)$

**Output:** Scalp electrode readings [time, n_eeg_channels]

**Status:** Toy head model projection; no volumetric conductivity

**Units:** Proxy voltage units (relative mV)

**Description:**
Projects laminar LFP to scalp electrodes using a simplified lead-field matrix. Not a realistic head model; suitable for relative visualization.

**Example:**
```python
EEG = signals.EEG  # [time, n_eeg_channels]
```

---

### 7. meg_proxy: Magnetoencephalogram

```python
cfg = cfg.probes(["meg_proxy"])
```

**Math form:** $Y_{\mathrm{MEG}}(t, m) = \sum_{n,s} L_{m,ns} o_n S_n(t)$

**Output:** Magnetometer readings [time, n_meg_channels]

**Status:** Toy dipole model; no volume conductor

**Units:** Proxy magnetic field units (relative to source)

**Description:**
Estimates magnetic field from dipoles with orientation weighting. Each source contributes based on its orientation relative to magnetometer. Simplified model for visualization.

**Example:**
```python
MEG = signals.MEG  # [time, n_meg_channels]
```

---

### 8. emm_proxy: Energetic/Metabolic Activity Metric

```python
cfg = cfg.probes(["emm_proxy"])
```

**Math form:** $\mathrm{EMM}(t) = \alpha \|S(t)\|_1 + \beta \|\phi(t)\|_1$

**Output:** Single activity metric [time]

**Status:** Proxy cost function for exploratory analysis; signaling-energy proxy

**Units:** Relative activity intensity (no physical units)

**Description:**
Estimates relative energy cost of network activity. Combines source magnitude (ion pump cost) and field magnitude (ATP for signal propagation). Weighted by parameters α and β.

**Example:**
```python
EMM = signals.EMM  # [time]
activity = EMM.mean()  # Mean metabolic proxy over time
```

---

## Probe Specifications

### Declaring Multiple Probes

```python
cfg = cfg.probes([
    "SPK",
    "Vm",
    "source",
    "LFP-proxy",
    "CSD-proxy",
    "EEG-proxy",
    "MEG-proxy",
    "EMM-proxy"
])
```

Or selectively:

```python
cfg = cfg.probes(["MUA-proxy", "LFP-proxy", "CSD-proxy"])
```

### Probe Report Structure

Each probe operator returns a JSON-safe report:

```python
@dataclass
class ProbeReport:
    kind: str  # "spk" | "vm" | "source" | "lfp_proxy" | ...
    method: str  # Computation method
    units_or_status: str  # Units or proxy declaration
    operator_status: str  # "simulated_proxy" for all v0.2.x
    physical_amplitude_claim_allowed: bool  # Always false for proxy
```

**Example:**
```json
{
  "kind": "lfp_proxy",
  "method": "point_or_finite_contact_phi_proxy",
  "units_or_status": "proxy_voltage_units",
  "operator_status": "simulated_proxy",
  "physical_amplitude_claim_allowed": false
}
```

---

## Claim Boundaries

⚠️ **All probe operators are computational proxies:**

- **No empirical validation:** Results are simulated, not measured
- **No physical amplitude:** Cannot claim mV or µV units without calibration
- **Relative metrics only:** Use for comparative analysis, not absolute scaling
- **Sign conventions declared:** CSD+ = inward current (extracellular source)
- **Spatial approximations:** Field solvers use convolution, not full PDE

**Safe claims:**
- "Spike rate increased by 20%"
- "LFP magnitude varies with depth"
- "EMM proxy indicates higher activity"

**Unsafe claims:**
- "LFP amplitude is 50 µV"
- "CSD source is located at 400 µm depth" (localization not solved)
- "EEG matches real recordings" (without validation)

---

## Readout Metrics

### Available Metrics (from ReadoutSpec)

| Metric | Operator | Description |
|--------|----------|-------------|
| `spike_rate_hz` | SPK | Mean firing rate (Hz) |
| `burst_frequency_hz` | SPK | Burst rate estimate |
| `max_spike_rate_hz` | SPK | Peak firing rate |
| `mean_V_m` | Vm | Mean membrane voltage (mV) |
| `min_V_m` | Vm | Min voltage (mV) |
| `max_V_m` | Vm | Max voltage (mV) |
| `mean_source` | source | Mean source magnitude |
| `mean_LFP` | LFP-proxy | Mean LFP amplitude |
| `mean_CSD` | CSD-proxy | Mean CSD magnitude |
| `mean_EEG` | EEG-proxy | Mean EEG amplitude |
| `mean_MEG` | MEG-proxy | Mean MEG amplitude |
| `mean_EMM` | EMM-proxy | Mean metabolic proxy |

**Example:**
```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("firing_rate", "spike_rate_hz"),
    jtfne.readout_spec("voltage", "mean_V_m"),
    jtfne.readout_spec("field_strength", "mean_LFP")
])
print(readouts.results)
```

---

## JSON Serialization

All probe outputs must be JSON-safe:

```python
import json
from jaxfne.io import json_safe

signals_dict = json_safe(signals.to_dict())
json.dumps(signals_dict, allow_nan=False)  # Must not raise
```

NaN or Inf values in signals will fail serialization.

---

## See also

- [Probe Operators Guide](../probe_operators.md) — Detailed mathematical descriptions
- [Fields API](fields.md) — Source projection and field computation
- [Core API](core.md) — Signal and readout containers
- [API reference](index.md)
