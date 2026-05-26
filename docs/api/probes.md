# Probes API

## Probe reports

Each probe operator returns a JSON-safe report declaring operator status, units, assumptions, and claim constraints.

### Probe report structure

```python
@dataclass
class ProbeReport:
    kind: str  # "spk" | "vm" | "source" | "lfp_proxy" | "csd_proxy" | "eeg_proxy" | "meg_proxy" | "emm_proxy"
    method: str  # Computation method (e.g., "threshold_or_emitter_spike_array")
    units_or_status: str  # Units or proxy declaration
    operator_status: str  # "simulated_proxy" for all v0.2.x operators
    physical_amplitude_claim_allowed: bool  # Always false for proxy operators
    # Additional fields per operator...
```

### Eight operator kinds

| Operator | Status | Math Form | Use Case |
|----------|--------|-----------|----------|
| **SPK** | Spike detection | $\mathrm{SPK}_n = \mathbb{1}[V_n \geq \theta_n]$ | Event-based readout |
| **Vm** | Voltage trace | $\mathrm{Vm}_n = V_n(t)$ | State readout |
| **source** | Source proxy | $S_n = f_{\mathrm{source}}(x_n, \theta_n)$ | Emitter-to-field mapping |
| **LFP-proxy** | Potential sampling | $\phi_{\mathrm{proxy}} = \sum W_{cn} S_n$ | Contact depth average |
| **CSD-proxy** | 2nd derivative | $\mathrm{CSD} \approx \nabla^2 \phi_{\mathrm{proxy}}$ | Laminar current profile |
| **EEG-proxy** | Scalp projection | $Y_{\mathrm{EEG}} = \sum L_s \phi_{\mathrm{proxy}}$ | Toy head model |
| **MEG-proxy** | Magnetometer | $Y_{\mathrm{MEG}} = \sum L_s o_n S_n$ | Orientation-weighted sources |
| **EMM-proxy** | Activity cost | $\mathrm{EMM} = \alpha\|S\|_1 + \beta\|\phi\|_1$ | Relative activity metric |

### Vocabulary

- **Proxy terminology:** All operators use `*-proxy` (e.g., `lfp_proxy`, `csd_proxy`) to denote computational operators.
- **Simulated status:** All v0.2.x operators are "simulated" or "proxy"—not empirically validated.
- **No truth_mode:** Public reports contain no internal `truth_mode` field.
- **Physical claims:** `physical_amplitude_claim_allowed: false` for all v0.2.x operators.

### Example report (JSON)

```json
{
  "kind": "lfp_proxy",
  "method": "point_or_finite_contact_phi_proxy",
  "units_or_status": "proxy_voltage_units_or_V_if_calibrated",
  "operator_status": "simulated_proxy",
  "field_solver_status": "laminar_proxy_no_pde",
  "physical_amplitude_claim_allowed": false
}
```

### Validation

```python
import json
from jaxfne.io import json_safe

# Ensure JSON-safe with no NaN/Inf
report = model.probe(signals, modes=["lfp_proxy"])
json.dumps(json_safe(report), allow_nan=False)
```

## See also

- [Probe operators guide](../probe_operators.md) — Detailed operator descriptions and mathematical forms
- Developer documentation — Contract validation and testing
- [API reference](index.md)
