# Calibration

jaxfne readouts are computational proxies by default. Physical-unit workflows require calibration.

## Calibration-ready design

jaxfne is designed to support future calibration workflows. The framework:

- **Preserves source identity** — track source origin (emitter type, cell type)
- **Declares assumptions** — metadata fields state conductivity, solver, geometry models
- **Supports geometry specification** — define layer/contact depths and spatial locations
- **Allows empirical mapping** — workflows can include measured-to-model correspondences

## Next steps for calibration

To prepare a workflow for calibration:

1. **Specify geometry:** Define layer depths, contact locations, tissue conductivity (if known)
2. **Document assumptions:** State source model, field solver, claim-status
3. **Collect reference data:** Identify empirical EEG/MEG/LFP/CSD for comparison
4. **Validate:** Compare proxy readouts to empirical data; compute residuals and alignment metrics

## Calibration Specification and Reporting (v0.2.5)

jaxfne v0.2.5 introduces calibration specification and reporting contracts. These allow workflows to declare calibration state without changing the default proxy readout behavior.

### CalibrationSpec

Declare calibration intent with `CalibrationSpec`:

```python
from jaxfne.validation import CalibrationSpec, make_calibration_report

# Declare uncalibrated proxy (default)
spec = CalibrationSpec(
    name="default_proxy",
    target="readout"
)

# Declare toy calibration (illustrative, pending validation)
spec = CalibrationSpec(
    name="toy_eeg_proxy",
    target="readout",
    mode="toy_scale",
    scale=1.0,
    units="proxy_V",
    reference="toy_leadfield"
)

# Declare empirical calibration candidate (metadata declared, pending validation in v0.2.5)
spec = CalibrationSpec(
    name="eeg_candidate",
    target="readout",
    mode="empirical_gain_candidate",
    scale=2.5,
    units="mV",
    reference="pilot_recording_2024"
)
```

### Supported Modes

- `uncalibrated_native` — Proxy readout, no calibration (default)
- `toy_scale` — Illustrative calibration, pending validation
- `relative_normalized` — Normalized relative to proxy baseline
- `empirical_gain_candidate` — Candidate gain estimate, pending validation
- `physical_units_candidate` — Candidate physical units, pending validation
- `calibrated_empirical` — Calibration metadata declared (pending validation in v0.2.5)

### Calibration Reports

Generate a calibration status report:

```python
report = make_calibration_report(spec, readout_kind="lfp_proxy")

# report contains:
# - calibration_name, target, mode, status
# - units, scale, reference, description
# - physical_amplitude_claim_allowed: false (always in v0.2.5)
# - calibration_claim_level: computational_proxy_with_declared_metadata
# - assumptions and warnings
```

### Important: v0.2.5 Behavior

- **All proxy readouts remain computational proxies** by default
- `physical_amplitude_claim_allowed` stays `false` for all modes in v0.2.5
- Calibration metadata is declared for future validation, validation pending
- Empirical calibration requires separate geometry, reference data, and validation evidence beyond the spec

## Current status (v0.2.3–v0.2.5)

- ✓ Metadata fields support future calibration annotations
- ✓ JSON output bundles preserve geometry and source information
- ✓ Calibration specification contracts: v0.2.5 (metadata only, no physical amplitude upgrade)
- ◐ Empirically validated calibration examples: v0.2.6–v0.2.7
- ◐ Empirically calibrated readouts: v0.3.x and beyond

## Example: Declaring a calibration-ready workflow

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",  # or specify σ in S/m if known
        depths=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],  # layer boundaries
        boundary="mean_zero_neumann",
        gauge="mean_zero"
    )
    .probe(
        name="calibration_ready",
        n_contacts=6,
        contact_depths=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    )
)

model = jtfne.construct(cfg)
signals = model.simulate(...)
manifest = model.manifest(signals, ...)

# Manifest includes geometry and metadata suitable for later validation
```

## References and further reading

- [Scope and limitations](../scope_and_limitations.md)
- [Output bundles](output_bundles.md)
- [Probe operators](probe_operators.md)
