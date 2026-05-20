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

## Current status (v0.2.3)

- ✓ Metadata fields support future calibration annotations
- ✓ JSON output bundles preserve geometry and source information
- ◐ Calibration example workflows: planned in v0.2.4–v0.2.5
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

- [Scope and limitations](scope_and_limitations.md)
- [Output bundles](output_bundles.md)
- [Probe operators](probe_operators.md)
