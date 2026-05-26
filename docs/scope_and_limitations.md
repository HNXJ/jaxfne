# Scope and limitations

jaxfne is a JAX-based source-to-field/readout layer for building and testing multimodal tensor-field workflows.

## What jaxfne is designed for

- **Reproducible proxy simulations** — deterministic, JAX-based, CPU-safe by default
- **Source-to-field organization** — structure emitter outputs into field/readout workflows
- **Multimodal readouts** — eight probe operators for spikes, voltage, LFP, CSD, EEG, MEG, EMM
- **JSON-safe output bundles** — strict serialization with metadata and validation tracking
- **Calibration-ready workflows** — framework supports future empirical validation

## Design scope and boundaries

jaxfne is designed for local/laminar circuit simulation and proxy readouts:

- **Scope:** Focused on local/laminar circuits; whole-brain simulation is future work
- **Calibration:** Computational scaffold designed with empirical validation workflows in mind; calibration requires external reference data
- **Physical units:** Default readouts are proxy-scale; physical-unit calibration is future work

## Output scope

**Default readouts are computational proxies.** Physical-unit EEG/MEG/LFP/CSD workflows require:

- Appropriate geometry specification (conductivity, layer structure)
- Empirical calibration data for mapping proxy outputs to physical units
- Validation against reference measurements
- Clear documentation of assumptions and solver methods

jaxfne provides the framework for these steps; users must supply the validation evidence.

## Next steps

- See [Calibration](calibration.md) for preparing calibration-ready workflows
- See [Tutorials](tutorials/index.md) for examples
