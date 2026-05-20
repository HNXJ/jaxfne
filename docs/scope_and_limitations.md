# Scope and limitations

jaxfne is a JAX-native source-to-field/readout layer for building and testing multimodal tensor-field workflows.

## What jaxfne is designed for

- **Reproducible proxy simulations** — deterministic, JAX-native, CPU-safe by default
- **Source-to-field organization** — structure emitter outputs into field/readout workflows
- **Multimodal readouts** — eight probe operators for spikes, voltage, LFP, CSD, EEG, MEG, EMM
- **JSON-safe output bundles** — strict serialization with metadata and validation tracking
- **Calibration-ready workflows** — framework supports future empirical validation

## What jaxfne is not

- **A whole-brain simulator** — focuses on local/laminar circuits
- **A substitute for empirical validation** — computational scaffold requiring external calibration
- **A complete physical-unit model** — default readouts are computational proxies

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
