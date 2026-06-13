# Scope

jaxfne provides compact TFNE source, field, probe, objective, and optimizer workflows.

## Truth gates (non-negotiable)

jaxfne is a **`computational_scaffold`**, not a calibrated biological simulator.
Every field/EEG/MEG/EMM/LFP/CSD output is a **`proxy_readout`** — a computational
diagnostic, never a solved PDE or sensor-level measurement. The following gates
are enforced in code and may be read but never escalated:

- **`field_solver_status = "laminar_proxy_no_pde"`** — the laminar field is a
  Gaussian-leadfield proxy with finite-difference CSD; no PDE is solved and no
  current source density `J_e` is synthesized.
- **`physical_amplitude_claim_allowed = False`** — all amplitudes are relative
  (uncalibrated) units only. No physical-amplitude or physical-measurement claim
  is made for any EEG/MEG/LFP/CSD proxy output.

### Modeling assumptions

- **local nonlinearity** — preserved within the reduced Izhikevich emitter
  dynamics at the single-unit level.
- **global linearity** — the source→field projection is an approximately linear
  (superposition-respecting) readout between populations.

These outputs are tools for structural and dynamical analysis, not evidence of
biological mechanism or amplitude. See the [API reference](api/index.md) for the
per-symbol gate annotations.

## Current use

- tutorial-scale neural simulations
- laminar and population proxy readouts
- JSON-safe reports
- deterministic seeds
- package-level optimization examples

## Calibration path

Physical-unit workflows need geometry, conductivity, calibration data, solver settings, and reference measurements. jaxfne keeps these fields explicit in reports so examples can grow into calibrated workflows.

## Related pages

- [Probe operators](probe_operators.md)
- [Quickstart](quickstart.md)
- [Tutorials](tutorials/index.md)
