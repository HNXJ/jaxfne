# Limitations and future plans

This page is the single location for jaxfne scope boundaries, proxy-readout limits,
and declared future field-computation regimes. Every other page describes the package
as it ships today.

jaxfne provides compact TFNE source, field, probe, objective, and optimizer workflows.

## Truth gates

jaxfne is a **`computational_scaffold`** for tensor-field neural workflows. Every
field/EEG/MEG/EMM/LFP/CSD output is a **`proxy_readout`** — a computational diagnostic
defined by explicit proxy equations. The following gates are enforced in code and may be
read but stay at their conservative defaults:

- **`field_solver_status = "laminar_proxy_no_pde"`** — the laminar field is a
  Gaussian-leadfield proxy with finite-difference CSD. The proxy operator is defined by
  its kernel equation; it stands in for a volume-conductor PDE solve.
- **`physical_amplitude_claim_allowed = False`** — amplitudes are relative (uncalibrated)
  units. EEG/MEG/LFP/CSD proxy outputs are reported in proxy units.

### Modeling assumptions

- **local nonlinearity** — preserved within the reduced Izhikevich emitter dynamics at
  the single-unit level.
- **global linearity** — the source→field projection is an approximately linear
  (superposition-respecting) readout between populations.

These outputs are tools for structural and dynamical analysis. See the
[API reference](api/index.md) for the per-symbol gate annotations.

## Current scope

- tutorial-scale neural simulations
- laminar and population proxy readouts
- JSON-safe reports
- deterministic seeds
- package-level optimization examples

## Calibration path

Physical-unit workflows require geometry, conductivity, calibration data, solver settings,
and reference measurements. jaxfne keeps these fields explicit in reports so examples can
grow into calibrated workflows as those inputs become available.

## Declared future field regimes

TFNE defines a sequence of field-computation regimes of increasing complexity. The shipped
package uses the laminar proxy regime above. Future regimes are documented here to record
the intended direction:

| Regime | Description | Status |
| --- | --- | --- |
| Laminar proxy | Gaussian-leadfield proxy + finite-difference CSD | shipped |
| Conservation diagnostics | Poynting-flux and field-diagnostic bookkeeping | partial |
| Elliptic field solver | Poisson/elliptic volume-conductor solve with boundary and gauge handling | reserved |
| Full electrodynamic solver | Calibrated physical-conductivity field solve | reserved |

The elliptic and electrodynamic regimes are reserved for a future release and require
boundary, gauge, residual, convergence, and calibration validation before any
physical-amplitude reporting. The [elliptic field equation specification](guides/poisson_admissibility.md)
documents the admissibility mathematics for that regime.

## Related pages

- [Probe operators](guides/probe_operators.md)
- [Quickstart](quickstart.md)
- [Tutorials](tutorials/index.md)
