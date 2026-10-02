# Limitations and future plans

This page is the single location for scope boundaries, proxy-readout limits,
and declared future field-computation regimes.

## Status fields

jaxfne is a **`computational_scaffold`** for tensor-field neural workflows. Every
field/EEG/MEG/EMM/LFP/CSD output is a **`proxy_readout`** — a computational diagnostic
defined by explicit proxy equations. Status fields are enforced in code; they read at their conservative defaults:

- **`field_solver_status = "linear_solver"`** — the laminar field is a
  Gaussian-leadfield proxy with finite-difference CSD. The proxy operator is defined by
  its kernel equation; it stands in for a volume-conductor PDE solve.
- **`physical_amplitude_calibrated = False`** — amplitudes are relative (uncalibrated)
  units. EEG/MEG/LFP/CSD proxy outputs are reported in proxy units.

### Modeling assumptions

- **local nonlinearity** — preserved within the reduced Izhikevich emitter dynamics at
  the single-unit level.
- **global linearity** — the source→field projection is an approximately linear
  (superposition-respecting) readout between populations.

See the
[API reference](api/index.md) for the per-symbol gate annotations.

## Current scope

- tutorial-scale neural simulations
- laminar and population proxy readouts
- JSON-safe reports
- deterministic seeds
- package-level optimization examples

## Calibration path

Physical-unit workflows require geometry, conductivity, calibration data, solver settings,
and reference measurements. Reports keep these fields explicit so examples can
grow into calibrated workflows as those inputs arrive.

## Declared future field regimes

TFNE defines field-computation regimes of increasing complexity; the shipped
package uses the laminar proxy regime above. Later regimes record
the intended direction:

| Regime | Description | Status |
| --- | --- | --- |
| Laminar proxy | Gaussian-leadfield proxy + finite-difference CSD | shipped |
| Conservation diagnostics | Poynting-flux and field-diagnostic bookkeeping | partial |
| Elliptic field solver | Poisson/elliptic volume-conductor solve with boundary and gauge handling | reserved |
| Full electrodynamic solver | Calibrated physical-conductivity field solve | reserved |

The elliptic and electrodynamic regimes are reserved for a future release and require
boundary, gauge, residual, convergence, and calibration validation before any
physical-amplitude reporting. The elliptic field equation specification (`docs/guides/poisson_admissibility.md` — repository-internal reference, excluded from the built site)
documents the admissibility mathematics for that regime.

### Biological calibration (canonical V1 `canonical-v1-column-1000n`)

The canonical V1 is a **qualitative laminar scaffold**: `qualitative_laminar_scaffold = true`, `quantitative_cell_fraction = false`, `quantitative_connectivity = false`. Per-layer fractions (`L1 E 0.50` etc.) and typed connection rules are scaffold values with `value_tag="relative"` and provenance in `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` / `jaxfne/configs/canonical-v1-column-1000n.json` and `jaxfne.builders.CANONICAL_LAYER_CELL_TYPE_FRACTIONS` (see [Scope & status](scope_and_status.md) and [Calibration — Biological status](guides/calibration.md#biological-calibration-status)). Reduced Izhikevich labels `E`/`PV`/`SST`/`VIP` are functional scaffold identities, not warranted literal cell-type identities. No kernel change.

PseudoGenome realization via `develop` does not establish effectiveness (`ΔX` under intervention); see [PseudoGenome guide](guides/jdna.md).

## Related pages

- [Probe operators](guides/probe_operators.md)
- [Quickstart](quickstart.md)
- [Tutorials](tutorials/index.md)
- Tensor Electromagnetics Scope (`docs/tensor_electromagnetics_scope.md` — repository-internal reference, excluded from the built site) — the same regime table, expanded into a scope ladder with per-stage requirements
- TFNE Operator Doctrine (`docs/operator_doctrine.md` — repository-internal reference, excluded from the built site) — per-stage operator rule referencing this scope table
