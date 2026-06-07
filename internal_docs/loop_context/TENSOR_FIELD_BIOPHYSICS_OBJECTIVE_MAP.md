# Tensor-field biophysics objective map — jaxfne

Planning document only. No solver implementation implied.

## Final scientific objective

jaxfne should become a compact JAX-native emitter → source → field/probe → objective scaffold for computational biophysics, with a staged path from proxy readouts to calibrated source bridges, electromagnetic admissibility checks, physical field solvers, probe validation, and empirical comparison.

## Core operator chain

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

## Mathematical closure

```text
Y = P o F o S o E
L = O(Y, target, gates, manifest)
theta_next = A(theta, L, constraints, key)
M_next = V(M, X, Q, Z, Y, L, theta_next)
```

Symbols: `E` emitter state; `S` source map; `F` field/proxy operator; `P` probe; `Y` readouts; `O` objective; `A` optimizer; `V` validation/manifest writer; `M` metadata bundle.

## Local nonlinear / global linear split

**Local nonlinear elements:** channels, membranes, dendrites, synapses, resets, receptor traces, local populations, metabolic/vascular local states.

**Global linear elements (fixed metadata):** source projection, volume conduction, laminar mixing, leadfield projection, sensor readout.

**Failure test:** if the global operator depends nonlinearly on distant activity, represent it as a local adaptive medium/state — not as fixed linear field propagation.

## Electromagnetic admissibility ladder

| Level | Name | Allowed claims | Required metadata | Required tests | Forbidden claims |
|---|---|---|---|---|---|
| P0 | proxy projection | proxy readout, laminar mixing | source mode, probe status, truth gates | finite outputs, JSON reports | solved field, calibrated amplitude |
| P1 | declared leadfield-like projection | EEG-like/MEG-like linear projection | leadfield shape, sensor count, gauge note | operator report, shape checks | real sensor-level EEG/MEG |
| P2 | boundary-normalized kernels | normalized kernel readout | boundary label, gauge, conductivity proxy | kernel sum/normalization checks | Maxwell/PDE solve |
| P3 | discrete volume-conductor solve | quasi-static solve receipt | mesh/geometry, conductivity, BC, gauge | residual, convergence | calibrated SI amplitude without calibration |
| P4 | differentiable adjoint solve | gradient-through-solve receipt | adjoint metadata, solver status | adjoint finite-diff spot checks | mechanism proof |
| P5 | external validation | comparison to external forward model | external reference, units, geometry | held-out comparison, nulls | superiority claims |

**Current default:** P0 (`laminar_proxy_no_pde`).

## Source calibration ladder

| Stage | Description |
|---|---|
| S0 | native reduced emitter drive |
| S1 | normalized proxy source |
| S2 | calibrated current bridge |
| S3 | morphology/area-aware membrane current |
| S4 | source-density conservation report |
| S5 | empirical/external calibration |

**Current default:** S0–S1 (uncalibrated proxy).

## Probe/readout ladder

| Readout | v0.3 default status | Stronger evidence required | Manifest fields required |
|---|---|---|---|
| SPK | supported | spike sorting validation | operator_status, method, finite |
| Vm | supported | unit calibration | units_or_status, calibration_status |
| source | supported | source calibration bridge | source_calibration_status, source_projection_mode |
| LFP-like | proxy | geometry + conductivity | field_solver_status, field_claim_level |
| CSD-like | proxy | sign convention + depth axis | CSD sign metadata, laminar axis |
| EEG-like | proxy | leadfield + head model | leadfield status, sensor geometry |
| MEG-like | proxy | leadfield + sensor geometry | leadfield status, units |
| EMM-proxy | normalized proxy | energy accounting with calibrated J,E | no metabolism claim; emm_proxy status |

## Jaxley / PyNWB integration

- **Jaxley:** optional detailed differentiable compartment/cell emitter bridge (`jaxfne/bridges.py`); lazy import; guarded unavailable error.
- **PyNWB:** optional export/archive target; lazy import; not root dependency.
- Neither is a top-level hard dependency.

## Stop rules

- No real EEG/MEG claim without geometry + leadfield + units + validation.
- No calibrated amplitude without source calibration + solver + units.
- No EMM physical power claim without calibrated J, E, and energy accounting.
- No mechanism proof without nulls, ablations, repeated seeds, and observed-data comparison.
- No PDE-solve claim without boundary, gauge, residual, convergence, and finite field report.

## Implementation roadmap

| Horizon | Work |
|---|---|
| Near-term | publication ED stack (ED5–ED10), manifest/hash receipts |
| Mid-term | config-first 0.3.28–0.3.34 ladder, probe contract matrix |
| Later | source calibration bridges |
| Later | experimental solver namespace |
| Later | PyNWB export receipts and external validation |

## Biophysics scoreboard (15 factors)

| Factor | score_now | target | next_action |
|---|---:|---:|---|
| source calibration | 30 | 90 | S2 bridge design + tests |
| field solver admissibility | 25 | 90 | P1–P2 metadata contracts |
| boundary/gauge | 20 | 85 | declare in field config + manifest |
| passivity/conservation | 25 | 85 | conservation proxy diagnostics |
| CSD sign convention | 55 | 90 | ED7 contract row + tests |
| probe geometry | 35 | 85 | leadfield metadata ladder |
| leadfield status | 30 | 85 | P1 receipts |
| EMM status | 40 | 80 | keep normalized proxy only |
| null/ablation status | 25 | 85 | ED9 |
| empirical comparison | 10 | 70 | post-scaffold phase |
| JAX transform safety | 60 | 90 | jit guards + lint |
| manifest completeness | 65 | 95 | ED5 + per-ED manifests |
| optional dependency laziness | 70 | 95 | ED4 regression |
| tutorial reproducibility | 45 | 90 | ED3 + ED8 |
| release archive | 15 | 100 | ED10 approval-gated |

## Validation

```bash
python3 scripts/publication_inventory.py
python3 -m mkdocs build --strict
```
