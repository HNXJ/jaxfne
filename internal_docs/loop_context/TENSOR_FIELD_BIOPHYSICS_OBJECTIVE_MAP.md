<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# Tensor-Field Biophysics Objective Map

## Final objective

`jaxfne` should become a compact JAX-native emitter-to-source-to-field/probe-to-objective scaffold for computational biophysics, with a staged path from proxy readouts to calibrated source bridges, electromagnetic admissibility checks, physical field solvers, probe validation, and empirical comparison.

## Operator chain

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

## Tensor closure

```math
Y = P o F o S o E
L = O(Y, target, gates, manifest)
theta_next = A(theta, L, constraints, key)
M_next = V(M, X, Q, Z, Y, L, theta_next)
```

Worded form:

```text
emitters produce state trajectories;
source maps declare source tensors;
field/proxy operators map source tensors into observable-like variables;
probe operators sample named readouts;
objectives score readouts under gates and nulls;
optimizers propose bounded parameters;
manifests record evidence, hashes, runtime, status, and validation.
```

## Local nonlinear / global linear split

Local nonlinear components:

```text
ion-channel gates, membrane dynamics, spike threshold/reset, synaptic saturation, dendritic integration, receptor traces, plasticity, local transfer functions, adaptive medium state
```

Global linear components under fixed metadata:

```text
source projection, laminar mixing, field spreading, fixed leadfield projection, sensor readout, fixed-kernel long-range coupling
```

Failure test: if a global operator changes its kernel as a nonlinear function of distant activity, represent it as local adaptive state, explicitly nonlinear component, or state-dependent operator with tests.

## Electromagnetic admissibility ladder

| Level | Name | Allowed now | Required to advance |
|---:|---|---|---|
| P0 | Proxy projection | simulated/proxy readout | finite arrays, shape, proxy labels, manifest |
| P1 | Declared leadfield-like projection | declared geometry proxy | source/probe geometry metadata, fixed operator |
| P2 | Boundary-normalized kernels | boundary-aware scaffold | normalization, conservation checks, sign convention |
| P3 | Discrete volume solve | numerical field-solve candidate | `K phi = q`, boundary, gauge, residual, convergence |
| P4 | Differentiable adjoint solve | differentiable solver candidate | VJP/adjoint checks, gradient checks, JIT/VMAP safety |
| P5 | External validation | externally compared solver/readout | named reference comparison or empirical held-out comparison |

Current default: P0.

## Source calibration ladder

| Level | Status | Meaning | Evidence needed |
|---:|---|---|---|
| S0 | native reduced drive | unitless/native model drive | finite outputs, seed, params |
| S1 | normalized proxy source | source-like tensor for scaffolds | source mode and normalization metadata |
| S2 | calibrated current bridge | native feature mapped to amperes | calibration factor/map, units, tests |
| S3 | morphology/area-aware current | membrane current with geometry | morphology, area, sign convention |
| S4 | source-density conservation report | source density integrates consistently | support and integral checks |
| S5 | empirical/external calibration | amplitude validated externally | observed or external comparison |

## Probe/readout ladder

| Readout | Default status | Stronger evidence required |
|---|---|---|
| `spk` | simulated spike/event readout | emitter specification and finite seed-controlled output |
| `vm` | voltage-like state trace | state units and calibration |
| `source` | declared proxy source | source mode, calibration, conservation |
| `lfp_like` | laminar potential-like proxy | calibrated potential and electrode geometry |
| `csd_like` | finite-difference proxy | spacing, sign convention, validated estimator/current density |
| `eeg_like` | proxy projection | head model, leadfield, sensors, units, validation |
| `meg_like` | proxy projection | orientation, magnetic forward model, sensors, units |
| `emm_proxy` | normalized within-run cost proxy | calibrated `J`, `E`, conductivity/admittivity, energy accounting |

## Jaxley rule

Jaxley is an optional detailed differentiable cell/compartment emitter bridge. Keep it lazy. Do not reimplement Jaxley internals. Bridge output enters TFNE through explicit source maps and metadata.

## PyNWB rule

PyNWB is an optional archive/export target. Keep it lazy. Export explicit schema, units/status, provenance, and session metadata when available. Validate read/write round trips. NWB export alone is not empirical data validity.

## Scope catalogue

| Stage | Goal | Gate |
|---|---|---|
| v0.3.x | evidence stack, tutorial atlas, proxy readout contracts | ED9/ED10, manifests, strict JSON, clean install |
| v0.4.x | experimental physical field solvers | stable source/field schemas, boundary/gauge/residual tests |
| v0.5.x+ | external comparisons, calibration, inverse modeling | named references, held-out data, uncertainty reports |

## Stop rules

- No real EEG/MEG language for proxy projections.
- No physical amplitude without calibration + units + solver + validation.
- No solver claim without boundary + gauge + residual + convergence.
- No mechanism claim without nulls + ablations + repeated seeds + empirical comparison.
- No optional dependency on root import.
- No local notebook solver/objective/readout engine when package APIs exist.
