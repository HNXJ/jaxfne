# jaxfne — minimal persistent context

## Purpose

jaxfne expresses neural biophysics as modular Tensor-Field Neural Equations (TFNE): nested biological semantics and geometry are preserved while numerical tensors may be computationally flattened.

Scientific grammar:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Evidence
```

Execution grammar:

```text
CircuitSpec -> construct -> Model -> simulate -> Signals
```

`CircuitSpec` includes supported Configuration and NeuronalTensor forms.

## Mathematical invariants

- Internal quantities may remain relative. Absolute units arise through explicit calibration transformations at semantic boundaries.
- **RBS (Relative Biophysical State):** `H` is a finite-dimensional relative hidden biophysical state — not intrinsically homeostasis and not intrinsically a scalar. Coordinates may be normalized ratios \(z/z^\*\) or reduced \(\mathcal R(\mathbf z)\). **RBD** is dynamics of \(\mathbf x\) and \(\mathbf H\); **HDP** is hidden-state dependent plasticity (\(\dot W\)). Homeostasis is a possible regime of selected kernels (e.g. `homeostatic_ei`), not the definition of `H`. Authority: `docs/doctrine/rbs_rbd_hdp.md`, `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md`.
- General adaptive dynamics are conceptually `dX/dt = F_X`, `dH/dt = F_H`, `dTheta/dt = F_Theta`. RBD with fixed `W` is valid; plasticity rules are realizations of this grammar, not separate architectural subsystems by default.
- Preserve biological identity, topology, signs, receptor/mechanism identity, geometry, locality, and declared parameter ownership through compilation and optimization.
- Source, field, probe, objective, and calibration semantics remain explicit. A projection, proxy, PDE solve, calibration, and validation status are distinct concepts.

## Authority

For current mathematical specification, use the repository's authoritative project-source set when present. For implemented truth, inspect live `jaxfne/` code and tests. For public explanation, inspect README/docs. For current repository state, use generated state/audit scripts when available.

Do not store SHAs, versions, benchmark timings, test counts, bug lists, implementation line numbers, or temporary release state in persistent doctrine.

## Evidence

Keep these distinct:

- SPECIFIED — required by authoritative specification.
- IMPLEMENTED — present in the checkout.
- TESTED — covered by an executable test/verification receipt.
- OBSERVED — measured in a named run/environment.

Scientific experiments preserve failed prospective receipts. Do not tune a frozen protocol/controller after observing its validation outcome unless a new protocol is explicitly declared.

## Repository behavior

- Read the smallest relevant skill under `skills/` for procedure.
- Verify unfamiliar public symbols against live code before using them.
- Prefer package-native scientific operators over notebook/script-local duplicate engines.
- Keep reusable plotting in the visualization layer.
- Use targeted tests during development; broader/release gates are separate evidence tiers.
- Do not commit, push, tag, release, or mutate remote state without explicit authorization.
- Public README/docs should be compact mathematical descriptions using positive definitions; engineering history and agent governance stay outside public scientific documentation.
