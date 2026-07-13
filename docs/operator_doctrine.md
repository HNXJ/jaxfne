# TFNE Operator Doctrine

**Status:** conceptual documentation, no new runtime behavior
**Scope:** indexes the per-stage operator contract (domain, codomain, tensor rank,
class, assumptions, units/status) for the seven-stage TFNE pipeline. Introduces no
new operators, claim levels, or grammar — every fact below already exists in code
or in the linked pages; this page is the single table that ties them together.

## Purpose

jaxfne computation is organized as one pipeline:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest
```

[Configuration Grammar](guides/configuration_grammar.md) already establishes that
this operator chain and the fluent `Configuration` builder chain are two views of
one system: `Configuration` is the declarative specification, `construct()` is the
compiler, and the operator chain below is what the compiler produces. This page
does not replace that framing. It adds the one piece that page leaves implicit — a
single per-stage contract table, gathered from where each fact is actually declared:

- [Tensor-Network Ancestry](tensor_network_ancestry.md) — the basis-transform
  cascade (Emitter basis -> Source basis -> Field basis -> Readout basis) and the
  bridge-terms table.
- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — the formal
  equation, term glossary, and run boundary for each stage.
- [Computation Basis](computation_basis.md) — canonical tensor shapes and the
  `field_regime` gating doctrine.
- [Source/Field Equations](source_field_equations.md) — source bookkeeping and the
  forbidden double-counting pattern.
- [Tensor-Field Workflows](guides/tensor_field_workflows.md) — the probe-operator
  tensor form.

This page indexes those contracts. It does not restate their derivations.

---

## The seven-stage contract table

| Stage | Domain | Codomain | Tensor rank | Class | Key assumption | Units / status |
|---|---|---|---|---|---|---|
| **Emitter** | parameters `theta`, input `u(t)`, initial state | state trajectory `z(t)` | rank-2 `[T, N]` | dynamical | Izhikevich default; explicit `PRNGKey` required; Relative-value state | mV / ms; `claim_level = "computational_scaffold"` |
| **Source** | state `z(t)`, drive `I(t)`, spatial basis `chi(x)` | source density `q(x,t)` | rank-2 `[T, N]` | source projection | no synaptic double-counting (see Source/Field Equations); Relative-value coefficients | relative units; `source_calibration_status` |
| **Field** | source density `q` | `lfp_proxy`, `csd_proxy` | rank-2 `[T, n_contacts]` | field projection (declared, not solved) | isotropic conductivity assumed; the field equation is declared, not solved (`field_solver_status = "linear_solver"`) | relative units; `physical_amplitude_calibrated = False` |
| **Probe** | field + source + state | channel readout `Y_c(t)` | depends on operator (eight operators) | readout | a proxy reading unless lead-field geometry is calibrated to real electrodes | relative units; see [Probe Operators](guides/probe_operators.md) |
| **Objective** | signals + declared targets | scalar/vector score | rank-0 / rank-1 | evaluation | strict mode raises on a non-finite score | dimensionless |
| **Optimizer** | objective + parameters | updated parameters | matches the parameter tensor | search / gradient | hard spike reset has no gradient without a surrogate; `Model.tune()` checks `gradient_path_safe()` | n/a |
| **Manifest** | full run state | JSON-safe report `dict` | n/a (a dict, not a tensor) | bookkeeping / truth gate | write-once; `allow_nan=False` | n/a |

Every row's validation status is reported at runtime by `jaxfne.operator_status()`.
See [Tensor Operator Registry](api/tensor_operators.md) for the live registry and
its symbol-to-stage mapping — this page does not duplicate that registry.

**On the name "Validation" for the final stage:** some external descriptions of this
pipeline label the seventh stage "Validation" rather than "Manifest". The codebase
has one final stage, `Manifest`, in the "bookkeeping / truth gate" class above; it
already includes the field-admissibility/source-conservation checks that live in
`jaxfne/validation.py` (`validate_source_field_status`, `make_field_operator_status`,
etc. — see [Field Validation](api/validation.md)) as part of producing that report.
There is no separate eighth "Validation" stage to implement; "Validation" and
"Manifest" name the same stage.

---

## Operator classes

The "Class" column above groups operators by what they do, separate from their
position in the pipeline:

| Class | Description | Example |
|---|---|---|
| Dynamical | Evolves per-neuron state forward in time | the Izhikevich emitter (`simulate_eig_izhikevich`) |
| Source projection | Maps state/drive into a spatial source density | `construct_source_tensor`, `project_laminar_sources` (source half) |
| Field projection | Maps a source density to a declared field proxy | `project_laminar_sources` (field half), `csd_tensor` |
| Readout | Extracts a channel-level quantity from field/source/state | the eight probe operators in [Probe Operators](guides/probe_operators.md) |
| Evaluation | Scores signals against declared targets | `rate_targets`, `rate_synchrony_targets` |
| Search / gradient | Updates parameters from an objective | `agsdr`, `gsdr`, `optax_adam`, `random_search` |
| Bookkeeping / truth gate | Produces a JSON-safe report without changing simulation state | `manifest`, `run_receipt`, `validate_source_field_status` |

A solver class — a stage that solves rather than declares the field equation — is
reserved and not implemented. See [Tensor Electromagnetics Scope](tensor_electromagnetics_scope.md)
and the "Full electrodynamic solver" row in [Limitations and future plans](limitations_and_future_plans.md).

---

## See Also

- [Configuration Grammar](guides/configuration_grammar.md) — the fluent/operator grammar reconciliation
- [Tensor Operator Registry](api/tensor_operators.md) — the live `operator_status()` registry and symbol mapping
- [Tensor-Network Ancestry](tensor_network_ancestry.md) — basis-transform cascade and historical context
- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — per-equation glossary and run boundary
- [Computation Basis](computation_basis.md) — canonical tensor shapes and the field-regime gating doctrine
- [Tensor Electromagnetics Scope](tensor_electromagnetics_scope.md) — reserved field-solver stages
- [Limitations and future plans](limitations_and_future_plans.md) — scope boundaries and reserved regimes
