# jaxfne Global Rules and Restrictions

## 1. Identity

`jaxfne` is a compact JAX-native Tensor-Field Neural Equations (TFNE) computational scaffold. Its purpose is to compose neural emitters, source maps, field/readout operators, probes, objectives, optimizers, and evidence manifests under explicit scientific claim gates.

Canonical import only:

```python
import jaxfne as jtfne
```

The package is not, by default, a validated biological mechanism, calibrated EEG/MEG instrument, metabolic model, or general physical electromagnetic solver.

## 2. Two grammars; do not conflate them

### Scientific operator grammar

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

### Software execution grammar

```text
CircuitSpec -> construct -> Model -> simulate -> Signals
                                      |
                                      +-> Analysis / Objective -> Optimizer
                                      +-> Vis / Export -> Manifest / Validation
```

`CircuitSpec` currently includes at least two real construction tiers:

- `Configuration`: compact/fluent circuit specification.
- `NeuronalTensor`: structured `Areas x Layers x NeuronTypes` specification with explicit topology/geometry semantics.

They converge to executable `Model` objects but are not declared lossless transforms of each other. Do not invent a converter or imply one is merely a wrapper around the other.

## 3. Truth gates

Default interpretation remains conservative:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

The current repository often stores the legacy value `field_solver_status="linear_solver"` for proxy projections. Treat this as compatibility metadata, not evidence of a solved PDE or linear system. Preferred semantic distinction for new work:

```yaml
field_operator_type: linear_projection | pde_solve | not_computed
field_solver_status: not_solved | experimental_pde_solver | validated_solver
amplitude_status: relative | calibrated
```

Until a compatibility migration is implemented, preserve existing public fields but never translate `linear_solver` into a physical-solver claim.

### Escalation requirements

A stronger physical field claim requires, as applicable:

1. explicit source definition and units/calibration;
2. geometry and material parameters;
3. governing equation;
4. boundary conditions;
5. gauge/reference condition where required;
6. numerical method;
7. residual and convergence evidence;
8. finite outputs and unit checks;
9. external or empirical validation.

No solver claim from projection alone. No physical amplitude from normalized/native-unit sources. No mechanism claim from qualitative resemblance alone.

## 4. JAX discipline

- Use `jax.numpy` for numerical kernels.
- Explicit PRNG keys; deterministic seed receipts.
- Pure kernels where practical.
- `lax.scan` for time recurrence.
- `vmap` for batches, trials, seeds, or homogeneous populations where shapes permit.
- `jit` hot numerical paths only.
- Keep plotting, filesystem I/O, JSON, NWB, and rich Python objects outside compiled kernels.
- Default float32; x64 is explicit opt-in and must be enabled before relevant arrays are created.
- CPU correctness precedes accelerator optimization.
- Prefer lower asymptotic complexity; avoid dense O(N^2) construction when sparse topology permits O(E) or better structured alternatives.

## 5. API discipline

- Preserve supported public APIs; use compatibility wrappers for migrations.
- Do not invent APIs from documentation prose.
- Public stable callables should execute supported behavior. Names that intentionally raise `NotImplementedError` should be implemented, demoted/fenced as experimental, or removed from the stable surface before final release.
- Optional dependencies remain lazy. Root import must not require Jaxley, PyNWB, Plotly, Matplotlib, or other optional ecosystems.
- Jaxley is a detailed differentiable emitter bridge, not a top-level dependency or an implementation target to duplicate.
- PyNWB is an optional archival/export target only when an explicit schema, units/status, provenance, and read/write validation exist.

## 6. Runtime/configuration discipline

The v0.4.8 snapshot contains both `RuntimeConfig` and `RuntimeConfiguration`. Treat this as current API reality, not as two different scientific concepts. New documentation must state their exact roles and avoid suggesting semantic differences that are not necessary to understand TFNE.

Long-term target: one canonical runtime semantics with compatibility adapters preserving older call sites.

Every tunable scientific parameter must be explicit in a public signature/configuration with a default. A callable intended for users must run with defaults or with a fully specified configuration.

## 7. Source bookkeeping

One declared source interpretation per run. The source operator must define:

```text
source_mode
source_calibration_status
support/normalization
sign convention
units or native/relative status
```

Never double count the same synaptic contribution through multiple source terms. Duplicated source gains/constants across dense/edge kernels are architectural debt: centralize or property-test equivalence.

## 8. Field/readout discipline

Proxy projections and PDE solves are different operator classes. Preserve this distinction in code, metadata, figures, docs, and paper prose.

Readouts such as LFP-like, CSD-like, EEG-like, MEG-like, EMM proxy, and spectrolaminar proxy remain proxy quantities unless the relevant calibration/solver/validation ladder is satisfied.

## 9. State-continuation discipline

A recurrent simulation continuation is valid only if the carried state is
sufficient for the Markov state of the selected kernel. Authority:
`artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` §2.3.

For RBD/HDP recurrent dynamics this may include membrane state, recovery
state, previous spikes, synaptic filter state, **Relative Biophysical State
(RBS)** coordinates \(\mathbf H\), weights \(\mathbf W\), and **delay history
\(\mathcal B_t\)** when finite edge delays are enabled. Restoring only
\((\mathbf H,\mathbf W)\) is partial hidden-state/parameter initialization
unless equivalence to full \(\mathcal X_t\) continuation is proven.

## 10. Documentation and context discipline

Repository state beats remembered context. Before publication or repo work verify branch/SHA/status/version from the checkout.

Do not require a URL in every individual function docstring. Instead require every supported public API surface to be discoverable from API documentation and test this coverage mechanically where possible.

Historical receipts, retired APIs, old versions, and completed plans must not remain mixed with current doctrine. Archive them outside current agent context.

## 11. Release stop rules

Stop and do not escalate claims if any of the following holds:

- NaN/Inf or implausible numerical blow-up;
- ambiguous source semantics;
- proxy labeled as physical measurement;
- solver claim without governing-equation/boundary/residual evidence;
- physical amplitude claim without calibration and units;
- mechanism claim without appropriate nulls/ablations/repeated seeds and comparison;
- unverified test result without command + environment + exit receipt;
- publication number/figure not generated from the frozen publication SHA.
