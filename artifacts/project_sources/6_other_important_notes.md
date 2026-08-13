# jaxfne Current State, Drift Register, Publication Roadmap, and Long-Term Plan

## 1. Snapshot authority

This source set was revised against the supplied `dev` snapshot whose `pyproject.toml` reports version `0.4.8`. Older project-source documents were generated around a `0.3.29` publication branch and are historical when they conflict with the current snapshot.

For any live repo task, verify branch/SHA/status independently; the zip snapshot does not establish the current remote branch head.

## 2. Current context corrections

### Corrected: canonical architecture

Use two layers of description:

```text
Scientific: Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
Software: CircuitSpec -> construct -> Model -> simulate -> Signals -> analysis/objective -> optimize -> export/validate
```

Do not use `Config -> Net -> Paradigm -> Trainer...` as the sole canonical grammar unless those names are verified as the actual supported API for the target checkout.

### Corrected: Configuration vs NeuronalTensor

Both are supported circuit-specification tiers converging on `Model`. `NeuronalTensor` is structured and information-rich; `Configuration` is compact/fluent. There is deliberately no assumed lossless converter.

### Corrected: field status

The v0.4.8 repository widely uses `field_solver_status="linear_solver"` for proxy operators while also documenting that those paths are not PDE solves. Treat this value as legacy compatibility metadata. A linear projection is not a numerical field solver.

### Corrected: physical solver roadmap

The current repository contains an experimental PDE solver path, so “physical solver is only future v0.4 work” is stale. However, experimental numerical solution is still not equivalent to calibrated/validated physical measurement.

### Corrected: PyNWB

The repository exposes NWB-related names but current agent context records them among public stubs that raise `NotImplementedError`. Therefore PyNWB integration is planned/incomplete unless a live checkout proves otherwise. Do not advertise it as a completed export path.

## 3. Drift / friction register

### P0 — semantic and mathematical closure

1. **`linear_solver` ambiguity.** Proxy projection and actual equation solving are semantically conflated in metadata. Preserve compatibility but introduce orthogonal operator/solver/amplitude status.
2. **HDP continuation semantics.** Partial \((\mathbf H,\mathbf W)\) restoration is not necessarily full \(\mathcal X_t\) continuation (see `4_tfne_theory_and_neural_tensor.md` §2.3, §8.6), including delay history \(\mathcal B_t\) when edge delays are enabled. Add a complete dynamic-state contract and equivalence test.
3. **HDP long-run stability domain.** The repository itself documents `K_w_ctrl` sensitivity/preset-specific stability concerns. Establish supported parameter domains and boundedness evidence rather than generalizing from short runs.
4. **Source proxy duplicated gain.** The repo context identifies a hard-coded spike/source gain that must remain synchronized across dense/edge kernels. Centralize it or property-test equivalence.
5. **Competing context grammars.** Retire old API grammars from current project sources.

### P1 — release/API closure

6. **Runtime naming duality.** `RuntimeConfig` and `RuntimeConfiguration` create teaching/API friction. Choose one long-term semantic owner and preserve the other through compatibility.
7. **Stable-looking public stubs.** `GLIFEmitter`, `LIFEmitter`, `write_nwb`, `read_nwb`, and experimental solver-related exports are recorded in current repo context as raising `NotImplementedError` on use. Implement or fence/demote before stable release.
8. **Truth metadata overloaded.** Split operator type, solver validation, and amplitude calibration rather than encoding them in one status string.
9. **Context volume.** Agent/governance history should not dominate current scientific context. Keep current invariants concise and generate volatile facts from scripts.

### P2 — cleanup

10. **Per-function documentation-link rule.** Replace with mechanically audited public API documentation coverage.
11. **Historical tutorial/version counts.** Remove from invariant context; generate inventories from the checkout.
12. **Completed plans/receipts.** Archive outside active agent context.

## 4. Immediate publication/release roadmap

### Phase A — semantic freeze (1–2 focused days)

Deliverables:

- adopt this revised source set;
- define canonical scientific/software grammars;
- document Configuration/NeuronalTensor relation;
- specify runtime roles;
- define proxy vs PDE vocabulary;
- define full simulation-state continuation;
- define source modes and source gain ownership.

Acceptance:

```text
zero known contradictory current doctrine
one definition per concept
no new scientific feature work
```

### Phase B — mathematical invariant suite (2–4 days)

Add tests for:

1. deterministic repeatability;
2. finite-state closure;
3. shape contracts;
4. linearity/superposition of declared linear operators;
5. zero-source response where mathematically applicable;
6. dense/edge equivalence where implementations should agree;
7. segmented full-state continuation equivalence;
8. explicit HDP nulls: \(N_W^{\mathrm{HDP}}\), \(N_H\), and \(N_{\mathrm{system}}\);
9. HDP boundedness over declared supported domains;
10. gradient checks for publication-used differentiable paths.

Acceptance: every invariant has an exact test command and passing receipt on the frozen candidate SHA.

### Phase C — API contraction (1–2 days)

- fence or implement stable-looking stubs;
- centralize duplicated scientific constants;
- preserve old names through wrappers;
- keep optional imports lazy;
- avoid new root-level API expansion;
- update docs with every semantic patch.

Acceptance: every stable public callable used in docs executes; clean root import succeeds without optional extras.

### Phase D — publication experiment matrix (~1 week compute/analysis)

#### E1 Minimal EI

Purpose: validate operator closure and numerical invariants.

#### E2 Laminar structured circuit

Purpose: validate NeuralTensor structure, laminar/cell-type accounting and readout composition.

#### E3 HDP adaptation / omission / oddball

Purpose: test timescale-dependent adaptation, perturbation/recovery and omission effects under explicit nulls/ablations and repeated seeds.

#### E4 Objective/optimization recovery

Purpose: demonstrate that a declared objective can recover a known synthetic parameter/target regime.

Required controls across the matrix:

```text
HDP weight-update null, RBS-dynamics null (`N_H`), or full-system null as appropriate
structure shuffle
cell-type/layer shuffle where meaningful
source/readout ablation
stimulus-target shuffle
seed replication
parameter sensitivity
```

Acceptance: all manuscript claims map to a result table, null/control and frozen artifact.

### Phase E — release candidate

Freeze an immutable SHA and run the complete validation matrix:

```text
compile
lint/static checks
full pytest
clean-venv root import
optional-dependency laziness tests
docs strict build
tutorial/Etude execution
strict JSON validation
artifact hashes
wheel/sdist build + install smoke
CPU reference runs
publication figure regeneration
```

Do not publish/tag/release until exact commands, environment and exit receipts are archived.

## 5. Methods-paper plan

### Central claim

jaxfne provides a typed, differentiable computational factorization that separates neural dynamics from source construction, field/readout operators, probes, objectives, optimization and evidence bookkeeping.

### Suggested manuscript

1. **TFNE formalism** — define `E`, `S`, `F`, `P`, `O`, `A` and their tensor contracts.
2. **Circuit specification** — Configuration and NeuralTensor as constructors into a common executable model space.
3. **JAX execution** — recurrence, batching, PRNG, differentiability and complexity discipline.
4. **Source/readout hierarchy** — native/normalized/calibrated source ladder; proxy vs PDE operator distinction.
5. **Objectives and optimization** — gated metrics, nulls, bounded search/gradient paths.
6. **RBS/RBD/HDP** — RBS definition, RBD with optional fixed \(W\), HDP plasticity subset, timescales, stability domain, full \(\mathcal X_t\) continuation semantics and nulls (`artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` §8).
7. **Validation** — mathematical properties, numerical tests, repeated seeds, ablations and convergence.
8. **Experiments** — EI, laminar, HDP/oddball, parameter recovery.
9. **Scope and limitations** — proxy semantics, calibration boundary, experimental solver status, mechanism-claim boundary.

The paper should be a mathematical/computational methods paper with the package as its executable reference implementation, not a catalog of package features.

## 6. Long-term plan

### Stage 1 — stable TFNE core

Goal: make the operator grammar and evidence contracts stable.

Freeze:

```text
Emitter/Source/Field/Probe interfaces
Signals semantics
CircuitSpec -> Model construction contract
Objective report schema
Manifest/validation schema
truth/readout status ontology
```

Success means new emitters/readouts can be added without changing the conceptual grammar.

### Stage 2 — RBS/RBD/HDP as a rigorously characterized dynamical family

Develop equilibrium analysis, local stability, timescale separation,
boundedness domains, omission/global-local oddball predictions, and comparison
against simpler adaptation alternatives. HDP is the plasticity subset
(\(F_W\)); RBD with \(\dot W=0\) remains in scope.

The goal is not to declare the equations biologically true; it is to make
mathematical predictions precise and falsifiable.

### Stage 3 — calibrated source bridges

Move selected source operators from native/relative status toward physical units only through explicit calibration maps, morphology/area information where required, conservation checks, and external comparison.

Jaxley is the natural detailed-emitter partner for this stage.

### Stage 4 — physical field validation

Promote experimental PDE operators only after:

```text
governing equations
geometry/material model
boundary/reference conditions
residual tests
mesh/grid convergence
gradient checks if differentiable
unit/calibration chain
comparison to an external solver/reference
```

Keep proxy and physical-solver APIs separate even after validation; proxies remain useful computational models.

### Stage 5 — empirical interfaces and inverse problems

After forward semantics are stable:

- optional PyNWB archival/import/export with round-trip tests;
- empirical dataset adapters;
- held-out fitting/evaluation;
- uncertainty and identifiability reports;
- inverse modeling under explicit priors;
- model comparison rather than single-model confirmation.

### Stage 6 — mature ecosystem boundary

Keep jaxfne compact:

```text
Jaxley: detailed cellular/compartment dynamics
PyNWB: archival data schema
external FEM/BEM/EEG/MEG tools: specialized physical forward models
jaxfne: composition, differentiable operators, objectives, optimization, evidence
```

Do not absorb mature external ecosystems merely to increase feature count.

## 7. Long-term scientific acceptance ladder

```text
L0 executable scaffold
L1 deterministic/operator validation
L2 nulls/ablations and parameter recovery
L3 numerical convergence / calibrated source bridge
L4 external solver comparison
L5 held-out empirical prediction
L6 mechanism discrimination against alternatives
```

Every paper/release should state which level each result actually reaches.

## 8. Final stop rule

Until the methods paper and stable release are frozen, prefer deletion, consolidation, tests, and semantic clarification over new features. The package already has sufficient breadth; publication value now comes from making the mathematics, implementation, documentation, tests and evidence artifacts describe the same system exactly.
