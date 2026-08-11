# jaxfne Project Sources — revised for dev snapshot v0.4.8

Basis: `jaxfne-dev.zip`, `pyproject.toml` version `0.4.8`, plus the six predecessor project-source documents supplied with the project. The repository snapshot is authoritative when older source text conflicts with it.

These files are intended to replace the older project-source bundle as current agent/project context. They separate invariant doctrine from volatile repository facts so future changes create less context drift.

## Files

1. `1_global_rules_and_restrictions.md` — identity, invariants, API/JAX discipline, truth gates, release rules.
2. `2_jaxfne_objective_grammar.md` — typed objective/optimizer semantics, nulls, gates, optimization evidence.
3. `3_jaxfne_visualization_rules.md` — visualization/readout doctrine, proxy-safe semantics, publication figures.
4. `4_tfne_theory_and_neural_tensor.md` — full mathematical theory: TFNE operator closure, NeuralTensor, dynamics, source/field/probe operators, HDP, differentiability and validation.
5. `5_docs_tutorials_etudes_and_suites.md` — executable documentation and evidence protocol.
6. `6_other_important_notes.md` — current drift register, context corrections, publication/release roadmap, long-term plan.

## Canonical distinction

Scientific grammar:

`Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation`

Software execution grammar:

`CircuitSpec -> construct -> Model -> simulate -> Signals -> Objective/Analysis -> Optimizer -> Manifest/Validation`

`CircuitSpec` may be a `Configuration` or a `NeuronalTensor`; neither is defined as a lossless representation of the other.

## Status vocabulary

- **verified-current**: directly supported by the v0.4.8 snapshot.
- **invariant**: project rule intended to survive versions.
- **compatibility**: retained behavior/name, not preferred new-code vocabulary.
- **experimental**: implemented but not promoted to validated/stable scientific semantics.
- **planned**: roadmap only; must not be described as implemented.
