# jaxfne Project Sources — mathematical authority

**Live checkout** (implementation facts, tests, APIs) outranks the version
label on this bundle. As of the RBS/RBD/HDP migration, `pyproject.toml` may
report a newer release (e.g. `0.4.15`) than the snapshot note below.

Basis: revised project-source set + live repository. When older source text
conflicts with live code or `docs/doctrine/rbs_rbd_hdp.md`, the live checkout
and doctrine file govern implementation; revise these sources to remove
contradiction.

Historical snapshot note: an earlier revision targeted dev `v0.4.8`.

These files are intended to replace the older project-source bundle as current agent/project context. They separate invariant doctrine from volatile repository facts so future changes create less context drift.

## Files

1. `1_global_rules_and_restrictions.md` — identity, invariants, API/JAX discipline, truth gates, release rules.
2. `2_jaxfne_objective_grammar.md` — typed objective/optimizer semantics, nulls, gates, optimization evidence.
3. `3_jaxfne_visualization_rules.md` — visualization/readout doctrine, proxy-safe semantics, publication figures.
4. `4_tfne_theory_and_neural_tensor.md` — full mathematical theory: TFNE operator closure, NeuralTensor, RBS/RBD/HDP, dynamics, source/field/probe operators, differentiability and validation.
5. `5_docs_tutorials_etudes_and_suites.md` — executable documentation and evidence protocol.
6. `6_other_important_notes.md` — current drift register, context corrections, publication/release roadmap, long-term plan.
7. `7_tfne_algebra.md` — canonical TFNE algebra, version `tfne/2`, status `SEALED LANGUAGE`: the specification language (`x:A:y`, `O`/`X` composition, rules and `$L`/`$R` binding, frontiers, `N`/`P`/`R` realization, `(s,h0,I)`, semantic failure classes). Supersedes the `tfne/1` text and older architectural TFNE vocabulary on naming scope; the operator factorization (`Emitter -> ... -> Manifest`) is preserved. The language seal implies no parser or compiler: `jaxfne.tfne` is a compiler against this language, and the current conformance boundary is recorded in `docs/doctrine/tfne_algebra.md`.
8. `8_atlas.md` — canonical Atlas for the 0.5.x programme: ten simulations S1–S10 (≡ AT-01…AT-10) exposing one object `G -D-> N -> (X,H,W,B) -> Q -> Phi -> Y` from one HH neuron to a 20-area JDNA system, the common measurement vector `Y`, and the claim boundaries. Requirement IDs and mutable coverage state live in `artifacts/programme/atlas_coverage.json`, not here.

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
