# Remaining work

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Stable authorized facts: `artifacts/fact_stack.md` (human edit only).
Evidence: git, tests, receipts, tags, PyPI — not this file.

**v0.4.24 published** (tag `v0.4.24`, peel `7f89eff`; origin/main @ `7f89eff`; GitHub release with CI bytes; PyPI `0.4.24` via trusted publishing run `34839857303`; RTD stable build `34547734` @ `7f89eff`).
Published predecessors: `v0.4.23`, `v0.4.22` (immutable).

---

MODE = TFNE/2 ADOPTION

`tfne/2` (`SEALED LANGUAGE`) replaced project source 7. The language is
authority; `jaxfne.tfne` is a compiler against it and conforms partially.
Measured delta and method:
`artifacts/programme/tfne2_language_supersession_receipt.md`; current standing
of every clause in `docs/doctrine/tfne_algebra.md` under "Compiler
conformance".

`tfne/2` S31 orders the remaining layers as: algebra (sealed) -> biological
definition library (`CTX`) -> realization/compiler -> scientific qualification.
The compiler items below are written in dependency order within their layer;
the ordering **between** the `CTX` item and the compiler items is a human
decision and is not settled here.

## Compiler conformance (`jaxfne.tfne`)

Each item conforms one clause and rewrites the tests that encode the superseded
rule. `tests/test_tfne_algebra.py` asserts `tfne/1` semantics as intended
behaviour, so these are test rewrites, not test additions.

TFNE2-01 (S14 exclusions) and TFNE2-02 (S6/S7 instance addressing) are closed —
receipt `artifacts/programme/tfne2_conformance_0102_receipt.md`.

- **TFNE2-08** — S10 ordered adjacency. **Meaning bug, do first.** Chained `O`
  applies each rule to every accumulated left operand, so `A O[ff] B O[ff] C`
  realizes `A>C` as well (12 edges where the language gives 8), and
  `L1 O[ff] ... O[ff] L6` realizes every layer onto every later layer instead
  of the laminar chain. The compiler fabricates projections the source does not
  declare. Fix `Ordered` expansion to bind the rule to its own adjacency, and
  expect `CORTEX`/`NESTED` corpus edge counts to change. Found while
  implementing TFNE2-01; measured in
  `artifacts/programme/tfne2_conformance_0102_receipt.md`.
- **TFNE2-03** — S20 typed natural ordering, declaration-independent
  (`L1<L2<L10`, `SEG.2<SEG.10`), with declared biological order overriding it.
  Changes realization indexing and `I` for existing specs.
- **TFNE2-04** — S9 frontiers: `in`/`out` as reserved interface path
  components, derived defaults `in({A O B}) = in(A)`, `out({A O B}) = out(B)`,
  X-composite union default, `E_FRONTIER_UNRESOLVED`. Carries S8: once
  frontiers exist, `{A O B} O C` must stop being edge-identical to
  `A O B O C`.
- **TFNE2-05** — S12 `$L` / `$R` rule metavariables and rule bodies that
  specify topology, mechanism, parameters, geometry and delay independently.
- **TFNE2-06** — S14/S25 statement atomicity: `;`-separated statements inside a
  composite, each atomic, a statement with an invalid resolved projection
  contributing nothing. Carries S6 prefix rule application `O[k](SEG^n)`.
- **TFNE2-07** — S25 semantic failure vocabulary (`E_ADDRESS_UNKNOWN`,
  `E_FRONTIER_UNRESOLVED`, `E_MECHANISM_UNRESOLVED`,
  `E_MECHANISM_NOT_PERMITTED`, `E_PROJECTION_REDUNDANT`,
  `E_EXCLUSION_UNKNOWN`, missing realization policy, ambiguous expansion,
  invalid proportion) as typed classes rather than prose `TFNEError`. Carries
  S13 projection-identity redundancy and S11 ungrouped-`X` rejection.

## Definition layer

- **CTX-01** — `tfne/2` S27 names `CTX` as the next definition candidate:
  populations and `P_{l,c}` including absent populations, `N`-scaling and
  allocation, `in`/`out` interfaces, local connectivity, biological identity
  vs dynamical realization, geometry required for observables. S28 keeps
  spectrolaminar qualification (SL0/SL1/SL2) outside the algebra, and forbids
  phenomenological visualization proxies as the objective defining `CTX`.
  Depends on TFNE2-04 for `in`/`out` if expressed in compiler-executable form.

## Independent of `tfne/2`

Both found while measuring the supersession; neither is caused by it.

- **TFNE-EXEC-01** — the claim that TFNE output executes in JaxFNE kernels is
  untested. `jaxfne/tfne.py` docstring and the doctrine pipeline table both
  assert it; `tests/test_tfne_algebra.py` states "no simulation kernels are
  touched", and no test calls `construct`/`simulate` on `to_neuronal_tensor`
  output. Either test it end to end or qualify the claim.
- **TFNE-IMPORT-01** — `jaxfne.tfne` is not reachable from the documented entry
  point. `import jaxfne; hasattr(jaxfne, "tfne")` is `False`; only
  `from jaxfne import tfne` works. `artifacts/context.md` declares `jaxfne` the
  only supported public entry point, and the doctrine page names
  `jaxfne.tfne` as the implementation. Decide the tier (public / advanced /
  internal) and register it in `jaxfne.public_surface`, or state that it is
  deliberately unexported.
