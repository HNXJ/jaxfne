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

Authorized priority order:

```
TFNE2-08 (done) -> TFNE-EXEC-01 (done) -> TFNE-IMPORT-01 -> remaining conformance gaps
```

The three clauses whose non-conformance changed realized biology are closed:
TFNE2-01 (S14 exclusions), TFNE2-02 (S6/S7 instance addressing), TFNE2-08 (S10
ordered adjacency). Receipts `tfne2_conformance_0102_receipt.md` and
`tfne2_conformance_08_receipt.md`. Execution qualification is next; do not
resume the conformance list ahead of it.

## Next

- **TFNE-IMPORT-01** — make the supported import path explicit and test it from
  outside the repository root. `import jaxfne; hasattr(jaxfne, "tfne")` is
  `False`; only `from jaxfne import tfne` works, and `artifacts/context.md`
  declares `jaxfne` the only supported public entry point. Decide the tier and
  register it in `jaxfne.public_surface`, or state that it is deliberately
  unexported. Note a bare `python script.py` outside the repository root
  resolves `jaxfne` from site-packages, which has no `tfne` module — so the
  test must run from outside the root to mean anything.
- **TFNE-PARAM-01** — rule parameters do not reach the executed model. Opened
  by TFNE-EXEC-01: `realize()` and `to_neuronal_tensor()` are two independent
  compilations of one source, and the tensor bridge builds `InterConnection`s
  without the rule's declared `weight`, so construction substitutes its own
  scaling and sign convention. Topology, cell-type split and mechanism
  identity all transfer; connection parameters do not. Decide whether the
  bridge should carry `weight`/`probability`/`delay`, or whether TFNE rule
  parameters are declared non-binding on execution — and say which in the
  doctrine. Divergence pinned by
  `test_declared_rule_weight_does_not_reach_the_executed_model`.

## Remaining measured conformance gaps

Each item conforms one clause. Standing of every clause is in the doctrine
page; none of the items below changes realized biology — the compiler rejects
what it cannot express rather than realizing a different nervous system.

- **TFNE2-03** — S20 typed natural ordering, declaration-independent
  (`L1<L2<L10`, `SEG.2<SEG.10`), with declared biological order overriding it.
  Changes realization indexing and `I` for existing specs.
- **TFNE2-04** — S9 frontiers beyond the derived ordered defaults that S10
  already uses: declared `in := [...]` / `out := [...]` bodies, the
  X-composite union default, `E_FRONTIER_UNRESOLVED`. Carries the rest of S8
  — only a declared frontier can make `{A O B} O C` differ in edge set from
  `A O B O C`.
- **TFNE2-05** — S12 `$L` / `$R` rule metavariables and rule bodies that
  specify topology, mechanism, parameters, geometry and delay independently.
  Also gives `X[k]` a way to select endpoints, which it currently lacks.
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
  Depends on TFNE2-04 for declared `in`/`out`.
