# TFNE2-05 — CLOSED. S12 rule bodies (`$L` / `$R`, scope addressing).

**Branch:** `dev`
**Scope:** rule-body grammar, `$L`/`$R` binding, per-statement mechanism.
S10 adjacency untouched; flat (params-only) rules keep exact legacy
behavior. No kernel semantics changed. JDNA untouched.

## Defect

S12 defines rules as reusable architectural-to-projection maps with
rule-bound `$L`/`$R` metavariables and bodies specifying topology,
mechanism, parameters, geometry, and delay independently. The compiler had
flat parameter maps only: `$` was a lexer error, every projection of a
rule shared one member-to-member mapping, and per-projection mechanism
was inexpressible.

## Delivered

Rule bodies as projection statements with the architectural invariant:

```
architectural relation + rule body -> explicit typed projections
```

without allowing the body to redefine the operands.

- Grammar: `O[cmc1] := [{L2,L3}>L3; L5<{L2,L5}]`. Endpoints are
  `$L`/`$R` (bare: whole operand; `.out`/`.in`: resolved interface),
  member refs, or `{a, b}` collections. Direction lives in each
  statement (bodies mix `>` and `<`); `[mech=NAME]` overrides the
  rule-level mechanism per statement. A rule-level `direction` alongside
  a body is refused rather than silently ignored.
- Binding: `$L`/`$R` bind the syntactic operand expansions; plain names
  resolve to immediate members (replica-aware) within their own side.
  Exact paths are accepted only at/under a member scope. Anything
  absent or outside is `E_ADDRESS_UNKNOWN`: expansion operates on
  resolved interfaces and never reconnects arbitrary descendants.
- `$L.out`/`$R.in` select declared-or-derived interfaces, so the TFNE2-04
  invariant carries forward: with `out[V1] := [L3]`, `$L.out` binds L3.
- Lexer gains one META token (`$` + name); a lone `$` stays a hard error.
- Bodies are canonical (`NF`) and digest-sensitive; flat-rule emission
  byte-identical to before, so no existing digest moves.
- Per-statement mechanism travels on the relation (not the rule) through
  `realize()`, origins, and `to_neuronal_tensor()`.

## What this changes

Only rules with bodies (new syntax; nothing existing parses as one).
Flat rules expand exactly as before (asserted by equivalence test).

```
O[cmc1] := [{L2,L3}>L3; L5<{L2,L5}]   on V1 O[cmc1] V2
  -> V.V1.{L2,L3} > V.V2.L3  (2x1) + V.V2.{L2,L5} > V.V1.L5  (2x1)
X[v3x1] := [{L2,L3}<>{L2,L3}]          on V1 X[v3x1] V2
  -> both halves under one group stem (2x2 each way)
```

## Tests

`tests/test_tfne_algebra.py`, section "9g" (10 functions):

- per-side scope addressing with edge identities (cmc1 shape);
- `<>` split halves sharing one group stem;
- `$L.out > $R.in` identical to the flat canonical rule;
- bare `$L`/`$R` bind whole operands;
- `$L.out` respects declared frontiers (no bypass);
- rule reuse across scales; absent member refused;
- exact paths outside operands refused;
- statement mechanism override through to `InterConnection`;
- rule-level direction + body refused;
- replay, digest sensitivity, equivalent serialization.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 110 passed

python -m pytest tests/test_tfne_algebra.py tests/test_jdna_completion.py -q
    -> 101 passed

python scripts/run_test_gate.py dev
    -> 248 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 4005 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+10 against the 3995 TFNE2-04 run: the §9g rule-body battery)
```

The receipt is necessarily edited after the gate it reports.

## Status

- TFNE2-05: **CLOSED**. `$L`/`$R`, collections, bidirectionality, reuse,
  nesting, frontier interaction, and absent-reference refusal all hold.
- Open and carried, not added: `X[k]`-rule frontier override (bodies now
  exist, but no syntax declares a composite interface from a rule —
  a language-design question, not a compiler gap); geometry/delay
  per-statement specification (rule-level params only); full S25
  vocabulary (TFNE2-07); S6 prefix application (TFNE2-06).
- PARAM-03 / PARAM-04 untouched per scope.
