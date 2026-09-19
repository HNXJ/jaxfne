# TFNE2-04 — CLOSED. S9 declared composition frontiers.

**Branch:** `dev`
**Scope:** frontier declaration, validation, and rule binding only. S10
adjacency semantics unchanged; no connectivity, parameter, or kernel
semantics changed. JDNA untouched (preservation verified, nothing added).

## Defect

S9 gives every composable typed `in`/`out` interfaces, declarable via
`in := [...]` / `out := [...]`, with ordered-composite and X-composite
derived defaults. The compiler had only the derived defaults: no declared
frontier could make `{A O B} O C` differ in edge set from `A O B O C`, and
`E_FRONTIER_UNRESOLVED` was unreachable.

## Delivered

`in[A] := [m1, ..., mk]` / `out[A] := [m1, ..., mk]` as named properties —
the same declaration surface as `order[A]`, because frontiers are interface
metadata, not structure. A declaration names a subset of the scope's
immediate members (replica-aware remainders, shared with S20.1); a declared
side overrides the derived default for that side only.

Semantics, exactly:

- derived frontier != declared frontier != all members. Derived: ordered
  chain ends (`in({A O B}) = in(A)`, `out = out(B)`), X-union of members.
  Declared: the named subset, subtree-inclusive. Rules never bind
  accumulated members (unchanged).
- A declaration that cannot be honoured exactly is refused as
  `E_FRONTIER_UNRESOLVED` — unknown member, descendant rather than member,
  duplicate member, empty interface, duplicate declaration, unknown or
  ambiguous scope, leaf scope, replica-instance scope. One sealed code for
  every frontier defect; nothing invented.
- Bare scope names resolve only when unambiguous against the final model;
  final uniqueness is enforced at end of resolve(), so a bare name matching
  two nested scopes always fails closed even if it matched one mid-expansion.
- Execution invariant: for `A O[k] B`, rule `k` binds exactly
  `out(A) -> in(B)` after frontier resolution. No accumulated-member
  fallback (was already true; pinned by tests).
- Normalization invariant: `in[A]`/`out[A]` statements are canonical
  (`NF`), so frontier semantics drive the digest. Different interfaces are
  different models.
- JDNA invariant: completion reads leaves only (`neuron_paths`,
  `s["geometry"]`); it cannot reinterpret a declared frontier. Verified by
  test, not by review.

## What this changes

Only specs that declare frontiers. The load-bearing case:

```
M := A O[k] B;  V := P O[j] M
derived:          P -> A (in(A))     3 edges
in[M] := [B]:     P -> B             4 edges
```

The inner rule still binds `A -> B` by name in both: declarations own only
the interface they name; S10 is untouched. Nested declarations compose:
derived frontiers resolve through declared ones (`in[A] := [P]` narrows
`W := A O B` to `P`; adding `in[W] := [A]` reopens `W` to `A` whole).

## Tests

`tests/test_tfne_algebra.py`, section "9f" (13 functions + 7 parametrized):

- derived chain binds head/tail; braced/unbraced edge-set coincidence (S8);
- in-only, out-only, both-sides restriction with edge identities;
- braced-vs-flat divergence under declaration (S8's case);
- nested precedence (narrow + reopen) with subtree-inclusiveness;
- group/nested mixed `{A X B} O C`, `A O {B X C}` endpoint binding;
- replica members (`out[V] := [SEG.3]`), joined-replica agreement;
- 7 malformed cases refused; ambiguous bare scope refused;
- normalization replay, digest sensitivity, equivalent serialization.

`tests/test_jdna_completion.py`:

- `test_declared_frontier_reaches_jdna_completion_unchanged`: edges differ
  upstream (3 vs 4, `V.M.A` vs `V.M.B`), per-leaf positions and origins
  identical under the same `K_D`.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 100 passed

python -m pytest tests/test_jdna_completion.py -q
    -> 15 passed

python scripts/run_test_gate.py dev
    -> 238 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3995 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+22 against the 3973 JDNA run: 21 frontier + 1 JDNA preservation)
```

The receipt is necessarily edited after the gate it reports.

## Status

- TFNE2-04: **CLOSED**. CTX-01's `in`/`out` dependency is unblocked.
- Open and carried, not added: X[k]-rule frontier override (needs rule
  bodies, TFNE2-05); S6 prefix application (TFNE2-06); full S25 vocabulary
  (TFNE2-07).
- PARAM-03 / PARAM-04 untouched per scope.
