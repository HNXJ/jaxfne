# TFNE2-06 — CLOSED. S6 prefix application, S14/S25 statement atomicity.

**Branch:** `dev`
**Scope:** `O[k](...)` prefix parsing/expansion, `;`-statements in braces
with atomic projection failure, `(`/`)` lexing. S10 adjacency, rule
bodies, frontiers, and all downstream compilation unchanged. JDNA
untouched.

## Defect

Two sealed forms were unexpressible. S6 defines `O[k](SEG^8)` as seven
ordered adjacencies, but `(` was not even lexed (the `_parse_atom` paren
branch was dead code) — prefix application was a lexer error. S14 states
that statements are atomic with invalid resolved projections contributing
nothing, but `;` inside `{...}` was a parse error, so composites held
exactly one expression and any invalid projection aborted the program.

## Delivered

Atomicity is contextual, never "has no children":

```
internal structure != exposed structure != atomicity
```

- Prefix: `O[k](SEG^n)` chains rule `k` over the instances
  (`SEG.1 O[k] SEG.2 ...`), exposing head as `fin` and tail as `fout`,
  so the chain composes as one operand. Degenerate `n=1` binds nothing.
  Each pair expands through the ordinary rule path (flat or body rules
  apply per pair with the instances as operands). Only `O[k]` over plain
  `A^n` is defined: `X[k](...)` is refused (X-join is `A^{nX}`), as are
  non-replication targets. Infix `A O[k] (group)` keeps its existing
  meaning — only a leading `O[k](` is the prefix form.
- Atomic braces: `{s1; s2}` expands each statement independently under an
  atomicity flag. A statement whose endpoint resolution yields nothing
  contributes no nodes, relations, or exclusions; the group still hands
  itself upward as one member (S8) with the surviving statements'
  frontiers combined by derived defaults. All other errors propagate:
  top-level (non-brace) invalid projections still abort, contradictory
  selections against existing objects still raise (absence is lenient,
  contradiction is not), and a dropped exclusion records nothing (so it
  cannot trip `E_EXCLUSION_UNKNOWN`). SEP is the separator, so newlines
  behave exactly like `;`, as at program level.
- `(`/`)` added to the lexer symbol table — purely additive, since both
  were hard errors before.

## What this changes

Only the newly expressible forms. Single-statement groups, bare chains,
and all rule paths behave exactly as before (asserted by the unchanged
suite). New behavior:

```
V := O[k](SEG^3)              -> 2 adjacencies, r0/r1 with instance labels
V := P O[j] O[k](SEG^3) O[j] Q -> P binds the head, the tail binds Q (4 edges)
V := {A O[k] B; Z.Q > B}      -> 2 edges (A>B); Z.Q leaves no trace
```

A definition expands at most once per scope (pre-existing rule), so two
brace statements may not reference the same definition — same as chains.

## Tests

`tests/test_tfne_algebra.py`, section "9h" (12 functions + 1 parametrized
count as listed):

- prefix adjacency identities; head/tail composition through a chain;
  degenerate singleton; non-replication and `X[k]` refusals; body-rule
  per-pair application with `[mech=]` override;
- atomic drop (no nodes/edges/relations) with siblings intact; group
  frontier from surviving statements; dropped exclusion without
  `E_EXCLUSION_UNKNOWN`; top-level invalid projection still raises;
  contradictory selection still raises;
- replay, digest sensitivity, `;`/newline equivalence.

`tests/test_jdna_completion.py`:

- `test_atomically_dropped_statement_leaves_no_jdna_trace`: completion
  identical with and without the dropped statement.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py \
                 tests/test_jdna_completion.py -q
    -> 137 passed

python scripts/run_test_gate.py dev
    -> 260 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 4018 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+13 against the 4005 TFNE2-05 run: 12 atomicity/prefix + 1 JDNA)
```

The receipt is necessarily edited after the gate it reports. One broad
run also caught a lint defect in the new code (ruff F541 placeholder-free
f-strings); fixed, re-verified (`ruff check jaxfne/` clean), then broad
re-ran green on the final bytes.

## Status

- TFNE2-06: **CLOSED**. The `X[k]` frontier-override question stays
  unresolved per instruction (not determined by this item).
- Next in stack order: TFNE2-07 (failure vocabulary — unblocks PARAM-03).
- PARAM-03 / PARAM-04 untouched per scope.
