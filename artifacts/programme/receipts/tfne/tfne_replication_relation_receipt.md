# S6.1 — replication with relation (`A^{nX}` / `A^{nO}`). CLOSED.

**Branch:** `dev`
**Scope:** TFNE parse/expand/emit only. No connectivity, parameter or kernel
semantics changed. Bare `A^n` behavior and every existing digest unchanged.

## Defect

S6 defined `A^n` (instances, no connectivity) and prefix application
`O[k](SEG^n)`, but gave no compressed form for instances developed under a
relation. `V1^{10X}` — ten instances joined by `X` — was inexpressible except
by writing all ten joins longhand.

## Delivered

Project source 7 carries S6.1 (amendment recorded in the source header):
`A^{n}` (instances only), `A^{nX}` (`A.1 X ... X A.n`), `A^{nO}`
(`A.1 O ... O A.n`). Compressed TFNE expands before realization.

Compiler (`jaxfne/tfne.py`):

- `Replicate` gains `rel: None | "X" | "O"`. Parse requires braces —
  `SEG^{10X}` — so `SEG^10 X Q` keeps its existing meaning (replication
  followed by cross composition). Anything else in braces (`{X}`, `{3Y}`,
  unclosed) is refused, not guessed.
- Expansion joins per-instance frontiers under bare-operator semantics: X
  exposes every instance on both frontiers (identical to `A^n`); O chains
  them, exposing head as `fin` and tail as `fout`, so a rule applied to the
  composite reaches exactly the chain ends.
- Emission keeps `A^n` for `rel=None`, so no existing normalization or digest
  moves; `A^{nX}`/`A^{nO}` are canonical and replay-stable.

## What this changes

Nothing for existing specs (verified: identical `neuron_paths` for `SEG^3`
vs `SEG^{3}`, identical frontiers for `SEG^3` vs `SEG^{3X}`). New behavior:

```
V := SEG^3   O[k] Q   ->  6 edges, pre {SEG.1, SEG.2, SEG.3}
V := SEG^{3X} O[k] Q  ->  6 edges, pre {SEG.1, SEG.2, SEG.3}
V := SEG^{3O} O[k] Q  ->  2 edges, pre {SEG.3}            (chain tail only)
V := P O[k] SEG^{3O}  ->  2 edges, post {SEG.1}           (chain head only)
```

## Tests

`tests/test_tfne_algebra.py`, section "9e":

- `test_replicate_with_X_matches_bare_replication_frontiers`
- `test_replicate_with_O_restricts_adjacency_to_chain_ends`
- `test_replicate_with_O_binds_incoming_to_chain_head`
- `test_replicate_relation_survives_normalization_replay`
- `test_bare_replication_followed_by_cross_is_not_a_relation`
- `test_malformed_replicate_relations_fail_closed` (3 parametrized)

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 79 passed

python scripts/run_test_gate.py dev
    -> 217 passed, 1 skipped, 2 deselected
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3973 passed, 75 skipped, 37 deselected, 4 xfailed
       All checks passed!; exit 0; zero failures
       (+22 against the 3951 S20 run: 8 replication + 14 completion)
```

## Status

- S6.1: **CLOSED**. `O[k](SEG^n)` prefix application stays open (TFNE2-06).
- Replication-with-relation is structural expansion (deterministic, no RNG);
  concrete contents remain JDNA's under `K_D` per the boundary doctrine.
