# TFNE2-03 — S20 canonical ordering. Typed natural ordering implemented.

**Branch:** `dev`
**Entry state:** `3aaad16`
**Scope:** realization indexing only. No connectivity, parameter or kernel
semantics changed.

## Defect

`tfne/2` S20 opens: "Source declaration order does not determine realization
indexing." It did. The same nervous system, with the same layers and the same
counts, realized to different neuron indices depending on the order the layers
appeared in the composite body:

```
V := L1 O L2 O L10   ->  ['V.L1','V.L2','V.L2','V.L10','V.L10','V.L10']
                         L1=(0,1)  L2=(1,3)  L10=(3,6)

V := L10 O L2 O L1   ->  ['V.L10','V.L10','V.L10','V.L2','V.L2','V.L1']
                         L1=(5,6)  L2=(3,5)  L10=(0,3)
```

Every path-to-slice answer, every `edge_pre`/`edge_post` id and the whole
neuron axis depended on how the source happened to be written.

## Delivered

`_natural_component_key` splits a component into digit and text runs, comparing
digit runs as integers and text runs as text, with a type tag on each run so
mixed tuples stay comparable. `_natural_path_key` applies it per path
component. So `L1 < L2 < L10` rather than the lexical `L1 < L10 < L2`, and
`SEG.2 < SEG.10`. A parent sorts before its own children because its key is a
proper prefix of theirs.

The sort is applied at exactly one place — where `ExplicitModel.order` is
built. That is deliberate. `explicit.order` feeds both the realization neuron
axis and `to_neuronal_tensor`; ordering them separately would let them drift,
and the specs address neurons by realized id, so a drift would wire the wrong
neurons while every count still matched. One chokepoint makes that
unrepresentable rather than merely unlikely.

Cell types keep the order their `C = {...}` enumeration gives them. Only object
paths are reordered.

## What this changes

Realization indexing for existing specs, as the item anticipated. The visible
case is the execution fixture: `V1 := L4 O[ff] L2` used to lay out L4 at
`(0,10)` and L2 at `(10,15)`; under S20 it is L2 at `(0,5)` and L4 at
`(5,15)`. The projection still runs L4 -> L2 — S10 adjacency binds by name and
is untouched — and the new layout is also the biologically conventional one.
`test_index_map_addresses_the_simulated_neuron_axis` was updated to the S20
layout with the reason recorded in its docstring.

`test_executed_neuron_order_matches_the_realization` passed unchanged, which is
the evidence that realization and execution did not drift apart.

## Not delivered, and why

S20's second sentence — a biological definition that "explicitly declares
meaningful order" overrides generic natural ordering — is **not** implemented.
The sealed language mentions it once, at line 807 of project source 7, and
specifies no syntax for declaring such an order. The obvious reading, that an
enumerated definition body *is* the declaration, is ruled out by S20's own
opening sentence.

So the override is currently unexpressible and nothing can override natural
ordering. This is a gap in the language rather than the compiler, and is
recorded as such in the doctrine's conformance table rather than being resolved
by inventing a syntax.

## Tests

Added to `tests/test_tfne_algebra.py`, section "9c. S20 canonical ordering":

- `test_declaration_order_does_not_determine_realization_indexing` — the two specs above must produce identical `neuron_paths`. This is the clause itself.
- `test_typed_natural_order_beats_lexical_order` — `L1`, `L2`, `L10` occupy slices in that order.
- `test_replication_indices_order_numerically` — `SEG^12` yields `SEG.1 … SEG.12` in numeric order; lexical order would have put `SEG.10` immediately after `SEG.1`.
- `test_natural_order_is_hierarchical_parent_before_child` — the key function directly: parent before child, `V.L2 < V.L10`, `A.9 < A.10`, `V1.L4 < V2.L1`.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 51 passed in 13.07s

python scripts/run_test_gate.py dev
    -> 193 passed, 1 skipped, 2 deselected in 101.97s
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3935 passed, 75 skipped, 37 deselected, 4 xfailed in 1540.14s
       All checks passed!; exit 0; zero failures
       (+4 against the previous run, exactly the four S20 tests added)
```

## Status

- TFNE2-03: **typed natural ordering CLOSED**; the declared-order override stays open as a language gap, listed in the doctrine's "Not yet conformant" table.
- Next: remaining conformance gaps TFNE2-04 through TFNE2-07, then `CTX-01`.
  TFNE-PARAM-03 remains blocked behind TFNE2-07; TFNE-PARAM-04 remains open and
  independent.
