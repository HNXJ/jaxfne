# TFNE2-03 — CLOSED. S20 canonical ordering, natural default and explicit override.

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

## The override, and the language amendment it required

S20's second sentence — a definition that "explicitly declares meaningful
order" overrides natural ordering — was unimplementable as written. The sealed
text said it once, at line 807 of project source 7, gave no syntax, and its own
opening sentence ruled out reading an enumerated body as the declaration.

Rather than invent a syntax, this was surfaced as a language gap. Hamm then
ruled: dedicated syntax, and ordinary enumeration or structural listing never
implies order — otherwise source-order dependence returns through another door,
which is exactly what S20 exists to prevent. Project source 7 now carries S20.1
defining `order[A] := [m1, ..., mk]`, with the amendment recorded in the
source's status header since the language is sealed.

`order` is a named property rather than a new operator, because ordering is
metadata and the algebraic vocabulary is scarce.

Implementation: `Program.orders` carries the declarations, `_declared_ranks`
validates them into `(parent, remainder)` ranks following the resolved parent
links, and `_ordering_key` ranks each edge of the parent chain by declared
rank where one exists and by natural key otherwise. Both branches are
3-tuples whose leading tag differs, so a natural key is never compared
against a declared rank. A declaration therefore applies only among the
siblings it names, and nested scopes order independently. Replicated
instances such as `SEG.1` are immediate members of their scope, so
`order[V] := [SEG.3, SEG.2, SEG.1]` reorders them; the remainder keeps its
dots in both validation and the ordering key.

Validation refuses anything it cannot honour exactly, because ordering a
partial declaration would index some members by declaration and the rest by
another rule:

```
E_ORDER_INCOMPLETE         omits an immediate member
E_ORDER_DUPLICATE_MEMBER   names one twice
E_ORDER_MEMBER_UNKNOWN     names a non-member
E_ORDER_NOT_IMMEDIATE      names a descendant rather than a member
E_ORDER_SCOPE_UNKNOWN      names no object
E_ORDER_SCOPE_AMBIGUOUS    a bare name matching two nested objects
E_ORDER_DUPLICATE          two declarations for one scope
E_ORDER_EMPTY              parse-time: `order[A] := []` declares no members
```

Root/top-level sibling order has no explicit override syntax; natural
ordering applies there. S20 only requires ordering within named scopes, so
this does not block closure. No `order[]` root form is claimed or supported.

Cell-type enumeration order is unchanged, retained as the temporary
compatibility exception S20.1 names.

## Tests

Added to `tests/test_tfne_algebra.py`, section "9c. S20 canonical ordering":

- `test_declaration_order_does_not_determine_realization_indexing` — the two specs above must produce identical `neuron_paths`. This is the clause itself.
- `test_typed_natural_order_beats_lexical_order` — `L1`, `L2`, `L10` occupy slices in that order.
- `test_replication_indices_order_numerically` — `SEG^12` yields `SEG.1 … SEG.12` in numeric order; lexical order would have put `SEG.10` immediately after `SEG.1`.
- `test_natural_order_is_hierarchical_parent_before_child` — the key function directly: parent before child, `V.L2 < V.L10`, `A.9 < A.10`, `V1.L4 < V2.L1`.

Section "9d. S20 explicit order declarations":

- `test_explicit_order_overrides_natural_order` — the clause itself.
- `test_same_order_declaration_survives_reserialization` — two programs differing in statement order and composite body, same `order[V]`, identical indexing. Serialization is not semantics.
- `test_nested_objects_order_independently` — `order[W.A]` reorders A while B keeps the natural default.
- `test_malformed_order_declarations_fail_closed` — five parametrized cases against their typed codes.
- `test_ambiguous_order_scope_is_refused`, `test_duplicate_order_declaration_for_one_scope_is_refused`.
- `test_order_declaration_survives_normalization_replay` — the canonical form carries `order[V]` and re-parses to itself; without this, two orders share one digest and provenance collides.
- `test_different_orders_give_different_digests` — `spec_hash` distinguishes the orders.
- `test_replica_instances_order_independently` — `order[V] := [SEG.3, SEG.2, SEG.1]` reorders `SEG^3`; replicas are immediate members despite the dot in their remainder.

Added to `tests/test_tfne_parameter_transfer.py`:

- `test_order_override_keeps_realization_and_execution_aligned` — the constructed neuron table still matches `I["neuron_paths"]` under an override. Without this, an override could move the realization axis while execution stayed put, wiring the wrong neurons with every count matching.
- `test_order_override_preserves_edge_identity` — with `order[V] := [B, A]`, every edge still runs from A's id range into B's. S20 reorders indexing; S10 binds by name, so reordering relabels edges rather than redirecting them.
- `test_order_override_does_not_change_edge_count_or_weights`.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 71 passed in ~16s

python scripts/run_test_gate.py dev
    -> 209 passed, 1 skipped, 2 deselected in 119.93s
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3951 passed, 75 skipped, 37 deselected, 4 xfailed in 1921.02s
       All checks passed!; exit 0; zero failures
       (+16 against the 3935 pre-override run: 13 override tests + 3
       normalization/replica tests)
```

The receipt is necessarily edited after the gate it reports.

## Status

- TFNE2-03: **CLOSED** — typed natural ordering and the `order[A]` override,
  with the S20.1 amendment in project source 7. The doctrine's S20 gap row is
  removed; the remaining exception is cell-type enumeration order, retained as
  the temporary compatibility behaviour S20.1 names.
- Next: remaining conformance gaps TFNE2-04 through TFNE2-07, then `CTX-01`.
  TFNE-PARAM-03 remains blocked behind TFNE2-07; TFNE-PARAM-04 remains open and
  independent.
