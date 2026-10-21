# TFNE2-08 — CLOSED. S10 ordered adjacency.

**Branch:** `dev`
**Entry state:** `24f2f11` (TFNE2-01, TFNE2-02 closed; S10 measured and queued)
**Scope:** S10 only. The other nine measured conformance items were left alone.

## Defect

`tfne/2` S10: a mixed-rule ordered chain `A O[k] B O[j] C` normalizes to the
adjacency-labelled sequence `(A,k,B,j,C)`, and each rule applies only to its own
adjacency.

The compiler generated `A>C` as well. This was undeclared topology, not a
missing feature: a specification asking for a chain realized a different neural
system. For `L1 O[ff] L2 O[ff] ... O[ff] L6` the compiler produced 15 forward
interlaminar relations where the source declares 5.

## Cause

`_Resolver.expand` returned a flat list of member paths, and `Ordered`/`Cross`
both did:

```python
left = self.expand(node.left, scope)
right = self.expand(node.right, scope)
if node.rule is not None:
    self._record_rule(node, left, right)
return left + right          # <- accumulates
```

A chain parses left-associatively, so for `Ordered(Ordered(A,k,B), j, C)` the
inner node returned `['A','B']` and became the left operand of `j`. The rule
bound `('A','B') -> ('C',)`. Every earlier operand in a chain stayed reachable
as a rule endpoint, because the return value carried structural membership and
nothing else. A composite had the same problem from the other direction: a
group returned `[group_path]`, and `scope_ids` expanded that to every neuron
underneath, so `{A O B} O C` reached C from both A and B.

## Fix

Expansion now returns members and composition frontiers separately:

```python
@dataclass(frozen=True)
class _Expansion:
    members: tuple[str, ...]   # what the expression contributes structurally
    fin: tuple[str, ...]       # in-frontier
    fout: tuple[str, ...]      # out-frontier
```

- An object is its own frontier on both sides.
- `Ordered` binds `left.fout -> right.fin` and propagates `fin=left.fin`, `fout=right.fout`. Accumulated operands are no longer reachable as endpoints, which is what makes adjacency exact.
- A brace returns `members=(group_path,)` with the body's frontiers, so it composes through `in(A)`/`out(B)` (S9 derived defaults) instead of being flattened. A named transparent definition is treated the same way.
- `Replicate` returns the whole instance set as both frontiers — instances carry no order and no connectivity, so replication behaviour is unchanged.
- `Cross` still binds whole operands and is otherwise untouched. `X` relates operands rather than an adjacency, and per-rule endpoint selection is `$L`/`$R` (TFNE2-05).

Only the derived ordered `in`/`out` defaults that S10 requires were implemented.
Declared `in := [...]` / `out := [...]` bodies, the X-composite union default and
`E_FRONTIER_UNRESOLVED` remain TFNE2-04.

## Acceptance

Configured relation count, realized projection identities and realized edge
count are reported together, so a correct neuron total cannot hide a topology
error.

| Case | Configured relations | Realized identities | Edges |
| --- | --- | --- | --- |
| `A O[ff] B O[ff] C` | 2 | `(A,B) (B,C)` | 8 |
| `A O[ff] B O[fb] C` | 2, rules `['ff','fb']` | `(A,B) (B,C)` | 8 |
| `L1 O[ff] ... O[ff] L6` | 5 | `(L1,L2) (L2,L3) (L3,L4) (L4,L5) (L5,L6)` | 20 |
| `{A O[ff] B} O[ff] C` | 2 | `(g0.A,g0.B) (g0.B,C)` | 8 |
| `E^4` | 0 | none | 0, 8 neurons, paths `E.1..E.4` |

No `A>C` identity appears in any case. The six-layer chain is 5 adjacencies,
not 15. The braced case binds `pre_scopes=('g0.B',)`, the composite's out
frontier, rather than `('g0',)`.

## Tests

Added to `tests/test_tfne_algebra.py`:

- `test_ordered_chain_binds_only_its_own_adjacency`
- `test_mixed_rule_chain_keeps_each_rule_on_its_adjacency` (asserts the bound scopes, not only the rule names)
- `test_six_layer_chain_is_five_adjacencies_not_fifteen`
- `test_braced_composite_composes_through_its_out_frontier`

Replication coverage was already present and is unchanged. No existing test
needed rewriting: all 26 passed before the new tests were added, so the fix
broke nothing that the corpus had pinned.

```
python -m pytest tests/test_tfne_algebra.py -q   ->  30 passed in 2.05s
python scripts/run_test_gate.py dev             ->  138 passed, 1 skipped, 2 deselected in 138.51s
                                                    public docs language audit: pass
                                                    vocabulary check: pass
```

Earlier clauses re-probed, still conformant:

```
[CONFORMS] S6/S7 replication addressing   ['E.1','E.2','E.3','E.4']
[CONFORMS] S14 exclusion subtracts        n_edges=0
[CONFORMS] S14 unmatched exclusion        E_EXCLUSION_UNKNOWN
[CONFORMS] S10 ordered adjacency          [('A','B'), ('B','C')], 8 edges
```

## Status

- TFNE2-08: **CLOSED**
- The three clauses that changed realized biology (S14, S6/S7, S10) are now all closed. Every remaining measured gap is a missing feature: the compiler rejects what it cannot express rather than realizing a different nervous system.
- Next by authorization: `TFNE-EXEC-01`, then `TFNE-IMPORT-01`, then the remaining conformance gaps. The conformance list is not to be resumed ahead of execution qualification.
