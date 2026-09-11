# 23-REP-02 — realized topology authority

Status: **CLOSED** after verification increment.

## Scope

Realized canonical topology is authoritative; dense `W` and `edge_list` are
backend-specific execution layouts, not co-equal permanent truths.

## Authority model (`model.static['representation']`)

| Path | `topology_authoritative` | `emitter_W_storage` | `dense_W_role` | `edge_list_role` |
| --- | --- | --- | --- | --- |
| Sparse / bounded-degree (placeholder W) | `edge_list` | `placeholder` | `execution_layout_on_demand` | `authoritative` |
| Dense below `_SPARSE_DIRECT_N` | `emitter_W` | `materialized` | `execution_layout_materialized` | `execution_layout_derived` |

Checkpoint JSON records `topology_authoritative` and `emitter_W_storage`.

## Refusal gates (no silent backend mismatch)

- **construct:** `_require_edge_list_backend` when rules/sparse-direct require `edge_list`.
- **simulate:** `_refuse_contradicted_dense_backend` when placeholder W + explicit `recurrent_backend='dense'`.

## Tests

- `tests/test_connectivity_scaling.py` — representation static (both paths), checkpoint authority, simulate refusal.

## Not in scope (separate items)

- Lazy omission of derived `edge_list` on dense path (future PARAM/memory work if U_k warrants).
- Sparse-direct vs dense equivalence — **23-REP-03**.
