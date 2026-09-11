# 23-COMPAT-JOM-01 receipt — CLOSED

**Branch:** `dev`  
**Scope:** JaxFNE 0.4.22 compatibility regressions reported from Jomission downstream evidence.

## Classification

| Report | Classification | Evidence |
| --- | --- | --- |
| `np.asarray(resolve_edge_delay_steps(...))` tracer failure under `jax.jit` + `simulate` | **JAXFNE_REGRESSION** | Reproduced `TracerArrayConversionError` on compact `uniform_zero` models; fixed by metadata-aware delay routing and eager-only host materialization |
| `tau_ms` shape `(0,)` validation / round-trip failure | **JAXFNE_REGRESSION** | `resolve_edge_tau_ms` already correct; `EdgeList.to_dict`/`from_dict` dropped compaction metadata and rejected `(0,)` placeholders — fixed |

No item classified `DOWNSTREAM_INVALID_USAGE` or `NOT_REPRODUCED`.

## Repairs (smallest generic)

- `jaxfne/emitters.py`: `_edge_delay_steps_numpy`, `_edge_delays_any_positive`, `_edge_delays_all_zero`, `_edge_max_delay_steps`, `_validate_edge_delays_nonnegative_eager`; kernel dispatch uses storage metadata instead of `np.asarray` on traced JAX arrays.
- `jaxfne/emitters.py`: `EdgeList.to_dict`/`from_dict` preserve `tau_storage`, `delay_storage`, `weight_storage`, mechanism/weight tables.
- `jaxfne/_model_simulate.py`: delay preflight uses `_edge_delay_steps_numpy` (skip under trace).

## Tests

`tests/test_compat_jom01_regressions.py` (4 tests): compact tau resolve, serialization round-trip, JIT simulate on compact zero-delay model, documented tracer hazard.

## Provenance

- Independent reproduction in-tree (no Jomission repo required for classification).
- Jomission migration suite re-run: **not executed** (downstream repo not present in workspace); repairs are generic engine fixes aligned with 0.4.22 intended semantics.
