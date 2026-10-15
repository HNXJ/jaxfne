# 23-PARAM-01 — class-shared edge weight storage

Status: **CLOSED** (first increment: derivable signed weight magnitudes).

## Scope

Compact `edge_list.weight` only when exactly derivable from declared
connection-rule magnitudes and presynaptic intrinsic sign (or legacy
exc/inh `receptor_index` for sign-only compiler paths). Per-edge fallback
when weights are inhomogeneous (e.g. canonical biophysics strengthening).

## Storage modes

| `weight_storage` | Derivation |
| --- | --- |
| `per_edge` | fallback — full `(E,)` array |
| `magnitude_times_presynaptic_sign` | `table[receptor_index] * sign[pre]` or scalar magnitude `* sign[pre]` |
| `sign_from_receptor` | `+mag` if `receptor_index==0` else `-mag` (legacy sign-only path) |

Kernels materialize via `resolve_edge_weight(..., presynaptic_sign=emitter.sign)`.

## W8 bounded_degree (N=1000, K_max=100)

| Metric | Pre-PARAM-01 (~REP-01 closed) | After |
| --- | --- | --- |
| M_persistent | ~1.34 MB | **0.944 MB** |
| edge_list.weight | 400 KB | **0 B** (compact) |

Receipt: `artifacts/audit/w10_param01_baseline.json`

## Tests

- `tests/test_param01_edge_weight_class_storage.py`
- Updated `tests/test_rep01_edge_class_storage.py`, `tests/test_connectivity_scaling.py`

## Not in scope

- Neuron-level emitter parameter class sharing (separate PARAM work if warranted).
- Weight compaction when canonical biophysics mutates PV↔E edge strengths inhomogeneously.
