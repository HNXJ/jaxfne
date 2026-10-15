# 23-REP-03 receipt — CLOSED (gate retained)

**Branch:** `dev`  
**Scope:** sparsity routing (`p_connect` honour/refuse) and dense vs sparse-direct equivalence at `_SPARSE_DIRECT_N`.

## Classification

$$
\boxed{\textbf{NOT\_BIT\_EXACT — threshold retained at } N=5000}
$$

Optimization to force equivalence is **not mandatory**. Lowering `_SPARSE_DIRECT_N` without new evidence remains blocked.

## Sparsity routing

| Requirement | Status | Evidence |
| --- | --- | --- |
| `p_connect` realized on sparse-aware path | **PASS** | `test_p_connect_fraction_is_realized_not_ignored` |
| `p_connect=0` yields zero within-area edges | **PASS** | `test_p_connect_zero_yields_no_within_area_edges` |
| `.network()` route refuses inert `p_connect` | **PASS** | `test_network_route_refuses_a_p_connect_it_cannot_honour` (both spellings) |
| Sparse-direct skips dense `W` at scale | **PASS** | `test_sparse_direct_skips_dense_W_at_scale` |
| Below threshold keeps dense `W` | **PASS** | `test_below_threshold_keeps_dense_W` |

## Equivalence gate (N=5000, p=0.02, seed=7 construct / seed=0 simulate)

| Metric | Dense backend | Sparse-direct (`edge_list`) |
| --- | --- | --- |
| `emitter.W` shape | `(5000, 5000)` | `(0, 0)` placeholder |
| Realized edges | 499,074 | 501,197 |
| Total spikes (10 ms) | 5,045 | 5,292 |
| Δ spikes | — | **+247** (not bit-exact) |

Backends differ in **topology realization** (dense `W` materialization vs authoritative `edge_list`) and **synaptic kernel** (instantaneous dense coupling vs per-edge exponential filter on `edge_list`). The gate test documents this divergence; it is not an ε-equivalence claim.

## Tests

- `tests/test_connectivity_scaling.py::test_sparse_direct_not_bit_exact_to_dense_at_threshold`
- `tests/test_connectivity_request_realization.py` (W13 routing contract)
- `tests/test_rep01_edge_class_storage.py` (threshold gate comment)

## Decision

Retain `_SPARSE_DIRECT_N = 5000`. Any future threshold change requires a separate bit-exact equivalence programme item with measured per-observable parity.

## Spawned

None — delayed registrable HDP remains under **23-DELAY-01**.
