# 23-REP-01 increment — lazy dense W materialization

Status: **PASS** (second increment; 23-REP-01 remains in stack until remaining gates close).

Anchor: `dev` after this increment.

## Scope

- Lazy dense `emitter.W` for checkpoint/inspect/tune/optim consumers without restoring permanent dual storage.
- Checkpoint metadata records `topology_authoritative` and `emitter_W_storage`.
- `p_connect=0` bounded-degree path unchanged (first increment @ `bfa48ee`).

## Evidence

### W8 bounded_degree_dual_storage (N=1000, K_max=100)

| Metric | Value |
| --- | --- |
| M_W | 0 B |
| M_persistent | 2,444,004 B (~2.44 MB) |
| M_construction_peak | 188,854,272 B |
| M_recording | 549,344 B |
| M_peak_run | 130,359,296 B |
| nxn_arrays | `[]` |
| backend_realized | edge_list |

### Bit-exact observables (p_connect=0, N=1000)

| Observable | construct | checkpoint round-trip |
| --- | --- | --- |
| spikes sum | 996.0 | 996.0 |
| V_m mean | -65.19081 | -65.19081 |

### Tests

- `tests/test_connectivity_scaling.py` — placeholder lazy materialization, checkpoint authority, synaptic_gain via edge_list, REP-02 representation static, REP-03 non-equivalence gate at `_SPARSE_DIRECT_N`.

## Not done (explicit)

- `0 < p_connect < 1` sparse-direct vs dense: **not bit-exact** at N=5000 (documented test); `_SPARSE_DIRECT_N` not lowered.
- Class-sharing (`receptor_index`, `tau_ms`, `delay_steps`) — later stack items.
