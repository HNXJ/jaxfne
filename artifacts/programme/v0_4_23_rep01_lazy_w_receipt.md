# 23-REP-01 increment — lazy dense W materialization

Status: **PASS** (second + third increments; 23-REP-01 remains in stack until remaining gates close).

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

## Third increment — class-shared edge execution layouts

When `tau_ms` matches the sign-only receptor map (2.0/5.0 ms) and `delay_steps`
are uniformly zero, construct compacts to:

- `tau_storage="sign_from_receptor"` with placeholder `tau_ms (0,)`
- `delay_storage="uniform_zero"` with placeholder `delay_steps (0,)`

Kernels resolve via `resolve_edge_tau_ms` / `resolve_edge_delay_steps` (lazy
materialization at execution, not dual persistent storage).

### W8 after class compaction (N=1000, K_max=100)

| Metric | Before class compact | After |
| --- | --- | --- |
| M_edge | 2,400,000 B | **1,600,000 B** |
| M_persistent | 2,444,004 B | **1,644,004 B** |
| M_W | 0 | 0 |
| nxn_arrays | `[]` | `[]` |

Cumulative vs pre-REP-01 baseline (~6.4 MB M_persistent, 4 MB M_W): **~74% persistent reduction**.

### p_connect path gates (tested, not optimized across)

| Path | Equivalence | Action |
| --- | --- | --- |
| `p_connect=0` + rules (bounded degree) | bit-exact | optimized |
| `0 < p_connect < 1` dense (N < `_SPARSE_DIRECT_N`) | self-consistent | no cross-path claim |
| sparse-direct vs dense at N=5000 | **not bit-exact** | `_SPARSE_DIRECT_N` stays 5000 |

## Not done (explicit)

- `receptor_index` class index (uint8) — inspect consumer needs per-edge indices; separate gate.
- Mechanism-table tau compaction when tau ≠ sign-only map.
- `0 < p_connect < 1` sparse-direct path — blocked until bit-exact vs dense is proven.
