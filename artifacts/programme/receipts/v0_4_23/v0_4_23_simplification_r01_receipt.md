# Simplification review R-01 — independent reproduction

**Branch:** `dev`  
**Input:** Gemini second-pass performance/simplification review (candidate evidence only)  
**Method:** code inspection + targeted probes; no wholesale plan adoption

## Score alignment

Gemini pass rated **86/100** by Hamm — agreed. Structural candidates reproduced; quantitative superiority claims largely **not** established.

---

## Priority 1 — code deletion / simplification

### A. Delay helpers (L-01) — **REPRODUCE + SIMPLIFY (scoped)**

| Metric | Value |
| --- | --- |
| Core cluster LOC | ~115 (`emitters.py` 1025–1115) |
| Duplicated metadata-branch LOC | ~35–45 |
| Mergeable | internal `_delay_storage_snapshot` factorization |
| Must not merge | `resolve_edge_delay_steps` (traced) ↔ host bridges; asymmetric JIT fallbacks in `_edge_delays_any_positive` / `_edge_delays_all_zero` |

**Status:** not implemented this increment (audit only). Gate: `test_compat_jom01_regressions.py`, delay continuation suite.

### B. JIT dispatch blocks (L-02) — **REPRODUCE + SIMPLIFY (deferred)**

| Metric | Value |
| --- | --- |
| Blocks | 4 in `_model_simulate.py` (276–309, 425–456, 494–536, 537–579) + `simulate_batch` variant |
| Duplicated skeleton LOC | ~134 JIT + ~18 eager |
| Estimated savings | ~75–110 LOC via `_dispatch_jit_cached(...)` |

**Status:** deferred to stack — consolidation is mechanical but touches cache keys / guard names.

### C. Izhikevich derivatives (L-03) — **IMPLEMENTED**

| | Before | After |
| --- | --- | --- |
| Definitions | 3 copies (~22 LOC) | 1 primitive + 1-line alias |
| Semantics | `du = a*(b*v-u)` vs `a*(h_k*b*v-u)` | unified `h_k=1.0` default |
| Tests | D1 static expression, HDP qualification | **19/19 pass** |

**Rejected claim:** “XLA guarantees identical machine code” — not asserted.

### D. `_RuntimeReportAdapter` (L-04) — **IMPLEMENTED (deleted)**

| | Before | After |
| --- | --- | --- |
| Callers | 1 (`_model_manifest.py`) | dict passed to `io.manifest` |
| LOC | 7-class wrapper | 0 |
| Tests | `test_manifest_json_safe` | **pass** |

### E. `FieldOutput` PyTree (L-06) — **VERIFY; defer simplify**

Manual `register_pytree_node` in `fields/proxy.py` (~25 LOC). JAX `>=0.4.25` supports `@register_pytree_node_class`, but `diagnostics: dict` is a non-array child — automatic dataclass registration may change aux handling.

**Status:** keep manual registration until tree-structure diff is demonstrated unchanged.

---

## Priority 2 — performance probes (measured)

Machine-readable: `artifacts/audit/simplification_r01_probe_results.json`  
Script: `scripts/perf/simplification_probe_r01.py`

### F. Compact tau decay — **REJECT general speed claim**

Random per-edge `tau_ms` (E=100k → 99,362 unique classes):

| E | max abs diff | t(per-edge) | t(class-gather) | ratio |
| --- | --- | --- | --- | --- |
| 1k | 0 | 0.26 ms | 0.82 ms | 0.31× (gather **slower**) |
| 10k | 0 | 0.19 ms | 0.92 ms | 0.21× |
| 100k | 0 | 0.30 ms | 1.56 ms | 0.19× |

Numerical equivalence holds; **compact path is slower** when few edges share tau classes. Worth revisiting only for `sign_from_receptor` / `from_mechanism_table` storage modes with small class tables.

### G. Scan recording — **INCONCLUSIVE (needs JAX-side probe)**

`record_edge_current|current_trace|u_trace` switches scan output arity (minimal 3 vs full 6 arrays per step). Python `tracemalloc` peak during a single eager call was **not** a reliable allocator proxy (JIT compile order dominated).

**Queued:** static specialization audit — confirm which `T×N` / `T×E` stacks are built when flags false.

### H. `run_trials` / vmap — **REJECT 5–10× claim**

Suite2, n=8, 8 trials, 20 ms (warm Python path):

| Path | median time |
| --- | --- |
| `run_trials` | 1.09 s |
| Python simulate loop | 1.00 s |
| `simulate_batch` (vmap) | 1.01 s |

Ratio `run_trials / simulate_batch` ≈ **1.08** — no material speedup at this scale. APIs differ; trajectory equivalence not part of this probe.

---

## Priority 3 — HDP kernel unification — **INSPECT ONLY; defer**

| File | LOC | Role |
| --- | --- | --- |
| `_hdp_registrable_kernel.py` | ~203 | zero-delay registered rule path |
| `simulate_edge_recurrent_izhikevich_hdp` | ~900+ | builtin rules, population, delays, boundary stabilization |

Shared structure: Izhikevich step + `syn_state` decay + `segment_sum` recurrent current (~15 lines). Semantic differences: population `bind_theta_to_plant`, boundary stabilization `r_bar`, delay ring buffer, closed-list `dw` rules vs `HDPRuleContext.step_fn`.

**Decision:** do not unify until **23-DELAY-01** delayed registrable probe is specified — avoids destabilizing newly qualified HDP-01 primitive.

---

## Rejected without new evidence

- 5–10× vmap speedup
- L2-cache / “5000 neurons fits L2” claims
- “Maximally efficient EdgeList”
- Small-delay FIFO/register specialization (D≤16)
- Global LRU `SimulationCache`
- EdgeList tagged-union redesign
- “Guaranteed identical machine code” for derivative merge

---

## Todo stack additions (trade-offs / deferred wins)

See `artifacts/todo_stack.md` — new `23-SIMP-*` queue items for L-01, L-02, L-05, L-06, G-audit, HDP-unify.
