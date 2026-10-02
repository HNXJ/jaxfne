# Bottlenecks 051 — ranked from matrix_051 (item 2)

Authority: `artifacts/perf/matrix_051.json` (14/14 MEASURED) against
`artifacts/perf/matrix_051_spec.json` (frozen). Thresholds per spec:
a phase is a bottleneck if ≥25% of warm-total time
(warm_median + probe + manifest; compile_est reported separately) or ≥25%
of peak memory in any cell. Env: jax/jaxlib 0.10.1, Windows-11, cpu:0,
jaxfne 0.5.0. Probes P1–P4 are read-only measurements (scripts in
`C:/Users/nejath/AppData/Local/Temp/opencode/probe_p*.py`, not repo scope).

## Warm-total shares (ms)

| cell | warm_med | probe | manif | warm-total | construct (1×) | first |
|---|---|---|---|---|---|---|
| base | 259.7 | 0.03 | 1.22 | 260.9 | 7500 | 1272 |
| n1 | 140.0 | 0.01 | 0.65 | 140.7 | 6300 | 738 |
| n10 | 183.9 | 0.01 | 0.69 | 184.6 | 6200 | 2603 |
| n1000 | 330.7 | 0.02 | 0.76 | 331.5 | 7700 | 1058 |
| n10000 | 45454.5 | 0.02 | 0.70 | 45455.2 | 22000 | 46473 |
| t10x | 275.4 | 0.02 | 0.74 | 276.2 | 6700 | 1283 |
| dt0025 | 214.4 | 0.02 | 0.84 | 215.3 | 8700 | 1042 |
| dt05 | 289.2 | 0.01 | 1.85 | 291.1 | 15400 | 1091 |
| rec_off | 241.8 | 0.02 | 0.83 | 242.6 | 10400 | 1472 |
| rec_full | 286.4 | 0.06 | 2.04 | 288.5 | 14300 | 4707 |
| mech_hdp | 563.4 | 0.01 | 0.69 | 564.1 | 4400 | 2505 |
| chunk_ref | 262.4 | 0.02 | 0.87 | 263.3 | 4600 | 1292 |
| chunk_k4 | 203.0(seg) | 0.02 | 0.82 | — | 6300 | 2884(chained) |
| at10 | 2491.1 | 0.02 | 184.07 | 2675.2 | 84010 | 3537 |

Step-count invariance at N=100: t10x (10k steps) 275ms ≈ base (1k steps)
260ms; dt0025 (4k steps) 214ms vs dt05 (200 steps) 289ms. Per-call fixed
cost ≈ 200–250ms dominates; stepping is marginal. Root cause below (B1).

## Ranked list

### B1. Per-call XLA recompilation on the default eager path — ENTER item 3
- Measure: P2 cProfile of a warm base-class call: 121/177ms (68%) in
  `pxla.compile`/`backend_compile_and_load` (39 cache misses — every call
  retraces the scan body). HDP warm (P3): 297/537ms (55%) recompile, 2
  compiles per call. Default `RuntimeConfig.jit=False` → no compiled cache.
- Expected gain: P1 (n=1000, 999k dense edges): `jit=True` warm 152ms vs
  eager 277ms = −45% of warm-total; first call 514ms (compile once).
  Probe bit-exact: spikes identical, V_m identical, maxabsdiff 0.0.
- Equivalence class: bit-exact (gate must hold on canonical configs; any
  tolerance need → STOP + report).
- Note: at n10000 recompile is only ~1s of 46s (~2%) — B1's gain is
  compile-dominated regimes (small/medium N, HDP).

### B2. AT-10 construct: connection-rule compilation — ENTER item 3 second
- Measure: construct 84.0s = 93% of AT-10 cell wall (84.0/90.2s).
  P4 attribution: `_construct_compile_connections` 81% of construct;
  `compile_connection_rules` 76%, of which `_select_indices` 45%
  (1960 calls over 20 identical areas — same local selections recomputed
  per area), `jnp.asarray/array` + getitem/take slicing the rest.
- Expected gain: cross-area selection dedup up to ~40% of construct
  (~34s of 84s; estimate, freeze exact target in opt051_2 spec).
- Equivalence class: bit-exact (realized edge-multiset equality gate +
  canonical trajectory identity).

### B3. Dense O(N²) warm execution + transient peak at 10k — CARRY (no fix)
- Measure: n10000 warm 45.5s = 100% of warm-total, 175× base; cell peak
  9.57GB vs boundary RSS ≤2.07GB → ≥78% of peak is transient inside
  simulate (arrays total 120MB). Builder warning recorded: this path is
  dense all-to-all (10⁸ edges); the sparse-aware builder is a different
  network, not an equivalent-preserving change.
- Expected gain under bit-exact: none (jit removes only the ~1s recompile,
  ~2% < 10% stop line). CARRY to a later release (report only).

### B4. HDP execution proper — CARRY, re-measure after B1
- Measure: HDP warm-total 564.1 vs base 260.9 → +303ms = 54% of HDP cell
  (crosses 25%). P3: ~297ms of the 537ms warm is recompile (B1's share);
  execution proper ≈ +100ms vs fixed-W.
- Expected gain of further HDP-kernel work: unattributed → no concrete
  change; CARRY with re-measure note after opt051_1 lands.

### B5. First-call trace/compile, one-time per process — CARRY
- Measure: 0.7–4.7s first calls (compile_est), paid once per process.
  Persistent cross-process compile cache = invalidation risk + complexity
  for a one-time cost. CARRY.

### B6. Recording volume — DECIDED: does not enter item 3 (2c)
- Measure: rec_full warm-total 288.5 vs rec_off 242.6 → recording share
  15.9% < 25% threshold. Peak +14% (493 vs 433MB), also below.
  The opt-in selective/downsampled recording API does NOT enter item 3.

### B7. Manifest at AT-10 (184ms, 6.9% of warm-total) — not a bottleneck
- Below threshold. CARRY (report only).

### B8. Chunked continuation — EXCLUDED from item 3, reassignment
- Measure: chained first call 2.88s vs single-call 1.29s (+123% overhead;
  no speed purpose — chunking is a memory tool). Spikes 173 vs 174 as-run:
  NOT bit-exact under this harness (identical seed per segment suspected;
  engine-level equivalence unestablished).
- No tolerance authorized → excluded. Report for reassignment (root-cause
  seed/key chaining across continuation segments first).

## Memory dimension
- Small cells: peaks 357–559MB; base construct delta +323MB = 74% of base
  peak (includes ~300MB process+JAX init — one-time, not recording;
  arrays ≤1.2MB). No recording-memory bottleneck.
- n10000: 9.57GB transient (B3). at10: 1.29GB peak vs 48MB arrays (27×
  transient in construct/simulate).

## 3d projection
After opt051_1 (B1) and opt051_2 (B2): remaining quantified gains are B3
(no equivalent fix), B5 (one-time), B6/B7 (below thresholds), B8
(excluded) — top remaining expected gain <10% → STOP, carry the rest.
Apply literally; a failed gate reverts its change and takes the next
candidate.
