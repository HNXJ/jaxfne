# HDP (Homeostasis-Dependent Plasticity) — Implementation Report

**Status:** Technical report · computational-control method, not a biological
mechanism claim · 2026-06-24
**Truth gates:** `claim_status=computational_control_proxy_not_biological_mechanism`,
`biological_learning_claim=False`, `mechanism_claim_status=not_claimed`.

## Summary

HDP is a per-neuron master-state (`H_i`) controller that drives excitatory and
inhibitory edge-weight ODEs, layered on top of the existing per-neuron
[homeostasis](guides/homeostasis.md) excitability controller (a separate,
mutually-exclusive mechanism). This report covers what was built, where it
lives, how it's verified, and its measured overhead.

## What was built

1. **Kernel** (`jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp`) —
   integrates `H_i` (five additive `dH/dt` terms: synaptic income `alpha`,
   bias `beta`, activity drain `gamma`, weight-budget drain `delta`, restoring
   control `K_ctrl`, plus a barrier term) and the weight ODEs
   `dw_E/dt = +K_HDP*(H_i-1)*w_E`, `dw_I/dt = -K_HDP*(H_i-1)*w_I`. Hard-bounds
   `H_i` to `[H_min, H_max]` and weights to `[w_floor, w_ceiling]`. Supports
   `record_dH_components`/`record_edge_current` diagnostics, `init_state`
   pause/resume, and `size_scale_override`.
2. **Generic builder** (`jaxfne.hdp_network`) — config-driven (no per-N
   functions), with two frozen tuned presets, `DEFAULT_HDP` (long-term stable,
   verified 20s/5-seed) and `DEFAULT_HDP_DESYNC` (faster, less-overdamped `H`
   dynamics; documented as a "current best candidate," not a finished point).
3. **Dispatch wiring** (`jaxfne/core.py`) — `RuntimeConfig.enable_hdp` /
   `hdp_params`, mutually exclusive with `enable_homeostasis`
   (`__post_init__` raises `ValueError` if both set), JIT-cached via a
   params-fingerprinted cache key, per-call diagnostics via
   `Model.last_hdp_diagnostics()`, and a `_simulate_hdp_metadata()` helper
   surfacing `Signals.metadata["hdp"]` with the same conservative claim
   framing as the homeostasis metadata block.
4. **Fluent verb** (`Configuration.hdp(relative_baseline=1.0, **kwargs)`) —
   mirrors `Configuration.homeostasis(...)`; `relative_baseline=1.0` is the
   identity baseline, deviating resolves `K_HDP = relative_baseline - 1.0` and
   activates the controller. Surfaced in `manifest()["hdp"]`
   (`jaxfne/io.py`).
5. **ED9 ablation evidence** (`scripts/ed9_hdp_evidence.py`) — null +
   ablation + repeated-seed evidence bundle on a deliberately imbalanced
   column, mirroring `scripts/ed9_homeostasis_evidence.py`. The grid is
   **3-way, not 4-way**: `null` / `h_dynamics` (H moves, weights frozen) /
   `both` (H moves and drives weights) — because HDP's weight term is itself
   gated on `H` deviating from 1.0, so "plasticity active with `H` pinned"
   collapses to the null and isn't a distinct condition.
6. **Docs** — [`docs/guides/hdp.md`](guides/hdp.md), wired into `mkdocs.yml`
   nav and cross-linked from `homeostasis.md`/`showcases.md`; `enable_hdp`/
   `hdp_params` documented in `docs/api/runtime.md`.

## Verification

- **Standalone kernel** (`tests/test_hdp_kernel_standalone.py`, 8 tests):
  null control (`H` pinned at exactly 1.0, weights bit-identical),
  `K_HDP=0` disabling plasticity independent of other gains, nonzero-gain
  finite/bounded trajectories, `record_dH_components`/`record_edge_current`
  trace shapes, `init_state` pause/resume determinism, `size_scale_override`
  changing dynamics, and `K_HDP<0` (anti-homeostatic stress mode) staying
  finite.
- **Dispatch wiring** (`tests/test_hdp_dispatch.py`, 7 tests): config
  propagation through `Configuration.runtime(...)`, the null-control
  invariant via `last_hdp_diagnostics()`, diagnostics passthrough through
  `Signals.metadata`, JIT-cache reuse across seeds (`N_compile==1`), the
  `synaptic_kernel="receptor_exponential"` guard, the
  `enable_homeostasis`/`enable_hdp` mutual-exclusivity guard, and cache-key
  isolation across `hdp_params` changes on a reused `Model`.
- **Parity coverage added to 5 existing homeostasis test files**: identity/
  deviation/manifest/mutual-exclusivity (`test_config_plasticity_homeostasis_baseline.py`),
  hard-bounded float32 under extreme drive
  (`test_homeostatic_stability_v042.py`), `_simulate_hdp_metadata` unit tests
  (`test_simulate_homeostasis_metadata_helper.py`), the canonical column with
  JSON-safe metadata (`test_canonical_biophysics.py`).
- **ED9 evidence harness** (`tests/test_ed9_hdp_evidence.py`): bundle
  structure, the null-control invariant (`H` pinned, `H_std=0`), `h_dynamics`
  moving `H`, and truth gates.
- **Full non-slow/non-notebook regression sweep**: `2413 passed, 70 skipped,
  237 deselected, 4 xfailed` — no regressions from the dispatch wiring.

## Performance

Measured on a 1000-neuron canonical laminar column, 1000 ms duration, CPU
(not JIT-compiled — direct `jtfne.simulate()` call):

| Path | Wall time | `V_m` finite |
|------|-----------|--------------|
| baseline (`recurrent_backend="edge_list"`) | 10.07 s | yes |
| HDP (`DEFAULT_HDP`-style gains) | 11.96 s | yes |

HDP adds ~19% wall-time overhead at this scale from the extra per-step `H`
integration and weight-ODE update over the edge list — not measured under JIT
warm-cache reuse, where the relative overhead is expected to shrink (the
per-step Python/dispatch overhead amortizes away; only the compiled kernel's
extra arithmetic remains). Not yet benchmarked at 10k neurons.

## Known gaps

- `simulate_batch()` (the vmap path) does not yet dispatch `enable_hdp` —
  only `homeostasis` is wired there today. Tests that would exercise an HDP
  equivalent of `test_batch_engages_homeostasis_and_null_matches` are not
  applicable until that's added.
- `DEFAULT_HDP_DESYNC` remains an explicitly documented "current best
  candidate," not a frozen/stable point like `DEFAULT_HDP` — its rate-spread
  band did not reach the originally requested `[0.8, 1.2]` floor.
- No 10k-neuron HDP perf/stability sweep yet (only the 1000-neuron
  measurement above and the smaller-N stability/standalone tests).
