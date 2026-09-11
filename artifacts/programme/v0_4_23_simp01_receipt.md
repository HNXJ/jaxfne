# 23-SIMP-01 receipt — CLOSED (kept)

**Branch:** `dev` (on top of `97609f6`)

## Delivered

Internal metadata snapshot only (`jaxfne/emitters.py`):
- New `_uniform_delay_value(edges) -> int | None`: compact-storage static
  delay from Python metadata (`delay_storage`/`uniform_delay_steps`);
  never touches JAX arrays, tracer-safe by construction.
- Four call sites now delegate their storage preamble and keep their own
  logic verbatim: `resolve_edge_delay_steps` (traced materialization),
  `_edge_delays_any_positive` / `_edge_delays_all_zero` (asymmetric
  JIT fallbacks preserved), `_edge_max_delay_steps` (tracer error path
  preserved). `_edge_delay_steps_numpy` / `_edge_delay_steps_host` /
  `_validate_edge_delays_nonnegative_eager` untouched.

## Measures
- Diff: +21/−16 (net +5; helper def + 4× (4-line preamble → 3-line
  delegation)). Concept 4 duplicated dispatches → 1.
- Verdict **KEEP**: identical behavior proven (see below), single source of
  truth for storage dispatch; adding a storage mode now touches 1 site.

## Exact-output evidence (before/after via stash)
- SHA256 before == after for delayed registrable kernel V/S/H_trace/w_trace
  (asymmetric 2/5/9, 40 steps) and compact-model V_m: all five digests match.

## Gates (after)
- `test_compat_jom01_regressions` (4) PASS
- `test_hdp_delayed_registrable` (7) PASS
- `test_hdp01_registrable_qualification` (6) PASS
- `test_hdp_finite_delay`, `test_continuation_contract`,
  `test_hdp_dispatch` (30 combined) PASS
- `TestHdpDelaySupported` (3, incl. fixed setup below) PASS

## Adjacent defect closed
HP-05 continuation `delay_state is not None` failure was a **test-setup
defect**: `replace(edges, delay_steps=ds)` without
`delay_storage="per_edge"` left `uniform_zero` storage, so delays never
engaged (bulk tests passed vacuously; continuation assert failed).
Fixed both nonzero-delay setups in
`tests/test_closure_hp_reconciliation.py`; verified `delay_state` now
carried (probe) and class green with delays genuinely engaged.

## Still open (pre-existing at `88fa347`, owned — not introduced here)
- Population `h_dim` IndexError (`_model_tune` boolean mask vs size-0
  compact weights): material, needs investigation; proposed owner
  23-EDGE-01 (edge selectors/transforms pipeline) or a dedicated defect item.
- Disconnected-null `diag is None` (identity `{"noise_scale": 0.0}` routes
  to baseline by HDP-01 design, so no HDP diagnostics): expectation-vs-design
  conflict; needs human decision on whether `disconnected_null` must force
  HDP engagement.
