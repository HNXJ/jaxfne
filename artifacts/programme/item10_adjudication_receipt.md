# Item-10 gate adjudication receipt (0.5.0)

Date: 2026-09-23. Human decision: option 1 — waive the absolute leg via the
recorded correction below (original FAIL immutable).
Candidate: `compute_conservation_proxy_diagnostics` jnp→numpy port
(`jaxfne/fields/proxy.py`, 14+/14-, function-confined).

## Immutable history (not rewritten)

- Frozen equivalence gate: all float diagnostics within rel 1e-5 **AND** abs
  1e-8 of pre-change values.
- Independent review verdict: `E_semantic = FAIL (strict)` — max abs
  deviation 2.38e-07 exceeded the abs leg ~24×; rel leg passed ~41×.
  `E_performance = PASS`.
- The original FAIL stands as the historical fact. This receipt authorizes a
  corrected specification; it does not relabel the previous result.

## Authorized correction (one-time, human)

Cross-implementation float comparisons use allclose semantics

    |x - y| <= max(eps_abs, eps_rel * max(|x|, |y|)),

with eps_rel=1e-5, eps_abs=1e-8. The absolute leg governs the near-zero
regime; it is not an additional mandatory constraint at O(1). Rationale: the
conjunctive reading demands sub-ulp agreement across different reduction
implementations; the implementations under test were observed to differ by
1–4 float32 ulps from reduction ordering, with no semantic shift (keys,
None-patterns, messages, version string, structure all identical; 42-test
behavior suite green unmodified).

Encoded for future agents in `tests/_numeric_gates.py`
(`cross_impl_close` / `assert_cross_impl_close`); unit-tested in
`tests/test_numeric_gates.py`. Do not reinterpret or widen without a new
recorded adjudication.

## Re-evaluation against the corrected gate (2026-09-23, this machine)

Oracle: installed jaxfne 0.4.25 `fields/proxy.py`, sha256 `77b4f0c8…`
verified identical to `git HEAD` version. Work tree: repo impl.
8-case matrix (finite (200,10)/(1000,100)/(400,1000), NaN-source, all-None,
1-D phi, tiny, zeros): keys/order/None-pattern identical everywhere;
19 floats compared, **0 corrected-gate failures**; worst rel 1.82e-07
(55× margin under eps_rel=1e-5); worst abs 5.96e-08 (near-zero arms exact).
**E_semantic (corrected) = PASS.**

## Performance re-evaluation (fresh processes, repo impl)

- Cold manifest 1n-equivalent: **1.7 ms** (pre-change 457.6 ms worker /
  ~520 ms reviewer; frozen baseline_050.json manifest 379 ms, untouched).
- Warm same-shape repeat: **1.5 ms** (gate ≤10 ms).
- Unseen shape (7n, 50 ms): **1.3 ms** (gate <100 ms; pre-change ~600 ms
  recompile per unseen shape, now eliminated).
- **E_performance = PASS.** Local-environment receipt only, same claim scope
  as `baseline_050.json`.

## Scope compliance

Single file changed (`jaxfne/fields/proxy.py`); `diagnostics.py`,
`_model_manifest.py`, tests unweakened (dedicated suite byte-identical);
no RNG/state; manifest observation-only so Δsimulation=0 by construction.
