# Port and oracle parity checks

Core routes of `jaxfne` against independent references, recomputed by
`scripts/score_port_parity_checks.py` with the inputs, tolerances and references of the
tests named below. `score.json` holds one `{error, tolerance, pass, reference}` row per check.
For a test that states `assert_allclose(rtol, atol)`, `error` is the worst ratio of
`|core - reference|` to `atol + rtol·|reference|`, so a passing test implies a passing row
(the limit is 1). `worst_abs_deviation` is the largest `|core - reference|` over all elements, in the
test's units; it need not be the element that sets the worst ratio.

Reference types: **external** (another library) and **transcription** (file-local float64
code of published equations or of a documented lane oracle).

| check | test | reference | type | bound |
|---|---|---|---|---|
| adex_single_cell_oracle | tests/test_emitters_adex.py:165 | float64 NumPy transcription of `caminos_jax.adex_step` (the CAMINOS project port, artifacts/etudes/papers/caminos/caminos_jax.py, which imports no jaxfne; not the authors' code) | transcription | rtol 1e-6, atol 1e-9; spikes exact |
| adex_network_oracle | tests/test_emitters_adex.py:196 | float64 hand-coded loop of the two-cell network, four cases | transcription | rtol 1e-6, atol 1e-9; spikes exact |
| multicompartment_hh_jaxley | tests/test_hh_multicompartment.py:44 | jaxley 0.14.0 jx.integrate, four compartments | external | rtol 1e-12, atol 1e-9 |
| gsdr_mcdp_oracle_zscored | tests/test_gsdr_mcdp.py:64 | float64 transcription of the lane oracle, seed 23 | transcription | rtol 1e-5 |

`multicompartment_hh_jaxley` needs the optional `jaxley` package; without it the row keeps
nulls with `"skipped": "jaxley not installed"`.

Runtime on the build machine, one run: adex single cell 1.4 s, adex network 2.6 s,
multicompartment 26.0 s, gsdr zscored 23.4 s.

Candidates not scored (budget: no check above 30 s; each needs a reference that is not the same code path):

| candidate | reason |
|---|---|
| tests/test_jaxley_emitter_bridge_e2e.py:60 | mechanistic only (finite V, spike above 0 mV); 115.5 s |
| tests/test_gsdr_mcdp.py:31 and :47 | transcription; 34.4 s and 33.1 s |
| tests/test_destexhe2009_port.py:76 | asserts a spike count only; no reference value |
| tests/test_intra_step_fixed_point_t22.py, `_oracle_first_step` | calls the core `adex_step`, so the same code path |
| tests/test_emitters_lif.py:288 and :419 | oracle side asserted inline on hand-set events |
| tests/test_istdp_vogels2011.py:202 | hand pairing values for the paper rule; open |

Rerun:

    python scripts/score_port_parity_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_port_parity_checks.py --check # recompute and exit 1 past rtol 1e-9
