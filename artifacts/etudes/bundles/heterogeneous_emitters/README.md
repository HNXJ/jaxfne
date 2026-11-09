# Heterogeneous emitters Étude

Frozen protocol: `docs/etudes/heterogeneous_emitters.md`  
Identity: `heterogeneous_emitters_v0415b`  
Claim: compositional closure `E → S → F → P` for distinct neural equations — not source equivalence.

Reproduce:

```bash
PYTHONPATH=. python3 scripts/run_heterogeneous_emitters_etude.py
```

| file | role |
|------|------|
| `figure.png` | Per-emitter spikes, declared Q, LFP-proxy |
| `figure_spectra.png` | Same spectral observation on each Q |
| `metrics.json` | Gates, hashes, rates, F–I, null control (frozen v0415b record; centroids superseded) |
| `provenance.json` | Distinct source status + observation receipts |
| `manifest.json` | Content hashes |
| `gap_review.md` | Classification including HEI stimulus ANALYSIS_GAP |
| `failed_v0415/` | Preserved nonfinite-Q regime (grammar ≠ dynamical validity) |
| `metrics_post_p048.json` | Live digest `rerun_etude_figures.py` checks; re-recorded 2026-11-09 after 2e1af6f8 |

Since 2e1af6f8 the HEI `bound_mode="stable"` run soft-bounds `x` to (-10, 10) (`tanh` codomain; was ±1e6).
Measured at 1ceb6923 on `simulate_hei` of the main run (x shape 2000 × 2; bound contact = |x| ≥ 0.999·bound, as in
`tests/test_homeostatic_ei_bound_mode_stable.py`): max |x| = 2.17, 0 of 4000 samples in contact; the `tanh` compression
alone moves the HEI `P_Q` centroid from 35.467 to 35.702 Hz. Rates, F–I, gates and the Izhikevich run are unchanged.

All readouts are relative proxies. `Q_Izh` and `Q_HEI` are not physically equated.
