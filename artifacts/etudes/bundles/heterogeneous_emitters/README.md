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
| `metrics.json` | Gates, hashes, rates, F–I, null control |
| `provenance.json` | Distinct source status + observation receipts |
| `manifest.json` | Content hashes |
| `gap_review.md` | Classification including HEI stimulus ANALYSIS_GAP |
| `failed_v0415/` | Preserved nonfinite-Q regime (grammar ≠ dynamical validity) |

All readouts are relative proxies. `Q_Izh` and `Q_HEI` are not physically equated.
