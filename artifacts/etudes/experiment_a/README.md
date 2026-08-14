# Experiment A (0.4.17-B)

Frozen protocol: `docs/etudes/experiment_a.md`  
Spec: `b0_protocol_spec.json`

## Checkpoints

| File | Checkpoint |
|------|------------|
| `b0_protocol_spec.json` | B0 protocol |
| `b1_canonical_receipt.json` | B1 canonical X/H/Q hashes |
| `b2_probe_receipt.json` | B2 independent probes |
| `b3_experiment_a_receipt.json` | B3 bundle receipt |
| `manifest.json` / `metrics.json` / `provenance.json` | B3 committed bundle |

Local/gitignored arrays: `canonical_source.npz`, `observations.npz`.

## Reproduce

```bash
python scripts/run_experiment_a.py
```
