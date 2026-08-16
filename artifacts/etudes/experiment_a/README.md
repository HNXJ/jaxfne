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

Reproduction is **verification-first**: the canonical arrays are regenerated
into the local gitignored `.npz` files, and every regenerated value is compared
against the committed receipts instead of rewriting them. Array SHA-256 hashes
must match byte-exactly; derived analytic metrics (spectral centroids, r90)
may differ only within the declared relative tolerance (rtol 1e-6) due to
environment float tail noise; run-stamp fields (`package_head`) and the
committed `test_evidence` annotation block are preserved as committed. Any real
drift raises a hard error. The runner exits non-zero unless all canonical
hashes verify. Committed receipt bytes are write-once provenance of the
original freeze and stay untouched across reviewer reruns.
