# HDP controllability & reachability Etude

Compact reproducible bundle for the HDP-MVC scientific argument.

## Contents

| file | role |
|------|------|
| `figure.png` | Main A–L scientific plate |
| `metrics.json` | Compact control + neurophysiology metrics |
| `manifest.json` | Frozen receipt hashes and runner pointers |

## Reproduce

From repository root:

```bash
python scripts/consolidate_hdp_controllability_etude.py
```

This packages the committed `metrics.json`, `figure.png`, and `manifest.json`. If local diagnostic receipts exist under `artifacts/msvc_hdp_diagnostic/`, consolidation verifies their hashes and refreshes the figure from the etude simulation receipt; otherwise the committed bundle remains authoritative.

Prior control-theoretic receipts are referenced by hash in `manifest.json` (local provenance, not committed).

## Claim status

- Representation: relative computational scaffold
- Field: laminar proxy readout (`field_claim_level=proxy_readout`), not calibrated empirical LFP/CSD
- Controller: prospectively frozen before MVC #2 validation

Branch `dev` @ `6b7a6b324d9e`
