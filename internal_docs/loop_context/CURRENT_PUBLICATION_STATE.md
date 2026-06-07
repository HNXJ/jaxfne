# Current publication state

Last updated: `2026-06-07` (live `cur` after ED9 merge).

## Freeze target

```text
branch: cur
sha: 5c78541 (re-verify with git rev-parse HEAD)
jaxfne_version: 0.3.29
```

## Inventory (live checkout)

```text
main figures: 8/8
extended data: 9/10
remaining: ED10 release archive receipt (approval-gated)
```

## Completed Extended Data

| ID | Purpose | Boundary |
|---|---|---|
| ed01 | Public API stability snapshot | local_api_snapshot_only |
| ed02 | JSON schema and config validation receipts | config_schema_validation_receipt |
| ed03 | Notebook execution receipts (tutorial atlas survey) | receipt_driven_structural_scan |
| ed04 | Optional dependency laziness (local subprocess receipts) | local_subprocess_optional_dep_receipt |
| ed05 | Local manifest/hash integrity receipt | local_artifact_integrity_receipt_only |
| ed06 | Benchmark scaling tables (local CPU runtime receipt) | local_cpu_runtime_receipt_only |
| ed07 | Probe/readout operator contract matrix | proxy_readout_operator_contracts_only |
| ed08 | Tutorial atlas coverage matrix (receipt-driven) | tutorial_atlas_coverage_receipt_only |
| ed09 | Failure modes and null controls (local receipt) | failure_null_local_receipt_only |

## Remaining Extended Data

| ID | Required next artifact | Scope |
|---|---|---|
| ED10 | `scripts/publication/ed10_release_archive_receipt.py`; `figures/publication/ed10_release_archive_receipt.png`; manifest/receipt | Release tag, wheel/sdist hashes, clean install smoke, archive/DOI metadata, exact repo SHA. Approval-gated. |

## Immediate next work

1. Regenerate `outputs/publication/*` by running fig01–fig08 and ed01–ed09 scripts on live checkout.
2. Implement ED10 only after explicit release/archive approval.
3. Repeat optional dependency laziness gate in a clean venv if root import loads `pandas` there.

## Status language

Use `implemented`, `receipt-driven`, `local artifact integrity`, `local CPU runtime receipt`, `failure/null local receipt`, and `proxy operator contract`. Do not use physical-field or empirical-validation language for current v0.3.x proxy results.
