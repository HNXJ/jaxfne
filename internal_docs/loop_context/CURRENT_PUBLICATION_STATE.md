# Current publication state

Last updated: live `cur` after ED10 merge.

## Freeze target

```text
branch: cur
sha: 1f5e599d4bc8045f822222945355667bfaea4c73
jaxfne_version: 0.3.29
```

## Inventory

```text
main figures: 8/8
extended data: 10/10
publication artifact stack: complete
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
| ed10 | Release/archive readiness receipt (evidence panel) | release_archive_evidence_receipt_only |

## Approval-gated (not executed by ED10)

- version tag at current HEAD
- wheel/sdist build and PyPI/TestPyPI publish
- GitHub release
- archive/DOI assignment
- full clean-room pip install smoke (isolated import smoke recorded in ED10)

## Immediate next work

1. Regenerate `outputs/publication/*` on live checkout before submission bundle export.
2. Obtain explicit approval before any tag, release, publish, or archive action.
3. Manuscript alignment pass: replace TBD slots with exact SHA/inventory receipts.

## Status language

Use `implemented`, `receipt-driven`, `release_archive_evidence_receipt_only`, and `pending_approval` for release fields. Do not use physical-field or empirical-validation language for current v0.3.x proxy results.
