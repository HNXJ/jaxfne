<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: cur
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# Current publication state

Last updated from attached zip: `2026-06-07T22:34:39Z`.

## Freeze target

```text
branch: cur
sha: e83df5d1a50dcb7a82c6feedc43ef3e45e586c78 (re-verify)
jaxfne_version: 0.3.29
checklist_generated_at_utc: 2026-06-07T22:25:39Z
zip_sha256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
```

The zip has no `.git` metadata, so live repo work must re-freeze branch/SHA before mutation.

## Inventory from attached zip

```text
main figures: 8/8
extended data: 8/10
figures/publication: 16 PNGs present
scripts/publication: 16 generator scripts present
tutorial files: 24 total, 18 notebooks
examples: 19 Python files
jaxfne Python modules: 39
tests: 138 test files
docs: 105 files
outputs/publication: not included in zip; local inventory created during inspection only
```

## Completed main figures

| ID | Purpose | Boundary |
|---|---|---|
| fig01 | TFNE operator grammar overview | computational_scaffold_only |
| fig02 | Source/field contracts and truth gates | no_pde_solver_claim |
| fig03 | Config to simulate backend flow | package_native_api |
| fig04 | Minimal install and smoke run | cpu_safe_tutorial |
| fig05 | Runtime scaling across N, Z, seeds | benchmark_receipt_required |
| fig06 | Eight proxy readout families | proxy_readouts_only |
| fig07 | Manifest and hash reproducibility | fixed_tag_sha_required |
| fig08 | Positioning vs adjacent tools | capability_comparison_not_superiority |

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

## Remaining Extended Data

| ID | Required next artifact | Scope |
|---|---|---|
| ED9 | `scripts/publication/ed09_failure_modes_and_nulls.py`; `figures/publication/ed09_failure_modes_and_nulls.png`; manifest/receipt | Failure modes, null controls, objective nulls, overinterpretation guards, JSON rejection, proxy/solver language guard. |
| ED10 | `scripts/publication/ed10_release_archive_receipt.py`; `figures/publication/ed10_release_archive_receipt.png`; manifest/receipt | Release tag, wheel/sdist hashes, clean install smoke, archive/DOI metadata, exact repo SHA. Approval-gated. |

## Immediate next work

1. Sync live `cur` and confirm whether ED8 branch is merged.
2. Regenerate `outputs/publication/*` by running fig01-fig08 and ed01-ed08 scripts.
3. Implement ED9.
4. Implement ED10 only after explicit release/archive approval.
5. Repeat optional dependency laziness gate in a clean venv; patch if `pandas` is loaded by root import there.

## Status language

Use `implemented`, `receipt-driven`, `local artifact integrity`, `local CPU runtime receipt`, and `proxy operator contract`. Do not use physical-field or empirical-validation language for current v0.3.x proxy results.
