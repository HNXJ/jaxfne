# JaxFNE v0.4.22 — independent seal scorecard

**Verdict: SEAL_GO**

Seal agent: independent verification per `artifacts/skills/jaxfne-seal/SKILL.md`.
Human authorization: **22-SEAL-01 AUTHORIZED** (publication gates remain separate).

## Candidate identity (immutable)

| Field | Value |
| --- | --- |
| Validated release candidate | `d849e6d91b72ce1ca2b736c03bdec8c1d89f55c0` |
| Candidate tree | `85a59d1055e90dc44d5275faba1cb013f6f07efd` |
| Package version @ candidate | `0.4.22` |
| Published predecessor | `v0.4.21` @ `88235208950425ce76723115f1cb61fe3a828ab2` |
| Integration baseline | `2338b00af5c7c822aa35af60d68be4d62f18469d` |
| Dev bookkeeping tip (pre-seal) | `00df47066439a48ba007a45260c2fb1e9ebce1d4` |

`00df470` is receipt/metadata-only after `d849e6d` (receipt + todo_stack). It does not
extend the scientifically validated candidate claim.

## Gate 0

| Check | Result |
| --- | --- |
| `scripts/harness/gate0_git_reality.py` | **PASS** |
| Authorities present | `artifacts/release/current_release_authorities.json` → v0.4.22 |
| Working tree @ seal time | CLEAN |

## C_core / C_release / C_receipt / C_head separation

| Identity | SHA | Notes |
| --- | --- | --- |
| C_release (validated candidate) | `d849e6d` | RC + CI + junit parity evidence anchor |
| C_receipt (release receipt) | names `d849e6d` | `artifacts/release/v0_4_22_release_receipt.json` |
| C_head (dev bookkeeping) | `00df470` → seal commit | metadata-only delta after candidate |
| origin/main (pre-FF) | `7118a67` | behind candidate; FF to `d849e6d` authorized |

## RC / CI / junit parity

| Gate | Evidence | Result |
| --- | --- | --- |
| RC gate | `artifacts/attestations/rc_gate_attestation.json` @ `d849e6d` | **PASS** (`pre_release_subsumes_ci=true`) |
| CI (Release & Scheduled) | run [34536095797](https://github.com/HNXJ/jaxfne/actions/runs/34536095797), `headSha=d849e6d`, conclusion `success` | **PASS** |
| JUnit parity | `artifacts/attestations/junit-parity.json` @ `d849e6d` | **PASS** (rc=ci=3869 nodes; 4 justified POSIX-only diffs per matrix) |
| Isolated wheel smoke | RC family `isolated_wheel_smoke` | **PASS** (`jaxfne-0.4.22`) |

## Acceptance goals (5/5 required)

Authority: `artifacts/private_acceptance/jaxfne_v0_4_22_programme_acceptance.md`

| # | Goal | Class | Evidence |
| --- | --- | --- | --- |
| 1 | Scientific numeric invariance vs v0.4.21 | **PASS** | Integration baseline `2338b00` RC+CI PASS pre-identity bump; `22-VERIFY-01` identity-sensitive suite (56 passed); version bump is identity-only per programme scope |
| 2 | New failure-path coverage | **PASS** | `tests/test_objective_null_reproducibility_v0330.py`, `tests/test_atlas_suite.py` in RC `pytest_broad` @ `d849e6d` |
| 3 | Classification closure (zero UNKNOWN) | **PASS** | `artifacts/audit/w3_broad_handler_tally.json` (132 handlers, no UNKNOWN bucket); `artifacts/audit/w7_dense_nxn_inventory.json` (16 sites, roles ∈ {AUTHORITATIVE, TEMPORARY, DERIVED, VALIDATION_ONLY}) |
| 4 | Public unfinished-work sweep | **PASS** | RC families `docs_language_audit`, `docs_build_strict`, `vocabulary_audit` all PASS @ `d849e6d` |
| 5 | Release verification chain | **PASS** | Full RC gate PASS; CI PASS; notebook gate PASS; package build + twine_check PASS; junit parity PASS |

## Reconciliation status

| Check | Result |
| --- | --- |
| `release_candidate_receipt_valid` | **true** (RC attestation matches `d849e6d`) |
| `release_target_reconciled` | **false until** `origin/main == d849e6d` (FF authorized; not claimed complete pre-FF) |
| Tag `v0.4.22` | **not created** (22-PUB-01 — separate human gate) |
| PyPI / GitHub release / RTD | **not authorized** (22-PUB-02..04) |

## Blockers

None at P0/P1 for seal. Publication remains blocked on explicit human authorization per todo stack.

## Seal declaration

**SEAL_GO** on validated candidate `d849e6d91b72ce1ca2b736c03bdec8c1d89f55c0`.

Seal does **not** authorize tag creation, GitHub release, PyPI upload, or production docs publication.
