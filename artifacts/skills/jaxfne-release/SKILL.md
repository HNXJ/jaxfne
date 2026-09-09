---
name: jaxfne-release
description: Release candidate preparation — packaging, gates, artifacts, and authorized publication steps.
metadata:
  audience: agents
---
# jaxfne release procedure

## WHEN
Release-checkpoint work: candidate identity, version consistency, test gates, packaging, hashes, clean install, and authorized tag/release/PyPI/docs steps.

## AUTHORITIES
1. Repository `artifacts/AGENTS.md` (release identities, completion rule).
2. Version-specific paths: `artifacts/release/current_release_authorities.json`.
3. Active task mode: `scratch/CURRENT_TASK.md` (Gate 0 reads `mode:`).

## RULES
- Distinguish C_core, C_release, C_receipt, and C_head explicitly in receipts.
- Read the current release receipt and acceptance list from `current_release_authorities.json`; do not assume a version from skill prose.
- Release/tag/push/upload happens only with explicit user authorization.

## STEPS
1. Run Gate 0 to verify branch alignment and required authorities for RELEASE mode.
2. Confirm candidate identity: version strings, changelog, and immutable candidate SHA.
3. Build and verify wheel/sdist packages; ensure zero private-tree leaks.
4. Run required gates (`scripts/run_test_gate.py` per scope), MkDocs strict, and clean-room install smoke.
5. Record exact immutable SHAs, package hashes, and release receipts.

## STOP
- Missing explicit authorization for any remote/release operation; gate confusion; hash mismatch.

## VERIFY
- Package hashes match the release receipt; clean checkout install passes smoke test.

## DONE
- Release candidate finalized and ready for independent seal.
