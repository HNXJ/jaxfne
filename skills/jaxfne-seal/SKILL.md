---
name: jaxfne-seal
description: Independent release seal verification for jaxfne.
metadata:
  audience: agents
---
# jaxfne seal procedure

## WHEN
Final verification and candidate sealing before tag, release publication, or PyPI upload.

## AUTHORITIES
1. Release receipt: `artifacts/release/v0_4_17_release_receipt.json`.
2. Issue log: `artifacts/issue_log/ISSUE_LOG.md`.
3. Pre-freeze CI logs and test gate receipts.

## RULES
- Gate 0 first.
- Explicitly distinguish C_core, C_release, C_receipt, and C_head.
- Seal agent must NEVER repair its own candidate.
- SEAL_NO_GO returns findings; SEAL_GO does not authorize public writes without user approval.

## STEPS
1. Execute Gate 0 (`scripts/harness/gate0_git_reality.py`).
2. Verify C_core exact diff against C_release (must be non-core/docs only).
3. Verify wheel and sdist hashes against release receipt.
4. Execute required test gates and docs strict build.
5. Issue verdict: SEAL_GO or SEAL_NO_GO.

## STOP
- Red CI, hash mismatch, or Delta C_core != 0.

## VERIFY
- Hash match on packages and clean git status.

## DONE
- Seal verdict declared with exact immutable SHAs.
