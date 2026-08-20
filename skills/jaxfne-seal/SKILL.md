---
name: jaxfne-seal
description: Independent release seal verification for jaxfne against the private 95-goal acceptance authority.
metadata:
  audience: agents
---
# jaxfne seal procedure

## WHEN
Final verification and candidate sealing before tag, release publication, or PyPI upload.

## AUTHORITIES
1. Final acceptance authority: `artifacts/private_acceptance/jaxfne_v0_4_17_final_100_goals.md`.
2. Release receipt: `artifacts/release/v0_4_17_release_receipt.json`.
3. Issue log: `artifacts/issue_log/ISSUE_LOG.md`.
4. Gate 0: `scripts/harness/gate0_git_reality.py`.

## RULES
- Gate 0 first.
- Explicitly distinguish C_core, C_release, C_receipt, and C_head.
- Seal agent must NEVER repair its own candidate.
- All 95 goals must be audited and classified into: PASS, PARTIAL, FAIL, DEFER.
- P0/P1 PARTIAL or FAIL blocks the seal (SEAL_NO_GO).
- SEAL_GO requires 100% PASS on all release-required goals and still does NOT authorize public writes without user approval.
- Public/private purity: Private goals, harness, and plans must not leak into public docs or packages.

## STEPS
1. Execute Gate 0 (`scripts/harness/gate0_git_reality.py`).
2. Verify C_core exact diff against C_release (must be non-core/docs only; Delta C_core = 0).
3. Audit all 95 goals in `jaxfne_v0_4_17_final_100_goals.md` with direct evidence for each.
4. Verify wheel/sdist packages, checksums, and exclusion of private trees.
5. Produce the complete 95-goal scorecard and declare SEAL_GO or SEAL_NO_GO.

## STOP
- Red CI, hash mismatch, Delta C_core != 0, or any P0/P1 goal failure.

## VERIFY
- Full 95-goal scorecard produced with direct evidence and clean package verification.

## DONE
- Seal verdict declared with exact immutable candidate SHAs.
