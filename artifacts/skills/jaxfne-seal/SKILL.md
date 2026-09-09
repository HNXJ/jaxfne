---
name: jaxfne-seal
description: Independent release seal verification — reconstruct target, verify evidence, classify blockers.
metadata:
  audience: agents
---
# jaxfne seal procedure

## WHEN
Final independent verification before tag, release publication, or PyPI upload.

## AUTHORITIES
1. Version-specific paths: `artifacts/release/current_release_authorities.json`.
2. Issue log: `artifacts/issue_log/ISSUE_LOG.md`.
3. Gate 0: `scripts/harness/gate0_git_reality.py`.
4. Repository `artifacts/AGENTS.md` (evidence discipline).

## RULES
- Gate 0 first.
- Reconstruct the declared candidate from receipts; do not inherit executor scores.
- Read acceptance goal count and goal list path from `current_release_authorities.json`.
- Seal agent must NEVER repair its own candidate.
- Classify each required goal: PASS, PARTIAL, FAIL, DEFER. P0/P1 PARTIAL or FAIL blocks seal (SEAL_NO_GO).
- SEAL_GO does not authorize public writes without user approval.
- Public/private purity: private goals and harness plans must not leak into public docs or packages.

## STEPS
1. Execute Gate 0 (`scripts/harness/gate0_git_reality.py`).
2. Reconstruct candidate identity from the release receipt named in `current_release_authorities.json`.
3. Verify C_core / C_release / C_receipt / C_head separation for the candidate.
4. Audit every required acceptance goal with direct evidence.
5. Verify distributed artifacts: wheel/sdist checksums, exclusions, downstream smoke where required.
6. Produce the complete scorecard and declare SEAL_GO or SEAL_NO_GO.

## STOP
- Red CI, hash mismatch, unresolved P0/P1 blocker, or missing required authority file.

## VERIFY
- Full scorecard produced with direct evidence and clean package verification.

## DONE
- Seal verdict declared with exact immutable candidate SHAs.
