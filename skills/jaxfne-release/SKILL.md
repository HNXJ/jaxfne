---
name: jaxfne-release
description: Stable-surface contraction, documentation freeze, packaging, and authorized release work.
metadata:
  audience: agents
---
# jaxfne release procedure

## WHEN
Release-checkpoint work: stable-surface contraction, documentation freeze, packaging, and release operations.

## AUTHORITIES
1. Repository `AGENTS.md` (release identities, completion rule).
2. Release receipt: `artifacts/release/v0_4_17_release_receipt.json`.

## RULES
- Distinguish C_core, C_release, C_receipt, and C_head explicitly.
- Delta C_core = 0 during release candidate polish.
- Release/tag/push/upload happens only with explicit user authorization.

## STEPS
1. Run Gate 0 to verify branch alignment.
2. Build and verify wheel/sdist packages; ensure zero leaks (.opencode, receipts, etc.).
3. Verify clean-room install, mkdocs strict, and test gates.
4. Record exact immutable SHAs and receipts.

## STOP
- Missing explicit authorization for any remote/release operation; gate confusion.

## VERIFY
- Package hashes match release receipt; clean checkout install passes smoke test.

## DONE
- Release candidate finalized and ready for independent seal.
