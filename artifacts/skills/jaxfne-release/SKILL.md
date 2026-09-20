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
- Record newly detected problems in the issue log immediately with IDs; a
  release with a non-empty Open section is blocked.

## STANDING ACCEPTANCE (every jaxfne release from 0.4.25 on)

- Docs: low-verbosity, simple, smooth; tables/lists/paragraphs/HTML figures
  interleaved; theme matched; TFNE grammar + terms consistent; left menu
  (`mkdocs.yml` nav) organized.
- Code: low complexity, optimal operation order, canonical flattening; JAX
  switches (float32/64, cuda/cpu/parallel-cpu/metal) via official
  mechanisms; official-doc conformance suffices, no bespoke JAX-plumbing
  tests.
- Stacks empty: release todo section drained item-by-item AND issue-log
  Open section empty with resolutions recorded (history preserved, never
  rewritten); problems detected mid-release get IDs and drain the same way.
- Single full release executes once, only when all acceptance verifies
  green on the sealed commit. No partial releases.

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
