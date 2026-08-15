---
name: jaxfne-release
description: Stable-surface contraction, documentation freeze, packaging, and authorized release work.
metadata:
  audience: agents
---
# jaxfne release procedure

## WHEN TO USE
Release-checkpoint work: stable-surface contraction, documentation freeze,
packaging, and explicitly authorized release operations.

## AUTHORITIES TO READ
1. Repository AGENTS.md (branches, root freeze, completion rule).
2. The release checkpoint definition and its required artifact list.

## INVARIANTS
- Release work is verification/contraction, not a reason to add breadth.
- Public surfaces are classified: stable, namespaced, experimental,
  compatibility, remove. Stable root symbols represent supported first-class
  TFNE concepts.
- Dev, broad, release, and publication gates are distinct; a repository-wide
  non-slow suite is not the curated dev gate.
- Release/tag/push/upload happens only with explicit authorization; never
  force-push unless explicitly requested and justified.
- Exact immutable SHAs and receipts are recorded for published artifacts.

## PROCEDURE
1. Classify public surfaces: stable, namespaced, experimental, compatibility, remove.
2. Prefer deleting/merging stale docs over rewriting duplicates.
3. Keep README compact, mathematical, neutral, and positive.
4. Verify package build/install/import, API snapshot, docs, tests,
   examples/publication artifacts, artifact hashes, version metadata, and
   clean Git state as required by the release checkpoint.
5. Record exact immutable SHA and receipts for published artifacts.

## STOP CONDITIONS
- Missing explicit authorization for any remote/release operation;
  dirty repository state; gate confusion (dev vs broad vs release vs publication).

## REQUIRED VERIFICATION
- Package build/install/import, API snapshot, docs, tests,
  examples/publication artifacts, artifact hashes, version metadata,
  clean Git state, and the recorded immutable SHA.

## FORBIDDEN INFERENCES
- Releasing without authorization; force-push without explicit request;
  calling the broad suite the curated dev gate.

## COMPLETION
- Release checkpoint verified, receipts recorded, exactly the authorized
  remote operations performed.