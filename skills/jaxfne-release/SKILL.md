---
name: jaxfne-release
description: Stable-surface contraction, documentation freeze, packaging, and authorized release work.
---
# jaxfne release procedure

1. Treat release work as verification/contraction, not a reason to add breadth.
2. Classify public surfaces: stable, namespaced, experimental, compatibility, remove. Stable root symbols must represent supported first-class TFNE concepts.
3. Prefer deleting/merging stale docs over rewriting duplicates. README remains compact, mathematical, neutral, and positive.
4. Separate dev, broad, release, and publication gates. Do not call a repository-wide non-slow suite the curated dev gate.
5. Verify package build/install/import, API snapshot, docs, tests, examples/publication artifacts, artifact hashes, version metadata, and clean Git state as required by the release checkpoint.
6. Release/tag/push/upload only with explicit authorization; never force-push unless explicitly requested and justified.
7. Record exact immutable SHA and receipts for published artifacts.
