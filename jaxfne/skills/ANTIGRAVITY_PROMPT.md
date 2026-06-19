# Antigravity prompt

Install these skills into `jaxfne/skills`, then use them to harden the repo.

## Task
Add the markdown skill files to `jaxfne/skills` with clear filenames and an index file. Keep them concise, actionable, and aligned with the existing TFNE/JAX scaffold doctrine.

## Required outcomes
- Remove silent analysis fallbacks that can mask failures.
- Replace dense connectivity construction with sparse or block-sparse logic where possible.
- Make batched simulation vectorized by default.
- Make projection normalization semantics explicit and add a density-preserving mode.
- Make runtime fallbacks transparent and strict-mode safe.
- Clarify public API contracts and eliminate accidental stub surfaces.
- Clarify parameter scope in builders and validation.
- Fence incomplete bridges and experimental solvers clearly.

## Working rules
- Follow the existing repo doctrine and keep proxy/scaffold wording intact.
- Preserve backward compatibility unless a real bug requires a wrapper.
- Do not invent APIs.
- Keep optional dependencies lazy.
- Update docs in the same change when a public function or notebook changes.
- Run tests after each meaningful edit and verify the actual result.
- Prefer sparse, vectorized, JAX-native implementations over dense Python loops.

## Suggested implementation order
1. `vis/tutorial_panels.py`
2. `core.py` connectivity and batching paths
3. `fields/proxy.py` projection semantics
4. `runtime.py` fallback transparency
5. builder and bridge contract cleanup

## Definition of done
The repo should be closer to a 100/100 enforcement state: fewer silent failures, faster large-N scaling, clearer semantics, and no public surface that looks complete but is only a stub.
