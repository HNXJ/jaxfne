# Antigravity prompt for jaxfne skills

Install the markdown skills from this bundle into `jaxfne/skills/` as flat root-level files.
Do not recreate nested skill folders inside the bundle or in the repo copy.

## Goal
Make the skills sufficient for both repo usage and repo understanding, and make them strong enough to enforce the review findings rather than merely describe them.

## Required skill set
- repository orientation and glossary discipline
- mandatory objective grammar discipline
- analysis integrity
- sparse connectivity
- batch-first simulation
- projection semantics
- runtime fallback transparency
- API contracts
- parameter semantics
- experimental fencing

## Working rules
- follow the repo doctrine and keep proxy/scaffold wording intact unless code and tests prove a stronger claim
- never invent APIs
- keep optional dependencies lazy
- preserve backward compatibility unless a real bug requires a wrapper
- update docs in the same change when a public function or notebook changes
- run tests after each meaningful edit and verify the actual result
- prefer sparse, vectorized, JAX-native implementations over dense Python loops
- make failures visible rather than silently repaired

## Repo hardening targets
1. remove silent analysis fallbacks that mask failures
2. avoid dense O(N^2) connectivity construction by default
3. make repeated simulations vectorized on the common path
4. expose projection normalization semantics explicitly
5. make runtime fallbacks transparent and strict-mode safe
6. clarify public API contracts and eliminate accidental stub surfaces
7. clarify parameter scope in builders and validation
8. fence incomplete bridges and experimental solvers clearly

## Definition of done
The repo should be easier to use, easier to understand, and harder to misuse.
A skill pass should make the repo safer, faster, and more semantically explicit rather than merely more verbose.
