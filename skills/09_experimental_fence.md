# Experimental Fence Skill

## Purpose
Keep incomplete bridges, solvers, and compatibility layers honest.

## Rules
- Experimental modules must be labeled as such in name, docstring, and docs.
- Incomplete physics/solver/bridge paths must not be described as stable or validated.
- The repo should preserve a clean boundary between proxy scaffolds and future physical or empirical work.
- If a feature is incomplete, fence it rather than letting it look finished.
- A placeholder is acceptable only when the fence is explicit and tested.

## Acceptance checks
- Experimental paths are clearly separated from stable package APIs.
- Tests and docs agree on current status.
- No stub is presented as a validated implementation.
