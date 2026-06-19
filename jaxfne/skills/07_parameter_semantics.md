# Parameter Semantics Skill

## Purpose
Remove ambiguity from builder and model parameters.

## Rules
- Any scalar parameter with multiple plausible meanings must state whether it is per-area, per-layer, per-column, or global.
- Reject ambiguous usage in validation rather than guessing.
- Every tunable parameter should have a default when sensible; required data inputs remain required.
- Public docstrings should explain the effect of the parameter on the resulting model or artifact.

## Acceptance checks
- Builder signatures are self-describing.
- The meaning of `n`, counts, densities, and layer fractions is explicit.
- Tests fail when a parameter is used with the wrong scope.
