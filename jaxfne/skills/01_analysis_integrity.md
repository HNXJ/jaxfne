# Analysis Integrity Skill

## Purpose
Prevent fabricated analysis outputs from masking upstream failures.

## Rules
- Never synthesize a fallback spectrum, trace, summary, or diagnostic that can be mistaken for a real result.
- If a computation fails in strict mode, raise immediately.
- If synthetic output is intentionally requested, label it synthetic in the API name, docstring, and rendered output.
- Separate data preparation from rendering so plotting code cannot silently repair broken analysis.

## Acceptance checks
- A failed upstream analysis cannot produce a visually valid but false diagnostic.
- Strict-mode tests fail loudly when spectral or trace computation fails.
- Every synthetic fallback is explicit and opt-in.
