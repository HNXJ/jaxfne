# Runtime Fallback Transparency Skill

## Purpose
Prevent silent degradation in JIT, VMAP, and precision helpers.

## Rules
- Fallbacks must report why they were triggered.
- Strict mode must fail instead of silently degrading to a slower or less faithful path.
- Precision-policy helpers must report the effective state, not only the requested state.
- Notebook convenience wrappers may exist, but they must not hide real tracing or precision failures.
- A fallback must never look like a successful optimization.

## Acceptance checks
- A failed trace or unsupported compilation path produces an explicit reason.
- Tests cover both the strict and fallback paths.
- Runtime helpers cannot masquerade as successful optimization when they did not optimize.
