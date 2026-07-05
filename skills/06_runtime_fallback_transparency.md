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

## Verified case in this repo (2026-07-05)

`jaxfne/runtime.py::safe_jit`/`safe_vmap` docstrings claimed to "fall back to
eager execution" on failure, but `jax.jit`/`jax.vmap` are lazy -- the
try/except only ever catches wrap-time failures (e.g. a malformed
`static_argnums`), which almost never happen; a function that traces but
fails on shape/control-flow propagates uncaught on the *first call* of the
returned wrapper, not caught by this guard at all. Fixed via docstring
narrowing (not a lazy-rewrap rewrite -- both functions have zero call sites
anywhere in this repo, so there was no live behavior to regression-test
against; rewriting the mechanism itself would have been unverifiable risk
for no live benefit). The corrected docstrings now state plainly what is and
isn't caught, so a future caller doesn't rely on a guarantee that was never
real.

Separately, `jaxfne/validation.py::CompilationRegistry.track_trace`
(the `N_compile <= 1` recompilation guard) was flagged as counting every
Python call as a "compile" rather than genuine XLA trace events. Investigated
against real usage rather than trusting the original synthetic reproduction
(which called the guard's output directly, bypassing `jax.jit`): all 5 real
call sites in `jaxfne/_model.py` wrap the guard's output with `jax.jit(...)`
*before* ever calling it, and a faithful reproduction of that pattern showed
the counter is correct (JAX's own internal cache prevents re-executing the
traced Python body on a cache-hit call). Conclusion: not a live bug in this
codebase's actual usage, only a footgun if a future call site skips the
`jax.jit(...)` wrap -- fixed via a docstring warning, not a mechanism
rewrite, since rewriting working machinery to guard against a hypothetical
misuse is a worse trade than documenting the real constraint.
