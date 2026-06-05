# 08_RISKS_AND_FRAGILITIES

|fragile spot|evidence|failure mode|smallest safe mitigation|
|---|---|---|---|
|runtime/module wrapper|`jaxfne/__init__.py` uses `sys.modules` wrapper swap in ZIP; live audit flags sys leak|public import oddities, namespace leaks, fragile cleanup|avoid coupled changes; if touched, import/API smoke + wrapper tests|
|objective null RNG (RESOLVED)|was `objectives.py:118,133,141,149` unseeded `np.random`|non-reproducible null statistics|✅ FIXED in B01/PR#22 — explicit keyword-only `rng` + dispatcher/factory `null_seed`; shared-generator ordering caveat documented|
|LIF/GLIF stubs|live audit: exported but loud NotImplementedError|public surface appears broader than implemented|document planned status or deprecate carefully; do not remove casually|
|JIT recompilation|live audit: 2 opt-in tests N_compile=2|compile overhead, unstable static arg boundary|reproduce then stabilize signatures; compare traces|
|dist cleanup zsh glob|release receipt: bare `*.egg-info` nomatch preserved stale wheels|wrong twine/dist validation|find-based cleanup|
|Etude long cells|known audit warnings: cell #6 13 lines, #32 397 lines|maintenance risk, hard review|only split when execution-equivalent|
|unsupported emitter family fallback|fixed by PR #21 in live audit|silent wrong emitter run if regressed|keep regression tests in release blocker set|
|physical-claim overreach|many docs discuss future solvers/EEG/MEG|reviewer misunderstanding if proxy label omitted|truth-gate lint + status tables|


## Guardrails

- Touch one fragile spot per PR.
- Prefer fail-loud errors over silent fallbacks.
- Preserve public APIs unless a breaking cleanup is explicitly approved.
- Optional deps stay lazy.
- Re-run focused tests and full suite if import/runtime wrappers are touched.
