# Evidence figure generators (deliberate vis-grammar exception)

These scripts produce **one-off release and documentation figures**, not
installable simulation-signal visualization. Per `AGENTS.md` jaxfne-modular-grammar
rule 2, direct `matplotlib` / `plotly` calls here are a **documented, maintainer-approved
exception** — they are intentionally **not** routed through `jaxfne/vis/*`.

**Step 7 re-scope (2026-07-07):** migrating all 18 figure scripts into
`jaxfne/vis/` is deferred to step 8+ (large refactor). This directory remains
the canonical home for evidence/release figure generation until that migration
is explicitly scheduled.

**Claim posture:** computational scaffold / proxy diagnostics only; outputs are
receipt artifacts for docs and release evidence, not calibrated field readouts.
