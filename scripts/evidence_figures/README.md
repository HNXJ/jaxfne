# Evidence figure generators (deliberate vis-grammar exception)

These scripts produce **one-off release and documentation figures**, not
installable simulation-signal visualization. Per `artifacts/AGENTS.md` jaxfne-modular-grammar
rule 2, direct `matplotlib` / `plotly` calls here are a **documented, maintainer-approved
exception** — they are intentionally **not** routed through `jaxfne/vis/*`.

**Step 7 re-scope (2026-07-07):** full migration of all 18 figure scripts into
`jaxfne/vis/` plotting modules remains deferred. **Progress (2026-07-07):** manifest
helpers live in `jaxfne/vis/evidence_manifest.py`; matplotlib save/close routes through
`jaxfne/vis/evidence_export.py` via `_figure_common.save_matplotlib_figure`.

**Claim posture:** computational scaffold / proxy diagnostics only; outputs are
receipt artifacts for docs and release evidence, not calibrated field readouts.
