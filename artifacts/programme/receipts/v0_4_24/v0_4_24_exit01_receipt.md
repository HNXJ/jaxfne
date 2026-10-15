# 24-EXIT-01 receipt — hostile completion check (feature freeze held)

**Branch:** `dev` @ `81e0325` (== `origin/dev`, Gate 0 PASS). No source
changes since the 3875-green broad gate (receipt/stack only). No design
work performed under EXIT; every check below attempts to disprove
completion.

## Goal-by-goal verdict (todo_stack v0.4.24 items == declared goals; no
local roadmap file resolves "§5.2" beyond this list)
- MIN/DOC/CODEMIN/TESTMIN/PKG/API: closed, receipts present, items removed.
- PRO-01: closed; 11 findings worked, 5 trade-offs dispositioned below.
- HDP-GEN-01: closed; zero ENGINE_GAP in-grammar, bounded claim receipted.
- AUDIT-01: OPEN by design (external party) — recorded, not blocking dev.

## Disproof attempts and results
- P0/P1/P2 hunt: issue log is v0.4.17-era DOC/FRICTION only (I-001..I-014,
  all MINOR/should-fix/docs-process, none core); `TODO|FIXME|XXX|HACK`
  in `jaxfne/`: zero defect markers (only paradigm labels + history note).
- ENGINE_GAP re-sweep: two critic rounds, second constrained to declared
  grammar → 0 gaps; boundaries (threshold/gain targets, exogenous ports,
  raw multi-tap, structural plasticity) explicitly unclaimed in GEN receipt.
- Equivalence: `test_equiv01_table` green (32-test spot re-run green).
- Continuation/RNG/delay/HDP: broad green incl. noisy/delayed/registered.
- Representation/memory: REC-01/W16-6/streaming tests green in broad.
- Package/surface: wheel excludes in force, 190 symbols, vocab + orphans
  green in broad.
- Undone simplifications: none — critic SUPERIOR findings all worked;
  REJECTED carry file:line evidence in PRO receipt.

## The five PRO trade-offs (genuine, none blocking a declared goal)
- `jnp.clip` monkey-patch (optional Jaxley): ACCEPT — removal breaks a
  tested interop path; scoped to first use.
- Host sync per continuation segment: ACCEPT — supports exotic step_index
  leaves; no measured segment-cost defect.
- Trace-time G_history size warn: ACCEPT, reclassified intentional —
  documented OOM guardrail with measured threshold, not noise.
- Null-HDP test pair: ACCEPT keep both — cheap, speed-vs-strength recorded.
- `build_laminar_column` naming + EXPERIMENTAL_INTERNAL in wheel: ACCEPT —
  compat aliases (48–202 files); sanity modules import stdlib+numpy+jax
  only, root-import heaviness fenced by
  `test_sanity_delta_optional_imports`.

## Candidate for VERIFY
`81e0325`, verified-indistinguishable from broad-gate state. Release-
mutating steps (tag, main merge, PyPI, RTD) require separate authorization
per AGENTS.md and are NOT taken here.
