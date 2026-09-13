# 24-API-01 receipt — public-surface classification (conservative)

**Branch:** `dev`. Compatibility is the invariant: no removals.

## Measurement: concepts vs names
190 public names = **187 distinct concepts**: exactly 3 true alias pairs
(same object): `Config`/`Configuration`, `Model`/`Net`, `Signal`/`Signals`.
No hidden duplicates beyond these (factory/class case-pairs like
`simulation`/`Simulation`, `objective`/`Objective` are distinct concepts).

## Classification of all 190
- KEEP (185): every name has external use in tests/docs/scripts/examples
  (zero names with zero external references); distinct useful concepts.
- PREFERRED + COMPAT_ALIAS (3 pairs): aliases carry heavy documented use
  (`Config` 202 files, `Signals` 67, `Net` 48) — removal would break
  legitimate users; aliases stay.
- DEPRECATE, already declared (2): `save_figure`/`save_figures` →
  documented replacements `jaxfne.vis.export_figure(s)` exist and are used;
  old paths still pass via compat test `test_root_export_api_v0338.py`.
  Removal = HUMAN_DECISION (breaks declared compat + downstream), not taken.
- REMOVE_INTERNAL_ONLY (0): none found — all 190 referenced externally.
- HUMAN_DECISION surfaced (0 new): the only removal candidates are the two
  already-deprecated exports; decision deferred to a human release call.

## Gates
`test_api_smoke` + `test_root_export_api_v0338`: 14/14 green (deprecated
paths verified working, replacements present).

## Verdict
Correct near-NO_CHANGE: the surface is already minimal under compatibility
(187 concepts, 0 internal-only exposures, deprecations already declared
with replacements). Symbol count is not a target.
