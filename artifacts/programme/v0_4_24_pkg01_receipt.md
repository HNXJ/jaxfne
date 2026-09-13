# 24-PKG-01 receipt — installed-package boundary audit

**Branch:** `dev`. Wheel membership ≠ repository membership.

## Ownership chains (import owner → public use → runtime dep → location → wheel)
- `protocol_c` / `protocol_d_biological_rbs` / `protocol_e_integration` /
  `experiment_a` → RESEARCH_PROTOCOL: owner = their own tests + frozen
  receipts; zero importers outside their trees; zero public symbols →
  repo-kept, wheel-excluded.
- `h3_decodability` / `h4_matrix` / `w1a` / `w1b` / `w2` / `w3` / `w3a` /
  `w3b` → HISTORICAL_EVIDENCE (fig06/publication evidence chain): same
  ownership shape → repo-kept, wheel-excluded.
- `experimental_hpc` → RUNTIME_REQUIRED: `core` / `_model` / `_signals`
  import its contracts at load; `validation` uses the field solver → stays.
- `tutorial_utils` / `bridges` / `analysis` → PUBLIC_USER_FEATURE:
  eagerly imported by `jaxfne/__init__`, root re-exports, documented →
  stay. (`export.export_tutorial_artifacts` also lazily needs tutorial_utils.)
- `vis/` → PUBLIC_USER_FEATURE, deliberately lazy (zero-graphics-overhead
  gate): stays; not moved for size.
- `publication/` → already wheel-excluded (empty dir); unchanged.

## Before / after (hatchling 1.29.0, pinned)
- Wheel bytes: 729,731 → 604,421 (−125,310, −17.2%).
- Wheel files: 174 → 126; installed py modules: 159 → 111 (−48).
- Installed LOC: −10,893 (protocol dirs 7,850 + h/w top files 3,043).
- sdist bytes: 38,040,322 — retains ALL excluded modules (reproducibility).
- Public symbols: 190, unchanged. Import behavior: `import jaxfne` unchanged.

## Gates
- Clean-install smoke (pruned wheel, `--target` dir): version 0.4.23,
  190 symbols, all 6 excluded modules absent, retained user features
  import, 12-cell construct→simulate→V_m(100,12) green.
- Repo tests (files untouched, still executable): 47/47 across api smoke,
  hpc contracts, identity, protocol_c c0, experiment_a b0, jaxley
  optional, kappa synchrony.

## Mechanism
`pyproject.toml` wheel `exclude` += 4 packages + 8 modules, with rationale
comment. No code moved; no sdist change; PRO-01 owns repo-side placement.
