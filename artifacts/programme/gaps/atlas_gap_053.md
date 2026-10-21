# Atlas gap matrix v2 — 0.5.3 third pass (items 8–10)

Authority: `artifacts/project_sources/8_atlas.md` (Y vector, inheritance
S1→S2-4→S5-7→S8-9→S10; OMITTED/REFUSED never synthesized; B is
observation-only; proxy != calibrated; bounded != stable); scenario code
in `artifacts/atlas/at07_at04_053.py` (`Y_KEYS_V2`, `measure_v2`,
`gap_matrix_v2`); schema v2 keeps every v1 name stable (proven by
`tests/test_atlas_at07_053.py::test_schema_v2_keeps_v1_and_v0_names_stable`
against `at01_at06_052.py` and `at01_at10_toy.py`). Extends
`atlas_gap_052.md` (v1: 41 IMPLEMENTED / 6 REFUSED / 31 OMITTED over the
six owned columns AT-01…AT-06); this file names each cell moved in
0.5.3. Regenerate the 0.5.3 columns from code (`gap_matrix_v2` over
`run_at07()` + `run_at04r2()`); prose here explains, never overrides.

No v0/v1 names or semantics changed (additive only).

Human decision outstanding: AT-07-R4 (B as input to dynamics,
candidate:true) is defer-or-promote — recorded REFUSED in every 0.5.3
output, never executed. AT-01-R4/R5, AT-04-R3 stay OUT_OF_SCOPE (0.5.2).

## 0.5.3 evidence (2026-09-25, suite2_net1 n=8, dt=0.5 ms, 1000 ms model-time)

| AT | status | wall_s | headline |
|----|--------|--------|----------|
| AT-07 | OK | 8.5 | fixed==w0 bit-exact; Hebbian dw 0.03; noisy differs + reproduces; clamp pinned f32-exact, unclamped moves; 6/6 declared PASS; H/W trajectories + stride-5 budget kept==full; B-as-input REFUSED |
| AT-04R2 | OK | 1.7 | same W0 verified; H0 1.0→0.0: 114→61 spikes (causal); disabled control X bit-identical (H effect needs HDP-gated path); W bit-fixed on control pair; 3/3 declared PASS; Phi→X REFUSED; H-freeze recorded unavailable |

Per-item wall far inside the 600 s budget; no scenario OVER_BUDGET.
Performance envelope (0.5.1 acceptance): the 0.5.1 matrix base cell
(100 neurons / 100 ms) measures ≈5.2 s wall (`matrix_051.json`); the
0.5.3 runs are n=8 toy size at 1000 ms model-time for a ≈10 s total —
inside the envelope by size, no overrun to record.

Legend: I = IMPLEMENTED (value present) · O = OMITTED (absent quantity) ·
R = REFUSED (refused capability). Never synthesized.

## Y × AT (v2: AT-01…AT-06 carried from v1, AT-07 new, AT-04 H/W moved)

| Y \ AT | AT-01 | AT-02 | AT-03 | AT-04 | AT-05 | AT-06 | AT-07 |
|---|---|---|---|---|---|---|---|
| X | I | I | I | I | I | I | I |
| H | O | O | O | I | O | O | I |
| W | O | O | O | I | O | O | I |
| Q | O | I | I | I | I | I | I |
| Phi_E | O | I | I | I | I | I | I |
| Phi_B | R | R | R | R | R | R | R |
| SPK | I | I | I | I | I | I | I |
| PSD | O | O | I | O | O | O | O |
| C | O | O | I | I | I | I | I |
| phi | O | O | I | O | O | O | O |
| E_reduction | I | O | O | O | O | O | O |
| T_compute | I | I | I | I | I | I | I |
| M_compute | I | I | I | I | I | I | I |

Counts over all seven columns: 52 IMPLEMENTED · 7 REFUSED · 32
OMITTED (91 cells; v1 owned-column baseline was 41 / 6 / 31 over AT-01…AT-06).

## Cells moved in 0.5.3 (8 O→I; every move names its evidence)

- AT-07: H (`run_at07` H trajectory + final under budgets), W (w
  trajectory + final; fixed/clamp bit-exact, plastic moves), Q (source
  representation per arm), Phi_E (lfp_proxy at RELATIVE_PROXY per arm),
  C (executed kappa per arm), M_compute (host peak bytes).
  X/SPK/T_compute already I in the v0 toy pass; PSD/phi/E_reduction
  stay O (owners below); Phi_B stays R (AT-07-R4 candidate).
- AT-04: H + W via the R2 arm (`run_at04r2`: H0 perturbation with
  H trajectory, W observed fixed on the control pair). Rest of the
  AT-04 column unchanged from v1.

## Why each remaining cell is what it is

- H, W = OMITTED on AT-01/02/03/05/06: H/W ownership + trajectories
  land where the 0.5.3 arms run (AT-07, AT-04-R2); other columns keep
  their v1 owners.
- Phi_B = REFUSED everywhere: no calibrated Phi_B beyond the current
  proxy exists; AT-07-R4 needs independent evidence + human
  authorization (defer-or-promote pending).
- PSD/phi = OMITTED except AT-03: spectral/phase operators stay where
  the oscillator lives; cross-area measures go to 0.5.4 (AT-08-R3).
- E_reduction = OMITTED except AT-01: the scale matrix continues in
  0.5.5 (incl. the coherent-regime inequality question).
- AT-04 Phi→X stays REFUSED-by-candidate (AT-04-R3 OUT_OF_SCOPE).
- No adaptation phenotype is claimed on AT-07 (acceptance needs none);
  boundedness is reported as finiteness only (bounded != stable).

## Burndown (what later releases still own)

- 0.5.4: C, phi area-indexed + cross-area, AT-08/AT-09 columns.
- 0.5.5: schema frozen; E_reduction scale matrix; AT-10 column; every
  REFUSED cell either promoted via the candidate path or marked
  OUT_OF_SCOPE by the human (AT-07-R4 decision due).
