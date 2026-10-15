# Atlas gap matrix v0 — 0.5.1 toy first pass (items 6+7)

Authority: `artifacts/project_sources/8_atlas.md` (Y vector, inheritance
S1→S2-4→S5-7→S8-9→S10); schema code in `artifacts/atlas/at01_at10_toy.py`
(`Y_KEYS`, `measure`, `gap_matrix`). This file is the burndown for
0.5.2–0.5.5: each release names the cells it moves. Regenerate the table
from code (`gap_matrix(run_all())`); prose here explains, never overrides.

Toy evidence (2026-09-24, Jaxley 0.14.0 present, dt=0.5 ms toy):

| AT | status | wall_s | SPK observed |
|----|--------|--------|--------------|
| AT-01 | OK (Jaxley bridge + reduced neuron) | 8.25 | [10,1] n=1; HH ref 12 steps, V −70.0…−59.7 mV |
| AT-02 | OK (2-neuron driven column) | 3.72 | [20,2] n=2 |
| AT-03 | OK (4-neuron E/I mix) | 3.26 | [20,4] n=3 |
| AT-04 | OK (drive_lo + drive_hi arms) | 0.27 | [20,2] n=2 (hi arm) |
| AT-05 | OK (8-neuron column) | 4.03 | [40,8] n=9 |
| AT-06 | OK (8-neuron + electrode probes) | 1.27 | [40,8] n=8 |
| AT-07 | OK (fixed-W pair, seeds 7/11) | 0.35 | both arms extracted |
| AT-08 | OK (2 areas × 2 neurons) | 2.06 | [20,4] n=3 |
| AT-09 | OK (2 areas × 2 neurons, fixed coupling) | 0.16 | [20,4] n=3 |
| AT-10 | OK (3 areas × 2 neurons composition) | 3.77 | [20,6] n=6 |

Total ≈ 27 s; every scenario far inside the 120 s budget. No scenario
recorded REFUSED (Jaxley present) or OVER_BUDGET.

Legend: I = IMPLEMENTED (value present) · O = OMITTED (absent quantity) ·
R = REFUSED (refused capability). Never synthesized.

## Y × AT (v0)

| Y \ AT | AT-01 | AT-02 | AT-03 | AT-04 | AT-05 | AT-06 | AT-07 | AT-08 | AT-09 | AT-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| X | O | O | O | O | O | O | O | O | O | O |
| H | O | O | O | O | O | O | O | O | O | O |
| W | O | O | O | O | O | O | O | O | O | O |
| Q | O | O | O | O | O | O | O | O | O | O |
| Phi_E | O | O | O | O | O | O | O | O | O | O |
| Phi_B | R | R | R | R | R | R | R | R | R | R |
| SPK | I | I | I | I | I | I | I | I | I | I |
| PSD | O | O | O | O | O | O | O | O | O | O |
| C | O | O | O | O | O | O | O | O | O | O |
| phi | O | O | O | O | O | O | O | O | O | O |
| E_reduction | O | O | O | O | O | O | O | O | O | O |
| T_compute | I | I | I | I | I | I | I | I | I | I |
| M_compute | O | O | O | O | O | O | O | O | O | O |

Counts: 20 IMPLEMENTED · 10 REFUSED · 100 OMITTED (130 cells).

## Why each non-implemented cell is what it is

- X: trajectory extraction not in toy pass → 0.5.2 inspection (item 6).
- H, W: ownership + recording budgets → 0.5.3 (AT-07-R2, AT-04-R2).
- Q, Phi_E: single source representation + field contract with epistemic
  level → 0.5.2 (AT-01-R2/R3, AT-02-R2, AT-06-R2/R4).
- Phi_B = REFUSED everywhere: no calibrated Phi_B beyond the current proxy
  exists in 0.5.1; candidate rows AT-01-R4, AT-07-R4, AT-10-R7. Needs
  independent evidence + human authorization before any engine work.
- PSD, C, phi: spectral / locality / phase operators → 0.5.2 (AT-03-R2,
  AT-05-R2, AT-06-R3) and cross-area measures → 0.5.4 (AT-08-R3).
- E_reduction: reduction rows with predeclared tolerances → 0.5.2
  (AT-01-R6) through 0.5.5 scale matrix.
- M_compute: measured in the 0.5.1 benchmark matrix (ENGINE item 1), not in
  the toy pass; wired into Y when schema goes v0 → v1.
- AT-04 Phi→X stays REFUSED-by-candidate (AT-04-R3, fails closed); the toy
  ran the correlation arms only. AT-07 plastic arms and AT-09 plastic
  coupling are OMITTED (0.5.3 / 0.5.4 own them); AT-10 genome development
  is OMITTED (0.5.5 owns AT-10-R1…R6).

## Burndown (cell moves each release owns)

- 0.5.2: X (inspection), Q, Phi_E, PSD, E_reduction (first reduction row),
  AT-01…AT-06 columns; M_compute wired in.
- 0.5.3: H, W (+trajectories/budgets), AT-07 column; AT-04 H arm.
- 0.5.4: C, phi area-indexed + cross-area, AT-08/AT-09 columns.
- 0.5.5: schema frozen; E_reduction scale matrix; AT-10 column; every
  REFUSED cell either promoted via the candidate path or marked
  OUT_OF_SCOPE by the human.
