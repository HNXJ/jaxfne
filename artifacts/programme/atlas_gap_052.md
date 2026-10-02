# Atlas gap matrix v1 — 0.5.2 second pass (items 9–13)

Authority: `artifacts/project_sources/8_atlas.md` (Y vector, inheritance
S1→S2-4→S5-7→S8-9→S10); scenario code in
`artifacts/atlas/at01_at06_052.py` (`Y_KEYS_V1`, `measure_v1`,
`gap_matrix_v1`); schema v1 keeps every v0 name stable (proven by
`tests/test_atlas_reduction_052.py::test_schema_v1_keeps_v0_names_stable`
against `at01_at10_toy.py`). Extends `atlas_gap_051.md` (v0: 20
IMPLEMENTED / 10 REFUSED / 100 OMITTED over 130 cells); this file names
each cell moved in 0.5.2. Regenerate the table from code
(`gap_matrix_v1` over `run_at01()…run_at06()`); prose here explains,
never overrides.

Human decision (0.5.2): AT-01-R4 (B beyond proxy), AT-01-R5 (explicit
calibration), AT-04-R3 (Phi→X feedback) are OUT_OF_SCOPE — no
independent evidence exists yet. The scenarios fail closed there
(REFUSED records, never substitutes).

## 0.5.2 evidence (2026-09-24, Jaxley present, dt=0.5 ms toy)

| AT | status | wall_s | headline |
|----|--------|--------|----------|
| AT-01 | OK | 13.06 | HH ref vs reduced: rate PASS (0.0 Hz diff), v_rest PASS (9.7 mV), v_peak FAIL (25.4 > 20 mV), spike-time FAIL (10.0 > 2.0 ms); B/calibration REFUSED |
| AT-02 | OK | 5.32 | delay 2.0 ms → 4 steps; absent≡0.0 bit-identical; delayed differs; superposition identity (err 4e-6); distance proxy recorded |
| AT-03 | OK | 3.88 | E↔I bidirectional pair, delay 4 steps; phase lag, cancellation index, band powers; single-mechanism grammar limitation declared |
| AT-04 | OK | 1.06 | stacked/swapped/overlap G arms move executed field; Phi→X REFUSED; G=[10,20] refused; H arm OMITTED (0.5.3) |
| AT-05 | OK | 9.55 | A_Phi over (N, executed-kappa) arms; coherent regime: ratio ≈ 1 recorded (inequality not reached, owned by 0.5.5) |
| AT-06 | OK | 0.89 | electrode chain declared; C(R,f) 6 cells in [0,1]; R=1.0 ≡ 1; assumptions labeled |

Total ≈ 34 s; every scenario far inside the 300 s budget. No scenario
OVER_BUDGET. Reduction row: HH→reduced rate PASS, v_peak FAIL (same
25.4 mV, recorded), reduced→population rate PASS (3.75 Hz), source
PASS (8% relative).

Legend: I = IMPLEMENTED (value present) · O = OMITTED (absent quantity) ·
R = REFUSED (refused capability). Never synthesized.

## Y × AT (v1, AT-01…AT-06 second pass; AT-07…AT-10 unchanged from v0)

| Y \ AT | AT-01 | AT-02 | AT-03 | AT-04 | AT-05 | AT-06 |
|---|---|---|---|---|---|---|
| X | I | I | I | I | I | I |
| H | O | O | O | O | O | O |
| W | O | O | O | O | O | O |
| Q | O | I | I | I | I | I |
| Phi_E | O | I | I | I | I | I |
| Phi_B | R | R | R | R | R | R |
| SPK | I | I | I | I | I | I |
| PSD | O | O | I | O | O | O |
| C | O | O | I | I | I | I |
| phi | O | O | I | O | O | O |
| E_reduction | I | O | O | O | O | O |
| T_compute | I | I | I | I | I | I |
| M_compute | I | I | I | I | I | I |

Counts over the six owned columns: 41 IMPLEMENTED · 6 REFUSED · 31
OMITTED (78 cells; v0 owned-column baseline was 12 / 6 / 60).

## Cells moved in 0.5.2 (29 O→I; every move names its evidence)

- AT-01: X (`run_at01` Vm/rate features), E_reduction (rate/v_peak/
  v_rest/spike-time verdicts incl. two FAILs), M_compute (host peak bytes).
- AT-02: X, Q (kernel individual-vs-superposed identity), Phi_E
  (distance-law proxy table), M_compute.
- AT-03: X, Q, Phi_E, PSD (4–12 vs 30–80 Hz band powers),
  C (dynamic cancellation index), phi (E–I phase lag), M_compute.
- AT-04: X, Q, Phi_E, C (source-alignment correlation across G arms),
  M_compute.
- AT-05: X, Q, Phi_E, C (executed kappa + coherent ratio), M_compute.
- AT-06: X, Q, Phi_E, C (C(R,f) proxy cells), M_compute.

## Why each remaining cell is what it is

- H, W = OMITTED everywhere: ownership + recording budgets arrive in
  0.5.3 (AT-07-R2, AT-04-R2).
- Phi_B = REFUSED everywhere: no calibrated Phi_B beyond the current
  proxy exists; candidate rows AT-01-R4, AT-07-R4, AT-10-R7 need
  independent evidence + human authorization (AT-01-R4/R5 confirmed
  OUT_OF_SCOPE for 0.5.2).
- Q, Phi_E = OMITTED on AT-01 only: the anchor runs HH-vs-reduced
  trajectories, no field contract on that rung.
- PSD/phi = OMITTED except AT-03: spectral/phase operators land where
  the oscillator lives; cross-area measures go to 0.5.4 (AT-08-R3).
- E_reduction = OMITTED except AT-01: the first reduction row; the
  scale matrix continues in 0.5.5.
- AT-04 Phi→X stays REFUSED-by-candidate (AT-04-R3 OUT_OF_SCOPE); the
  H-perturbation arm is OMITTED (0.5.3 owns AT-04-R2).

## Burndown (what later releases still own)

- 0.5.3: H, W (+trajectories/budgets), AT-07 column; AT-04 H arm.
- 0.5.4: C, phi area-indexed + cross-area, AT-08/AT-09 columns.
- 0.5.5: schema frozen; E_reduction scale matrix (incl. the coherent-
  regime inequality question); AT-10 column; every REFUSED cell either
  promoted via the candidate path or marked OUT_OF_SCOPE by the human.
