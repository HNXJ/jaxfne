# Atlas gap matrix v3 — 0.5.4 fourth pass (items 6–8)

Authority: `artifacts/project_sources/8_atlas.md` (Y vector, inheritance
S1→S2-4→S5-7→S8-9→S10; OMITTED/REFUSED never synthesized; B is
observation-only; proxy != calibrated; bounded != stable); scenario code
in `artifacts/atlas/at08_at09_054.py` (`Y_KEYS_V3`, `measure_v3`,
`gap_matrix_v3`); schema v3 keeps every v1/v2 name stable and in order
(proven by `tests/test_atlas_at0809_054.py::test_schema_v3_keeps_v1_and_v2_names_stable`).
Extends `atlas_gap_053.md`; this file names each cell moved in 0.5.4.
Regenerate the 0.5.4 columns from code (`gap_matrix_v3` over
`run_at08()` + `run_at09()`); prose here explains, never overrides.

No v1/v2 names or semantics changed (additive only: 10 area-indexed
and cross-area cells appended).

Human decision outstanding: none new in 0.5.4. AT-07-R4 (B as input to
dynamics, candidate:true) stays REFUSED; AT-01-R4/R5, AT-04-R3 stay
OUT_OF_SCOPE (0.5.2).

## 0.5.4 evidence (two columns n=8, dt=0.5 ms, 1000 ms model-time, repeated A1 pulses)

| AT | status | wall_s | headline |
|----|--------|--------|----------|
| AT-08 | OK | 9.6 | same network, 3 arms; A1 driven (122 spikes all arms); downstream A2: fixed 67, local-only 67, full 73 — propagated spike delta 0 with fixed coupling, +6 from coupling plasticity; propagated field delta 0.0095 without spike-count change (the R4 dissociation); 3/3 declared PASS; Phi_B REFUSED |
| AT-09 | OK | 11.1 | cross-only arm: cross W moves (max 1.48), members bit-frozen; member-only arm mirrors; frozen arm: dW = 0 while H deviates 0.0094 (R2 separation); C_12 0.50/0.45/0.45 across arms (R3); 2/2 declared PASS |

Per-scenario wall far inside the 600 s budget; no scenario OVER_BUDGET.
Performance envelope (0.5.1 acceptance): toy two-area runs at 1000 ms
model-time complete in ~10 s wall — inside the envelope by size
(scaling_054.json records the k-area scaling probe separately), no
overrun to record.

Legend: I = IMPLEMENTED (value present) · O = OMITTED (absent quantity) ·
R = REFUSED (refused capability). Never synthesized.

## Y × AT (v3: AT-08 and AT-09 new; prior columns unchanged)

| Y \ AT | AT-08 | AT-09 |
|---|---|---|
| X | I | I |
| H | I | I |
| W | I | I |
| Q | I | I |
| Phi_E | I | I |
| Phi_B | R | R |
| SPK | I | I |
| PSD | O | O |
| C | I | I |
| phi | I | I |
| E_reduction | O | O |
| T_compute | I | I |
| M_compute | I | I |
| SPK_A1 | I | I |
| SPK_A2 | I | I |
| H_A1 | I | I |
| H_A2 | I | I |
| Phi_A1 | I | I |
| Phi_A2 | I | I |
| W_12 | I | I |
| W_21 | I | I |
| C_12 | I | I |
| dphi_12 | I | I |

Ownership-scoped W trajectories (per-projection masks) are exercised
in AT-09; AT-08 records the same owned ranges without mask scoping on
the full-adaptation arm.
