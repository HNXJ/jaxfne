# Izhikevich 2003 P3b result: STOPPED at the equivalence gate

`PREREG_P3b.md` signed v4, gate amendment v5 (Hamm, 2026-10-13). No scored run took place. P3
(signed v7) keeps its verdict.

## v4 gate: FAIL (`scores/p3b_gate_v4.json`, code `9ff85e7f` plus the uncommitted arm)

| Step | First divergent row | Counts (ref / kernel) | Bar |
|---|---|---|---|
| 1 (50 cells, thal 6) | 270 | 954 / 958 | rounding-flip bar fails |
| 2 (full net) | 615 | 7421 / 7445 | extra u at rtol 1e-9 before row 615 fails |

The reference fails the same bar against itself when thal moves by one ulp: step 1 diverges at row
310 (counts 954 / 965), and v differs by more than 1e-9 from row 71. That is the basis of v5.

## v5 gate: FAIL (`scores/p3b_gate_v5.json`, code `9b6b4a77` in the work clone; integrated as `87353ec9`)

| Step | R | Rasters before R | Extra u before R (rtol 1e-6) | Counts within 1 % | Verdict |
|---|---|---|---|---|---|
| 1 | 157 | identical | agrees | 954 / 958 | PASS |
| 2 | 446 | identical | fails | 7421 / 7445 | FAIL |

Per the amendment, a v5 failure stops P3b.

## Diagnostic (unscored)

Step 2, rows before R, 1803 spikes with extra u:
- 1-ulp control against the reference: max relative difference 1.4e-8, none over 1e-6.
- Kernel against the reference: max 3.0e-5, 5 over 1e-6, the worst at row 445, cell 800.

The JAX arm drifts from the NumPy reference faster than a one-ulp input change does. A likely source
is the summation order of `S @ fired` over up to 1000 terms (inferred, untested). The S1 reference
arm result (`RESULT_P3.md`) stands as the answer to the update-order question: N3 fails on every seed
under `net.m`'s order in the NumPy transcription.
