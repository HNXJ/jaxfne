# E2 Whole-Epoch Scientific Synthesis (2026-08-25, HEAD d84fa80)

## Final E2 outcomes

| Arm | Verdict | Subclass | Independent rescoring |
|---|---|---|---|
| V1 PING (7 arms x 5 seeds @ theta*) | NEGATIVE | NEGATIVE_NOT_PING_LIKE | 0/5 PING_LIKE frozen-only, separate implementation (`v1_rescored_frozen_only.json`) |
| V2 SSA (20 reps x 5 blocks @ theta*) | NEGATIVE | FAIL_SI_gate + FAIL_swap_asymmetry + SIGN_deviant_below_standard | agreement (`v2_rescored_frozen_only.json`) |
| S3 recovery (V2 sub-gate) | not passed | delta_rec +4.0 Hz p<0.001 BUT rho = -0.77 (inverted ladder), I_rec guard fails — reported as-is | — |
| S4 mechanism matrix | CONFIRMATORY_DEFERRED_v3 | per factor_staging in spec chain v2+ | n/a |

## Coherent mechanistic story across V1+V2 (diagnosis, not repair)

Both arms characterize the SAME regime: at theta*, the 1000-neuron E/I circuit is a
~7 Hz globally synchronous population-pulse system (participation ~1.0, FF~0) with
E-leading-I timing (+6.8 ms phase / +16 ms pulse pairs). In V1 the gamma-band spectral
"peaks" (36.1 / 43.3 Hz) are harmonic comb teeth of that slow rhythm; PING gates fail on
multiple independent dimensions (prominence gray band, negative AC sidepeak at gamma
period, cycle count, dphi outside window). In V2, stimulus identity (spatial pattern A vs
B on disjoint E subpopulations) produces strongly asymmetric responses (swap asymmetry
0.426 >> 0.10) and the SI sign is INVERTED (deviant below standard, pooled -0.084,
BCa [-0.108,-0.034]) — pattern-selective attenuation, not deviance detection.

## What this epoch establishes (bounded claims)

1. The preregistered PING-like regime was NOT found at theta* under frozen conjunctive
   criteria, robust to executor defects and to relaxed/frozen-only rescoring.
2. No stimulus-specific adaptation (SSA) was found; identity-driven attenuation dominates.
3. Both negatives are load-bearing: falsification discipline preserved end-to-end
   (INVALID/UNRESOLVED never collapsed into negatives; negatives never tuned).
4. Calibration honesty: E2a adequacy was synthetic-proxy and six-way tied (theta*
   lexicographic); confirmatory adequacy V1 5/5, V2 G_A/G_B all-pass.

## Executor-defect disclosure for Supplement

Executor defects occurred in V1 (direction/units prose, vacuous collapse key via
'dphi' vs 'dphi_deg', four invented conjuncts, salted hash(), undeclared consumed keys)
and were repaired as infrastructure before V2 execution (R-A..R-D rules promoted;
e2_exec_lib JSON-generated gates, typed units, fold_in/splitmix64 seeds, jit/DCE
memory-safe blocks, atomic resumable receipts). Primary V1 negative survived removal of
all defective criteria; V2 executed only after LFNI gate + adversarial code review passed.

## Manuscript boundary sentences (verbatim-ready)

- V1: "In the preregistered confirmatory arm (V1: 7 arms x 5 seeds at the tie-break-selected
  operating point theta*), networks met adequacy gates in 5/5 seeds but satisfied none of the
  four frozen PING-classifier criteria in any seed (verdict NEGATIVE_NOT_PING_LIKE), instead
  exhibiting a ~7 Hz globally synchronous population-pulse rhythm whose gamma-band spectral
  peaks fall below the frozen prominence gate; ... these results therefore support only the
  bounded statement that no PING-like signature was detected at theta* under the frozen
  criteria, carry no implication about other regions of the parameter space, and are
  independent of the CL-06 NO_WAVE result."
- V2: "In the confirmatory SSA arm (V2: 20 replicates x 5 blocks under e2_ssa_spec.v6),
  networks met adequacy gates in 20/20 replicates but showed no stimulus-specific adaptation:
  pooled SI = -0.084 (BCa [-0.108, -0.034], deviant below standard) with role-reversal
  asymmetry 0.426 >> 0.10 (frozen S2 falsifier), i.e., spatial-identity attenuation rather
  than deviance detection; verdict NEGATIVE with failure taxonomy retained."
- Joint: "Across both preregistered phenotypes, the TFNE scaffold at theta* supports
  synchronized population pulsing with pattern-selective attenuation, and neither PING-like
  gamma nor SSA is licensed at this operating point; these results are independent of, and do
  not alter, any E1 claim including CL-06 NO_WAVE."

## Disposition recommendation

Supplement-first placement (both arms negative; methods-level interest in the harness
discipline and the pulse-regime characterization). Main-text entry would require a new
separately declared experiment (new development/confirmation split) targeting a
PING-capable or SSA-capable regime — explicitly NOT a repair of these preregistered
negatives.

## Open items carried forward

- e2_ssa_spec.v3 mechanism-matrix (S4, 8 cells) remains deferred; requires its own staged
  freeze if pursued.
- JDNA define/inherit compiler concept stays deferred in artifacts/etudes.
- Harness rule candidates R-A..R-D documented in this synthesis (Section 4, R-A..R-D); see `artifacts/publication/publication_evidence_index.json` for evidence mapping.
