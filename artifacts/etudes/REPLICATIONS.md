# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v4 | not reproduced: T1, T3 pass; T2, T4 fail; S1 disagrees, S2 and S3 agree | `pcl-port` 5640ce18: `artifacts/etudes/pcl/LEDGER.md`, `prereg/P3_fig3_result.md`, `prereg/P3_fig3_diagnostic.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-methods | P5, prereg v6 + amendment v7 (R4) | reproduced under v7: R1–R3 pass (v6); R4 fails under v6 (every p from 0.02 oscillates) and passes under v7 (drive p·NE, onset at p = 0.04; p = 0.07 counts equal v6 in all five seeds) | `l2-caminos` 1e768f9d: `artifacts/etudes/caminos/LEDGER.md`, `RESULT_P3.md` | `F:\warehouse\caminos\scored\`, `…\v7_20261012\` | none yet (P7) |
| 0.5.7 | Izhikevich 2003, Fig. 2 and 3 | T-port | P3, prereg draft v7 | not scored: deferred to 0.5.7 (Hamm 2026-10-13) | `artifacts/etudes/izhikevich2003/PREREG_P3.md` | `calibration/calibration_v7_15f36087.json` (gate passed at θ = 4) | none yet (P7) |

Decisions (Hamm 2026-10-13):

- PCL Fig. 3 ships at its v4 verdict and the lane moves on. A new prereg for T2 and T4 would be
  written after seeing their outcome; the P6 library path waits for a later release.
- Izhikevich 2003 ships unscored in 0.5.6: the P3 prereg is drafted and calibrated, not signed.
  Signature, S1 reference arm and scored runs move to 0.5.7.
- CAMINOS P6 (library path) moves to 0.5.7.

B1: neither paper has a library-path (P6) attempt yet, so neither counts as replicated
with jaxfne.
