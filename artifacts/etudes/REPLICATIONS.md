# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v4 | not reproduced: T1, T3 pass; T2, T4 fail; S1 disagrees, S2 and S3 agree | `pcl-port` 5640ce18: `artifacts/etudes/pcl/LEDGER.md`, `prereg/P3_fig3_result.md`, `prereg/P3_fig3_diagnostic.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-methods | P5, prereg v6 + amendment v7 (R4) | reproduced under v7: R1–R3 pass (v6); R4 fails under v6 (every p from 0.02 oscillates) and passes under v7 (drive p·NE, onset at p = 0.04; p = 0.07 counts equal v6 in all five seeds) | `l2-caminos` 1e768f9d: `artifacts/etudes/caminos/LEDGER.md`, `RESULT_P3.md` | `F:\warehouse\caminos\scored\`, `…\v7_20261012\` | none yet (P7) |

B1: neither paper has a library-path (P6) attempt yet, so neither counts as replicated
with jaxfne.
