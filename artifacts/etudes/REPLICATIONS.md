# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v4 | not reproduced: T1, T3 pass; T2, T4 fail; S1 disagrees, S2 and S3 agree | `pcl-port` 3d4a4b90: `artifacts/etudes/pcl/LEDGER.md`, `prereg/P3_fig3_result.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-methods | P5, prereg v6 | not reproduced: R1–R3 pass; R4 fails (every p from 0.02 oscillates) | `l2-caminos` 9fe22858: `artifacts/etudes/caminos/LEDGER.md`, `RESULT_P3.md` | `F:\warehouse\caminos\scored\` | none yet (P7) |

B1: neither paper has a library-path (P6) attempt yet, so neither counts as replicated
with jaxfne.
