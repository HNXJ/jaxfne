# PCL R4 preregistration: Fig. 3 on the core event scan, exact and approximate arms

Status: **draft, unsigned** (2026-11-03). No scored run may use it before Hamm signs it. Signed v4 and v5 and their
FAIL verdicts (`P3_fig3_result.md`, `P3_fig3_v5_result.md`) stand; withdrawn v6 stays withdrawn. R5 scores this
protocol, exact and approximate arms apart (design study `r2_r4_design_study_2026-10-27.md` §6, option (b) ruled by
Hamm 2026-10-27).

## What is known before scoring

- R2 and R3 (`CALIBRATION.md`): the numba oracle equals the authors' C++ on weights at rtol 1e-12 after `shapes.h5`
  and after each of training files 0-4, learning on; the core event scan equals the C++ after training file 0
  (max relative weight difference 5.3e-15) with the oracle's spike counts. Neither test used `test7`.
- The C++ runs `"static"` inhibition without local inhibition (R1 F1); the v4 and v5 runs used the port's alias
  (local on). The oracle under the C++ setting (`F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal/`, 3168 runs)
  was scored for T2 only (`P3_fig3_diagnostic.md` Q7): simple 15/16 points, shape guard fails (SSE 0.244 against
  flat 0.233); complex 2/15 points, curve flat near 0.95. So T2 under the C++ setting is known to fail for both
  sets. T1, T3 and T4 under that setting have not been computed.
- The core arm costs about 0.16 s per event on CPU (R3, learning on); an oracle `test7` file takes about 22 s, so
  a core file is expected to take about 2 h (inferred from the R3 ratio, 306×). The full 3168-file set on the core
  is out of reach (about 6000 CPU h).
- F6 (R1): the notebook's column (0,0) receives 0 events in `test7` and fires 0 spikes; column (4,4) is driven.

## Common settings

- Network: the authors' trained cSTDP3 (Zenodo, CC-BY 4.0), learning off, maps in the libstdc++ order (v4).
- Inhibition: the C++ setting (`run_tuning.py --inhibition no-local`), which R2 and R3 matched exactly.
- Stimuli, response window, cell sets, inclusion floors: v4 verbatim (simple column (4,4), complex columns
  (2,2), (2,3), (3,2), (3,3)). Column (0,0), the notebook's, is reported beside (4,4) and not scored.

## R5-exact (prospective)

- Files: the 16 `pilot` tasks of `run_tuning.py` (8 orientations × sf indices 0 and 9 × phase 0 × trial 0),
  fixed here before any core run on `test7`.
- Arms: `--engine core` and `--engine numba`, same feed, from one frozen clone at the signed SHA.
- Pass: on every file, every one of the 6336 cells has the same spike count and the same spike times (integer
  µs) in both arms. No tolerance.
- Negative control: the core arm with the alias (`--inhibition full`) against the numba arm under `no-local`
  must differ on at least one file; if it does not, the test cannot fail and R5-exact is void.
- A pilot (unscored) times one file on the core before the scored run.

## R5-approximate

- Runs: the numba oracle under `no-local` on all 3168 tasks (the existing `diag_runs/fig3_cSTDP3_nolocal/` is
  rerun from the signed SHA and its counts must equal the existing files, else the rerun is scored and the
  difference reported).
- Tests: T1, T3 and T4 as v4 (T4 with the v5 criterion, no sign requirement), T2 as v5. T1, T3 and T4 are
  prospective; T2 is known to fail (above) and is scored for the record.
- The oracle stands for the core on the full set only if R5-exact passes; otherwise R5-approximate is reported
  as the oracle's and the core claim is withheld.

## Outcome language

- R5-exact pass: "the jaxfne core event scan reproduces the port oracle event-exactly on 16 `test7` stimuli of the
  authors' trained network"; with R2 and R3, the chain to the C++ is stated with its scope (weights, learning on,
  `shapes.h5` and training files 0-4).
- R5-approximate: per set and per test, with numbers. Fig. 3 counts as reproduced only if T1-T4 pass for both
  sets; otherwise the paper stays pending (Hamm 2026-11-03) and each failed test is reported as failed.

## For Hamm (recommendation first)

1. Inhibition: the C++ setting `no-local` (recommended: it is what the C++ runs and what R2 and R3 matched); or
   the port alias `full`, as v4 and v5 (then the exact chain to the C++ does not cover it).
2. Full set: the oracle stands for the core given R5-exact (recommended; compute); or the core on the full set
   (about 6000 CPU h, or a GPU port of the scan first).
3. Exact subset: the 16 `pilot` tasks (recommended); or a seeded draw of n files declared here.
