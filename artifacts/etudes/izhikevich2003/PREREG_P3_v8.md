# Izhikevich 2003 P3 v8: Fig. 2 amendment (post hoc)

Status: **draft v8.2**, unsigned (C4 per Hamm's call; ready for review and signature). Hamm (2026-10-13) asked for this draft after the v7 FAIL and the
unscored diagnostics. **Post hoc:** the change was chosen knowing its outcome
(`DIAGNOSTIC_P3.md`), so a pass under v8 describes the kernel and is not a prospective test. The
v7 verdict (FAIL: C4, C8, N2) stays on record unchanged.

v8 review (Muse, 2026-10-13): REVISE. The C4 "time to steady firing" estimand was trivial as
written. Computed correctly on the kernel receipts, it does not separate FS from RS, so it is
withdrawn. C4 is open (below).

Everything not named here is `PREREG_P3.md` signed v7 verbatim: kernel, dt, float32, initial
state, time convention, windows, burst definition, the other criteria, the scorer, and Fig. 3.

## Changes

| Id | v7 | v8 | Reason |
|---|---|---|---|
| C8 input | I = 0 for 10 ms, 5 for 1 ms (assumed), then 0 (311 ms) | I = 0 for 10 ms, 2 for 1 ms, then 0 (311 ms) | The figure draws no amplitude. A pulse of 3–5 makes the resonator fire once (float64 Euler in the kernel's order, `DIAGNOSTIC_P3.md`). A pulse of 2 gives a damped subthreshold oscillation, the behaviour the paper names |

The rows are unchanged:
- event row 110 (the pulse end);
- n_steps 3220 (`run_izh.py:56`);
- the C8 pass rule: no spike, ≥ 2 maxima after the pulse, the second lower;
- the reported-only frequency.

Only the amplitude in the 10 pulse rows changes.

## C4 (Hamm 2026-10-13: relative criterion)

| | v7 | v8 |
|---|---|---|
| C4 pass | rate ≥ 50 Hz and ISI ratio ≤ 1.2 | rate ≥ 50 Hz, and the C4 ISI ratio (last/first, v7's formula) strictly below both the C1 (RS) and the C5 (LTS) ISI ratios from the same scored run set |

FS adapts 86 % from the first to the last ISI at I = 10 from rest (ratio 1.86). That is the
smallest of the three adapting types (RS 2.53, LTS 4.39 on the v7 receipts), so v8 reads
"practically without adaptation" as "adapts least". The C1 and C5 ratios are used whether those criteria pass or fail. If either ratio is undefined
(fewer than 2 ISIs in its window), or the C4 ratio is, C4 fails. A C1 or C5 run that is missing
(incomplete) makes C4 incomplete. C4 needs no new run: the v7 C1, C4 and C5 runs are used.

## N2 (Hamm 2026-10-13)

No change. N2 stays FAIL, recorded as a paper claim the published program does not show: `net.m`
itself gives per-cell ISI CV 0.39, CV2 0.42 and Fano factor 0.19.

## Runs and scoring

- One new run, C8 at I = 2, to `F:\warehouse\izh2003\v8\fig2`. C1–C7 are v7's runs and verdicts.
- One-SHA rule: the v7 directories keep their single SHA (`423a2f88`), and the v8 directory is
  internally one SHA. v8 completes only when both hold. Fig. 3 verdicts are reused from v7 unchanged.
  The v8 report lists both SHAs.
- Scorer: a `v8` subcommand of `run_izh.py` that reads the v7 score for C1–C7 and the v8 C8 run.
  Every v7 path stays byte-identical, with a test on the v7 receipts.

## Outcome language

- C1–C8 pass under v8: "Izhikevich 2003 Fig. 2 firing patterns reproduce on the jaxfne Izhikevich
  kernel under amendment v8 (post hoc: C8 pulse amplitude and a relative C4 criterion, both chosen
  after the v7 FAIL)".
- Fig. 3 keeps the v7 language: N1, N3 and N4 pass, and N2 fails as a paper claim.
