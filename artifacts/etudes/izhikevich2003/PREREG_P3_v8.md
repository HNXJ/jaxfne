# Izhikevich 2003 P3 v8: Fig. 2 amendment (post hoc)

Status: **draft v8**, unsigned. Hamm (2026-10-13) asked for this draft after the v7 FAIL and the
unscored diagnostics. **Post hoc:** both changes were chosen knowing their outcome
(`DIAGNOSTIC_P3.md`), so a pass under v8 is a description of the kernel. It is not a prospective
test. The v7 verdict (FAIL: C4, C8, N2) stays on record unchanged.

Everything not named here is `PREREG_P3.md` signed v7 verbatim: runs, kernel, dt, initial state,
time convention, windows, burst definition, the other criteria and the scorer. Fig. 3 is unchanged.

## Changes

| Id | v7 | v8 | Reason (from `DIAGNOSTIC_P3.md`) |
|---|---|---|---|
| C8 input | I = 5 for 1 ms (assumed) | I = 2 for 1 ms | The figure draws no amplitude. A pulse of 3–5 makes the resonator fire once; 2 gives a damped subthreshold oscillation, the behaviour the paper names |
| C4 pass | rate ≥ 50 Hz and last/first ISI ≤ 1.2 | rate ≥ 50 Hz and time to steady firing ≤ 30 ms (10 % of the window) | Last/first is set by the three-spike onset from rest (4.2 → 7.6 ms); the train is periodic after it. Time to steady firing separates FS (14 ms) from RS (63 ms) and LTS (none) |

Time to steady firing is the time, from the input event, of the first spike from which every later
ISI in the window lies within 10 % of the window's last ISI. A cell with fewer than 3 spikes fails.
Times come from integer rows, as in v7.

## N2 (Hamm 2026-10-13)

No change. N2 stays FAIL and is recorded as a paper claim the published program does not show:
`net.m` itself gives per-cell ISI CV 0.39, CV2 0.42 and Fano factor 0.19 (S1 reference arm and
kernel alike).

## Runs and scoring

- Re-run C4 and C8 only, on one code SHA, to `F:\warehouse\izh2003\v8\fig2`. C1–C3 and C5–C7 are
  v7's runs and verdicts.
- Scorer: `run_izh.py score` with a `--prereg v8` switch that applies the two rows above and
  leaves every other check byte-identical.
- Tests:
  - Time to steady firing on constructed spike trains: 10 at constant ISI gives 0 ms; one long
    first ISI then constant; a monotonically lengthening train has no steady time.
  - The v7 scorer's output on the v7 receipts must be unchanged.

## Outcome language

- C1–C8 pass under v8: "Izhikevich 2003 Fig. 2 firing patterns reproduce on the jaxfne Izhikevich
  kernel under amendment v8 (post hoc: C4 estimand and C8 pulse chosen after the v7 FAIL)".
- Fig. 3 keeps the v7 language: N1, N3, N4 pass, and N2 fails as a paper claim.
