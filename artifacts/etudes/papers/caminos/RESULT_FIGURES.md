# CAMINOS figure verdicts: Figures 2-6, S3 (paper scale)

Status: partial. Figs 2A bottom, 2B bottom and 6B are not run; S1 scored 2026-11-02 against criteria declared first. Verdicts are descriptive. The paper gives
no numbers, so the tolerances were chosen after seeing the data; they are not pre-registered.

## Setup

- Code: `run_figA.py` and `cellrep_figures.py` at the commit that adds this file. Parameters: cellrep set, dt 1e-4 s,
  T 12 s, 2 s discard (10 s kept). Seeds 1, 2, 3. Fractions 0, 0.25, 0.5, 0.75, 1.0.
- Silencing removes the fraction of the population (mmc2 439-445). Replacement swaps spikes for Poisson
  trains at the group's paired-baseline mean rate: PV 7.901 Hz, SOM 5.122 Hz (mmc2 1772-1774).
- Metrics: f_pk = argmax of the LFP magnitude spectrum in 25-45 Hz; band power = integral of |X|^2 over
  25-45 Hz (the paper's Fig 4 metric, `caminos_analysis.band_power`); max E rate.

## Tolerances (set after the data)

- Frequency change "not substantial": within 10% of baseline.
- "Minor" change: band power within 20% and frequency within 10%.
- "Abolished": band power at most 10% of baseline.
- Amplitude increase: peak |X| (5-60 Hz) at least 1.2 times baseline.
- Seizure-like: max E rate at least 50 Hz.

## Verdicts

| Panel | Paper claim | Replication (seed means) | Verdict |
|---|---|---|---|
| Fig 3A SOM silencing | gradual gamma reduction | band power 81%, 67%, 51%, 38% of baseline at 0.25, 0.5, 0.75, 1.0 | pass |
| Fig 3A | abolished at most suppression | 38% of baseline at 100% | fail |
| Fig 3A | no substantial frequency change | 34.1, 37.4, 37.4 Hz (+1%, +11%, +11%) to 75%; 17.6 ± 20.9 Hz at 100% | partial |
| Fig 3B PV silencing | minor change at 20-50% | 25%: band −5%, f −3%; 50%: band −19%, f −8% | pass |
| Fig 3B | periodic 10-15 Hz synchrony at about 60% | 13.6 Hz at 75%; 50% still 30.9 Hz | qualitative pass; transition location unresolved on the 25% grid |
| Fig 3B | seizure-like at more than 80%, high E rate | max E 328 Hz at 100%; 22 Hz at 75% | pass at 100% only; the ~150 Hz component was not checked |
| Fig 4A SOM replacement | gradual amplitude decrease, no frequency or stability change | band power −15%, −30%, −41%, −58%; max E 4.2-4.6 Hz (baseline 4.7) | pass to 75%; 100% f 23.8 Hz (−29%, sd 15.8) |
| Fig 4B PV replacement | frequency falls about 35 → 10 Hz, gradually | 28.1, 20.8, 12.2, 5.3 Hz | pass for the trend; endpoint 5 Hz, not 10 |
| Fig 4B | amplitude increases | paper metric (band power): −6%, −74%, −67%, −99%; peak |X|: 1.18×, 1.10×, 3.3×, 0.69× | fail on the paper metric; pass on peak |X| at 75% only |
| Fig 4B | seizure-like at more than 80% | max E 327 Hz at 100% | pass at 100% |
| S3 | after SOM silencing at 6.04 s, control dips; silenced rebounds | control dips to −1.1 Hz below its pre-onset mean; silenced rebounds +2.3 Hz at 6.08-6.11 s; traces identical before onset (all seeds) | partial |
| S3 | prolonged deactivation; asynchronous regime | outside the 5.9-6.2 s window | not tested |

## Figures 2, 5, 6A (paper scale, 2026-10-31)

Same setup and tolerances; seeds 1, 2, 3; T 12 s. Runner modes `fig2a`, `fig2b`, `fig5a`, `fig5b`, `fig5c`,
`fig6a`. Receipts `F:\warehouse\caminos\figB_out\` (one JSON per mode). Delay = wrapped SOM-minus-PV phase at
f_pk (ref. 42 estimator) divided by 2π f_pk. Peak |X| over 5-60 Hz.

| Panel | Paper claim (mmc2) | Replication (seed means) | Verdict |
|---|---|---|---|
| Fig 2A top | f_pk rises monotonically, about 7 to 80 Hz, over feedforward 0-10 Hz (286-288) | silent at 0-1 Hz; 13.6, 25.6, 33.7, 40.4, 46.9, 51.9, 57.1, 59.4, 63.4 Hz at 2-10 Hz | pass for the trend; range 14-63 Hz, not 7-80 |
| Fig 2A top | PV-to-SOM delay falls from about 10 to 4 ms (293-295) | 9.8 ms at 2 Hz to 3.9 ms at 10 Hz, monotonic | pass |
| Fig 2B top | top-down to SOM: amplitude rises, frequency falls, E rate falls, SOM lag persists (314-321) | f_pk 33.7 to 23.9 Hz; peak \|X\| 1.0 to 3.7x; E 2.58 to 1.86 Hz; delay 6.8 to 3.7 ms, positive throughout | pass (band power falls once f_pk leaves 25-45 Hz) |
| Fig 5A | PV-to-SOM coupling abolishes gamma (474-478) | band power 0.65, 0.50, 0.45, 0.43, 0.40 of baseline at 2-10 nS | partial: reduced 60%, not abolished |
| Fig 5B | SOM-to-SOM coupling, relatively minor reduction (479-481) | band power 0.72 of baseline at 10 nS; f_pk within 10% | partial: smaller than 5A, but past the 20% "minor" tolerance |
| Fig 5C | swap: SOM leads PV; stronger locking than control (490-496) | SOM-parameter cells (PV slot) at 0.28 rad, PV-parameter cells at 0.68-0.78 rad, every seed; PPC 0.30-0.75 per type against -0.04 to 0.08 in the Fig 1C control | pass |
| Fig 1C | PPC PV > SOM > E; SOM about 7 ms after PV (250-262) | control (mode `fig1c`, `F:\warehouse\caminos\figC_out\c1\`): PPC PV 0.035, 0.080, 0.055 is highest in every seed; SOM above E in seed 2 only; SOM-minus-PV phase 5.0, 6.9, 5.3 ms | partial: PV > SOM > E in 1 of 3 seeds |

| Fig 6A bottom | higher SOM/PV slows the rhythm, about 40 to 15 Hz, amplitude rises (549-552) | f_pk 38.2 to 18.0 Hz over ratio 0.3-4; peak \|X\| 1.0 to 4.4x | pass |

PPC uses the first 0.3 s after the discard (`FIG5C_RASTER_WINDOW_S`) in both 1C and 5C, so the control's
small values are noisy; negative values are within its sampling error.

## Figure 3 raster fractions and the finer PV grid (2026-10-31)

Same setup; seeds 1, 2, 3; T 12 s. Modes `silence_som` (0, 0.5, 0.9), `silence_pv` (0, 0.4, 0.55, 0.6, 0.65, 0.7,
0.9), `replace_som` (0, 0.5, 0.9), `replace_pv` (0, 0.4, 0.7). Receipts `F:\warehouse\caminos\figC_out\c3a`, `c3b`,
`c4a`, `c4b`. High-frequency component = argmax of |X| over 60-250 Hz.

| Panel | Paper claim (mmc2 361-386, 437-445) | Replication (seed means) | Verdict |
|---|---|---|---|
| Fig 3A (SOM 50%, 90%) | amplitude falls; no substantial frequency change | band power 0.67, 0.44 of baseline; f_pk 37.4, 38.3 Hz (+11%, +14%) | partial: frequency past the 10% tolerance |
| Fig 3B top (PV 40%) | minor change at 20-50% | f_pk 31.6 Hz (-6%); band power 0.91 | pass |
| Fig 3B middle (PV 70%) | sudden transition near 60% to periodic 10-15 Hz synchrony | f_pk 30.9 Hz at 55%, 23.5 ± 5.8 Hz at 60%, 15.5 Hz at 65%, 14.5 Hz at 70% with peak \|X\| 3.3x | pass |
| Fig 3B bottom (PV 90%) | seizure-like beyond 80%: about 150 Hz, high E rate | max E 309 Hz, mean E 13.9 Hz (baseline 2.6); 60-250 Hz peak at 153 Hz | pass |
| Fig 4A rasters (SOM 50%, 90%) | no change of frequency or stability | f_pk 36.3, 33.4 Hz; max E 4.2, 4.5 Hz | pass at the raster fractions |
| Fig 4B rasters (PV 40%, 70%) | frequency falls toward 10 Hz | f_pk 25.4, 13.1 Hz; peak \|X\| 1.07x, 3.25x; band power 0.56, 0.52 | pass for frequency; amplitude open (below) |

These runs settle the earlier Fig 3B rows: the transition lies between 55% and 65%, and the seizure state carries
the 150 Hz component.

## Figure S1: connection probability (criteria declared 2026-11-02, before the sweep output was read)

Run: `run_figA.py <out> 0.01,0.02,0.04,0.08,0.12,0.16,0.20,0.25,0.30 1,2,3 s1` at 95d2161b (frozen WSL clone),
cellrep set, GPU float32, T 12 s, 2 s discard; drive round(p·NE) private Poisson sources at 4 Hz (Hamm 2026-11-02,
O-73). Claims: legend (mmc2.pdf p. 20) and Discussion (p. 10): frequency rises with p; seizure-like, hyper-synchronous
activity above 25%; in the realistic range 1-15% the frequency stays in the gamma range and the delay is maintained.
Status stays descriptive: the paper gives no numbers; the seizure tolerance is the one above; the gamma range is
fixed here as 30-80 Hz. Estimators: f_pk = magnitude argmax over 5-100 Hz, max E rate, PV-to-SOM delay = wrapped
SOM-minus-PV phase at f_pk (ref. 42) / 2π f_pk; seed means unless stated.

| Id | Claim | Criterion |
|---|---|---|
| S1-F | frequency rises with p | seed-mean f_pk at p = 0.20 above p = 0.04, and no step between consecutive p in 0.02-0.20 falls by more than 10% |
| S1-S | seizure-like above 25% | max E ≥ 50 Hz in every seed at p = 0.30, and below 50 Hz in every seed at p ≤ 0.20; onset at a different grid step is partial |
| S1-G | gamma at 1-15% | seed-mean f_pk in 30-80 Hz at p = 0.01, 0.02, 0.04, 0.08, 0.12 |
| S1-D | delay maintained at 1-15% | PV-to-SOM delay positive (SOM after PV) in every seed at p = 0.01-0.12 |

Result (receipts `F:\warehouse\caminos\s1_95d2161b\`: `s1.json` sha256 7dc1b25f…afc7, `out.log`; 27 runs, 53 min; seed means over seeds 1, 2, 3).

| p | 0.01 | 0.02 | 0.04 | 0.08 | 0.12 | 0.16 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|---|
| f_pk (Hz) | 12.2 | 41.2 | 41.7 | 46.1 | 49.5 | 52.0 | 53.7 | 52.2 | 100.0 |
| max E (Hz), seed range | 5.4-7.1 | 5.3-5.4 | 21.7-22.2 | 28.4-28.8 | 37.2-38.0 | 51.6-53.6 | 83.0-96.7 | 130.6-163.7 | 303.7-398.6 |
| mean E (Hz) | 1.56 | 2.86 | 2.60 | 3.03 | 3.68 | 4.55 | 5.73 | 7.02 | 196.1 |
| delay (ms), seed range | 10.2-11.0 | 5.3-6.0 | 3.0 | 2.1 | 0.9 | 0.5 | 0.3 | 0.7-0.8 | -0.01-0.03 |

| Id | Verdict | Evidence |
|---|---|---|
| S1-F | pass | 53.7 Hz at 0.20 against 41.7 Hz at 0.04; no step in 0.02-0.20 falls (smallest step +1.1%, 0.02 to 0.04) |
| S1-S | partial | every seed at or above 50 Hz from p = 0.16 (51.6-53.6 Hz), two grid steps before the paper's 25%; the hyper-synchronous state (mean E 196 Hz, f_pk at the 100 Hz edge of the search band) appears only at 0.30 |
| S1-G | fail | 12.2 Hz at p = 0.01 (11.1-13.3 Hz per seed); 41.2-49.5 Hz at 0.02-0.12 |
| S1-D | pass | positive in every seed at 0.01-0.12; it shrinks from 10.7 ms to 0.9 ms, so "maintained" holds for the sign only |

The 50 Hz max-E tolerance comes from Fig 3B; mean E stays at 4.6-7.0 Hz from 0.16 to 0.25, so the S1-S onset depends on
that tolerance. At p = 0.01 the drive is 80 Poisson sources (round(p·NE)), the weakest of the grid.

## Core-route equivalence (T-30; declared 2026-11-02, before the core output was read)

Question: does `run_cell(route="core")` (jaxfne construct and `Model.simulate`, fc0e13df) reproduce the oracle
sweeps that the verdicts above rest on? Reference: the frozen oracle PV-silencing sweep `figC_out/c3b/silence_pv.json`
(fractions 0, 0.4, 0.7, 0.9; seeds 1, 2, 3). Core run: `run_figA.py ... silence_pv` with `FIGA_ROUTE=core` at
08780699, same fractions, seeds 1, 2, 3, T 12 s, GPU float32. The two routes draw connectivity, drive and replacement
from different random streams (`run_core_raster` docstring), so equivalence is statistical.

Criterion (T-R28 ruling: within the frozen result's between-seed range): for each fraction and each metric (f_pk,
25-45 Hz band power, max E rate, mean E, PV and SOM rates), the core seed mean lies inside the oracle's per-seed
[min, max]. Equivalence holds when every cell is inside. A cell outside goes to attribution (parameters, drive, delay,
random streams) before any rescoring (todo T-30). With three oracle seeds the range is narrow (rates within about 1%),
so cells outside are expected by chance alone; each is reported with its distance from the range.

## Replica score (Cell Reports main text)

17 result panels: 1B, 1C, 2A top, 2A bottom, 2B top, 2B bottom, 3A, 3B, 3C top, 3C bottom, 4A, 4B, 5A, 5B,
5C, 6A bottom, 6B. Excluded: 1A and 6B top (schematics), 1D (mouse data), 6A top (literature densities).
Credit pass 1, partial 0.5: 1B, 2B top, 3B, 3C bottom, 5C and 6A bottom pass; 1C, 2A top, 3A, 3C top, 4A, 4B,
5A and 5B partial; 2A bottom, 2B bottom and 6B not run. Score 10/17 = 59/100.
1B and 1C rest on the cellrep baseline (f_pk 33.7 Hz, delay 6.8 ms against the text's 6-7 ms, mmc2 804).
The table rows labelled Fig 3A/3B and Fig 4A/4B above score the spectra of
panels 3C top/bottom and 4A/4B.

## Open

- Fig 4B amplitude: the paper's own metric falls in our runs. The paper's text (mmc2 433) says amplitude rises.
  The pass on peak |X| holds only at 75%. Decide which metric is the claim before calling Fig 4B reproduced.
- S3 beyond 6.2 s: needs a longer window.
