# CAMINOS figure verdicts: Figures 2-6, S3 (paper scale)

Status: partial. Figs 2A bottom, 2B bottom, 6B and S1 are not run. Verdicts below are descriptive. The paper gives
no numbers, so the tolerances were chosen after the data were seen; they are not pre-registered.

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

PPC uses the first 0.3 s after the discard (`FIG5C_RASTER_WINDOW_S`) in both 1C and 5C, so the control's
small values are noisy; negative values are within its sampling error.
| Fig 6A bottom | higher SOM/PV slows the rhythm, about 40 to 15 Hz, amplitude rises (549-552) | f_pk 38.2 to 18.0 Hz over ratio 0.3-4; peak \|X\| 1.0 to 4.4x | pass |

## Replica score (Cell Reports main text)

17 result panels: 1B, 1C, 2A top, 2A bottom, 2B top, 2B bottom, 3A, 3B, 3C top, 3C bottom, 4A, 4B, 5A, 5B,
5C, 6A bottom, 6B. Excluded: 1A and 6B top (schematics), 1D (mouse data), 6A top (literature densities).
Credit pass 1, partial 0.5: 1B, 2B top, 5C and 6A bottom pass; 1C, 2A top, 3C top, 3C bottom, 4A, 4B, 5A and
5B partial; 2A bottom, 2B bottom, 3A, 3B (rasters at 40, 70, 90% not run) and 6B not run. Score 8/17 = 47/100.
1B and 1C rest on the cellrep baseline (f_pk 33.7 Hz, delay 6.8 ms against the text's 6-7 ms, mmc2 804).
The table rows labelled Fig 3A/3B and Fig 4A/4B above score the spectra of
panels 3C top/bottom and 4A/4B.

## Open

- Fig 4B amplitude: the paper's own metric falls in our runs. The paper's text (mmc2 433) says amplitude rises.
  The pass on peak |X| holds only at 75%. Decide which metric is the claim before calling Fig 4B reproduced.
- Fig 3B transition near 60%: needs a finer PV grid (55, 60, 65, 70%).
- S3 beyond 6.2 s: needs a longer window.
