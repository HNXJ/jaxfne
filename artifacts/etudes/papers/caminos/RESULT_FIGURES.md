# CAMINOS figure verdicts: Figures 3, 4, S3 (paper scale)

Status: partial. Figs 1D, 2, 5, 6 and S1 are not run. Verdicts below are descriptive. The paper gives
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

## Open

- Fig 4B amplitude: the paper's own metric falls in our runs. The paper's text (mmc2 433) says amplitude rises.
  The pass on peak |X| holds only at 75%. Decide which metric is the claim before calling Fig 4B reproduced.
- Fig 3B transition near 60%: needs a finer PV grid (55, 60, 65, 70%).
- S3 beyond 6.2 s: needs a longer window.
