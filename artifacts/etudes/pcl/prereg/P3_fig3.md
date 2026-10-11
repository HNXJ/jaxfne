# L1-P3 preregistration: PCL Fig. 3 (simple and complex cell tuning)

Status: draft v2 for Hamm's batch sign-off. It revises v1 after the second review on
2026-10-11, which found v1 failed. No scored analysis runs before the signature. Pilots
are allowed and are not scored: timing, determinism and the axis and timing checks of the
input.

## Source claims (N'dri et al. 2025, Fig. 3 and Results; Europe PMC PMC12500994)

The paper states these qualitatively and gives no index or threshold:

- C1. Simple and complex cells are tuned for orientation and spatial frequency.
- C2. Simple cells are tuned for spatial phase. Complex cells are approximately phase
  invariant ("virtually flat phase tuning curve").
- C3. Complex cells show frequency doubling relative to simple cells under counterphase
  gratings.
- Averages run over the 64 simple cells and 32 complex cells and 2 trials.

Every numerical criterion below is ours, not the paper's.

## Network, stimuli, data

- Network: the authors' trained cSTDP3 (Zenodo, CC-BY 4.0), learning off.
- Simulator: the event-exact port (stage A, numba) at `pcl-port` 5d101eb3, driven by
  `port/run_tuning.py`. Outputs and manifest are in `F:\warehouse\pcl\runs\fig3_cSTDP3\`.
- Lateral and top-down maps load in the libstdc++-emulated order. This is a declared
  assumption; S1 tests whether it matters.
- Stimuli: `test7`, 8 orientations × 18 spatial-frequency indices × 11 phase indices ×
  2 trials. The axes were measured from the events: level 2 is spatial frequency, level 3
  is phase, and phase 10 repeats phase 0, so phases 0–9 are used.
- Each stimulus spans about 1.82 s of events (47975–1866025 µs in the first file).
- The two trials are near-identical event realisations: the column-40 count correlation
  is 0.95 between them. So trials are pooled and are never used as independent folds.
- Response: spikes from the first event to the last event of the stimulus file.

## Cell sets (fixed before scoring)

- **Simple:** the 64 cells of simple column (4,4).
  - Reason: the authors' `tuningCurves.ipynb` analyses `neuron_id = 2560`, which is
    column 40 = (4,4) if ids run column × 64 + depth. That convention is an assumption.
  - The stimulated patch drives several columns; (4,4) is one of the central ones.
  - S3 repeats the analysis on columns (3,3), (3,4) and (4,3).
- **Complex:** the 32 cells of the complex column whose input zone centre lies nearest
  simple column (4,4), by the port's geometry. On a tie, all tied columns are used and the
  count is reported.
- **Inclusion:** a cell enters a test only if it fires at least 50 spikes over all 1440
  stimuli used (8 × 18 × 10, trials pooled). If more than half of a set is excluded, every
  test on that set fails.

## Estimands and criteria

### Preference folds

- T1 and T2 fit preferences on even phases (0, 2, 4, 6, 8) and score on odd phases
  (1, 3, 5, 7, 9).
- Within a fold, the preference is (ori\*, sf\*), the argmax of the count summed over that
  fold's phases.

### Null

Each cell gets an (ori, sf) preference drawn uniformly at random, 1000 times. The aligned
population curve is recomputed each time, giving a null distribution for every
statistic. Random draws are used rather than a permutation across cells, because a
permutation is no null when the cells of a column share a preference.

| Id | Statistic (scored fold) | Pass, for each cell set separately |
|---|---|---|
| T1 | Population orientation curve at sf\*, aligned to ori\*. Statistic R_ori = mean response at Δori = 0 / mean at Δori = 90° | R_ori ≥ 1.5 and above the 99th percentile of its null |
| T2 | Population spatial-frequency curve at ori\*, aligned to sf\*. Statistic R_sf = mean response at Δsf = 0 / mean at \|Δsf\| ≥ 6 | R_sf ≥ 1.5 and above the 99th percentile of its null |
| T3 | Phase modulation at (ori\*, sf\*) fitted on all phases. Excess modulation per cell, X = PMI − E[PMI under Poisson with the cell's mean count], where PMI = (max − min)/(max + min) over phases 0–9 and the expectation is from 1000 Poisson draws | median X simple ≥ 0.15; median X complex ≤ 0.05; median X simple − median X complex ≥ 0.15 |
| T4 | Response-time spectra at (ori\*, sf\*, phase\*), trials pooled, as below | simple: A(ω) > A(2ω), with A(ω) above the 99th percentile of its null; complex: A(2ω) > A(ω), with A(2ω) above the 99th percentile of its null |

### T4 frequency doubling

- **ω** is the grating's contrast-reversal temporal frequency, measured from the input.
  For counterphase gratings, the summed event rate has its fundamental at 2ω.
  - So ω = half the summed-event-rate fundamental. It must equal the fundamental of the
    ON-event train of the single most active pixel, within one frequency bin.
  - If the two disagree, T4 fails.
- **Amplitudes:** for each cell, the Fourier amplitude of its spike train (sum of
  exp(−2πift) over spikes, divided by duration) at ω and 2ω. A(f) is the mean of these
  amplitudes across the cells in the set.
- **Null:** 1000 surrogates in which each cell's spikes are placed uniformly at random in
  the stimulus window, keeping each cell's count.

T3 and T4 use preferences fitted on all phases, because phase is the tested axis. Their
nulls carry the correction for noise.

## Sensitivity (reported, not scored)

- S1: lateral and top-down maps zeroed (local inhibition only). If T1–T4 agree with the
  full run, the map-order assumption does not affect Fig. 3.
- S2: preferences fitted on all phases (the authors' procedure), beside the folded result.
- S3: the other three central simple columns.

## Outcome language

- All tests pass for both sets: "Fig. 3 tuning reproduced qualitatively on the authors'
  trained network in the jaxfne port."
- Any failure is reported with its numbers and is not tuned away.
- Training from scratch (L1-G3) is scored later on the same T1–T4.
