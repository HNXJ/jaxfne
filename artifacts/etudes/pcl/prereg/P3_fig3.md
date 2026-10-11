# L1-P3 preregistration: PCL Fig. 3 (simple and complex cell tuning)

Status: draft v4 for Hamm's batch sign-off. It revises v1 (FAIL), v2 (PASS with notes)
and v3 (PASS with notes, three blocking) after the reviews of 2026-10-11. No scored analysis runs before the signature. Pilots
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
- **Complex:** the 128 cells of complex columns (2,2), (2,3), (3,2) and (3,3).
  - The rule is geometric: these are the complex columns whose input zone is centred
    nearest simple column (4,4).
  - From `pcl_event_ref.py:409-411`, complex column cx pools simple columns cx..cx+3, so
    its centre lies at cx + 1.5. Columns 2 and 3 tie on both axes.
  - The paper averages over 32 complex cells. We use the four tied columns rather than pick
    one of them.
- **Inclusion.** Each test has a floor on the spikes it actually uses:
  - T1 and T2: at least 20 spikes in the fit fold, summed over every stimulus in it
    (8 ori × 18 sf × 5 even phases × 2 trials).
  - T3: at least 50 spikes at (ori\*, sf\*) over phases 0–9 and both trials, about 5 per
    phase.
  - T4: at least 10 spikes at (ori\*, sf\*, phase\*) over both trials.
  - A cell whose L is not finite is excluded.
  - Every exclusion is counted and reported. If more than half of a set is excluded from a
    test, that test fails for the set.

## Estimands and criteria

### Preference folds

- T1 and T2 fit preferences on even phases (0, 2, 4, 6, 8) and score on odd phases
  (1, 3, 5, 7, 9).
- Within a fold, the preference is (ori\*, sf\*), the argmax of the count summed over that
  fold's phases.

"Generalises" here means across spatial phase: the fit and scored folds are different
phases of the same (ori, sf) stimuli.

### Nulls for T1 and T2 (both must be exceeded)

- **N-label.** The scored fold's (ori, sf) labels are shuffled within each cell, and the
  fit-then-score procedure is rerun. This is the null of "no structure in the scored
  fold".
- **N-pref.** Fitted preferences are permuted across the cells of the set, and the scored
  curves are recomputed.
  - This is the null of "the tuning is a stimulus-wide drive effect shared by all cells".
    An example is input event density that varies with sf.
  - It is exceeded only when each cell's own preference predicts its own held-out
    response better than another cell's preference does.
- Each null uses 1000 draws. The distribution of fitted preferences is reported beside
  them.

R_sf's flank is computed per cell, as the mean of each cell's own responses at
|Δsf| ≥ 6, and then averaged over cells.

| Id | Statistic (scored fold) | Pass, for each cell set separately |
|---|---|---|
| T1 | Population orientation curve at sf\*, aligned to ori\*. R_ori = mean response at Δori = 0 / mean at Δori = 90° | R_ori ≥ 2, above the 99th percentile of N-label and of N-pref |
| T2 | Population spatial-frequency curve at ori\*, aligned to sf\*. R_sf = mean response at Δsf = 0 / mean per-cell flank | R_sf ≥ 2, above the 99th percentile of N-label and of N-pref |
| T3 | Per cell, excess modulation X = PMI − PMI₀, where PMI = (max − min)/(max + min) of the count over phases 0–9 at (ori\*, sf\*) (all phases, trials pooled) and PMI₀ is the mean PMI of 1000 draws of 10 Poisson counts with the cell's mean count per phase | median X simple ≥ 0.15; median X complex ≤ 0.5 × median X simple. The Poisson null makes the comparison fair across the two sets' different counts; it is conservative, since the data are sub-Poisson |
| T4 | Per cell, the log ratio L = log(A(2ω)/A(ω)) of its spike train at (ori\*, sf\*, phase\*), trials pooled, as below | median L simple < 0 < median L complex, and median L complex − median L simple above the 99th percentile of a null that permutes cells between the two sets 1000 times |

### T4 frequency doubling

- **ω** is the grating's contrast-reversal temporal frequency, measured from the input.
  - It is measured separately for every stimulus file, as the fundamental of the
    ON-event train of the single most active pixel in that file, searched over 1–3 Hz on a
    0.001 Hz grid.
  - In three test7 files it is 1.98–1.99 Hz (second review).
  - A file whose ω falls outside 1.9–2.1 Hz is reported, and its cells are excluded from
    T4.
  - The window holds about 3.6 cycles, so the frequency resolution is about 0.55 Hz.
    Amplitudes are therefore evaluated at exactly ω and 2ω, not at FFT bins.
- **Amplitude:** A(f) = |Σ exp(−2πift)| over a cell's spikes, divided by the duration.
- T3 and T4 fit preferences on all phases, because phase is the tested axis. phase\* is the
  argmax over phases 0–9 of the trial-pooled count at (ori\*, sf\*); ties go to the lowest
  phase.
- A file with no ON event uses all its events for ω. The two trials' ω are averaged, and a
  difference over 0.01 Hz is reported.
- Ratios R_ori and R_sf: 0/0 counts as 1 and x/0 (x > 0) as infinite.
- Spike floors count every spike in the pooled trials, including spikes repeated across
  trials.

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
