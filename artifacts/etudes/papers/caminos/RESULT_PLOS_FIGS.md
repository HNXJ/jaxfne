# CAMINOS PLoS CB figure verdicts: Figs 2A-F, 3A-F, 4A bottom (network, paper scale)

Status: descriptive, unscored by prereg. The paper gives no numbers for these panels; the tolerances are those of
`RESULT_FIGURES.md` (set after the Cell Reports data, reused unchanged here). The network stars in each panel are
oscillation frequency; amplitude and maximum E rate are mean-field lines, checked here on the network.

## Setup

- Paper: Tahvili, Vinck, di Volo, PLoS Comput Biol 2026, DOI 10.1371/journal.pcbi.1014378 (PMC13286281 XML,
  `PORT_SPEC.md`). Claims read from the Fig 2-4 captions and Results paragraphs.
- Code: `run_figA.py` with `FIGA_PARAM_SET=default` (NE 8000, NPV 750, NSOM 750), JAX CUDA float32, dt 1e-4 s,
  T 12 s, 2 s discard. Seeds 1, 2, 3. Silencing fractions 0, 0.2, 0.4, 0.6, 0.8, 0.9; randomization fractions
  0, 0.4, 0.9 at the paired-baseline group rate (PV 11.40 Hz, SOM 7.11 Hz); Fig 4A SOM/PV 0.5, 1, 2, 4 at 1500
  interneurons.
- Scorer: `score_plos_figs.py`, summary `scores/plos/plos_figs_2_4.json` (input sha256 per file). Receipts
  `F:\warehouse\caminos\figP_out\` (p2a, p2b, p3a, p3b, p4a and logs). p2b reran after a CUDA illegal-address fault
  at SOM 80 %, seed 3 (`p2b_silence_som.crash1.log`); the rerun completed all 18 runs.
- Metrics as in `RESULT_FIGURES.md`: f_pk (25-45 Hz argmax), band power 25-45 Hz and peak |X| (5-60 Hz) as ratios
  to the seed's fraction-0 run, max E rate (1 ms-smoothed population rate per cell; it can exceed 1/Tref = 200 Hz
  and is not the mean-field fixed point).

## Seed means

| Panel | value | f_pk Hz | band power | peak \|X\| | max E Hz | E / PV / SOM Hz |
|---|---|---|---|---|---|---|
| 2A PV silenced | 0, 0.2, 0.4, 0.6, 0.8, 0.9 | 34.1, 31.0, 27.1, 22.2, 9.1, 5.1 | 1, 0.94, 0.87, 0.44, 0.35, 0.15 | 1, 0.98, 0.98, 0.68, 0.83, 1.27 | 8.0, 12.6, 25.9, 241.6, 308.2, 319.6 | E 1.11 → 18.46 |
| 2B SOM silenced | 0, 0.2, 0.4, 0.6, 0.8, 0.9 | 34.1, 35.0, 37.3, 39.3, 42.8, 43.0 | 1, 1.08, 1.18, 1.30, 1.23, 1.00 | 1, 0.90, 0.98, 1.10, 0.95, 0.92 | 8.0, 8.6, 7.5, 6.4, 6.0, 4.8 | SOM 7.11 → 0.61 |
| 3A PV randomized | 0, 0.4, 0.9 | 34.1, 23.7, 10.4 | 1, 0.49, 0.27 | 1, 1.08, 0.66 | 8.0, 18.4, 312.8 | E 1.11 → 7.12 |
| 3B SOM randomized | 0, 0.4, 0.9 | 34.1, 34.5, 35.8 | 1, 1.21, 1.27 | 1, 0.89, 0.88 | 8.0, 7.0, 3.6 | E 1.11 → 0.85 |
| 4A SOM/PV | 0.5, 1, 2, 4 | 39.4, 34.1, 26.6, 21.1 | 1, 0.81, 0.69, 0.35 | 1, 1.09, 1.32, 0.66 | 4.8, 8.0, 18.3, 261.6 | PV 8.87 → 30.16 |

Seed sd: f_pk at most 1.71 Hz except 3A 90 % (3.7) and 2B 90 % (2.9); full values in the summary JSON.

## Verdicts

| Panel | Paper claim | Replication | Verdict |
|---|---|---|---|
| 2A | PV silencing lowers the frequency (stars) | 34.1 to 5.1 Hz, monotonic | pass |
| 2A | larger oscillation amplitude | peak \|X\| 0.68-0.98 to 80 %, 1.27 at 90 %; band power falls to 0.15 | fail (rises at 90 % only) |
| 2A | higher maximum E rate | 8.0 to 319.6 Hz, monotonic | pass |
| 2A | seizure-like regime beyond 70 % | max E 241.6 Hz at 60 %, 25.9 Hz at 40 % | partial: transition between 40 and 60 %, earlier than 70 % |
| 2B | SOM silencing lowers the amplitude progressively | band power 1.00-1.30, peak \|X\| 0.90-1.10; SOM rate falls to 0.61 Hz | fail |
| 2B | frequency rises slightly, far less than under PV silencing | +26 % at 90 % against −85 % for PV | pass for direction and relative size; +26 % is past the 10 % tolerance |
| 3A | PV randomization qualitatively like PV silencing | f_pk 23.7, 10.4 Hz; max E 312.8 Hz at 90 % | pass |
| 3B | SOM randomization qualitatively like SOM silencing | band power 1.21, 1.27; f_pk +1 %, +5 % | pass (both leave gamma intact here) |
| 3B | silencing or randomizing SOM abolishes rhythmic activity | band power at least 1.0 at 90 % in 2B and 3B | fail |
| 4A | higher SOM/PV lowers the frequency (stars) | 39.4 to 21.1 Hz over 0.5-4 | pass |
| 4A | higher SOM/PV raises the amplitude | peak \|X\| 1.09, 1.32 at 1, 2; 0.66 at 4 with max E 261.6 Hz | partial: rises to ratio 2; ratio 4 enters a high-rate state |

Score: 6 pass, 2 partial, 3 fail of 11 claims. Replica panels (`dashboard/sources.json`): 2A partial, 2B fail (its
headline amplitude claim fails), 3A pass, 3B fail, 4A bottom partial. Panels 2C-F and 3C-F are scored in their own
section below. Both fails on SOM amplitude contrast with the Cell Reports parameter
set, where SOM silencing and replacement lower band power (`RESULT_FIGURES.md` Fig 3A, 4A).

## Panels 2C-F and 3C-F: criteria declared before scoring (2026-11-02)

Declared before any scorer read the raster run `F:/warehouse/caminos/plosr_7f7a660e/` (`run_plos_rasters.py` at
7f7a660e, seeds 1-3, T 12 s, manifest sha256 800ede5f…). Status stays descriptive: the claims come from the Fig 2-3
captions and Results text (no numbers given); the tolerances are those above, set after the Cell Reports data, and
the network scalars of the same conditions were already seen in the seed-means table (figP runs). Panel layout:
C, E PV and D, F SOM, at 40 % (C, D) and 90 % (E, F); Fig 2 silenced, Fig 3 randomized; top row mean field, bottom
row network. The run has no mean field for randomization, so the Fig 3 top rows are not built.

Estimands, per seed, then the seed mean (seeds 1-3): over the last 0.5 s, the 1 ms-binned population rates (network
`rate_hz`; mean field `nu_hz`) give the mean rate per population, the frequency f (argmax of |FFT| of the
mean-subtracted E trace in 5-100 Hz, 2 Hz resolution) and the E amplitude (peak-to-peak); over the 10 s kept window,
the stored `f_pk_hz` and `band_power_25_45` as ratios to the seed's baseline. A silenced population's network mean
is per unsilenced cell, `rate_hz` × N / (N − round(f N)) (silencing masks round(f N) cells, `caminos_jax.manip_mask`),
since the mean field's rate is per active cell (amended before scoring, same day).

| Panel | Claim | Criterion |
|---|---|---|
| 2C | PV silenced 40 %: mean field and network agree in mean rate and frequency | each population's network mean within 20 % of the mean field's; f within 10 % |
| 2D | SOM silenced 40 %: amplitude falls; network amplitude stays above zero | band power below 1 and above 0.10; mean-field E amplitude below the mean-field baseline |
| 2E | PV silenced 90 %: seizure-like, E at the inverse refractory period; mean-field fixed point at that rate | network mean E at least 10 times the seed baseline; mean-field E mean at least 100 Hz with CV over time below 0.05 |
| 2F | SOM silenced 90 %: amplitude reduced | band power below 1; mean-field E amplitude below the mean-field baseline |
| 3C, 3E | PV randomized 40 %, 90 %: like PV silencing at the same fraction | f_pk and band power change from baseline in the same direction as 2C, 2E; 3E also meets the 2E network criterion |
| 3D, 3F | SOM randomized 40 %, 90 %: like SOM silencing at the same fraction | f_pk and band power change from baseline in the same direction as 2D, 2F |

A panel passes when every criterion in its row holds on the seed mean; partial when its claims split; fail otherwise.
Figures: one raster (seed 1, last 0.5 s, every 40th E cell and every 5th PV and SOM cell) with the three population
rates per panel, and the mean-field traces above the 2C-F panels.

### Scored (2026-11-02)

Scorer `score_plos_rasters.py` (31c23fbc) on the run above, every input checked against the manifest sha256; summary
`scores/plos/plos_rasters_2_3.json` (sha256 82260303…); figures `F:/warehouse/caminos/plosr_7f7a660e/figs/fig2_CF.png`
(a783d267…) and `fig3_CF.png` (593e1fe5…). Seed means, seeds 1-3; MF = mean field.

| Panel | Criterion | Value | Holds | Verdict |
|---|---|---|---|---|
| 2C | rates within 20 % of MF | E 2.04 vs 3.01, PV 21.09 vs 13.64, SOM 13.96 vs 19.38 Hz (e 0.32, 0.55, 0.28) | no | fail |
| 2C | f within 10 % of MF | 26.7 vs 20.0 Hz (e 0.33) | no | |
| 2D | 0.10 < band power < 1 | 1.18 | no | partial |
| 2D | MF E amplitude below MF baseline | 0.46 vs 1.19 Hz | yes | |
| 2E | network E ≥ 10 × baseline | 18.2 × (seeds 18.2, 13.9, 22.5) | yes | partial |
| 2E | MF E fixed point ≥ 100 Hz, CV < 0.05 | mean 13.0 Hz, CV 1.22: a 10 Hz limit cycle | no | |
| 2F | band power < 1 | 0.996 (seeds 1.054, 0.942, 0.993) | yes | pass |
| 2F | MF E amplitude below MF baseline | 0 vs 1.19 Hz (MF flat over the last 0.5 s) | yes | |
| 3C | f_pk and band power move as 2C | f_pk 0.70, band 0.49; 2C 0.79, 0.87 | yes | pass |
| 3D | f_pk and band power move as 2D | f_pk 1.01, band 1.21; 2D 1.09, 1.18 | yes | pass |
| 3E | moves as 2E | f_pk 0.30, band 0.27; 2E 0.15, 0.15 | yes | partial |
| 3E | network E ≥ 10 × baseline | 8.1 × (seeds 11.3, 6.9, 6.2) | no | |
| 3F | f_pk and band power move as 2F | f_pk 1.05 with 2F 1.26 (same direction); band 1.27 against 2F 0.996 | no | fail |

Panels: 3 pass (2F, 3C, 3D), 3 partial (2D, 2E, 3E), 2 fail (2C, 3F). The 2F network pass sits within 0.4 % of its
threshold, with seed 1 above 1, and the 3F fail is the same near-unity band power seen from the other side: SOM
silencing at 90 % leaves gamma band power unchanged on the network (the O-83 finding above). The MF baseline itself
is off the network (E 1.55 vs 1.11 Hz, PV 8.74 vs 11.47 Hz; f 34 vs 32 Hz), so 2C fails on a gap present before
any silencing. Under PV silencing at 90 % the MF cycles at 10 Hz and the network E trace peaks at 10, 10 and 14 Hz on seeds 1-3 (figure 2E), where the
paper describes an MF fixed point at the inverse refractory rate; the MF start state is assumed
(`caminos_meanfield.DEFAULT_START`), and whether a second, high-rate fixed point exists was not tested.

## O-83 attribution: the SOM amplitude fails follow the 2 ms synaptic delay (2026-11-02)

Method (`calibration/attribute_o83.py`): the PLoS `default` set with one parameter family swapped to its `cellrep`
value, T 12 s, SOM silenced at 0 and 90 %; band power 25-45 Hz at 90 % over the same seed's 0 %. WSL CUDA float32.

Step 1, seven families, seed 1 (bafa4797; `F:/warehouse/caminos/o83_bafa4797/o83.json`, sha256 a8a8008e…):

| Family swapped | band ratio at 90 % | baseline E / PV / SOM Hz | readout |
|---|---|---|---|
| delay (`tau_d_s` 2 ms → 0) | 0.374 | 0.97 / 9.76 / 4.76 | valid |
| sizes (NPV 800, NSOM 600) | 0.679 | 1.08 / 10.81 / 6.42 | valid |
| adaptation | 0.854 | 0.76 / 8.77 / 11.31 | valid |
| connectivity | 0.905 | 0.31 / 5.56 / 0.04 | invalid: SOM silent at baseline |
| weights | 0.939 | 0.30 / 4.53 / 1.12 | valid |
| leak_threshold | 1.320 | 1.45 / 13.45 / 10.38 | valid |
| syn_tau | 0.238 | 196.07 / 184.44 / 193.82 | invalid: high-rate state, f_pk 100 Hz |

Step 2, seeds 1-3 (8ed3f29b; `F:/warehouse/caminos/o83_seeds_8ed3f29b/o83_seeds.json`, sha256 9ae770d2…):

| Set | band ratio at 90 %, seeds 1/2/3 | SOM at 90 % Hz | E at 90 % Hz |
|---|---|---|---|
| default | 1.054 / 0.942 / 0.993 | 0.58-0.65 | 1.22-1.31 |
| cellrep | 0.453 / 0.430 / 0.445 | 0.54-0.55 | 2.68-2.72 |
| default with delay 0 | 0.374 / 0.392 / 0.512 | 0.54-0.59 | 1.21-1.30 |

The delay swap alone moves the default set to the cellrep reduction on every seed, with SOM silenced and E inside
1-3 Hz. The 2 ms delay is the paper's stated value (PLoS CB Table 2; CALIBRATION §4, F1), so the port follows the
paper here: under the stated delay, SOM silencing leaves gamma intact; without it, gamma falls by half or more. The 2B
and 3B fails stand as scored.
