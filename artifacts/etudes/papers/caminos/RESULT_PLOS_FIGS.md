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
- Scorer: `score_plos_figs.py`, summary `scores/plos_figs_2_4.json` (input sha256 per file). Receipts
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
headline amplitude claim fails), 3A pass, 3B fail, 4A bottom partial. Panels 2C-F and 3C-F (rasters and mean-field
traces at 40 and 90 %) stay not run: the network runs exist in the receipts, the raster figures and the mean-field
comparison are not built. Both fails on SOM amplitude contrast with the Cell Reports parameter
set, where SOM silencing and replacement lower band power (`RESULT_FIGURES.md` Fig 3A, 4A).
