## Question

Does the authors' trained Predictive Coding Light network (N'dri, Barbier, Teulière and Triesch
2025, Nat Commun 16:8880, Fig. 3), run in a jaxfne port, give simple cells tuned to orientation,
spatial frequency and phase, and complex cells tuned to orientation and spatial frequency,
approximately phase invariant and frequency doubling under counterphase gratings?

The paper states these claims qualitatively and gives no index or threshold. Every numerical
criterion on this page is ours, fixed in a signed preregistration before scoring.

## Formulation

Event-driven leaky neurons (reference C++ code, Zenodo 10.5281/zenodo.16836801). On each input
of weight $w$ at time $t$, with the previous input at $t'$ and the last spike at $t_s$:

$$V \leftarrow V\,e^{-(t-t')/\tau_m}, \qquad A \leftarrow A\,e^{-(t-t')/\tau_{SRA}}$$

$$V \leftarrow V + w - \eta_{RP}\,e^{-(t - t_s)/\tau_{RP}} - A$$

$V$ is clamped at −20; a spike occurs when $V > \theta$, then $V \leftarrow V_{reset}$ and
$A \leftarrow A + \Delta_{SRA}$. Inhibitory inputs subtract $w$. Complex cells have no $A$ term.

| Parameter | Simple | Complex |
|---|---|---|
| $\theta$ | from each trained network's config | from each trained network's config |
| $V_{reset}$ | −10 | −10 |
| $\tau_m$, $\tau_{RP}$, $\tau_{SRA}$ (ms) | 18, 5, 100 | 50, 5, — |
| $\eta_{RP}$ | 10 | 10 |

Architecture: one 9 × 9 × 64 simple layer on DVS events (receptive field 10 × 10 × 2
polarities), one 6 × 6 × 32 complex layer (receptive field 4 × 4 × 64 simple cells), with local,
lateral and top-down inhibition loaded from the trained network cSTDP3. Learning is off.

Statistics (preregistration v4): $R_{ori}$ = response at the preferred orientation over the
response 90° away; $R_{sf}$ = response at the preferred spatial frequency over the per-cell
flank; $X$ = excess phase modulation over a Poisson null; $L = \log(A(2\omega)/A(\omega))$ of the
spike train under counterphase gratings at $\omega$ ≈ 2 Hz.

## Implementation

Standalone event-exact port on lane `pcl-port` (`artifacts/etudes/papers/pcl/port/`), numba on CPU,
driven by `port/run_tuning.py`; it reproduces the C++ update order. It imports no jaxfne code,
so under rule B1 the paper is not yet replicated *with jaxfne*.

## Deviations and limitations

- T2 (spatial frequency) is above both nulls in both sets and under the preregistered ratio
  of 2.
- T4 orders the sets as predicted (the difference 0.83 beats its null, 0.42), and the complex
  median L is −0.19: the complex cells are less phase-locked than the simple cells without
  frequency doubling.
- Lateral and top-down maps load in an emulated libstdc++ order, a declared assumption.
  Sensitivity S1 (maps zeroed) disagrees with the full run, so this assumption is not excluded.
- The simple-cell set (column 4,4) follows an id convention inferred from the authors'
  notebook; S3 repeats the analysis on neighbouring columns.
- An unscored diagnostic of the failure is in `prereg/P3_fig3_diagnostic.md`.

## Reproduce

```
python artifacts/etudes/papers/pcl/port/run_tuning.py --subset all
python artifacts/etudes/papers/pcl/port/run_tuning.py --subset all --inhibition local-only   # S1
python artifacts/etudes/papers/pcl/port/fig3_analysis.py <run dir> <stimulus dir> --signed --out <scores.json>
```

Network and stimuli: Zenodo 10.5281/zenodo.16813454 (cSTDP3, `test7` gratings: 8 orientations ×
18 spatial frequencies × 10 phases × 2 trials). Scoring seed 0, 1000 null draws. Runs:
`F:\warehouse\pcl\runs\fig3_cSTDP3\`; port at `pcl-port` 5d101eb3, scored at f6d9dc38.
