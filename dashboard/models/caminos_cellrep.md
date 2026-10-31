## Question

Does the E–PV–SOM AdEx network reproduce the figure claims of Tahvili, Vinck and di Volo 2025
(Cell Reports 44, 116131): gamma that tracks feedforward and top-down drive (Fig. 2), the distinct
effects of silencing (Fig. 3) and spike-time randomization (Fig. 4) of PV and SOM cells, the role
of the PV–SOM asymmetric connectivity (Fig. 5), and the slowing with the SOM/PV density ratio
(Fig. 6)?

## Formulation

The network of the CAMINOS reference model (see that page) with the Cell Reports parameter set
(`cellrep` in `caminos_jax.py`, from cell 1 of the authors' Cell Reports notebook): 8000 E, 800 PV, 600 SOM cells,
a 4 Hz feedforward drive onto E and PV, dt 0.1 ms. Each manipulation changes one factor:
feedforward rate, top-down rate onto SOM, the silenced or Poisson-replaced fraction of a group,
PV-to-SOM or SOM-to-SOM conductance, swapped PV and SOM cell parameters, or the SOM/PV ratio at
1400 interneurons.

## Implementation

Standalone JAX port (`artifacts/etudes/papers/caminos/cellrep_figures.py`, driver `run_figA.py`),
WSL2 CUDA, float32, seeds 1–3, 12 s per run with 2 s discarded.

## Deviations and limitations

- Verdicts are descriptive: the paper gives no numbers, and the tolerances were set after the
  data were seen. No preregistration covers these figures.
- The PV and SOM silencing and replacement grids step by 25%, so the rasters at 40, 70 and 90%
  are not reproduced.
- Fig. 4B amplitude rises on peak |X| but falls on the paper's band-power metric; which one is
  the claim is open.
- Figs. 2A and 2B bottom (transient drive) and 6B (transient E-to-E increase) need protocol
  values the text does not state.

## Reproduce

```
FIGA_T_S=12 python artifacts/etudes/papers/caminos/run_figA.py <out> <values> 1,2,3 <mode>
```

Modes: silence_pv, silence_som, replace_pv, replace_som, s3, fig2a, fig2b, fig5a, fig5b, fig5c,
fig6a. Verdicts and the replica score: `artifacts/etudes/papers/caminos/RESULT_FIGURES.md`.
Receipts: `F:\warehouse\caminos\figB_out\`.
