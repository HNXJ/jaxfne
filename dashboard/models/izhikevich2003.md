## Question

Do the library's four Izhikevich cell-type presets reproduce the firing patterns of Izhikevich
(2003, IEEE Trans Neural Netw 14:1569)? No comparison has been run. This page is a
demonstration that records what the presets contain; scoring the firing patterns is a candidate
first étude.

## Formulation

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I, \qquad \frac{du}{dt} = a(bv - u)$$

At $v \ge 30$ mV: $v \leftarrow c$, $u \leftarrow u + d$.

| Preset | Pattern | $a$ | $b$ | $c$ | $d$ | Drive |
|---|---|---|---|---|---|---|
| `E_RS` | regular spiking | 0.02 | 0.20 | −65 | 8 | 5.0 |
| `PV_FS` | fast spiking | 0.10 | 0.20 | −65 | 2 | 3.0 |
| `SST_LTS` | low-threshold spiking | 0.02 | 0.25 | −65 | 2 | 3.5 |
| `VIP_IS` | intrinsic spiking / bursting | 0.02 | −0.10 | −55 | 6 | 3.0 |

## Implementation

`jaxfne/presets.py` (`CELL_TYPE_PRESETS`), public through `jaxfne`. The module labels them
metadata only, with no biological calibration or empirical validation; drives are in native
current units.

## Deviations and limitations

- No run compares a preset's firing pattern with the paper.
- RS, FS and LTS match the paper's Fig. 2. `VIP_IS` does not appear in the paper, and its
  `source` field says so (`artifacts/etudes/izhikevich2003/SOURCE_P1.md`).
- The mapping of patterns to PV, SST and VIP interneurons is the library's labelling.

## Reproduce

There is no run to reproduce. The parameters are in `jaxfne/presets.py`.
