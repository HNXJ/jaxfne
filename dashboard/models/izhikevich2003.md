## Question

Does jaxfne's existing Izhikevich kernel reproduce the paper's single-cell firing patterns (Fig. 2) and the 1000-cell network rhythm (Fig. 3)? The port spec is `artifacts/etudes/papers/izhikevich2003/PORT_SPEC.md`.

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

The scored arm will be the existing dense kernel `jaxfne.emitters.simulate_eig_izhikevich` with
per-cell $a, b, c, d$ arrays (`PORT_SPEC.md`); a NumPy transcription of the paper's program is the
reference arm. The table above lists the library's cell-type presets (`jaxfne/presets.py`,
`CELL_TYPE_PRESETS`), which the module labels metadata only, with no biological calibration or
empirical validation; drives are in native current units.

## Deviations and limitations

- No run compares a preset's firing pattern with the paper.
- RS, FS and LTS match the paper's Fig. 2. `VIP_IS` does not appear in the paper, and its
  `source` field says so (`artifacts/etudes/papers/izhikevich2003/SOURCE_P1.md`).
- The mapping of patterns to PV, SST and VIP interneurons is the library's labelling.

## Reproduce

Nothing is scored yet. The commands arrive with the P4 build.
