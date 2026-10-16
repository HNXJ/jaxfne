## Question

Does an E–PV–SOM spiking network with the published parameters produce gamma at about 35 Hz,
SOM firing after PV within each cycle, the rate order PV > SOM > E, and an oscillation onset
above a critical connection probability p (Tahvili, Vinck and di Volo 2026, PLoS Comput Biol,
Fig. 1, the "reference model" introduced in their Cell Reports 2025 paper)?

## Formulation

Adaptive exponential integrate-and-fire neurons (PLoS CB Methods, Eqs. 15–18):

$$C\,\frac{dV}{dt} = g_L(E_L - V) + g_L\,\Delta\,e^{(V - V_T)/\Delta} - w + I_{syn} + I_{ext}$$

$$\tau_w\,\frac{dw}{dt} = a\,(V - E_L) - w, \qquad w \leftarrow w + b \text{ at each spike}$$

A spike is registered when $V \ge V_T + 5\Delta$; $V$ is reset to $V_{reset}$ and held for
$T_{ref}$. Synapses are conductance based:

$$I_{syn} = g_E\,(E_E - V) + g_I\,(E_I - V), \qquad \tau_X\,\frac{dg_X}{dt} = -g_X + Q\sum_j \delta(t - t_j - \tau_d)$$

| Parameter | E | PV | SOM |
|---|---|---|---|
| $V_T$ (mV) | −50 | −50 | −50 |
| $\Delta$ (mV) | 2 | 0.5 | 2 |
| $T_{ref}$ (ms) | 5 | 5 | 5 |
| $\tau_w$ (ms) | 500 | 500 | 500 |
| $a$ (nS) | 4 | 0 | 4 |
| $b$ (pA) | 50 | 0 | 80 |
| $C$ (pF) | 200 | 200 | 200 |
| $g_L$ (nS) | 10 | 10 | 10 |
| $E_L$ (mV) | −65 | −65 | −55 |
| $V_{reset}$ (mV) | −65 | −65 | −65 |

Synapses (Table 2): delay 2 ms; $\tau_E$ = 2 ms, $\tau_I$ = 8 ms; $E_E$ = 0 mV, $E_I$ = −80 mV;
Q(PV, SOM → E, PV) = 5 nS, Q(E → E) = 2 nS, Q(E → PV, SOM) = 3 nS, Q(PV → SOM) = Q(SOM → SOM) = 0;
external Q = 2.5 nS onto E and PV. Network: 8000 E, 750 PV, 750 SOM; every pair connects
independently with p = 0.07; external drive 560 Poisson sources at 4 Hz per E and PV cell.
Euler, dt = 0.1 ms.

## Implementation

Standalone JAX port on lane `l2-caminos` (`artifacts/etudes/papers/caminos/caminos_jax.py`), WSL2 CUDA,
float32. It imports no jaxfne code, so under rule B1 the paper is not yet replicated *with
jaxfne*. The library path (P6) is in build: the AdEx conductance kernel
`jaxfne.emitters.simulate_adex` matches a float64 transcription of this port's cell and
recurrent step within rtol 1e-6 (`tests/test_emitters_adex.py`).

## Deviations and limitations

- Values come from the PLoS CB Methods. That they equal the Cell Reports network is an
  assumption until its STAR Methods are read.
- Autapses are removed (our choice; the source states no rule).
- Each E and PV cell gets its own 560 sources; the text also allows shared sources. Shared
  input is untested: sensitivity S4 was invalid, since E fell silent at the reduced drive.
- R4 (onset with p) fails with the drive fixed at 560 sources: every p from 0.02 oscillates.
  It passes under amendment v7, which scales the drive as round(p·NE) sources, an assumption
  the paper does not state for the sweep. At p = 0.03, 2 of 3 seeds fail only the 20 Hz
  band-edge rule.
- The port reads an arrival in the same step's voltage update and clamps 50 steps after a
  spike; Brian2 differs by one step in each (sensitivity S5).

## Reproduce

```
python artifacts/etudes/papers/caminos/run_caminos.py --seeds 1 2 3 4 5 --p 0.07 --T 12.0 --signed --out <dir>
python artifacts/etudes/papers/caminos/run_caminos.py --seeds 1 2 3 --p <p> --T 12.0 --ext-sources pNE --signed --out <dir>
python artifacts/etudes/papers/caminos/caminos_analysis.py <dir> --ext-sources pNE
```

Seeds 1–5 (p = 0.07) and 1–3 (p sweep 0.02–0.10). Environment: WSL2 CUDA JAX, float32,
dt 0.1 ms, 12 s per run with the first 2 s discarded. Receipts: `F:\warehouse\caminos\scored\`.
