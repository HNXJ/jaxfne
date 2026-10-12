## Question

How closely does a reduced jaxfne neuron follow a Hodgkin–Huxley reference cell under the same
current step (Atlas scenario AT-01)? The reference is Jaxley's HH channel model (Hodgkin and
Huxley 1952). This page is a demonstration: a reference check with tolerances we declared,
with no figure from a paper as its target.

## Formulation

Reference (Jaxley HH, through `jaxfne.bridges.hh_jaxley_reference_trace`, 10 µA step):

$$C_m \frac{dV}{dt} = -\bar g_{Na} m^3 h (V - E_{Na}) - \bar g_K n^4 (V - E_K) - g_L (V - E_L) + I_{inj}$$

with first-order gating $\dot x = \alpha_x(V)(1 - x) - \beta_x(V)\,x$ for $x \in \{m, h, n\}$.

Reduced arm: the one-cell E configuration `suite2_single_neuron_config` (baseline drive 4.0),
run in native proxy units.

Declared before the run (`artifacts/atlas/at01_at06_052.py`): 50 ms, dt 0.5 ms, spike threshold
−20 mV; tolerances: first-spike time 2 ms, rate 10 Hz, peak 20 mV, rest 10 mV.

## Results against the declared tolerances

| Measure | HH reference | Reduced | Difference | Tolerance | Verdict |
|---|---|---|---|---|---|
| Rate (Hz) | 20.0 | 20.0 | 0.0 | 10 | pass |
| Peak $V$ (mV) | 39.58 | 13.68 | 25.91 | 20 | fail |
| Rest $V$ (mV) | −63.59 | −74.37 | 10.77 | 10 | fail |
| First spike (ms) | 10.5 | 0.5 | 10.0 | 2 | fail |

Source: `artifacts/atlas/results/at01_055.json`. No overall verdict is recorded; the failures
stand as recorded (caveats AT-01-R6 and AT-00-R4).

## Deviations and limitations

- The reduced arm runs in proxy units, so its voltages carry no calibration; the comparison
  shows how far the proxy sits from a physical cell.
- Each arm fires one spike in 50 ms, so the rate criterion rests on a single spike.
- Jaxley's HH has no M-current, so $I_M$ is dropped (AT-01-R2).
- One run, seed 7; the reference is deterministic.

## Reproduce

```
python scripts/emit_atlas_results_055.py --only at01
```

Requires the optional Jaxley install; without it the reference arm is refused with no
substitute. Output: `artifacts/atlas/results/at01_055.json`.
