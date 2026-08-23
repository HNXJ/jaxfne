# Homeostasis

jaxfne includes a **homeostatic excitability controller**: a minimal computational
control method that keeps each neuron's firing rate near a set-point by adapting an
intrinsic excitability bias. It is one extra parameter per emitter, and switching
it on does three useful things at once:

- **Eliminates hyperactivity** — runaway/saturated units are pulled back down.
- **Eliminates hypoactivity** — silent units are nudged up into a working range.
- **Models short-term adaptation** — the slow rate trace gives spike-frequency
  adaptation for free.

It is a *control method*, not a claimed biological plasticity mechanism — a
restoring controller on excitability, useful as a stabilizer and as an adaptation
proxy. jaxfne is the mathematical backend; the controller's role in a model is
whatever the configuration makes it.

## The control law

Per neuron, a slow activity trace `r` tracks recent firing; a restoring bias `g`
is added to the input current:

```
r ← decay · r + (1 − decay) · activity          # slow rate trace
g  = clip(k_gain · (r_star − r), g_min, g_max)   # over-active → g < 0, under-active → g > 0
I  ← I + g                                        # excitability bias into the input current
```

`k_gain` is the single dial. **`k_gain = 0` disables the controller — a clean
null control** for ablation experiments. The default `k_gain = 1.0` is a gentle
in-band nudge.

## Built-in emitter (per-step kernel)

Enable it on the runtime; the built-in Izhikevich kernel applies the bias every
step. After simulating, read the controller diagnostics off the model.

```python
import jaxfne as jtfne
jtfne.enable_x64()

cfg = (
    jtfne.build_laminar_column(n=1000)
      .set_emitter("izhikevich", "cortical_eig")
      .probes(["spikes", "V_m"])
      .field(domain="laminar_column", conductivity="proxy")
      .runtime(
          duration_ms=1000.0, dt_ms=0.5, seed=0,
          enable_homeostasis=True,
          homeostasis_params={"k_gain": 1.0, "r_star": 0.01},
      )
)
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model)

diag = model.last_homeostasis_diagnostics()   # {"g_bias": ..., "r_trace": ...}
```

`homeostasis_params` keys:

| Key | Meaning |
|-----|---------|
| `r_star` | Target activity set-point the controller restores toward |
| `k_gain` | Restoring gain (`0` = disabled / null; default `1.0`) |
| `tau_r_ms` | Time constant of the slow rate trace |
| `alpha` | Trace update mixing factor |
| `g_min`, `g_max` | Clamp on the excitability bias |
| `r_max` | Activity-trace ceiling |

`model.last_homeostasis_diagnostics()` returns the per-neuron `g_bias` and
`r_trace` from the most recent `enable_homeostasis=True` run (or `None` if the
last run had it off).

## Jaxley emitter (outer-loop windowed)

The same control law wraps a [Jaxley](jaxley_interop.md) emitter, so you keep
Jaxley's channel/morphology toolset and gain the homeostasis. Because Jaxley's
`integrate` is a closed scan, the feedback runs at **window cadence** rather than
per step, and the controller is parameterized by firing rate in **Hz** (which is
dt-independent):

```python
net = ...  # a Jaxley HH Network with recordings on the controlled compartments
bridge = jtfne.JaxleyBridge(model=net)

signals, diag = bridge.simulate_homeostatic(
    duration_ms=400.0,
    base_current_nA=0.01,
    target_rate_hz=40.0,    # set-point in Hz
    k_gain=0.2,             # 0 = null
    window_ms=20.0,
    return_diagnostics=True,
)
# diag["g_bias"], diag["r_trace"], diag["rate_hz"] -> [n_windows, n_cells]
```

The injected current is hard-bounded (`current_clip_nA`) and finiteness is
verified (`strict_finite=True` raises rather than masking) — the controller keeps
the implicit solver in a finite, stable regime even under heterogeneous drive. See
`JaxleyBridge.simulate_homeostatic` in the Bridges API (`docs/api/bridges.md` — repository-internal reference, excluded from the built site).

!!! warning "Stay in the monotonic f-I band (Jaxley path)"
    Jaxley single-compartment HH has a non-monotonic f-I curve (high current →
    depolarization block). Keep `base_current_nA` and the bias range inside the
    monotonic band (roughly 0.002–0.02 nA for the default compartment), so the
    controller operates where firing rate increases with current.

## Stability

Both paths are float32-safe: the built-in kernel hard-bounds its state
(`v`/`u`/synaptic variables) so the dynamics stay finite even under extreme drive,
and the Jaxley path hard-bounds the injected current and checks finiteness. The
controller therefore doubles as a numerical stabilizer.

## Using it as evidence

Because `k_gain = 0` is a true null, the controller is built for clean
comparisons: run with `k_gain = 0` and `k_gain > 0`, hold everything else fixed,
and attribute the difference (rate spread collapsing toward the set-point, silent
units recovering) to the controller. Report the null alongside the result.

## See also

- Bridges API (`docs/api/bridges.md` — repository-internal reference, excluded from the built site) — `simulate_homeostatic` parameters in full.
- [Configuration Grammar](configuration_grammar.md) — where the runtime and emitter fit in the chain.
- [Jaxley Interoperability](jaxley_interop.md) — real channels/morphology as emitters.
- [HDP](hdp.md) — the more structured per-neuron `H_i` controller (also works tensor-first, on a `NeuronalTensor`-built `Model`) if a single `k_gain` dial here isn't enough.
