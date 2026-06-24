# HDP (Homeostasis-Dependent Plasticity)

jaxfne includes a second, more structured excitability/plasticity controller:
**HDP**. Where [homeostasis](homeostasis.md) adapts an intrinsic excitability
bias per neuron, HDP gives each neuron a slow **master state** `H_i` that both
reflects the neuron's own activity/synaptic budget AND drives its excitatory
and inhibitory edge weights. It is one extra set of gains per emitter, and like
homeostasis it is a *control method*, not a claimed biological plasticity
mechanism — jaxfne is the mathematical backend.

`enable_homeostasis` and `enable_hdp` are mutually exclusive `RuntimeConfig`
fields (they are two distinct controllers); enabling both raises `ValueError`.

## The control law

Per neuron, a master state `H_i` (default 1.0) integrates five additive terms:

```
tau_i * dH_i/dt = alpha*I_syn_i + beta - gamma*r_i - delta*W_i
                  + K_ctrl*(1 - H_i) - dC/dH_i
```

`tau_i = tau_0_ms * size_i**2` (size depends on cell type, or an explicit
`size_scale_override`). The five terms are: synaptic income (`alpha`), a
constant bias (`beta`), an activity drain (`gamma`), a weight-budget drain
(`delta`), and a restoring control term (`K_ctrl`) plus an optional barrier
term. `H_i` then drives excitatory/inhibitory weight ODEs:

```
dw_E/dt = +K_HDP * (H_i - 1) * w_E
dw_I/dt = -K_HDP * (H_i - 1) * w_I
```

**The null control:** `alpha = beta = gamma = delta = C_spike = 0.0` holds
`H_i` pinned at exactly `1.0` forever, which makes the `K_HDP`-scaled weight
term identically zero regardless of `K_HDP`'s value. Equivalently, `K_HDP = 0`
disables plasticity outright even if `H_i` is moving — these are two
independent ablation axes (whether `H` moves vs. whether weights respond to
it), unlike homeostasis's single `k_gain` dial.

## Built-in emitter (per-step kernel)

Enable it on the runtime; the built-in Izhikevich kernel applies the weight
ODEs every step. After simulating, read the controller diagnostics off the
model.

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
          enable_hdp=True,
          hdp_params={"K_HDP": 0.01, "tau_0_ms": 200.0, "K_ctrl": 5.0,
                      "barrier_c": 0.01, "barrier_d": 0.01},
      )
)
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model)

diag = model.last_hdp_diagnostics()   # {"H_trace": ..., "w_trace": ..., ...}
```

`hdp_params` keys:

| Key | Meaning |
|-----|---------|
| `K_HDP` | Weight-ODE gain (`0` = plasticity disabled / null; default in presets `0.01`) |
| `tau_0_ms` | Base time constant multiplying `size_i**2` |
| `alpha`, `beta`, `gamma`, `delta`, `C_spike` | The five `dH/dt` income/drain terms (all default `0.0` — the null) |
| `K_ctrl` | Restoring control gain pulling `H_i` back toward 1.0 |
| `barrier_c`, `barrier_d` | Barrier-term coefficients near the `H_min`/`H_max` clamps |

`model.last_hdp_diagnostics()` returns `H_final`/`H_trace` (shape
`(n_steps, n_neurons)`) and `w_final`/`w_trace` (shape `(n_steps, n_edges)`)
from the most recent `enable_hdp=True` run, or `None` if the last run had it
off.

## Fluent verb

`Configuration.hdp(relative_baseline=1.0, **kwargs)` mirrors
`Configuration.homeostasis(...)`: `relative_baseline=1.0` is the identity
baseline (`enable_hdp=False`, unchanged `simulate()` output); deviating from
`1.0` (or passing any `hdp_params` override) resolves
`K_HDP = relative_baseline - 1.0` and activates the controller. The spec is
visible in `manifest()["hdp"]` from the first call.

```python
cfg = jtfne.Configuration()...hdp(K_HDP=0.01, alpha=0.05, gamma=0.5, K_ctrl=0.15)
```

## Tuned presets

`jaxfne.hdp_network.DEFAULT_HDP` and `DEFAULT_HDP_DESYNC` are frozen, verified
starting points for a canonical laminar column (see
`jaxfne/hdp_network.py` docstrings for the stability/desync tradeoffs each was
tuned for) — never merge them; pick the one matching your goal
(long-term-stable vs. faster-desynchronizing `H` dynamics).

## Stability

The built-in kernel hard-bounds its state (`v`/`u`/synaptic variables, `H_i`
clamped to `[H_min, H_max]`, weights clamped to `[w_floor, w_ceiling]`) so the
dynamics stay finite even under extreme drive — verified in
`tests/test_homeostatic_stability_v042.py`'s HDP parity tests.

## Using it as evidence

Because the null control is exact (`H` pinned, weights frozen), HDP is built
for clean ablation comparisons. `scripts/ed9_hdp_evidence.py` runs a 3-way
ablation grid (`null` / `h_dynamics` / `both`) over repeated seeds on a
deliberately imbalanced column and reports rate-spread reduction alongside
`H_mean`/`H_std`, with the same conservative truth gates as
`scripts/ed9_homeostasis_evidence.py`.

## See also

- [Homeostasis](homeostasis.md) — the simpler, single-dial excitability controller.
- [Configuration Grammar](configuration_grammar.md) — where the runtime and emitter fit in the chain.
- [HDP Implementation Report](../HDP_REPORT.md) — what was built, how it's verified, and measured overhead.
