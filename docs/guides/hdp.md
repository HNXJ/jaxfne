# HDP (Homeostasis-Dependent Plasticity)

**Stabilize population activity and adapt synaptic weights** with a structured
H-state. For each modeled biological entity, `H_i(t)` is a finite-dimensional
hidden biophysical state propagated with the neural dynamics. Its coordinates
provide latent variables that adaptation rules can use.

The current scalar HDP realization uses one homeostatic-like/resource-regulatory
coordinate, `d_H=1`, and couples activity, synaptic budget, and weight
adaptation in one loop. The generalized H-state supports multiple coordinates,
optional coupling, and adaptation-specific readouts.

`enable_homeostasis` and `enable_hdp` are mutually exclusive `RuntimeConfig`
fields; enabling both raises `ValueError`.

## The control law

The scalar compatibility form uses a per-neuron master state `H_i` (default
1.0) and integrates the implemented income, spending, restoration, and barrier
terms:

```
tau_i * dH_i/dt = alpha*I_syn_i + beta - gamma*H_i*r_i - delta*W_i
                  + rho_passive/H_i**2 + K_ctrl*(1 - H_i) - dC/dH_i
```

`tau_i = tau_0_ms * size_i**3` (cube law, verified 2026-06-25; size depends on
cell type via `size_scale_by_cell_type`, or an explicit `size_scale_override` —
default E `relative_size=5.0` integrates `H` exactly $5^3/1^3 = 125$ times slower
than `relative_size=1.0`). The terms are synaptic income (`alpha`), a constant
bias (`beta`), an H-taxed activity drain (`gamma`), a weight-budget drain
(`delta`), passive income (`rho_passive`), and a linear restoring control
(`K_ctrl`), plus an optional barrier term.

```
Delta_H_ij = H_post - H_pre
w_ij = q_ij * m_ij,  q_ij in {-1, +1}

dm_ij/dt = q_ij * K_HDP * phi(Delta_H_ij) * m_ij
           + K_w_ctrl * (m0_ij - m_ij)
```

For the difference family, `phi(x) = x` for `signed_linear` and
`phi(x) = x*abs(x)` for `signed_quadratic`. `hebbian_product` is a separate
product modulation with `phi = H_pre*H_post`; it does not compare pre/post
homeostatic state.

Use explicit null names:

```
N_W^HDP       K_HDP * phi(Delta_H_ij) * m_ij = 0
N_H           dH_i/dt = 0 and C_spike * spike_i = 0
N_system      (X, W, H)_HDP == (X, W, H)_baseline
              under matched initial state, inputs, and PRNG
```

`K_HDP=0` nulls the difference/product weight term, but does not disable the
H equation or `K_w_ctrl`. `K_ctrl` and `K_w_ctrl` are independent controls:
the former restores `H` toward 1, while the latter restores edge magnitude
`m_ij` toward its declared baseline `m0_ij`.

### Generalized H-state

The general H-state associated with entity `i` is:

$$H_i(t)\in\mathbb{R}^{d_H}.$$

The canonical sparse edge-list execution propagates it with:

$$H_{t+1}=F_H(H_t,X_t,U_t,\Theta_t).$$

Adaptation rules may consume the full state:

$$\dot{\Theta}=G_\Theta(H_{\mathrm{pre}},H_{\mathrm{post}},X,\Theta).$$

`h_state_dim=1` preserves the external legacy shape `(n_neurons)`. For
`h_state_dim>1`, `H` has shape `(n_neurons, h_state_dim)` and its coordinates
follow componentwise dynamics by default. An optional `h_state_coupling`
matrix supplies the initial supported coupling mechanism; omitting it means
zero coupling.

The current scalar HDP weight rule uses one supported adaptation-specific
readout:

$$h_i=R_H(H_i).$$

The initial implementation supports the linear form `r^T H_i`, configured with
`h_state_readout`; when omitted, the first coordinate is selected. The readout
serves the current scalar weight rule while the general H-state contract
remains vector-valued for future adaptation rules.

```python
runtime_hdp = jtfne.RuntimeConfig(
    enable_hdp=True,
    hdp_params={
        "h_state_dim": 2,
        "h_state_readout": [0.5, 0.5],
        "h_state_coupling": [[-0.02, 0.01], [0.01, -0.02]],
        "noise_scale": 0.0,
    },
)
signals, state = jtfne.simulate(
    model, duration_ms=10.0, dt_ms=0.5, runtime=runtime_hdp,
    return_state=True,
)
```

The continuation state carries vector `H` without a second state
representation. Ordinary and continuation HDP dispatch share the same
deterministic per-step PRNG contract, so matched scalar runs remain identical.
Readout and coupling dimensions are validated explicitly.

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
| `K_HDP` | Gain on the selected difference/product weight term (`0` = `N_W^HDP`; does not disable H dynamics or `K_w_ctrl`) |
| `tau_0_ms` | Base time constant multiplying `size_i**3` (cube law) |
| `size_scale_by_cell_type`, `size_scale_override` | Per-cell-type (or per-neuron) `size_i` for the cube-law `tau_i`; must be forwarded explicitly — see note below |
| `alpha`, `beta`, `gamma`, `delta`, `C_spike` | H income/spending terms; `C_spike` is the discrete spike drain |
| `rho_passive` | Passive H income term, stronger at low positive H |
| `K_ctrl` | H-state restoring control gain pulling `H_i` back toward 1.0 |
| `K_w_ctrl` | Independent edge-magnitude restoring gain toward `abs(edges.weight)` |
| `barrier_c`, `barrier_d` | Barrier-term coefficients near the `H_min`/`H_max` clamps |
| `h_state_dim` | H-state dimensionality; `1` keeps legacy `(n_neurons)`, larger values use `(n_neurons, h_state_dim)` |
| `h_state_readout` | Linear readout vector for the current scalar HDP weight rule; defaults to the first coordinate |
| `h_state_coupling` | Optional square component-coupling matrix; omitted means zero coupling |
| `record_weight_trace` | Default `True`. Set `False` to skip stacking the per-step, per-edge weight trace (`w_trace`) -- see below |

`model.last_hdp_diagnostics()` returns `H_final`/`H_trace` (shape
`(n_steps, n_neurons)` for scalar H or `(n_steps, n_neurons, h_state_dim)` for
vector H) and `w_final`/`w_trace` (shape `(n_steps, n_edges)`)
from the most recent `enable_hdp=True` run, or `None` if the last run had it
off.

**Memory at scale:** `w_trace` is `(n_steps, n_edges)`, which dominates
memory for large networks run over many steps (e.g. 10,000 steps x
2,000,000 edges x 4 bytes = 80GB -- a real reproduced OOM). Scalar `H_trace`
and spike/voltage traces use `(n_steps, n_neurons)`; vector `H_trace` uses
`(n_steps, n_neurons, h_state_dim)`. These remain ~100x smaller at a typical
`max_in_degree` and are not the source of this. If you don't need the
full per-step weight history, set `hdp_params={"record_weight_trace": False, ...}`
-- `w_final` (the terminal weight state) and HDP's actual dynamics are
unaffected either way; only `w_trace` becomes `None`.

## Barrier equilibrium (a numerical-methods result, independent of biology)

The barrier term contributes an asymmetric double-barrier potential

$$C(H) = \frac{\text{barrier\_c}}{H - H_{min}} + \frac{\text{barrier\_d}}{H_{max} - H}$$

to `dH/dt` (as `-dC/dH`), repelling `H` from both clamp boundaries. Setting
`dC/dH = 0` gives the potential's minimum:

$$\frac{\text{barrier\_c}}{(H-H_{min})^2} = \frac{\text{barrier\_d}}{(H_{max}-H)^2}$$

Solving for the minimum to sit exactly at the target equilibrium `H* = 1`
requires:

$$\frac{\text{barrier\_d}}{\text{barrier\_c}} = \left(\frac{H_{max}-1}{1-H_{min}}\right)^2$$

At the canonical `H_min=0.1`, `H_max=10.0`, this ratio is `((10-1)/(1-0.1))^2 = 100`.
This is a property of the barrier potential *alone* (independent of `alpha`/
`beta`/`gamma`/`delta`/`K_ctrl`/`rho_passive`), and it holds regardless of the
network, the drive, or any other simulation parameter -- a genuine
force-balance result, not a tuned/empirical fact. It is not a
proof of coupled-system convergence or asymptotic stability. **`jaxfne.hdp_network.DEFAULT_HDP`
ships `barrier_c=barrier_d=0.01` (ratio 1, not 100)** -- solving the same
minimum-condition equation with equal coefficients gives a pure-barrier
equilibrium of `H ≈ 5.05`, not `1.0`. `DEFAULT_HDP`'s real recovery-to-1
comes from its `K_ctrl=5.0` linear restoring term, not the barrier; the
barrier there is a boundary safety net, dormant under normal operation
(confirmed: `tests/test_hdp_barrier_equilibrium.py` reproduces both the
ratio-100 (`H→1.0`) and ratio-1 (`H→5.05`) cases directly against the
kernel, with everything else (`alpha=beta=gamma=delta=K_ctrl=rho_passive=0`)
held at the null so the barrier is the only active force).

## Tensor-first: enabling HDP on a NeuronalTensor-built Model

The recipe above builds the `Model` via `Configuration`. HDP works identically
on a `Model` built from a `NeuronalTensor` (the declarative `Areas x Layers x
NeuronTypes` data model — see [`docs/api/neuronal_tensor.md`](../api/neuronal_tensor.md)):
construct the tensor with `RuntimeConfiguration` (which has no HDP field),
then pass an explicit `runtime=RuntimeConfig(enable_hdp=True, ...)` override
to `simulate()` — this works with zero new public API because the explicit
`runtime=` kwarg overrides any `Configuration`-derived metadata entirely:

```python
import jaxfne as jtfne
jtfne.enable_x64()

E, PV = jtfne.NeuronType.make("E", fraction=0.9), jtfne.NeuronType.make("PV", fraction=0.1)
tensor = jtfne.NeuronalTensor(areas=[
    jtfne.Area(name="V1", layers=[jtfne.Layer(name="L4", n_neurons=100, neuron_types=[E, PV])])
])
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))

runtime_hdp = jtfne.RuntimeConfig(enable_hdp=True, hdp_params={
    "K_HDP": 0.01, "tau_0_ms": 200.0, "K_ctrl": 5.0,
    "size_scale_by_cell_type": {"E": 2.0, "PV": 1.0},   # tau_i = tau_0_ms * size_i**3
})
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0, runtime=runtime_hdp)
diag = model.last_hdp_diagnostics()
```

`size_scale_by_cell_type`/`size_scale_override` are free-form `hdp_params`
keys — they must be explicitly forwarded by `core.py`'s internal HDP dispatch
to reach the kernel (fixed 2026-06-25; previously silently dropped). Verify
with `grep -n size_scale_by_cell_type jaxfne/core.py` before relying on a new
`hdp_params` key reaching the simulation.

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
state remains numerically bounded even under extreme drive — verified in
`tests/test_homeostatic_stability_v042.py`'s HDP parity tests.
Bounded is not synonymous with locally stable, asymptotically stable, or
empirically stable over a specified horizon.

## Using it as evidence

The H-state and HDP weight-term nulls are separate controls for clean
ablation comparisons; neither should be called full-system null equivalence.
`scripts/ed9_hdp_evidence.py` runs a 3-way ablation grid
(`null` / `h_dynamics` / `both`) over repeated seeds on a
deliberately imbalanced column and reports rate-spread reduction alongside
`H_mean`/`H_std`, with the same conservative truth gates as
`scripts/ed9_homeostasis_evidence.py`.

## See also

- [Homeostasis](homeostasis.md) — the simpler, single-dial excitability controller.
- [Configuration Grammar](configuration_grammar.md) — where the runtime and emitter fit in the chain.
- [HDP Implementation Report](../HDP_REPORT.md) — what was built, how it's verified, and measured overhead.
