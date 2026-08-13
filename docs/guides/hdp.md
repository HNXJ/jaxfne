# RBS, RBD, and HDP

**Relative Biophysical State (RBS)** is a finite-dimensional hidden biophysical
state \(\mathbf H_i(t)\in\mathbb R^{d_H}\) carried with the neural dynamics.
**Relative Biophysical Dynamics (RBD)** is the coupled evolution of activity
\(\mathbf x\) and RBS. **HDP (Hidden-state Dependent Plasticity)** is the
optional subset in which persistent parameters (e.g. synaptic weights) evolve
from RBS and activity:

\[
\dot{\mathbf x}_i = F_x(\mathbf x_i,\mathbf I_i;\mathbf H_i),
\qquad
\dot{\mathbf H}_i = F_H(\mathbf H_i,\mathbf x_i,\mathbf I_i,\ldots),
\qquad
\dot{\mathbf W} = F_W(\mathbf W,\mathbf H,\mathbf x,\ldots)\quad\text{(HDP only)}.
\]

RBS is **relative biophysical state**, not necessarily a normalized copy of
individually identifiable physical variables. Coordinates may be
\(H_k=z_k/z_k^\star\) or reduced \(H_k=\mathcal R_k(\mathbf z)\). Homeostasis
is a possible **regime** of selected \(F_H\) or kernel-specific mechanisms (see
[Homeostasis](homeostasis.md), `homeostatic_ei`); it is **not** the definition
of RBS.

RBD remains meaningful with \(\dot W=0\). Adaptation, memory, and delayed
recurrent coupling are RBD phenomena and do not require plasticity.

Full Markov continuation requires the complete dynamical state
\(\mathcal X_t=(\mathbf x_t,\mathbf H_t,\mathbf W_t,\mathcal B_t,\ldots)\),
including delay history \(\mathcal B_t\) when finite edge delays are enabled
(Protocol D). See `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md`
§2.3 and `docs/doctrine/rbs_rbd_hdp.md`.

Public API names (`enable_hdp`, `hdp_params`, `DEFAULT_HDP`, `h_state_*`) are
compatibility surfaces; the acronym **HDP** denotes Hidden-state Dependent
Plasticity.

`enable_homeostasis` and `enable_hdp` are mutually exclusive `RuntimeConfig`
fields.

For the principal executable generalized-\(H\) demonstration, see the
[controllability / reachability étude](../etudes/hdp_controllability_reachability.md).

## Locality and shape

| locality | RBS \(H\) shape | role |
|----------|-----------------|------|
| `node` | \((N, d_H)\) or \((N,)\) when \(d_H=1\) | per-neuron RBS on the edge-list kernel |
| `population` | \((d_H,)\) for a single population summary | population-local RBS (supported \(d_H=2\) in current release) |

Public population semantics: set `h_state_locality="population"` with the
adaptive-parameter coefficients (`controller_*`, channel masks, bounds).
The runtime resolves the internal dispatch; callers do not supply MVC-specific
rule identifiers.

## Continuation scope

| capability | status |
|------------|--------|
| Node-local RBS continuation (`DynamicState`, `return_state=True`) | supported for scalar and vector \(d_H\) on the edge-list HDP kernel |
| Population-local RBS continuation | not supported in 0.4.13+ (explicit error) |
| Continuation with nonzero edge delays | not supported until \(\mathcal B_t\) carry is implemented |

## Scalar node HDP kernel (compatibility form)

The scalar node realization uses one RBS coordinate per neuron, \(d_H=1\), and
couples activity, synaptic budget, and weight adaptation in one loop. The
generalized contract supports \(d_H>1\) with optional readout and coupling.

## Node control law

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
RBS coordinates.

Use explicit null names (see `4_tfne_theory_and_neural_tensor.md` §8.4):

```
N_W^HDP       K_HDP * phi(Delta_H_ij) * m_ij = 0
N_H           dH_i/dt = 0 and C_spike * spike_i = 0   (RBS-dynamics null)
N_system      full X_t, W match baseline under matched PRNG/inputs
```

`K_HDP=0` nulls the difference/product weight term, but does not disable the
RBS equation or `K_w_ctrl`. `K_ctrl` and `K_w_ctrl` are independent controls:
the former restores `H` toward 1, while the latter restores edge magnitude
`m_ij` toward its declared baseline `m0_ij`.

### Generalized RBS

The general RBS associated with entity `i` is:

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
`h_state_readout`; when omitted, the first coordinate is selected.

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
| `K_HDP` | Gain on the selected difference/product weight term (`0` = `N_W^HDP`; does not disable RBS dynamics or `K_w_ctrl`) |
| `tau_0_ms` | Base time constant multiplying `size_i**3` (cube law) |
| `size_scale_by_cell_type`, `size_scale_override` | Per-cell-type (or per-neuron) `size_i` for the cube-law `tau_i`; must be forwarded explicitly — see note below |
| `alpha`, `beta`, `gamma`, `delta`, `C_spike` | RBS income/spending terms; `C_spike` is the discrete spike drain |
| `rho_passive` | Passive RBS income term, stronger at low positive H |
| `K_ctrl` | RBS restoring control gain pulling `H_i` back toward 1.0 |
| `K_w_ctrl` | Independent edge-magnitude restoring gain toward `abs(edges.weight)` |
| `barrier_c`, `barrier_d` | Barrier-term coefficients near the `H_min`/`H_max` clamps |
| `h_state_dim` | RBS dimensionality; `1` keeps legacy `(n_neurons)`, larger values use `(n_neurons, h_state_dim)` |
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

$$\frac{\text{barrier\_d}}{\text{barrier\_c}} = \left(\frac{H_{max}-1}{1-H_{min}}\right)^2$$

Solving for the minimum to sit exactly at the target equilibrium `H* = 1`
requires ratio \(100\) at canonical `H_min=0.1`, `H_max=10.0`. See
`tests/test_hdp_barrier_equilibrium.py`.

## Tensor-first: enabling HDP on a NeuronalTensor-built Model

The recipe above builds the `Model` via `Configuration`. HDP works identically
on a `Model` built from a `NeuronalTensor` — pass an explicit
`runtime=RuntimeConfig(enable_hdp=True, ...)` override to `simulate()`:

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
    "size_scale_by_cell_type": {"E": 2.0, "PV": 1.0},
})
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0, runtime=runtime_hdp)
diag = model.last_hdp_diagnostics()
```

## Fluent verb

`Configuration.hdp(relative_baseline=1.0, **kwargs)` mirrors
`Configuration.homeostasis(...)`: `relative_baseline=1.0` is the identity
baseline (`enable_hdp=False`); deviating activates the HDP controller. The spec
is visible in `manifest()["hdp"]` from the first call.

## Tuned presets

`jaxfne.hdp_network.DEFAULT_HDP` and `DEFAULT_HDP_DESYNC` are frozen,
verified starting points — see `jaxfne/hdp_network.py` docstrings.

## Stability

The built-in kernel hard-bounds its state so trajectories remain numerically
finite under extreme drive — verified in
`tests/test_homeostatic_stability_v042.py`. Bounded is not synonymous with
locally stable, asymptotically stable, or empirically stable over a specified
horizon.

## Using it as evidence

The RBS-dynamics null (`N_H`) and HDP weight-term null (`N_W^HDP`) are separate
controls; neither is full-system null equivalence.
`scripts/ed9_hdp_evidence.py` runs a 3-way ablation grid over repeated seeds.

## See also

- [Doctrine: RBS/RBD/HDP](../doctrine/rbs_rbd_hdp.md)
- [Homeostasis](homeostasis.md) — kernel-specific homeostatic excitability controller (distinct from generic RBS)
- [Configuration Grammar](configuration_grammar.md)
- [HDP controllability / reachability étude](../etudes/hdp_controllability_reachability.md)
