# jaxfne

**JAX Field Neural Equations**: a JAX-native source-to-field neurophysiology engine for Tensor-Field Neural Equations.

```python
import jax
import jax.numpy as jnp
import jaxfne as jtfne

cfg = jtfne.configuration()
cfg = cfg.network(
    name="V1",
    kind="cortical_column",
    n=1000,
    layers=["L1", "L2/3", "L4", "L5", "L6"],
    cell_types={"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03},
)
cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
cfg = cfg.field(domain="laminar_column", conductivity="isotropic", boundary="mean_zero_neumann", gauge="mean_zero")
cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"])

model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.05, plasticity=1.0, seed=0)
signals = model.simulate(sim)
readout = model.record(signals, modes=["spikes", "V_m", "CSD", "LFP"])
```

## Doctrine

`jaxfne` is not primarily a neuron model, optimizer, plotting library, or data format. It is the composition layer:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

JAX handles array compilation and batching. Jaxley can provide detailed emitters. Optax can provide differentiable optimizers. `jaxfne` handles TFNE source-to-field/readout contracts, diagnostics, and manifests.

## Skeleton status

This is an initial design skeleton for brainstorming. It intentionally favors a small number of cohesive files:

```text
jaxfne/
  __init__.py
  core.py
  emitters.py
  fields.py
  optim.py
  bridges.py
  io.py
```

The code runs a minimal JAX-native Izhikevich EIG scaffold, source projection, simple laminar field/probe placeholders, objective hooks, and manifest writing. Scientific field claims remain `truth_safe_unverified` until source calibration, boundary/gauge, solver residual, and validation gates are fully implemented.
