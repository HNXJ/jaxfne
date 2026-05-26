# 100-neuron Network


Build, simulate, and inspect a 100-neuron balanced excitatory/inhibitory (75E / 25I) network.
Extract all eight proxy readouts from population activity.

## Open as Colab notebook

**Recommended:** Open the full interactive tutorial in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/notebooks/03_network_100_ei_multimodal.ipynb)

Or download and run locally: `notebooks/03_network_100_ei_multimodal.ipynb`

## Network configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        name="network_100_ei",
        kind="balanced_ei_population",
        n=100,
        cell_types={"E": 0.75, "I": 0.25},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="declared_proxy",
        gauge="mean_zero",
    )
    .probe(
        name="multimodal_100_ei",
        modes=[
            "spikes",
            "V_m",
            "source",
            "phi_e",
            "J_e",
            "CSD",
            "LFP",
        ],
    )
)

model = jtfne.construct(cfg)
```

## Simulate and extract readouts

```python
signals = model.simulate(jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42))

# Apply all eight probe operators
from jaxfne.fields import (
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
)

spk_readout = spk_probe(signals.spikes)
vm_readout = vm_probe(signals.V_m)
source_readout = source_probe(signals.sources[0])
lfp_readout = lfp_proxy_probe(signals.field.lfp_proxy)
csd_readout = csd_proxy_probe(signals.field.csd_proxy)
eeg_readout = eeg_proxy_probe(signals.field.lfp_proxy)
meg_readout = meg_proxy_probe(signals.field.lfp_proxy)
emm_readout = emm_proxy_probe(signals.field.lfp_proxy)
```

## Analyze population dynamics

```python
# Population spike counts and rates
total_spikes = jnp.sum(signals.spikes)
excitatory_spikes = jnp.sum(signals.spikes[:, :75])
inhibitory_spikes = jnp.sum(signals.spikes[:, 75:])
pop_rate = jnp.mean(jnp.sum(signals.spikes, axis=1))

# Voltage statistics
vm_mean = jnp.mean(signals.V_m)
vm_std = jnp.std(signals.V_m)

print(f"Total spikes: {total_spikes}")
print(f"Excitatory: {excitatory_spikes}, Inhibitory: {inhibitory_spikes}")
print(f"Population rate: {pop_rate} spikes/timestep")
print(f"Voltage: {vm_mean:.2f} ± {vm_std:.2f} mV")
```

## Key observations

- 100 neurons with JAX vmap ensures efficient CPU computation
- Balanced E/I network maintains stable asynchronous activity
- Population-level field projections emerge from neural sources
- All eight proxy operators scale smoothly to population level
- Output bundle remains JSON-serializable and reproducible

## Next step

Progress to [V1 six-layer column](04_v1_column.md) for structured laminar networks.
