# jaxfne

JAX-based tools for TFNE source, field, probe, objective, and optimizer workflows.

The full pipeline is one linear chain — each step returns the input to the next:

```text
setup -> config -> construct -> simulate -> visualize -> tune/objective -> optimize -> export
```

## Install

```bash
pip install -U jaxfne
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .[dev,viz,opt]
```

## Minimal example

```python
import jaxfne as jtfne
jtfne.enable_x64()

# Canonical laminar cortex — no arguments required for the default prior.
cfg = (jtfne.build_laminar_column(n=1000, ei_profile="canonical")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
print(signals.V_m.shape, signals.spikes.sum())
```

`ei_profile="canonical"` applies the verified ground-truth E:I gradient (E peaks
deep, I peaks superficial, ≈77E:23I); `ei_profile="flat"` (the default) keeps the
legacy depth-invariant composition. See the [Quickstart](quickstart.md) for the
flat path and tuning.

## Main pages

- [Install](install.md)
- [Quickstart](quickstart.md)
- [Probe operators](probe_operators.md)
- [Tutorials](tutorials/index.md)
- [API reference](api/index.md)
- [Changelog](changelog.md)
