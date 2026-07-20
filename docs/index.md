# jaxfne

**JAX-based simulation of Tensor-Field Neural Equations** — emitter-to-source-to-field
readouts for computational electrophysiology. Declarative circuit definition,
canonical cortical priors, and optimization over population readouts, at population
and field scale.

jaxfne and [Jaxley](https://jaxley.readthedocs.io) are complements, not competitors:
Jaxley provides single/multi-compartment biophysical detail, jaxfne provides
population/field-scale, tensor-algebraic circuit definition and source-to-sensor-proxy
readouts. A Jaxley model plugs directly into jaxfne as an emitter via `JaxleyBridge`
and uses the same readout stack as built-in Izhikevich emitters. Details:
[Jaxley interoperability](guides/jaxley_interop.md).

Scope & status: [Scope & status](scope_and_status.md).

## Install

```bash
pip install -U jaxfne
pip install "jaxfne[viz]"   # optional plotting
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e ".[dev,viz]"
```

## Minimal example

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
print(signals.get("spk").shape)
```

Prefer the fluent `Configuration` builder or Jaxley bridge? See [Quickstart](quickstart.md).

## Main pages

- [Quickstart](quickstart.md) — three build paths, canonical column, HDP
- [Tutorials](tutorials/index.md)
- [API reference](api/index.md)
- [Scope & status](scope_and_status.md) — Relative vs Absolute value reference
- [Changelog](changelog.md)
- AI-oriented workflow notes: [For AI agents](for_ai_agents.md)
