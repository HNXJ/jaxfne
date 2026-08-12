# jaxfne

**Tensor-Field Neural Equations (TFNE)** — JAX simulation for layer-resolved neural
circuits, source operators, field proxies, probes, objectives, and evidence.

**Scientific grammar:** Emitter → Source → Field → Probe → Objective → Optimizer → Evidence

**Execution grammar:** CircuitSpec → `construct` → `Model` → `simulate` → `Signals`

[Jaxley](https://jaxley.readthedocs.io) provides compartmental biophysical detail;
jaxfne provides population/field-scale circuits and proxy readouts. Jaxley models
attach as emitters via [Jaxley interoperability](guides/jaxley_interop.md).

[Scope & status](scope_and_status.md) · [Public API contract](public_surface_contract.md) (0.4.13)

## Install

```bash
pip install -U jaxfne
pip install "jaxfne[viz]"
```

## Minimal example

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
```

## Main pages

- [Quickstart](quickstart.md) — build paths, paradigms, H-state adaptation
- [Tutorials](tutorials/index.md) — usage of the grammar
- [Études](etudes/index.md) — demonstrated scientific propositions
- [API reference](api/index.md)
- [H-state / HDP guide](guides/hdp.md)
