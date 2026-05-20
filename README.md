# jaxfne

**JAX Field Neural Equations** (`jaxfne`) is a compact JAX-native source-to-field/readout engine for Tensor-Field Neural Equations (TFNE).

It provides a practical object-oriented API for building reproducible computational neurophysiology workflows:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

`jaxfne` is designed for laminar proxy simulations, source-field metadata, readouts, objective reports, receipts, manifests, and benchmarkable JAX execution paths.

## Installation

From PyPI:

```bash
pip install jaxfne
```

In Colab:

```python
%pip install jaxfne
```

From a local checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .
```

Optional development tools:

```bash
pip install -e .[dev]
```

Optional optimizer / bridge extras are installed only when needed:

```bash
pip install -e .[opt]
pip install -e .[jaxley]
```

## Minimal example

```python
import json
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        name="V1_proxy",
        kind="cortical_column",
        n=100,
        layers=["L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(
        name="laminar_probe",
        modes=["spikes", "V_m", "source", "CSD", "LFP"],
        n_contacts=16,
    )
)

model = jtfne.construct(cfg)

sim = jtfne.simulation(
    duration_ms=300.0,
    dt_ms=0.1,
    seed=0,
    record_sources=True,
    record_fields=True,
)

signals = model.simulate(sim)
receipt = model.run_receipt(signals)

readouts = model.compute_readout(
    signals,
    [
        jtfne.readout_spec("rate", "spike_rate_hz"),
        jtfne.readout_spec("source", "source_abs_mean"),
        jtfne.readout_spec("csd", "csd_abs_mean"),
        jtfne.readout_spec("lfp", "lfp_abs_mean"),
    ],
)

manifest = model.manifest(signals, readouts)
json.dumps(manifest, allow_nan=False)

print("jaxfne", jtfne.__version__)
print("spikes:", signals.spikes.shape)
print("Vm:", signals.V_m.shape)
print("source:", None if signals.sources is None else signals.sources.shape)
if signals.field is not None:
    print("LFP:", signals.field.lfp.shape)
    print("CSD:", signals.field.csd.shape)
print("receipt_id:", receipt.receipt_id)
for result in readouts:
    print(result.name, result.metric, result.value, result.status)
```

## What it supports

`jaxfne` supports compact TFNE-style computational workflows with:

- Izhikevich emitter scaffolds;
- dense and edge-list recurrent paths;
- scan-backed JAX simulation kernels;
- native stimulus / drive schedules;
- laminar source geometry metadata;
- source proxy traces;
- LFP-proxy and CSD-proxy laminar readouts;
- readout specifications;
- objective reports;
- run receipts;
- strict JSON-safe manifests;
- CPU-first validation and optional accelerator execution through JAX.

The standard v0.2.0 field mode is a **laminar proxy readout**:

```text
source_projection_mode = proxy_no_field_solve
field_solver_status = laminar_proxy_no_pde
field_claim_level = proxy_readout_only
```

This is the intended basis for practical laminar spectrolaminar proxy simulations.

## Scope and capabilities

`jaxfne` is designed for building and testing source-to-field/readout workflows on CPU-first infrastructure.

Typical uses:

- Reproducible proxy simulations with local-global organization
- Source/readout bookkeeping and JSON-safe output bundles
- Objective scaffolds and optimization experiments
- Performance benchmarking with deterministic PRNG
- Integration with Jaxley-style models and other JAX workflows

**Default readouts are computational proxies.** Physical-unit EEG/MEG/LFP/CSD workflows require appropriate geometry, calibration, and validation for the intended use. See [Scope and limitations](docs/scope_and_limitations.md) for details.

## Documentation

Full documentation, tutorials, guides, and API reference are available at:

**[jaxfne.readthedocs.io](https://jaxfne.readthedocs.io/)**

Or in the `docs/` directory of the repository.

## Package layout

```text
jaxfne/
  __init__.py     public API exports
  core.py         configuration, model, simulation, signals, receipts, readouts
  emitters.py     Izhikevich and recurrent emitter kernels
  fields.py       laminar proxy source/field/readout logic
  objectives.py   objective and evaluation scaffolds
  optim.py        optimizer specs and optional Optax guard
  runtime.py      JAX runtime, dtype, device, and reproducibility reports
  io.py           JSON-safe manifests, hashes, save/load helpers
  validation.py   invariant and claim-gate checks
  bridges.py      optional external backend guards
```

## Validation

Core local checks:

```bash
python -m compileall -q jaxfne tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q --tb=short
```

Run examples:

```bash
python examples/00_minimal_column.py
python examples/01_source_field_manifest.py
python examples/02_spectrolaminar_oddball_scaffold.py
```

## Documentation

Release notes and detailed version history live outside the README:

- `CHANGELOG.md`
- `docs/`
- `examples/`
- `scripts/benchmark_scan_backends.py`

## License

MIT License.
