<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI package"></a>
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/pyversions/jaxfne" alt="Python versions"></a>
  <a href="https://jaxfne.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/jaxfne/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml"><img src="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://jaxfne.readthedocs.io/contributing/"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="contributions welcome"></a>
  <a href="https://github.com/HNXJ/jaxfne/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

# jaxfne

`jaxfne` is a compact JAX package for Tensor-Field Neural Equations (TFNE): a typed computational chain from neural emitters to source tensors, field-proxy operators, probe readouts, objective reports, optimizers, and run manifests.

Built for computational and systems neuroscientists who want differentiable, JAX-based circuit models — a canonical cortical-column prior with the real laminar E:I gradient, an interactive 3D viewer for inspecting circuit structure before you simulate, and a direct bridge into [Jaxley](https://jaxley.readthedocs.io) so single- and multi-compartment biophysical (HH, conductance-based) neurons slot into the same field/readout pipeline as the built-in point-neuron emitters.

## Install

```bash
pip install jaxfne
```

Visualization extras:

```bash
pip install "jaxfne[viz]"
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e ".[dev,viz]"
```

Canonical import:

```python
import jaxfne as jtfne
```

## Object grammar

Every jaxfne program is the same linear chain. Each step returns the input to the
next, so the whole pipeline reads as one fluent sequence:

```text
setup  ->  config  ->  construct  ->  simulate  ->  visualize  ->  tune/objective  ->  optimize  ->  export
enable_x64   Configuration   Model       Signals     vis.*           Objective         Model.tune   manifest / save_*
```

The full typed chain underneath that sequence:

```text
Config
  -> Runtime(seed, dtype, backend, jit, vmap, duration_ms, dt_ms)
  -> Identity(area, layer, cell_type, unit_id, position)
  -> Emitter(theta_e, state_0, drive, noise, key)
  -> SourceMap(source_mode, source_calibration_status, support, normalization)
  -> FieldProxy(kernel, geometry_metadata, field_solver_status, field_claim_level)
  -> Probe(kind, selector, channel_geometry, units_status, method)
  -> Signals(spk, vm, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, spectrolaminar_proxy, emm_proxy)
  -> Objective(metrics, targets, gates, nulls, rejection_reasons)
  -> Optimizer(search_space, budget, key, constraints)
  -> Manifest(run_id, version, repo_sha, runtime_report, artifact_paths, asset_hashes, truth_gates)
  -> Validation(finite_outputs, strict_json, png_assets, notebook_receipts, optional_dependency_laziness)
```

## Minimal workflow

```python
import jaxfne as jtfne

jtfne.enable_x64()                                   # setup: x64 before arrays

cfg = jtfne.build_laminar_column()                   # config: V1, n=1000, flat E:I (legacy default)
cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))

model   = jtfne.construct(cfg)                        # construct: Configuration -> Model
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)  # simulate -> Signals
spk = signals.get("spk")                             # (n_steps, n_neurons) spike raster
vm_e = signals.get("vm", cell_type="E")              # membrane voltage for E cells
```

## Canonical cortex (default prior)

The verified ground-truth laminar prior is a first-class API surface — no source
editing required. Excitatory fraction rises with depth (L6 ≈90% E), inhibition
peaks superficially (L1 50% I, no PV), PV concentrates at L4; overall ≈77E:23I.

```python
# Canonical laminar cortex: real per-layer composition + laminar placement.
cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")

# Multi-area hierarchy with the canonical prior in each area:
cfg = jtfne.build_multi_area_columns(["V1", "V4", "PFC"], ei_profile="canonical")
```

`ei_profile="flat"` (the default) preserves the legacy depth-invariant
composition and `uniform3d` placement **unchanged**; `ei_profile="canonical"`
auto-routes to laminar placement so each neuron keeps its layer label and the
per-layer E:I gradient is expressed. The exported constants
`jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`, `jtfne.CANONICAL_Z_BANDS`, and
`jtfne.DEFAULT_LAYERS` document the prior directly.

## Adjustable structure: layers, connectivity, homeostasis, plasticity

Every knob below is a chainable `Configuration` call, not a source edit.

```python
# Per-layer neuron count and composition: layer_fractions sets each layer's
# relative depth band (band width -> per-layer neuron count out of n);
# layer_cell_type_fractions sets each layer's own E/PV/SST/VIP split.
cfg = jtfne.build_laminar_column(
    "V1", n=2000,
    layers=["L1", "L2/3", "L4", "L5", "L6"],
    layer_fractions={"L1": (0.00, 0.08), "L2/3": (0.08, 0.40), "L4": (0.40, 0.55),
                      "L5": (0.55, 0.80), "L6": (0.80, 1.00)},
    layer_cell_type_fractions={
        "L1":   {"E": 0.20, "PV": 0.10, "SST": 0.10, "VIP": 0.60},
        "L2/3": {"E": 0.45, "PV": 0.30, "SST": 0.20, "VIP": 0.05},
        "L4":   {"E": 0.55, "PV": 0.35, "SST": 0.08, "VIP": 0.02},
        "L5":   {"E": 0.85, "PV": 0.10, "SST": 0.04, "VIP": 0.01},
        "L6":   {"E": 0.90, "PV": 0.06, "SST": 0.03, "VIP": 0.01},
    },
)

# Explicit within-layer and between-layer connections: .connections() compiles
# real edges at construct() time (status flips declared -> compiled, with an
# exact edge count), distinct from the blanket within_connectivity/within_gain
# the builder already applied above.
cfg = (cfg
    .connections(name="L4_recurrent", source={"layer": "L4"}, target={"layer": "L4"},
                 probability=0.2, weight=0.5)                              # within-layer
    .connections(name="L4_to_L23_feedforward", source={"layer": "L4"}, target={"layer": "L2/3"},
                 probability=0.3, weight=0.4, sign="excitatory")            # between-layer
)

# Homeostasis: a real per-neuron rate-feedback kernel, active in simulate()
# once declared (clip(k_gain * (r_star - r), g_min, g_max) intrinsic bias).
cfg = cfg.homeostasis(relative_baseline=1.0, r_star=8.0, k_gain=1.0)

# Plasticity: records intent in the manifest from the first call, but is
# declaration-only here — simulate() does not consume it. The actual STDP
# weight-update kernel (update_stdp_weights_jax) runs through the separate
# run_stdp_stream entry point, not through Model.simulate().
cfg = cfg.plasticity(relative_baseline=1.0)
```

## Interactive 3D network (dark theme)

Inspect circuit geometry before you simulate: `jtfne.vis.visualize_network_3d`
takes a constructed `Model` (or a `Signals` run) and renders an interactive
Plotly figure — per-layer depth, per-cell-type color/symbol, optional synaptic
edges — on a dark background by default, and exports a self-contained,
pannable/zoomable HTML file for sharing outside a notebook.

```python
cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")
cfg = cfg.set_emitter("izhikevich", "cortical_eig").probes(["spikes", "V_m"])
model = jtfne.construct(cfg)

jtfne.vis.visualize_network_3d(
    model,
    title="Canonical cortical column",
    output_html="network3d.html",   # interactive, dark-themed, open in any browser
)
```

## Jaxley interoperability

[Jaxley](https://jaxley.readthedocs.io) and jaxfne are complementary: Jaxley
builds differentiable, multi-compartment, conductance-based neuron and network
models (HH and other biophysical channels); jaxfne organizes the resulting
voltages into the same source/field/readout/objective chain used by its built-in
emitters — LFP-proxy, CSD-proxy, EEG-proxy, spectrolaminar readouts, manifests.
A Jaxley model is a drop-in emitter: build the morphology and channels in
Jaxley, then hand it to `JaxleyBridge` for one-call integration into `Signals`.

```python
import jaxley as jx
from jaxley.channels import HH
import jaxfne as jtfne

# Build a Jaxley emitter: single HH compartment, recorded and stimulated.
cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
cell.insert(HH())
cell.record("v")
cell.stimulate(jx.step_current(i_delay=10, i_dur=50, i_amp=0.1, delta_t=0.025, t_max=100))

# One call: integrate the Jaxley model and convert to jaxfne Signals.
sig = jtfne.JaxleyBridge(model=cell).simulate(duration_ms=100.0, dt_ms=0.025)

sig.V_m.shape                                         # [T, N] proxy voltage
sig.metadata["physical_amplitude_calibrated"]         # False (proxy gate, never escalated)
fig = jtfne.vis.vm(sig)                               # plot like any native tfne run
```

`JaxleyBridge` also exposes `.simulate_homeostatic(...)`, which keeps Jaxley's
channels/morphology untouched while layering tfne's windowed homeostatic
controller on top — useful for holding a population of biophysical cells near
a target firing rate without hand-tuning per-cell drive. See
[`docs/guides/jaxley_interop.md`](docs/guides/jaxley_interop.md) for the full
bridge surface (manual `jx.integrate()` conversion, the array-first trace
bridge for Jaxley-shaped data without installing Jaxley, and the homeostasis
example). All Jaxley-bridged output stays on the same conservative truth gates
as the rest of jaxfne: proxy voltage, not a calibrated biophysical recording.

## Tune toward a target

```python
obj = jtfne.rate_synchrony_targets()                 # defaults: 10 Hz, kappa 0 (async-irregular)
result = model.tune(obj, optimizer="AGSDR", steps=50)
tuned = result.model

manifest = jtfne.manifest(cfg, signals=signals)      # export: strict JSON-safe run manifest
```

## Checkout validation

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests -q --tb=short
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
PYTHONPATH=. python scripts/audit_notebook_grammar.py --check
mkdocs build --strict
```
