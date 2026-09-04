# Canonical Visualization Atlas (`jaxfne.vis.atlas_suite`)

The **JaxFNE Canonical Visualization Atlas** provides a unified, deterministic 6-panel visual grammar for inspecting, comparing, and reporting any realized JaxFNE circuit ($N \ge 1$).

Every atlas generation produces self-contained interactive HTML panels, a cryptographic provenance manifest (`manifest.json`), and an index dashboard (`index.html`).

---

## Semantic Panel Roles & Evidence Taxonomy

The atlas enforces strict separation between direct simulation observations (**OBSERVED**) and computed post-hoc metrics (**DERIVED**):

| # | Panel Filename | Semantic Role | Evidence Level | Generator Function | Degradation Invariant ($N=1$, 0 edges, silence) |
|---|---|---|---|---|---|
| 1 | `network_3d.html` | Realized 3D architecture | **OBSERVED** | `plot_network_3d(model)` | Single point in space if $N=1$; no edges drawn if uncoupled. |
| 2 | `connectivity.html` | Realized synaptic matrix | **OBSERVED** | `plot_connectivity(model)` | Empty adjacency matrix card with explicit zero-edge note. |
| 3 | `raster.html` | Microsecond spike events | **OBSERVED** | `plot_raster(signals, model)` | Clean axes showing 0 events if network is silent. |
| 4 | `traces.html` | Somatic membrane potentials | **OBSERVED** | `plot_membrane_potentials(signals, model)` | Somatic $V_m(t)$ for representative units (or unit 0). |
| 5 | `spectral.html` | Power spectral density / time-frequency | **DERIVED** | `plot_psd` (+ `plot_spectrogram`) | Welch PSD fallback when duration is too short for 2D spectrogram. |
| 6 | `operating_point.html`| Firing rates & silence fraction | **DERIVED** | Spike count summary | Per-cell-type rate bar chart + % silent units; counts from `model.summary()`. |
| * | `field.html` *(optional)* | Laminar LFP / CSD proxy readouts | **DERIVED** | `plot_lfp` / `plot_csd` | Emitted only when model contains a non-empty field proxy. |

---

## Quickstart

Building an atlas requires only a realized `Model` and simulated `Signals`:

```python
import jaxfne as jtfne
from jaxfne.vis import build_atlas

# 1. Realize circuit
tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model = jtfne.construct(
    tensor,
    jtfne.RuntimeConfiguration(seed=0, duration_ms=500.0, dt_ms=0.5),
)

# 2. Simulate
signals = jtfne.simulate(model)

# 3. Generate canonical atlas
manifest = build_atlas(
    model,
    signals,
    out_dir="docs/_static/atlas/v1_column",
    title="Canonical V1 Column (1000n)",
)

print(f"Atlas suite: {manifest['suite']}")
print(f"Cryptographic hash: {manifest['sha256']}")
for p in manifest["panels"]:
    print(f" - {p['panel']}: {p['file']} ({p['evidence']}, {p['status']})")
```

---

## Interactive Examples

The following standalone interactive panels are generated live from the canonical 1000-neuron column simulation:

- [Index Dashboard (`index.html`)](../_static/atlas/index.html)
- [Panel 1: Network 3D Architecture (`network_3d.html`)](../_static/atlas/network_3d.html)
- [Panel 2: Realized Connectivity (`connectivity.html`)](../_static/atlas/connectivity.html)
- [Panel 3: Spike Raster (`raster.html`)](../_static/atlas/raster.html)
- [Panel 4: Membrane Traces (`traces.html`)](../_static/atlas/traces.html)
- [Panel 5: Spectral Dynamics (`spectral.html`)](../_static/atlas/spectral.html)
- [Panel 6: Operating Point & Silence (`operating_point.html`)](../_static/atlas/operating_point.html)

---

## Provenance and Integrity

Every HTML panel embeds a structured provenance card containing:
- `config_hash`: Unique hash of the circuit specification.
- `neurons` & `edges`: Realized entity counts (`configured != realized`).
- `steps` & `dt_ms`: Exact temporal integration parameters.
- `evidence`: `OBSERVED` vs `DERIVED` badge.
- `jaxfne`: Installed library version string.
- `calibration`: Explicit proxy disclosure: `relative_proxy_readout (never calibrated physical units)`.

The output directory also contains `manifest.json`, recording SHA256 hashes and byte counts for mechanical verification in continuous integration.
