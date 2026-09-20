# Canonical Visualization Atlas (`jaxfne.vis.atlas_suite`)

The **JaxFNE Canonical Visualization Atlas** provides a unified, deterministic 7-panel layout for inspecting, comparing, and reporting any realized JaxFNE circuit ($N \ge 1$).

Every atlas generation produces self-contained interactive HTML panels, a cryptographic provenance manifest (`manifest.json`), and an index dashboard (`index.html`).

---

## Semantic Panel Roles & Evidence Taxonomy

The atlas enforces strict separation between direct simulation observations (**OBSERVED**) and computed post-hoc metrics (**DERIVED**):

| # | Panel Filename | Semantic Role | Evidence Level | Generator Function | Degradation Invariant ($N=1$, 0 edges, silence) |
|---|---|---|---|---|---|
| 1 | `schema.html` | Block schematic (areas/layers/classes/links) | **OBSERVED** | `network_hspice_plotly(model)` | Single box, no arrows if uncoupled. |
| 2 | `network_3d.html` | Realized 3D architecture | **OBSERVED** | `plot_network_3d(model)` | Single point in space if $N=1$; no edges drawn if uncoupled. |
| 3 | `raster.html` | Microsecond spike events | **OBSERVED** | `plot_raster(signals, model)` | Clean axes showing 0 events if network is silent. |
| 4 | `lfp.html` | Laminar LFP/CSD proxy readouts | **DERIVED** | `plot_lfp` / `plot_csd` | Explicit omission card when no field is recorded — never substituted. |
| 5 | `h_dynamics.html` | Recorded hidden-state trajectory | **DERIVED** | HDP `H_trace` / homeostasis trace | Explicit omission card when no H is recorded. |
| 6 | `hdp.html` | Mutable weight diagnostics | **DERIVED** | HDP `w_trace` | Explicit omission card when HDP is off or the trace is unrecorded — never inferred from activity. |
| 7 | `oscillatory.html` | Power spectral density / time-frequency | **DERIVED** | `plot_psd` (+ `plot_spectrogram`) | Welch PSD fallback when duration is too short for 2D spectrogram. |

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
    jtfne.RuntimeConfiguration(seed=0, duration_ms=200.0, dt_ms=0.5),
)

# 2. Simulate
signals = jtfne.simulate(model)

# 3. Generate canonical atlas
manifest = build_atlas(
    model,
    signals,
    out_dir="docs/_static/atlas",
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
- [Panel 1: Circuit Schematic (`schema.html`)](../_static/atlas/schema.html)
- [Panel 2: Network 3D Architecture (`network_3d.html`)](../_static/atlas/network_3d.html)
- [Panel 3: Spike Raster (`raster.html`)](../_static/atlas/raster.html)
- [Panel 4: LFP Proxy (`lfp.html`)](../_static/atlas/lfp.html)
- [Panel 5: H Dynamics (`h_dynamics.html`)](../_static/atlas/h_dynamics.html)
- [Panel 6: HDP Plasticity (`hdp.html`)](../_static/atlas/hdp.html)
- [Panel 7: Oscillatory Response (`oscillatory.html`)](../_static/atlas/oscillatory.html)

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
---

## Reproducing the published atlas

Both public presentations of the atlas come from one implementation and one
pinned configuration, so they cannot drift apart:

```bash
python scripts/generate_readme_atlas.py
```

That regenerates the interactive panels under `docs/_static/atlas/` **and**
rasterizes the same figure objects to `docs/assets/readme/*.png`, which is what
the GitHub README displays inline. The README stills are therefore reproducible
from a clean checkout rather than hand-added binaries.

The pinned configuration is:

| Field | Value |
|---|---|
| Neuronal tensor | `canonical-v1-column-1000n` |
| Runtime | `RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)` |
| `config_hash` | `e701098092814baa` |
| Realized | 1000 neurons, 215785 edges, 2000 steps |

The generator verifies `config_hash` before writing and aborts on drift, so a
change to the canonical column has to be an explicit decision rather than a
silent republish. PNG rasterization needs `kaleido` (`pip install ".[viz]"`);
`--html-only` skips it.

### A note on panel 6's name

Panel 6 is `state_summary`, not `operating_point`. It reports the time-averaged
firing rate per cell type and the fraction of silent units over the simulated
window. That is a descriptive summary of the realized trajectory: no fixed
point is solved for, no equilibrium condition is checked, and no stationarity
or burn-in criterion is applied. Genuine operating-point analysis --
linearization about an equilibrium across a tonic-drive grid -- lives in
`jaxfne.w3a_stability_analysis.analyze_operating_point`, and the two must not be
confused.
