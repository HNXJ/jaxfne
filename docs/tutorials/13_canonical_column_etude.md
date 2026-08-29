# Étude: Canonical Cortical Column

A single end-to-end étude that walks the full objective grammar on one model:
**Configuration → Construct → Simulate → Visualize → Tune → Post-tune.** It builds
the canonical 1000-neuron laminar column, drives it into a plausible firing regime,
renders proxy readouts, fits a firing-rate target with a black-box optimizer, and
writes a truth-gated run manifest.

Everything here is a **computational scaffold**: the laminar fields are proxies
(`field_solver_status="linear_solver"`), not a solved volume conductor, and no
calibrated-amplitude or mechanism claim is made.

> **Biological calibration status — canonical V1 (`canonical-v1-column-1000n`).** `qualitative_laminar_scaffold = true`, `quantitative_cell_fraction = false`, `quantitative_connectivity = false`. The per-layer fractions shown below (`L1 E 0.50` etc.) and typed connection motifs are **scaffold values, not quantitatively calibrated** against empirical V1 stereology or connectomics. Seed provenance: `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` (PseudoGenome, generative rules with `fraction_tolerance` bands, `fraction_jitter_sigma=0.01`, `value_tag="relative"`) and `jaxfne/configs/canonical-v1-column-1000n.json` (realized NeuronalTensor); builder provenance: `jaxfne.builders.CANONICAL_LAYER_CELL_TYPE_FRACTIONS` (`~66E:34I` variant) / `CANONICAL_Z_BANDS`. Labels `E`/`PV`/`SST`/`VIP` are **reduced Izhikevich scaffold identities** (heterogeneous `a`/`b`/`c`/`d`/`drive`/`sign` in `IZHIKEVICH_CELL_TYPE_DEFAULTS`), not warranted literal cell-type identities. No kernel change. Full disclosure: [Scope & status](../scope_and_status.md) and [Calibration — Biological status](../guides/calibration.md#biological-calibration-status).

| Step | What it does |
|------|--------------|
| 1 Config | Fluent canonical cfg: thickness-proportional counts + per-layer E/I |
| 2 Construct | 1000-neuron model, ~66E:34I, E-deep / I-superficial gradient |
| 3 Simulate | Drive sweep → operating point ~18 Hz; sanity gates; `*_proxy` fields |
| 4 Visualize | Spiking, rate, PSD, LFP/CSD-proxy, spectrolaminar, interactive 3D |
| 5 Tune | AGSDR fits `drive_gain` to a target rate |
| 6 Post-tune | Before/after readout + truth-gated manifest |

---

## The canonical column

Two laws define the reference column (the default prior for laminar work):

1. **Excitation peaks deep.** The E-fraction rises monotonically with depth, to
   95% E in L6.
2. **Inhibition peaks superficial.** The I-fraction is highest superficially
   (L1/L2/L3, 50% I each) and falls with depth to 5% in L6; the largest
   inhibitory *count* sits in the dense superficial L2. PV peaks at L2/L3
   (25% each); L6 carries no PV.

Overall composition is ~66% E : 34% I, in the realistic cortical range for the builder-constant variant. This
table is `jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS` — query the live
constant rather than copying these numbers if you need them elsewhere.

> **Scaffold provenance & calibration.** The table above is the **builder scaffold** (`jaxfne/builders.py:55`, ~66E:34I). The **genome / realized-tensor** provenance used by `load_canonical_neuronal_tensor('canonical-v1-column-1000n')` and `load_canonical_pseudogenome('canonical-v1-column-1000n')` is a related but distinct scaffold: `L1 {E:0.50, SST:0.15, VIP:0.35}`, `L2 {E:0.648, PV:0.20, SST:0.10, VIP:0.052}`, `L3 {E:0.80, PV:0.08, SST:0.08, VIP:0.04}`, `L4 {E:0.75, PV:0.18, SST:0.04, VIP:0.03}`, `L5 {E:0.88, PV:0.06, SST:0.04, VIP:0.02}`, `L6 {E:0.90, PV:0.0533, SST:0.0267, VIP:0.02}` (~75.8E:25.2I realized; `value_tag="relative"`), with `fraction_tolerance` bands and `fraction_jitter_sigma=0.01` declared in `jaxfne/jdna/genomes/canonical-v1-column-1000n.json`. **Both are qualitative scaffolds, not quantitatively calibrated** against empirical composition (`quantitative_cell_fraction = false`; `quantitative_connectivity = false`; see header box and [Calibration — Biological status](../guides/calibration.md#biological-calibration-status)). Reduced Izhikevich labels `E`/`PV`/`SST`/`VIP` below are functional heterogeneity tags (distinct `a`/`b`/`c`/`d`/`drive`), not warranted literal cell-type identities.

```python
import jaxfne as jtfne
jtfne.enable_x64()  # x64 before building arrays

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]

# z-bands: band WIDTH is proportional to neuron count (count ∝ thickness)
ZBANDS = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

# Per-layer cell-type composition: I-fraction high superficial -> low deep
LAYER_CELL_TYPES = {
    "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},  # 50% I
    "L2": {"E": 0.50, "PV": 0.25, "SST": 0.10, "VIP": 0.15},  # 50% I (I-count peak)
    "L3": {"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10},  # 50% I
    "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},  # 30% I (PV feedforward)
    "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},  # 15% I
    "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},  # 5% I (E-fraction peak, no PV)
}
```

---

## 1. Configuration

```python
cfg = (
    jtfne.laminar_cortex_config(
        seed=0, duration_ms=1000.0, dt_ms=0.5,
        areas=["V1"], layers=LAYERS, n=1000, emitter="izhikevich",
        baseline_drive_by_cell_type={"E": 5.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0},
    )
    .layer_fractions(layer_fractions=ZBANDS)        # width ∝ count -> per-layer counts
    .area_layer_cell_types("V1", LAYER_CELL_TYPES)  # per-layer E/I composition
)
```

The global `cell_types=` weight is intentionally **not** used for composition: it
spreads cell types uniformly and produces an over-inhibitory gradient. Always set
composition per layer with `.area_layer_cell_types(...)`.

## 2. Construct

```python
model = jtfne.construct(cfg)

# neuron_table() returns a list of dict rows: neuron_id, area, layer, cell_type, x, y, z
import collections
counts = collections.Counter((r["layer"], r["cell_type"]) for r in model.neuron_table())
```

Verified composition (per-layer counts, I% by layer):

```text
 layer     E    PV   SST   VIP   tot   I%
    L1    50     5    10    35   100   50
    L2   125    62    25    38   250   50
    L3   100    50    30    20   200   50
    L4    70    20     5     5   100   30
    L5   170    10    10    10   200   15
    L6   142     0     8     0   150    5
                                  1000  ~34   ->  ~66E : 34I
```

E-fraction rises with depth (L1 50% → L6 90%); the inhibitory *count* peaks in L2.

## 3. Simulate

Find a plausible operating point by sweeping the per-cell-type baseline drive,
then run the full length. Sanity gates (Izhikevich, native units): resting
membrane ≈ −67 mV, spike peak +30 mV (hard reset), mean rate in ~8–25 Hz, all finite.

```python
import numpy as np
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.5, seed=0)
signals = jtfne.simulate(model, sim)

rate = jtfne.tutorial_utils.population_rate_hz(np.asarray(signals.get("spikes")), 0.5)
# drive=5 -> ~18 Hz; drive sweep: 4 -> ~13 Hz, 6 -> ~23 Hz, 8 -> hot, 0 -> silent
```

Proxy fields are computed automatically and carry the `*_proxy` suffix
(`lfp_proxy`, `csd_proxy`, `source_proxy`), each shaped `(n_steps, n_contacts)`.
`eeg_proxy`/`meg_proxy` are **not** auto-computed — they require an explicit
lead-field (`eeg_proxy_transform(source, leadfield)`).

```python
lfp = np.asarray(signals.get("lfp_proxy"))   # (n_steps, n_contacts)
csd = np.asarray(signals.get("csd_proxy"))
```

### Layer-balanced drive (avoid layer-rate bias)

Per-layer firing should be similar across depth, or any laminar readout is biased
toward the hotter layers. Under uniform drive this column is intrinsically
*superficial-hotter* (L2/L3 fire faster than L5/L6) — no layer is silent, but the
bias is real. Flatten it with **graded per-layer drive**, applied to the **same
constructed model** (no rebuild) via `with_emitter_parameters`. A gentle
proportional-control loop converges in a few steps:

```python
layer_of = np.array([r["layer"] for r in model.neuron_table()])
masks = {L: (layer_of == L) for L in LAYERS}
drive = np.zeros(len(layer_of))
TARGET = 10.0

def per_layer_rates(layer_drive):
    for L in LAYERS:
        drive[masks[L]] = layer_drive[L]
    m = model.with_emitter_parameters(drive_per_neuron=drive)   # reuse, no rebuild
    spk = np.asarray(jtfne.simulate(m, sim).get("spikes"))      # (n_steps, N)
    per_neuron = spk.sum(axis=0) / (1000.0 / 1000.0)            # Hz per neuron
    return {L: float(per_neuron[masks[L]].mean()) for L in LAYERS}

layer_drive = {L: 5.0 for L in LAYERS}
for _ in range(30):
    r = per_layer_rates(layer_drive)
    if max(abs(r[L] - TARGET) for L in LAYERS) <= 3 and (max(r.values()) - min(r.values())) <= 4:
        break
    for L in LAYERS:  # gentle gain (0.35) + tight clip avoids overshoot under recurrence
        factor = np.clip((TARGET / max(r[L], 0.3)) ** 0.35, 0.8, 1.25)
        layer_drive[L] = float(np.clip(layer_drive[L] * factor, 0.0, 30.0))
```

Converged graded profile (superficial gets *less* drive, deep gets *more* — cancels
the bias without altering the structural E/I gradient):

```text
layer  drive  rate_Hz
   L1   4.79    10.6
   L2   4.08    12.9
   L3   4.17    12.5
   L4   4.26    12.5
   L5   4.47    11.7
   L6   4.54    11.8     ->  spread ~2.3 Hz, all layers within 10 ± 5 Hz
```

## 4. Visualize

All `jtfne.vis.*` take a `Signals` object and return matplotlib figures;
`visualize_network_3d` returns an interactive Plotly scene with HTML export.

```python
jtfne.save_figure(jtfne.vis.raster(signals), "raster.pdf")
jtfne.save_figure(jtfne.vis.rate(signals), "rate.pdf")
jtfne.save_figure(jtfne.vis.psd(signals), "psd.pdf")
jtfne.save_figure(jtfne.vis.lfp(signals), "lfp_proxy.pdf")
jtfne.save_figure(jtfne.vis.csd(signals), "csd_proxy.pdf")
jtfne.save_figure(jtfne.vis.spectrolaminar_suite(signals), "spectrolaminar.pdf")

# Interactive 3D column (pan/zoom; shows the depth gradient directly)
jtfne.vis.visualize_network_3d(
    model.neuron_table(), title="Canonical V1 column (66E:34I)",
    show_layers=True, show_column_shells=True, output_html="network_3d.html",
)
```

## 5. Tune

The Izhikevich hard spike reset is non-differentiable, so gradient optimizers are
gated off. Use the black-box AGSDR optimizer to fit a scalar `drive_gain` to a
firing-rate target. The optimizer reuses the constructed model across generations —
no rebuild.

```python
N = len(model.neuron_table())
objective = jtfne.rate_targets(groups={"all": np.arange(N)}, targets_hz={"all": 10.0})

result = model.tune(
    objectives=objective,
    optimizer=jtfne.agsdr(seed=0),
    parameters={"drive_gain": (0.3, 1.5)},   # multi-parameter form
    generations=8, population_size=6, simulation=sim,
)
tuned = result.model                           # best model, ready to simulate
best = result.to_dict()["best_parameters"]     # e.g. {"drive_gain": ~0.79}
# baseline ~15 Hz -> tuned ~10 Hz; squared-relative-error score ~1e-6
```

## 6. Post-tune

Compare the tuned model and write a strict, truth-gated run manifest.

```python
sig_tuned = jtfne.simulate(tuned, sim)

manifest = tuned.manifest(
    signals=sig_tuned, readout=None,
    objective={"kind": "group_rate_targets", "target_hz": 10.0, "group": "all"},
    tuning={"optimizer": "agsdr", "parameter": "drive_gain",
            "best_gain": float(best["drive_gain"])},
)
jtfne.save_json(manifest, "tuned_run_manifest.json")  # allow_nan=False; finite-checked
```

Manifests carry the same default claim fields as [Scope & status](../scope_and_status.md):

```text
claim_level:                   computational_scaffold
field_solver_status:           linear_solver
field_claim_level:             proxy_readout
physical_amplitude_calibrated: False
# Biological calibration (canonical V1):
#   qualitative_laminar_scaffold = true
#   quantitative_cell_fraction    = false
#   quantitative_connectivity     = false  (see header box; scaffold provenance, not calibrated measurements)
# Reduced labels E/PV/SST/VIP are functional scaffold identities, not warranted literal cell-type identities
```

---

## Notes on scale and claims

- **Reuse, don't rebuild.** `construct()` is the expensive step (~40 s at 10k,
  ~2 s at 1k); `simulate()` is comparatively cheap. For sweeps, seeds, drive, or
  trials, reuse the constructed model (vary the simulation, or adjust emitter
  parameters with `with_emitter_parameters`); only rebuild when the structure
  changes (counts, layers, cell types, connectivity).
- **Spectrolaminar structure is scale-dependent.** A clean depth × frequency
  separation needs large populations and multiple trials; at 1000 neurons the
  spectrolaminar panel is pipeline-correct but not a substitute for a larger,
  multi-trial run via `tutorial_utils.spectrolaminar_from_trials`.
- **Proxy language only.** Use "simulated", "proxy", "scaffold",
  "computational diagnostic". The laminar fields here are proxies, not a solved
  field, and amplitudes are uncalibrated. Likewise, `E`/`PV`/`SST`/`VIP` labels denote reduced Izhikevich dynamical heterogeneity, not warranted literal cell-type identity.
- **Scaffold, not calibrated biology.** The fractions and motifs above are qualitative scaffold values (`quantitative_cell_fraction = false`, `quantitative_connectivity = false`) with declared provenance — do not present as empirically calibrated V1 composition or connectivity.

## Next step

For the multi-trial laminar readout, see
[Suite No. 2 (Corticospectrolaminar Motif)](07_jaxfne_suite_no_2_spectrolaminar_motif.md);
for multi-area routing, see [V1-PFC Dual Column](05_v1_pfc_dual_column.md).
