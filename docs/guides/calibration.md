# Calibration

When you have **empirical EEG, MEG, LFP, or CSD data**, jaxfne's default proxy
readouts need a calibration step before amplitudes or units are meaningful.
This guide describes the metadata and workflow hooks the package provides —
without claiming shipped outputs are already physically calibrated.

## Calibration-ready design

jaxfne is designed to support calibration workflows. The package:

- **Preserves source identity** — track source origin (emitter type, cell type)
- **Declares assumptions** — metadata fields state conductivity, solver, geometry models
- **Supports geometry specification** — define layer/contact depths and spatial locations
- **Allows empirical mapping** — workflows can include measured-to-model correspondences

## Next steps for calibration

To prepare a workflow for calibration:

1. **Specify geometry:** Define layer depths, contact locations, tissue conductivity (if known)
2. **Document assumptions:** State source model, field solver, status
3. **Collect reference data:** Identify empirical EEG/MEG/LFP/CSD for comparison
4. **Validate:** Compare proxy readouts to empirical data; compute residuals and comparison metrics

## Calibration Specification and Reporting

jaxfne provides calibration specification and reporting contracts. These allow workflows to declare calibration state without changing the default proxy readout behavior.

### CalibrationSpec

Declare calibration intent with `CalibrationSpec`:

```python
from jaxfne.validation import CalibrationSpec, make_calibration_report

# Declare uncalibrated proxy (default)
spec = CalibrationSpec(
    name="default_proxy",
    target="readout"
)

# Declare toy calibration (illustrative, pending validation)
spec = CalibrationSpec(
    name="toy_eeg_proxy",
    target="readout",
    mode="toy_scale",
    scale=1.0,
    units="proxy_V",
    reference="toy_leadfield"
)

# Declare empirical calibration candidate (metadata declared, validation pending)
spec = CalibrationSpec(
    name="eeg_candidate",
    target="readout",
    mode="empirical_gain_candidate",
    scale=2.5,
    units="mV",
    reference="pilot_recording_2024"
)
```

### Supported Modes

- `uncalibrated_native` — Proxy readout, no calibration (default)
- `toy_scale` — Illustrative calibration, pending validation
- `relative_normalized` — Normalized relative to proxy baseline
- `empirical_gain_candidate` — Candidate gain estimate, pending validation
- `physical_units_candidate` — Candidate physical units, pending validation
- `calibrated_empirical` — Calibration metadata declared (validation pending)

### Calibration Reports

Generate a calibration status report:

```python
report = make_calibration_report(spec, readout_kind="lfp_proxy")

# report contains:
# - calibration_name, target, mode, status
# - units, scale, reference, description
# - amplitude_status: false (always)
# - calibration_model_status: computational_proxy_with_declared_metadata
# - assumptions and warnings
```

### Important: Behavior

- **All proxy readouts remain computational proxies** by default
- `amplitude_status` stays `false` for all modes
- Calibration metadata is declared for future validation, validation pending
- Empirical calibration requires separate geometry, reference data, and validation evidence beyond the spec

## Biological calibration status

Field-amplitude calibration (above) is distinct from **biological calibration** — whether the circuit's cell-type composition and connectivity are quantitatively fitted to empirical biology.

### Canonical V1 column (`canonical-v1-column-1000n`)

The shipped canonical V1 is intentionally scoped as a **qualitative laminar scaffold**:

| Axis | Value | Interpretation |
|------|-------|----------------|
| `qualitative_laminar_scaffold` | `true` | Laminar structure is declared (6 bands L1–L6, depth-graded E:I, typed E/PV/SST/VIP populations, canonical E->I / I->E / cross-layer E->E motifs). Suitable for method development and structural diagnostics. |
| `quantitative_cell_fraction` | `false` | Per-layer fractions declared in `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` (PseudoGenome, generative) and `jaxfne/configs/canonical-v1-column-1000n.json` (NeuronalTensor, realized; see provenance below) are **scaffold values, not calibrated measurements**. Example realised fractions (NeuronalTensor / PseudoGenome): `L1 {E:0.50, SST:0.15, VIP:0.35}`, `L2 {E:0.648, PV:0.20, SST:0.10, VIP:0.052}`, `L3 {E:0.80, PV:0.08, SST:0.08, VIP:0.04}`, `L4 {E:0.75, PV:0.18, SST:0.04, VIP:0.03}`, `L5 {E:0.88, PV:0.06, SST:0.04, VIP:0.02}`, `L6 {E:0.90, PV:0.0533, SST:0.0267, VIP:0.02}` (overall ~75.8E:25.2I realized). The builder constant `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` (`jaxfne/builders.py:55`) is a related but distinct scaffold variant (~66E:34I). None are warranted as stereologically calibrated V1 composition. |
| `quantitative_connectivity` | `false` | The 48 typed connection rules (within-layer E->PV/SST/VIP, PV/SST/VIP->E, PV->PV and cross-layer E->E such as L4->L2/L3, L2->L3/L5, L6->L4/L1) and their weights/delays/gains (`w_mech=0.45`, `dT_ms` 2.0/5.0, `value_tag="relative"`) are qualitative motif scaffolds, not fits to empirical connectomics or paired recordings. |

**Provenance.** Generative provenance is `PseudoGenome canonical-v1-column-1000n` (`schema_version pseudogenome_v1`, `development_parameters.fraction_jitter_sigma=0.01`, `fraction_tolerance` bands per layer, `value_tag="relative"` on all numeric fields; realized via `develop(genome, seed=K_D)` -> `NeuronalTensor` -> `construct` -> `Model` with `source_calibration_status="uncalibrated_izhikevich_native_current"`). File provenance: `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` (generative rules) and `jaxfne/configs/canonical-v1-column-1000n.json` (realized NeuronalTensor snapshot); builder provenance: `jaxfne.builders.CANONICAL_LAYER_CELL_TYPE_FRACTIONS` / `CANONICAL_Z_BANDS`. Treat all three as **scaffold provenance**, not calibration evidence.

**Reduced-emitter caveat.** The default emitter is a reduced Izhikevich point neuron. Labels `E`/`PV`/`SST`/`VIP` are **functional scaffold identities** (distinct `a`/`b`/`c`/`d`/`drive`/`sign` rows in `IZHIKEVICH_CELL_TYPE_DEFAULTS`) that provide dynamical heterogeneity, not warranted literal cell-type identities. A label match does not imply transcriptomic, morphological, or biophysical identity with the named biological class. Do not present results as if `PV` in the model is proven to be biological PV interneurons; report as "PV-like reduced scaffold" unless an independent calibration/validation study supplies that warrant. No kernel change.

To claim quantitative biological correspondence, supply an explicit layer-resolved validation study (stereology, connectomics, or transcriptomic mapping) and tag the derived circuit with its own citation and `value_tag` — do not reinterpret the shipped scaffold as already validated.

## Current status

- ✓ Metadata fields support calibration annotations
- ✓ JSON output bundles preserve geometry and source information
- ✓ Calibration specification contracts (metadata only, no physical amplitude upgrade)
- ✓ Biological calibration status declared for canonical V1 (qualitative scaffold only: cell-fraction and connectivity both `false` — this section)
- ◐ Empirically validated calibration examples: planned
- ◐ Empirically calibrated readouts: planned

## Example: Declaring a calibration-ready workflow

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",  # or specify σ in S/m if known
        depths=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],  # layer boundaries
        boundary="mean_zero_neumann",
        gauge="mean_zero"
    )
    .probe(
        name="calibration_ready",
        n_contacts=6,
        contact_depths=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    )
)

model = jtfne.construct(cfg)
signals = model.simulate(...)
manifest = model.manifest(signals, ...)

# Manifest includes geometry and metadata suitable for later validation
```

## References and further reading

- [Scope and limitations](../limitations_and_future_plans.md)
- [Output bundles](output_bundles.md)
- [Probe operators](probe_operators.md)
