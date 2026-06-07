<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# JAXFNE Core Doctrine

## Package identity

`jaxfne` is a compact JAX-native TFNE source-to-field/readout scaffold. It is an evidence-generation and assumption-auditing layer for computational neurophysiology. It is not currently a validated biological simulator, calibrated EEG/MEG solver, metabolic simulator, Maxwell solver, or physical-amplitude package.

## Canonical API use

```python
import jaxfne as jtfne
```

Avoid alternate aliases, wildcard imports, and invented APIs.

## Current API posture from inspected zip

- Package version: `0.3.29`.
- Root `__all__` inspection found 149 exported symbols in the local environment.
- `jtfne.Config` and `jtfne.Paradigm` are present.
- `jtfne.Net` was not present as a root export in the inspected zip; use `Model`/`construct` unless a live checkout verifies `Net` as public.
- `jtfne.vis.visualize_network_3d` is present.
- The README uses `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export`; keep this as target object grammar, but verify root exports before writing code that names `Net`.

## Truth gates

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Stronger wording requires run-specific solver, geometry, boundary condition, gauge condition, residual, convergence, source calibration, units, finite fields, and external or empirical validation.

## Operator grammar

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

| Operator | Current package locus | Evidence required |
|---|---|---|
| Emitter | `jaxfne.emitters`, `jaxfne.core`, optional Jaxley bridge | finite state, seed, dtype, params, update rule |
| Source | `jaxfne.fields`, source reports | one source mode, no synaptic double count, calibration status |
| Field/proxy | `jaxfne.fields.proxy`, `jaxfne.fields.diagnostics` | solver/proxy status, finite arrays, row/kernel normalization when applicable |
| Probe | `jaxfne.fields.probes`, `jaxfne.core.Signals` | kind, method, shape, units/status, field/source status |
| Objective | `jaxfne.objectives`, `jaxfne.core.Objective` | finite metrics, gates, null status, rejection reasons |
| Optimizer | `jaxfne.optim` | seed, bounds, budget, objective report, finite status |
| Manifest/validation | `jaxfne.io`, `jaxfne.validation` | strict JSON, hashes, gate preservation, runtime receipt |

## Source bookkeeping

Use one source mode per run:

```text
Mode A: total membrane-current source
Mode B: decomposed electrical-source mode
```

Rules:

- Avoid double-counting synaptic current.
- Native reduced-emitter current is not amperes unless calibrated.
- Export `source_mode`, `source_calibration_status`, and source support/normalization metadata.
- Treat LFP/CSD/EEG/MEG/EMM outputs as proxy readouts unless calibration and solver evidence exist.

## Probe/readout operators

Required readout kinds:

```text
spk, vm, source, lfp_like, csd_like, eeg_like, meg_like, emm_proxy
```

Each readout report includes:

```text
kind, shape, units/status, method, assumptions, source calibration, field status, finite-output status, artifact paths, interpretation level
```

## Public API compatibility

- Preserve public names unless a breaking cleanup is explicitly requested.
- Add compatibility wrappers when helpers move.
- Optional dependencies stay lazy.
- Core root import must not require visualization extras or optional ecosystem packages.
- Reusable visualization and tutorial helper logic belongs in package APIs, not notebook-local scientific engines.

## Artifacts

Release-facing workflows export:

```text
manifest.json
validation_report.json
metrics.json when objective/optimizer evidence exists
asset_hashes.json
figures/*.png
plotly/*.html optional
```

All JSON must pass strict serialization with finite values only.
