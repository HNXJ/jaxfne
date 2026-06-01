# JAXFNE Core Doctrine

## Package identity

`jaxfne` is a compact JAX-native TFNE source-to-field/readout scaffold. It is a bridge and evidence-generation layer, not a validated biological simulator or full EEG/MEG forward solver.

## Canonical API use

```python
import jaxfne as jtfne
```

Avoid alternate aliases, wildcard imports, and invented APIs.

## Truth gates

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Use evidence/status wording. Reserve stronger interpretation for runs with solver, calibration, geometry, boundary, gauge, residual, units, and validation evidence.

## Source bookkeeping

Use one source mode per run:

```text
Mode A: total membrane-current source
Mode B: decomposed electrical-source mode
```

Avoid double-counting synaptic current. Native reduced-emitter current is not amperes unless calibrated. Export source calibration status.

## Probe/readout operators

Required readout kinds:

```text
spk, vm, source, lfp_like, csd_like, eeg_like, meg_like, emm_proxy
```

Each readout report includes shape, units/status, method, assumptions, source calibration, field status, finite-output status, and artifact paths.

## Public API compatibility

- Preserve public names unless a breaking cleanup is explicitly requested.
- Prefer compatibility wrappers for moved helpers.
- Keep optional dependencies lazy.
- Core `import jaxfne as jtfne` must not require visualization extras.
- Reusable visualization such as `jtfne.vis.visualize_network_3d(...)` belongs in `jaxfne.vis`, not notebook-local code.

## Reporting and artifacts

Release-facing workflows export:

```text
manifest.json
validation_report.json
metrics.json when objective/optimizer evidence exists
asset_hashes.json
figures/*.png
plotly/*.html optional
```

All JSON uses finite values and strict serialization.