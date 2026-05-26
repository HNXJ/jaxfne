# v0.2.0 Spectrolaminar Public Path

## Overview

This example demonstrates a minimal computational scaffold for peri-event spectrolaminar objective reporting using the jaxfne v0.2.0 pipeline.

**Pipeline:** Emitter → Source → Field → Probe → Objective → Manifest

## What It Shows

- **Small cortical column** simulation with Izhikevich neurons
- **Oddball paradigm** with peri-event windowing:
  - Baseline: -500 to 0 ms
  - Event: 0 to 500 ms  
  - Post: 500 to 1000 ms
  - Full peri-event: -500 to +1000 ms
- **Laminar diagnostics:** alpha/beta power proxy, gamma power proxy, synchrony diagnostic
- **Objective evaluation** using Phase E canonical grammar:
  - Loss: profile_score with window metadata
  - Regularizer: synchrony with bin/window parameters
  - Gate: synchrony threshold check
- **JSON manifest** with complete model/field metadata
- **Truth gates frozen** throughout (computational_scaffold status, proxy_readout_only field claim, etc.)

## How to Run

```bash
python examples/02_spectrolaminar_oddball_scaffold.py
```

Expected output directory: `outputs/v020_spectrolaminar_public_path/`

## Generated Outputs

The example produces strict JSON files (no NaN/Inf):

- **manifest.json** — Full model/field/source/probe metadata with truth gates
- **metrics.json** — Windowed spectrolaminar diagnostics (baseline, event, post, full peri-event)
- **objective_report.json** — Objective evaluation with acceptance_decision
- **validation_report.json** — Truth gate audit and condition vocabulary
- **asset_hashes.json** — SHA256 checksums for file integrity

All outputs use the **manifest-first** architecture: clean JSON evidence over visualization.

## Truth Status

All outputs preserve the v0.2.0 scientific status:

- **truth_mode:** truth_safe_unverified
- **claim_level:** computational_scaffold
- **source_calibration_status:** uncalibrated_izhikevich_native_current
- **field_solver_status:** laminar_proxy_no_pde (no PDE solution)
- **field_claim_level:** proxy_readout_only (forward-field model only)
- **physical_amplitude_claim_allowed:** False

This is a computational scaffold, not a neurophysiological model or biological validation.

## Limitations

- Small network (32 neurons) for fast demonstration
- Izhikevich native current source; no biological calibration
- Proxy conductivity kernel (not physical)
- Synchrony and spectral diagnostics are simplified proxies
- No empirical validation or mechanism claims
- CPU execution only (JAX default float32)

## Next Steps

- Use this scaffold as a basis for hypothesis-driven objective specifications
- Extend to multi-area networks with explicit null modes
- Add empirical validation pipelines (separate phase)
