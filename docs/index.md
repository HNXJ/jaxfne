# jaxfne

**JAX Field Neural Equations** (`jaxfne`) is a compact JAX-native source-to-field/readout engine for Tensor-Field Neural Equations (TFNE).

**Current version:** v0.2.3

## What it does

jaxfne provides a practical object-oriented API for building reproducible computational neurophysiology workflows:

```text
Emitter → Source → Field → Probe → Objective → Optimizer
```

It is designed for laminar proxy simulations, source-field metadata, readouts, objective reports, receipts, manifests, and benchmarkable JAX execution paths.

## Getting started

- **[Installation](install.md)** — Install from PyPI or GitHub
- **[Quickstart](quickstart.md)** — Minimal working example
- **[Probe Operators](probe_operators.md)** — Eight readout channels and their status
- **[Roadmap](ROADMAP.md)** — v0.2.4–v0.2.21 strategic phases

## Scientific scope

jaxfne is a computational scaffold for constructing, testing, and reporting TFNE source-to-field/readout models under explicit assumptions.

**Default v0.2.x readouts are proxy readouts unless a run supplies calibration, source-conservation, geometry, and solver evidence sufficient for stronger physical claims.**

All runs carry conservative default claim-status metadata:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
source_calibration_status: uncalibrated_izhikevich_native_current
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
```

Practical use: reproducible proxy simulations, source/readout bookkeeping, objective scaffolds, performance benchmarking, and manifest-driven model comparison.

Physical-amplitude CSD/LFP claims and mechanism-level interpretation are reserved for models with empirical calibration, validation datasets, nulls, and ablations.

## Key features

- Izhikevich emitter scaffolds
- Laminar source geometry metadata
- Source proxy traces
- LFP-proxy and CSD-proxy laminar readouts
- Readout specifications and objective reports
- Run receipts and JSON-safe manifests
- CPU-first validation and optional accelerator execution through JAX

## License

MIT License.
