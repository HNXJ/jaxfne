# Scope & status

Every jaxfne output is one of two kinds:

- **Relative** — a value computed by the simulator, meaningful for comparison
  within and across runs. This is the default for every signal, readout, and
  amplitude jaxfne produces.
- **Absolute** — a value tied to real-world physical units through an explicit
  calibration step you supply. jaxfne never produces Absolute values on its own.

## What jaxfne is

jaxfne is a computational scaffold for laminar population models in JAX:
configure a circuit, simulate spikes and membrane traces, and extract field
readouts (LFP, CSD, EEG/MEG-style projections, spectrolaminar PSD summaries).
It targets method development, circuit-level diagnostics, and reproducible
pipelines.

## Status fields

These fields are stamped on every manifest and cannot be set to a stronger
value without an explicit calibration step:

| Field | Default | Meaning |
|------|---------|---------|
| `physical_amplitude_calibrated` | `False` | Amplitudes are Relative unless you supply calibration |
| `claim_level` | `computational_scaffold` | Simulated workflow |
| `field_claim_level` | `proxy_readout` | LFP/CSD/EEG outputs are Relative-value channels |
| `field_solver_status` | `linear_solver` | Laminar readouts use a Relative projection, not a full elliptic solve |

## Plausible ranges (Izhikevich path)

Quick checks on a finished run. The mV values below are the model's nominal
internal units (the Izhikevich preset's declared numerical scale), not
calibrated physiological measurements; jaxfne outputs remain relative/proxy
quantities per the truth gates above.

- Resting membrane voltage ≈ −66 (nominal model mV)
- Spike peak ≈ +30, then hard reset to `c`
- Mean population rate roughly 8–25 Hz for canonical drive
- `|Vm| > 150` or NaN/Inf indicates a numerical blowup

## Jaxley bridge

Jaxley emitters integrated through `JaxleyBridge` produce the same Relative
voltage and field readouts as built-in emitters, unless you supply separate
calibration.

## Going further

- [Limitations and future plans](limitations_and_future_plans.md) — reserved regimes
- [Calibration](guides/calibration.md) — how to add an Absolute calibration step
- [Computation basis](computation_basis.md) — regime reference
