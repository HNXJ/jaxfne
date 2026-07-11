# Scope & status

**Authoritative summary** of what jaxfne claims — and does not claim — for
simulated outputs. Link here instead of repeating the full gate block on every page.

## What jaxfne is

jaxfne is a **computational scaffold** for laminar population models in JAX:
configure a circuit, simulate spikes and membrane traces, and extract **proxy**
field readouts (LFP, CSD, EEG/MEG-style projections, spectrolaminar PSD summaries).

It is designed for **method development, circuit-level diagnostics, and
reproducible pipelines** — not as a substitute for validated biophysical recordings
or PDE field solvers without additional calibration evidence.

## Default truth gates

These conservative defaults are enforced in code and manifests; they are **not**
escalated without explicit evidence and review:

| Gate | Default | Meaning |
|------|---------|---------|
| `claim_level` | `computational_scaffold` | Simulated workflow, not validated biology |
| `field_solver_status` | `linear_solver` | Laminar readouts use proxy projection, not a full elliptic solve |
| `field_claim_level` | `proxy_readout` | LFP/CSD/EEG outputs are `*_proxy` channels |
| `physical_amplitude_calibrated` | `False` | Amplitudes are relative unless you add calibration |

## Language to use

**Prefer:** simulated, proxy, scaffold, computational diagnostic.

**Avoid without receipts:** validated, physical, proved mechanism, calibrated LFP/EEG,
PDE solve, biological learning claim.

## Plausible sanity checks (Izhikevich path)

Quick checks on a finished run — not publication claims:

- Resting membrane voltage ≈ −66 mV
- Spike peak ≈ +30 mV (hard reset)
- Mean population rate roughly 8–25 Hz for canonical drive
- `|Vm| > 150` or NaN/Inf → numerical blowup, not a scientific result

## Jaxley bridge

Jaxley emitters integrated through `JaxleyBridge` inherit the same gates: proxy
voltage and field readouts unless you supply separate calibration.

## Going further

- [Limitations and future plans](limitations_and_future_plans.md) — reserved regimes
- [Calibration](guides/calibration.md) — hooks for empirical mapping
- [Computation basis](computation_basis.md) — regime gating doctrine
