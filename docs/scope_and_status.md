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

### Biological calibration status (canonical V1 column)

The canonical V1 column (`canonical-v1-column-1000n`) is a **qualitative laminar scaffold**, not a quantitatively calibrated cortical circuit. Its biological calibration status is declared explicitly:

| Axis | Status | Value | Meaning |
|------|--------|-------|---------|
| `qualitative_laminar_scaffold` | `true` | Scaffold: E fraction rises with depth, inhibition peaks superficial, 6 laminar bands, typed E/PV/SST/VIP populations, E->I and I->E motifs declared. Useful for structural and dynamical method development. |
| `quantitative_cell_fraction` | `false` | Per-layer fractions (`L1 E 0.50`, `L2 E 0.648`/`VIP 0.052`, `L3 E 0.80`, `L4 E 0.75`, `L5 E 0.88`, `L6 E 0.90`, etc. in `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` and `jaxfne/configs/canonical-v1-column-1000n.json`; builder constant `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` is the `~66E:34I` variant) are illustrative scaffold values — **not** quantitatively calibrated against stereological counts. Do not treat as empirical V1 composition. |
| `quantitative_connectivity` | `false` | Within-area and cross-layer connection rules (48 typed rules such as E->PV/SST/VIP within-layer and E->E cross-layer) are **qualitative motif scaffolds** (probabilities, weights, delays are `relative` value-tags) — not fitted to paired-recording or connectomic measurements. |

These flags are **informational provenance**, not a solver or kernel setting. No dynamics, solver, or numerical kernel changes. All value-tags remain `"relative"` unless you supply an explicitly validated calibration.

**Reduced emitter caveat.** The shipped emitter is a **reduced Izhikevich point-neuron** (`a`/`b`/`c`/`d`/`drive` per label). Labels `E`/`PV`/`SST`/`VIP` are **functional scaffold identities** for dynamical heterogeneity (regular-spiking vs fast-spiking vs LTS vs VIP-like), not warranted literal transcriptomic or morphologically validated cell-type identities. See [Emitters API](api/emitters.md) and [Calibration guide](guides/calibration.md#biological-calibration-status).

## Plausible ranges

*Izhikevich path — nominal mV ranges for the shipped preset; see body for calibration caveat.*

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
