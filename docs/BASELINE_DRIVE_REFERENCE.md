# Baseline Drive Reference for Izhikevich Neurons

**Purpose:** Eliminate silent-neuron populations in jaxfne networks by providing subthreshold baseline current per cell type.

**Generated:** 2026-06-09 via `scripts/characterize_neuron_io_curves.py`  
**Truth status:** computational_scaffold (uncalibrated Izhikevich native units)

> **Partially reproducible:** `scripts/characterize_neuron_io_curves.py` exists
> in this repo and can be re-run. However, its recorded output artifacts
> (`outputs/neuron_io_reference.json`, `outputs/neuron_io_baseline_reference.json`)
> are not preserved (verified absent from working tree and full git history
> 2026-07-03) — the numeric tables below are a historical record of a prior run,
> not values you can currently diff against a saved artifact.

---

## Rheobase Values (Minimum DC for First Spikes)

| Cell Type | Rheobase (nA) | First Firing Rate | Notes |
|-----------|---------------|------------------|-------|
| E         | 4.17          | 8.00 Hz          | Regular-spiking excitatory |
| PV        | 4.17          | 29.50 Hz         | Fast-spiking parvalbumin+ |
| SST       | 0.83          | 10.50 Hz         | Low-threshold somatostatin+ |
| VIP       | 24.14         | 9.11 Hz          | Burst-like (negative b param) |

---

## Recommended Baseline Drive (80% of Rheobase)

| Cell Type | DC (nA) | Rationale |
|-----------|---------|-----------|
| E         | 3.33    | Subthreshold; reduces silent neurons |
| PV        | 3.33    | Subthreshold; maintains fast-spiking dynamics |
| SST       | 0.67    | Very conservative (low rheobase) |
| VIP       | 19.31   | Requires high DC due to burst dynamics |

**Scope:** Use 80% of rheobase to raise excitability **without** forcing tonic spiking. Network synaptic input can then reliably activate neurons above this baseline.

---

## Baseline Adjustment Guide

If min_neuron_rate gate (≥1.0 Hz) still fails after applying recommended baseline:

| Multiplier | E (nA) | PV (nA) | SST (nA) | VIP (nA) | Status |
|-----------|--------|--------|----------|----------|--------|
| 0.5       | 2.08   | 2.08   | 0.42     | 12.07    | Very conservative |
| 0.7       | 2.92   | 2.92   | 0.58     | 16.90    | Conservative |
| **0.8**   | **3.33** | **3.33** | **0.67** | **19.31** | **← Recommended** |
| 0.9       | 3.75   | 3.75   | 0.75     | 21.72    | Moderate |
| 1.0       | 4.17   | 4.17   | 0.83     | 24.14    | Full rheobase |
| 1.1       | 4.58   | 4.58   | 0.92     | 26.55    | Aggressive |
| 1.2       | 5.00   | 5.00   | 1.00     | 28.97    | Very aggressive |
| 1.5       | 6.25   | 6.25   | 1.25     | 36.21    | Maximum tested |

**Iteration protocol:**
1. Start with 0.8 (recommended)
2. If gates fail → try 0.9
3. If still failing → try 1.0 (full rheobase)
4. If still failing → problem is not DC; investigate network connectivity or emitter parameters

---

## Usage in Delta-Test

### In Notebook Global Config

```python
GLOBAL = {
    # ... other parameters ...
    
    # Baseline drive (from neuron I-O characterization)
    "baseline_drive_by_cell_type": {
        "E": 3.33,
        "PV": 3.33,
        "SST": 0.67,
        "VIP": 19.31,
    },
}
```

### In Network Construction

```python
cfg = jtfne.laminar_cortex_config(
    areas=["V1", "V4", "MT", "FEF", "PFC"],
    layers=["L1", "L2/3", "L4", "L5A", "L5B", "L6"],
    cell_types={"E": 0.78, "PV": 0.10, "SST": 0.08, "VIP": 0.04},
    n=200,
    baseline_drive_by_cell_type=GLOBAL["baseline_drive_by_cell_type"],
    emitter="izhikevich",
)
```

### Optional: Stochastic Baseline

Set noise amplitude to same values for ~2 Hz expected stochastic firing:

```python
noise_amplitude_by_cell_type = {
    "E": 3.33,
    "PV": 3.33,
    "SST": 0.67,
    "VIP": 19.31,
}
```

---

## Expected Outcome (After Baseline Application)

### Baseline Simulation (TFNE_SMOKE=0, 1000 ms)

- **Baseline mean rate:** ~7–9 Hz (from 2 Hz stochastic base + network activity)
- **Baseline min rate:** ≥ 1.0 Hz (all neurons have DC support)

### AGSDR Tuning (64 candidates, gain 0.2–3.0)

- **Target gate (7.5 ± 1.5 Hz):** Expected PASS ✓
- **Min neuron rate gate (≥ 1.0 Hz):** Expected PASS ✓

> Note: "AGSDR Tuning" here refers to the sanity-checker's connectivity-gain
> **grid search** (`np.linspace` over 64 gain candidates) — a deterministic
> sweep, not the AGSDR (Adaptive Genetic Stochastic Delta Rule) optimizer.
> For the canonical AGSDR tuning path see `jtfne.agsdr()` + `Model.tune`.

### Delta-Test Status

- **Score:** 100/100 ✓
- **Release:** Unblocked (pending final validation)

---

## Scientific Notes

### Why VIP Requires Much Higher Baseline

VIP neurons use `b = -0.10` (negative recovery feedback), creating burst-like dynamics. This parameter creates a bifurcation that requires higher baseline current (~5–6× other types) to reliably evoke activity. This is **expected behavior**, not a bug:

- **E/PV/SST:** Positive `b`; tonic firing with moderate DC
- **VIP:** Negative `b`; bursting behavior; requires high DC or oscillatory input

### Rheobase Interpretation

Rheobase (minimum DC for spiking) varies by cell type due to parameter differences:

- **SST:** Lowest rheobase (0.83 nA) — most excitable to DC
- **E/PV:** Higher rheobase (4.17 nA) — less DC-sensitive
- **VIP:** Highest rheobase (24.14 nA) — requires strong baseline for bursting

Using 80% of rheobase provides a **safe margin** below full excitation while still raising the threshold for network-driven spiking.

---

## Files

- **Characterization script:** `scripts/characterize_neuron_io_curves.py`
- **Raw I-O curves:** `outputs/neuron_io_reference.json`
- **Baseline variants:** `outputs/neuron_io_baseline_reference.json`
- **This guide:** `docs/BASELINE_DRIVE_REFERENCE.md`
