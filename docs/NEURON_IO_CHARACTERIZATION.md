# Izhikevich Neuron Input-Output Characterization

## Purpose

Systematically map constant DC input (baseline drive) to steady-state firing rates for each jaxfne cell type (E, PV, SST, VIP). This eliminates the silent-neuron blocker in delta-test AGSDR by ensuring all neurons have baseline activity.

## Scientific Context

### Izhikevich Model

The reduced Izhikevich model is a two-variable ODE:

```
dv/dt = 0.04*v² + 5*v + 140 - u + I
du/dt = a*(b*v - u)

If v >= 30 mV:
    v := c
    u := u + d
```

Where:
- `v`: membrane potential (mV)
- `u`: recovery variable (sodium channel inactivation + potassium channel activation)
- `I`: external input (DC current + synaptic input)
- `a, b, c, d`: cell-type parameters

### Cell-Type Parameters

| Type | a    | b     | c    | d   | Description | Firing Mode |
|------|------|-------|------|-----|-------------|-------------|
| E    | 0.02 | 0.20  | -65  | 8.0 | Regular-spiking excitatory | Tonic |
| PV   | 0.10 | 0.20  | -65  | 2.0 | Fast-spiking parvalbumin+ | Tonic (very fast) |
| SST  | 0.02 | 0.25  | -65  | 2.0 | Low-threshold somatostatin+ | Tonic |
| VIP  | 0.02 | -0.10 | -55  | 6.0 | Intrinsic spiking / bursting | **Burst-like** (requires higher DC) |

**Note:** VIP uses negative `b` parameter, creating burst dynamics. Requires higher DC drive (0–100 nA) and longer observation windows (10 sec) to reveal firing patterns. Expected behavior: sparse high-frequency bursts rather than regular tonic spiking.

## Characterization Method

1. **Single-neuron isolation**: Initialize each cell type at rest (v=-65 mV, u=0)
2. **DC sweep**: Vary constant input current (drive) from 0 to 20 nA in 25 steps
3. **Simulation**: 5-second traces at 0.1 ms time step, Euler integration
4. **Steady-state extraction**: Discard first 1 second (transient), measure spike rate on remaining 4 seconds
5. **Interpolation**: Map target firing rates (1, 2, 4, 5, 10, 20, 50 Hz) to corresponding DC values

## Expected Results

### Firing Rate Thresholds

Each cell type has a **rheobase** (minimum DC to evoke any spiking):

- **E neurons**: Low threshold (~4 nA for first spikes; ~5 nA for 10 Hz)
- **PV neurons**: Similar threshold (~4–5 nA)
- **SST neurons**: Similar or slightly higher (~5–6 nA)
- **VIP neurons**: May differ due to negative `b` parameter (intrinsic bursting)

### Expected DC Values for 2 Hz Baseline

These values ensure all neurons have minimum 1–2 Hz baseline activity:

- **E**: ~4.5 nA (estimate)
- **PV**: ~3.5–4.5 nA (faster-spiking)
- **SST**: ~5.0–6.0 nA
- **VIP**: ~4.0–5.0 nA (depending on burst dynamics)

## Usage in Delta-Test

### Problem Being Solved

Delta-test full mode (TFNE_SMOKE=0) fails the min_neuron_rate gate (≥1.0 Hz) because:
- Network baseline has zero-firing (silent) neurons
- Connectivity gain scaling (AGSDR, 0.2–3.0×) cannot activate silent neurons
- Min_rate gate becomes unsatisfiable

### Solution

1. **Characterize** individual neuron I-O curves (this script)
2. **Set baseline DC drive** per cell type (from reference table)
3. **Inject DC current** in network construction
4. **Re-run AGSDR** with new baseline
5. **Expected outcome**: min_neuron_rate ≥ 1.0 Hz gate PASSES

### Configuration Example

```python
# In laminar_cortex_config or network builder:
baseline_drive_by_cell_type = {
    "E": 4.5,      # nA (from characterization)
    "PV": 4.0,     # nA
    "SST": 5.5,    # nA
    "VIP": 4.5,    # nA
}

# Inject during simulation:
# for each neuron, add baseline_drive_by_cell_type[cell_type] to I_input

# Also set noise amplitude to same values (for ~2 Hz expected baseline):
noise_amplitude_by_cell_type = baseline_drive_by_cell_type
```

## Truth Status

- **claim_level**: computational_scaffold
- **scope**: Baseline neural excitability, not biological calibration
- **physical_amplitude_calibrated**: False

These are Izhikevich native units (not calibrated to biological pA or nA). They are a computational proxy for relative excitability differences between cell types.

## Iteration Loop If Gates Still Fail

If after applying baseline DC, the min_neuron_rate gate still fails:

1. **Check**: Did DC injection actually apply? Verify network I_input computation.
2. **Increase DC**: Scale baseline_drive by 1.5–2.0× (but keep in GLOBAL config)
3. **Add noise**: Set noise_amplitude = baseline_drive to create stochastic baseline
4. **Profile**: Check per-neuron rates to find outliers (e.g., neurons with 0 Hz even after DC)
5. **Report blocker**: If no DC value works, may indicate network connectivity or parameter issue requiring redesign

## Files

- **Script**: `scripts/characterize_neuron_io_curves.py`
- **Output**: `outputs/neuron_io_reference.json` (DC values for target rates)
- **Table**: Printed reference table (DC vs firing rate for each cell type)
- **Usage**: Pass baseline_drive values to network construction
