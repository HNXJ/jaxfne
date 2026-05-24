# v0.3.1 Single Izhikevich Neuron

**Scenario status:** tutorial evidence generator on stable `jaxfne==0.2.30`  
**Canonical import:** `import jaxfne as jtfne`  
**Truth gates:** `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`, `physical_amplitude_claim_allowed=False`

Open in Colab: `notebooks/v030_01_single_neuron_izhikevich.ipynb`

## Learning objectives

1. Simulate a single reduced Izhikevich emitter for 1000 ms at `dt_ms=0.1` with deterministic seed and float32 runtime.
2. Explain voltage-like state, recovery/reset, spike events, and native current status without treating native current as amperes.
3. Generate SPK, Vm, source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, and EMM-proxy reports from one declared source/field state.
4. Export strict JSON artifacts, PNG figures, claim gates, and SHA-256 hashes.

## Biological/computational question

How does a reduced spiking emitter create a reproducible state trajectory that can be routed through the TFNE source-to-field/probe contract without making physical-amplitude or mechanism claims?

## Mathematical glossary flow

### Reduced voltage equation

\[
\frac{dv}{dt}=0.04v^2+5v+140-u+I_{\mathrm{native}}
\]

`v` is the voltage-like reduced state, `u` is the recovery state, and `I_native` is native model drive. In word form: **voltage-like change equals quadratic excitation plus linear excitation plus baseline drive minus recovery plus native drive**. The implementation path is the JAX Izhikevich emitter kernel used by `model.simulate`. Claim boundary: `I_native` is not a physical ampere current unless a calibration bridge is supplied.

### Reset equation

\[
\text{if }v\ge 30:\quad v\leftarrow c,\qquad u\leftarrow u+d
\]

`c` is reset voltage and `d` is recovery increment. In word form: **a spike event is threshold crossing followed by voltage reset and recovery increment**.

## Configuration block

```python
import jaxfne as jtfne

run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=0)

cfg = (
    jtfne.configuration()
    .network(name="v030_01_single_neuron", kind="isolated_neuron", n=1, cell_types={"E": 1.0})
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
    .probe(name="single_neuron_laminar_proxy", modes=["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
    .update_metadata(dx_mm=0.010, dy_mm=0.010, dz_mm=0.010,
                     geometry_mode="declared_tutorial_metadata_not_solved_3d_grid")
)
```

## Simulation block

```python
model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=0, runtime=run)
signals = model.simulate(sim)
```

## Probe/readout block

Use `jaxfne.fields` probe operators to produce JSON-safe reports for all eight readouts. Use names `spk`, `vm`, `source`, `lfp_proxy`, `csd_proxy`, `eeg_proxy`, `meg_proxy`, and `emm_proxy` in machine-readable reports. Use display labels such as “simulated LFP-like readout” in prose and figure titles.

## Manifest and claim gates

The tutorial must assert:

```python
manifest["truth_mode"] == "truth_safe_unverified"
manifest["claim_level"] == "computational_scaffold"
manifest["field_solver_status"] == "laminar_proxy_no_pde"
manifest["physical_amplitude_claim_allowed"] is False
```

## Figures

Required PNGs:

- `figures/voltage_trace.png`
- `figures/spike_raster.png`
- `figures/csd_proxy_heatmap.png`

## Interpretation

The expected successful run has finite arrays and mean firing rate between 2 and 25 Hz. This supports a tutorial-level demonstration of the emitter-to-probe pipeline. It does not validate a biological cell type, current amplitude, CSD amplitude, EEG, MEG, or metabolism.

## Failure modes

- Firing rate outside 2–25 Hz: retune native drive or use a declared null-control tutorial instead.
- NaN or Inf in any signal: reject the run and inspect dt, drive, and dtype.
- Missing JSON-safe manifest: reject publication until serialization is fixed.
- Probe report lacks proxy status: reject publication until report metadata is explicit.

## Exercises

1. Change `seed` and verify that claim gates remain unchanged.
2. Compare `duration_ms=1000` and `duration_ms=2000` while holding `dt_ms=0.1`.
3. Replace the raster title with “SPK readout” and explain why it remains simulated/proxy.

## What this tutorial does NOT claim

It does not claim a calibrated membrane current, solved Poisson field, real LFP/CSD/EEG/MEG, biological metabolism, or mechanism proof.

## Script

Run:

```bash
PYTHONPATH=. python examples/v030_01_single_neuron_izhikevich.py
```
