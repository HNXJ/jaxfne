# v0.3.2 Single-Neuron Parameter Sweep

**Scenario status:** tutorial evidence generator on stable `jaxfne==0.2.30`  
**Canonical import:** `import jaxfne as jtfne`  
**Truth gates:** `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`, `physical_amplitude_claim_allowed=False`

Open in Colab: `notebooks/v030_02_single_neuron_parameter_sweep.ipynb`

## Learning objectives

1. Sweep reduced Izhikevich parameters `a`, `b`, `c`, `d`, and native-drive gain while preserving fixed duration, dt, dtype, and seed.
2. Show how recovery speed, reset value, adaptation increment, and native drive alter firing rate and voltage-like trajectories.
3. Use finite/rate/JSON/claim gates to reject unstable or overclaiming tutorial runs.
4. Use the public `jtfne.with_emitter_parameters(model, a=..., drive_scale=...)` helper to modify emitter parameters without internal dataclass replacement.

## Biological/computational question

Which reduced-emitter parameters most strongly control firing rate in a single-neuron scaffold, and how can this be demonstrated without treating a parameter fit as biological mechanism proof?

## Mathematical glossary flow

### Recovery equation

\[
\frac{du}{dt}=a(bv-u)
\]

`u` is recovery state, `a` controls recovery speed, and `b` couples voltage-like state to the recovery target. In word form: **recovery change equals recovery speed times the gap between voltage-coupled target and current recovery**.

### Reset/adaptation equation

\[
\text{if }v\ge 30:\quad v\leftarrow c,\qquad u\leftarrow u+d
\]

`c` is the reset value and `d` is the post-spike recovery increment. In word form: **after a spike, the voltage-like state is reset and recovery is incremented, changing future excitability**.

## Configuration block

Use the same configuration as v0.3.1, then sweep scalar replacements of `a`, `b`, `c`, `d`, and drive scale via the public `jtfne.with_emitter_parameters(base_model, a=..., drive_scale=...)` helper. This keeps tutorial code clean without internal dataclass access.

## Simulation block

Every condition must use:

```python
duration_ms = 1000.0
dt_ms = 0.1
seed = 0
run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=seed)
```

## Probe/readout block

The tutorial records firing rate, spike count, voltage-like min/mean/max, and finite checks for each condition. Optional extension: add all eight probe reports per condition once artifact size is acceptable.

## Manifest and claim gates

The sweep passes only if every condition has finite arrays and 2–25 Hz firing rate unless the tutorial is explicitly reframed as a silence/instability/null-control lesson.

## Figures

Required PNGs:

- `figures/firing_rate_sweep.png`
- `figures/voltage_sweep_decimated.png`

## Interpretation

The sweep is a parameter-sensitivity demonstration. It can show computational effects of reduced-emitter parameters. It does not identify biological channel kinetics, validate a cell type, or prove a mechanism.

## Failure modes

- Some conditions exceed 25 Hz: reduce drive or label them as instability examples.
- Some conditions are silent: either retune or explicitly teach silence/null behavior.
- Parameter replacement fails: use `jtfne.with_emitter_parameters(base_model, ...)` which is public and tested.

## Exercises

1. Add a condition with stronger adaptation and predict the firing-rate direction before running it.
2. Add a weak-drive condition and classify it as valid activity or null-control.
3. Compare two seeds and explain which differences are stochastic and which are parameter-driven.

## What this tutorial does NOT claim

It does not claim calibrated biological parameter values, conductance-level channel mechanisms, real field amplitudes, or empirical fit.

## Script

Run:

```bash
PYTHONPATH=. python examples/v030_02_single_neuron_parameter_sweep.py
```
