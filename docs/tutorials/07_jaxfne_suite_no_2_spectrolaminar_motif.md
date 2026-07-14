# Suite No. 2 Upgrade: Izhikevich to V1-V4 Spectrolaminar Proxy Readouts

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)

Pipeline: `Emitter -> Source -> Field -> Probe -> Objective -> Optimizer`.

## Learning objectives

1. Configure one reduced Izhikevich emitter through `jaxfne`.
2. Compare E/PV/SST/VIP reduced-emitter presets through package-level arrays.
3. Build `net1`, a 100-emitter uniformly sampled 3D column.
4. Generate raster, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, and EMM-proxy figures without tutorial-local plotting functions.
5. Configure a V1-V4 six-layer scaffold with feedforward and feedback metadata.
6. Tune solver-tuned noise amplitude toward a firing-rate target with the Suite No. 2 AGSDR-Adam utility.

## Question

How far can a compact `jaxfne` grammar carry the workflow from one reduced emitter to a two-area laminar scaffold while preserving reusable source/field/probe readouts?

## Mathematical glossary flow

### Reduced Izhikevich emitter

Formal equation:

$$\frac{dv_i}{dt}=0.04v_i^2+5v_i+140-u_i+I_i(t), \qquad \frac{du_i}{dt}=a_i(b_i v_i-u_i)$$

Reset rule:

$$v_i\ge 30 \Rightarrow v_i\leftarrow c_i,\quad u_i\leftarrow u_i+d_i$$

Terms: $v_i$ is the reduced voltage-proxy state, $u_i$ is recovery, $a,b,c,d$ set class-specific dynamics, and $I_i(t)$ combines base drive, recurrent input, and stochastic drive.

Worded equation: voltage changes according to intrinsic recovery, a nonlinear voltage term, and input current; threshold events reset the emitter state.

Implementation: `jaxfne.emitters.IzhikevichEmitter`, `jaxfne.emitters.simulate_eig_izhikevich`, and Suite No. 2 config builders.

### Source/readout projection

Formal equation:

$$Y_c(t)=\sum_n W_{cn}S_n(t)$$

Terms: $S_n(t)$ is a source proxy from emitter $n$, $W_{cn}$ is a declared projection weight, and $Y_c(t)$ is a channel/contact readout.

Worded equation: each readout channel is a weighted sum of source activity.

Implementation: `jaxfne.fields.LinearReadout`, `jaxfne.fields.project_laminar_sources`, and `jaxfne.vis.*` figure functions.

## Package-level workflow

The notebook uses only package-level helpers:

```python
import jaxfne as jtfne

cfg_net1 = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=1000.0, dt_ms=0.1)
model_net1 = jtfne.construct(cfg_net1)
bundle_net1 = jtfne.suite2_run_bundle(model_net1, seed=7, duration_ms=1000.0, dt_ms=0.1)
fig_panel = jtfne.vis.spectrolaminar_suite(bundle_net1["signals"], max_freq_hz=80.0)
```

V1-V4 scaffold:

```python
cfg_v1v4 = jtfne.suite2_v1_v4_config(seed=7, n_per_area=400, duration_ms=1000.0, dt_ms=0.1)
model_v1v4 = jtfne.construct(cfg_v1v4)
```

## Latest canonical network (recommended for the spectrolaminar crossover)

For the depth × frequency crossover (deep alpha/beta vs superficial gamma), use the
canonical laminar column with a realistic E/I gradient and the multi-trial **LFP**
pipeline. The column follows the ground-truth composition, `jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`
(E peaks deep, I peaks superficial; PV peaks at L2/L3, absent at L6; ~66E:34I overall):

```python
import jaxfne as jtfne
from jaxfne import tutorial_utils as tu
from jaxfne.vis.tutorial_panels import spectrolaminar_suite_3panel
jtfne.enable_x64()

# Per-layer neuron counts (count ∝ thickness) and the E/I gradient.
LAYER_COUNT_FRAC = {"L1": 0.10, "L2": 0.25, "L3": 0.20, "L4": 0.10, "L5": 0.20, "L6": 0.15}
LAYER_CELL_TYPES = {
    "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},  # 50% I
    "L2": {"E": 0.50, "PV": 0.25, "SST": 0.10, "VIP": 0.15},  # I-count peak (dense L2)
    "L3": {"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10},
    "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},  # PV feedforward
    "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},
    "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},  # E-fraction peak, no PV
}

cfg = tu.make_laminar_column_config(
    areas=("V1",), area_x_rel=(0.0,),
    layer_count_frac=LAYER_COUNT_FRAC,
    layer_cell_type_frac=LAYER_CELL_TYPES,
    n_neuron_per_column=1000,        # 1k is tutorial-scale; crossover needs oscillatory regime + low κ
    dt_ms=0.1, duration_ms=1000.0,
    n_trials=10, n_contacts=32, freq_count=96, seed=0,
)
model  = tu.build_laminar_column(cfg)            # build ONCE
trials = tu.simulate_laminar_trials(model, cfg, n_trials=10)   # reused across trials

# Spectrolaminar from the LFP proxy (NOT CSD — CSD suppresses the deep alpha/beta band)
scores, specs = tu.summarize_spectrolaminar_similarity(trials, cfg, signal_key="lfp_contacts")
figs = spectrolaminar_suite_3panel(specs, model, cfg, areas=["V1"])   # canonical 3-panel suite
```

`spectrolaminar_suite_3panel` renders the canonical motif: **(A)** superficial-vs-deep
relative-power spectra, **(B)** depth × frequency relative-power heatmap, **(C)**
alpha-beta (deep) / gamma (superficial) band power crossing at the L4 reference.

### Superficial → deep descending drive

The canonical microcircuit is completed by an intra-column **L2/3 (E) → L5/6 (E)**
descending pathway, so superficial activity drives the deep layers (countering the
intrinsic superficial-hotter bias). It is part of the default `connectivity_spec`
(`L23_to_deep_*` rules) and is tunable via `base_control['supra_to_deep_gain']`:

```python
cfg = tu.make_laminar_column_config(
    areas=("V1",), area_x_rel=(0.0,),
    layer_count_frac=LAYER_COUNT_FRAC, layer_cell_type_frac=LAYER_CELL_TYPES,
    n_neuron_per_column=1000, dt_ms=0.1, duration_ms=1000.0,
    n_trials=10, n_contacts=32, freq_count=96, seed=0,
    base_control={"supra_to_deep_gain": 1.0},   # 0.0 disables; >1 strengthens
)
```

Effect on per-layer E firing (deep / superficial rate ratio): `gain=0.0 → ~0.96`
(deep slightly cooler); `gain=1.0 → ~1.34` (deep driven above superficial). Raise
the gain for a stronger descending drive; deep firing scales monotonically with it.

A symmetric **inhibitory** L2/3 (I) → L5/6 (E) pathway (`L23_inh_to_deep_*`) mirrors
the excitatory rules; its source cell types are the interneurons (PV/SST/VIP), so it
contributes inhibition. Tune it with `base_control['supra_to_deep_inh_gain']`
(default 1.0). With both pathways at gain 1.0 the deep E rate is balanced against
superficial (the inhibition offsets the excitatory drive).

Pass `trials=` to `spectrolaminar_suite_3panel` to add a fourth panel — **firing rate
by depth** (E / I / total), per depth bin:

```python
figs = spectrolaminar_suite_3panel(specs, model, cfg, areas=["V1"], trials=trials)
```

Two readout prerequisites (verify both — see the validation gates below):

```python
import numpy as np
# 1) Low synchrony: kappa ~ 0 (else a global rhythm biases the readout)
kappa = jtfne.kappa_synchrony(np.asarray(trials["spikes"])[0], cfg.dt_ms)
# 2) Balanced per-layer rates (~10 Hz, no silent layer): apply graded per-layer drive
#    on the SAME built model via model.with_emitter_parameters(drive_per_neuron=...)
```

The crossover is **not guaranteed by column geometry or N alone**: in an
asynchronous-irregular regime the LFP is broadband at every depth. A deep-α/β vs
superficial-γ crossing requires band-limited **oscillations** localized by layer
while global κ stays low — see `skills/jaxfne-spectrolaminar-suite/SKILL.md`.
Multi-trial averaging and ≥1k neurons help **estimate** the readout, not create the regime.

## Homeostasis-Dependent Plasticity (HDP) for long-duration spectrolaminar runs

The two readout prerequisites above — low synchrony and balanced per-layer rates —
get harder to hold by hand as run duration grows: a drive correction tuned at a short
window can drift out of the target rate band over a longer one, and weight plasticity
left unconstrained can run away. **Homeostasis-Dependent Plasticity (HDP)** is
a per-neuron master state `H_i` (default 1.0) that all of a neuron's incoming
excitatory/inhibitory weight updates read from, and that synaptic drive and the
neuron's own spiking feed back into — keeping a population stationary over many
seconds via automatic feedback rather than hand-retuning the drive correction.

```python
I_syn_i = sum_j w_ji * x_j                                 # incoming synaptic current
tau_i * dH_i/dt = alpha*I_syn_i - gamma*r_i - delta*W_i + K_ctrl*(1 - H_i) - dC/dH_i
dw_E^ij/dt = +K_HDP * (H_i - 1) * w_E^ij                   # excitatory incoming edges
dw_I^ij/dt = -K_HDP * (H_i - 1) * w_I^ij                   # inhibitory incoming edges
```

`K_ctrl*(1-H_i)` is the linear restoring force that pins the equilibrium at `H_i=1`;
`tau_i = tau_0_ms * size_i**2` makes larger (e.g. E) cells integrate `H_i` more
slowly. Implementation: `jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp`. A
single config-driven builder for HDP-ready laminar columns of any size lives in
`jaxfne.hdp_network` (`HDPColumnConfig` + `build_model`/`apply_drive_correction`/`run`) —
size is a config field rather than a per-N function.

**HDP integration into `core.py`/`RuntimeConfig` remains an open task**, so it is driven
directly against a `jtfne.construct()`-built `Model`'s `emitter`/`edge_list` params,
then hand-assembled into a `jtfne.Signals` container for `jtfne.vis.spectrolaminar_suite`:

```python
import jax, jax.numpy as jnp, numpy as np
import jaxfne as jtfne
from jaxfne.hdp_network import (
    HDPColumnConfig, build_model, apply_drive_correction,
    layer_size_scale_override, run, BASE_HDP_KWARGS_DEFAULT, DEFAULT_HDP,
)

cfg = HDPColumnConfig(n_neurons=1000, duration_ms=5000.0, dt_ms=0.5, seed=0)
model = build_model(cfg)                          # build ONCE
model = apply_drive_correction(model, cfg)
size_override = layer_size_scale_override(model, cfg)   # deep-layer (L5/L6) size inflation

hdp_kwargs = dict(BASE_HDP_KWARGS_DEFAULT); hdp_kwargs.update(DEFAULT_HDP)
out = run(model, cfg, duration_ms=5000.0, seed=0,
          hdp_kwargs=hdp_kwargs, size_scale_override=size_override)
spikes, H_trace, w_trace = out["spikes"], out["diagnostics"]["H_trace"], out["diagnostics"]["w_trace"]
```

`DEFAULT_HDP` (`K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, barrier_c=barrier_d=0.01`) is a
5-seed, 20-second-validated stability point: rates stay flat in-band, `H` pins at
~1.0000-1.0029, weight saturation ~10.5% (active, away from the floor/ceiling). It is the
config to reach for when the goal is a **long, stationary** run.

### Avoiding HDP-induced oversynchrony

`DEFAULT_HDP`'s strong restoring force (`K_ctrl=5.0`) combined with `tau_i =
tau_0_ms * size_i**2` (E cells default to `size=5`, so `tau_i=5000` ms) makes `H_i`
almost static (`H_std ≈ 0.0006`). Lacking a per-neuron variability driver, population
spiking reads as near-regular ("ECG-like") rather than async-irregular — exactly the
high-synchrony failure mode the **low synchrony** prerequisite above warns about.
`jaxfne.hdp_network.DEFAULT_HDP_DESYNC` (paired with `DRIVE_SCALE_DESYNC=1.2`) trades
some of that overdamping for faster `H` integration (`tau_0_ms=5`, 40x faster) plus a
genuine rate-drain term (`gamma=0.5`), so `H_i` fluctuates instead of sitting pinned:

```python
from jaxfne.hdp_network import (
    DEFAULT_HDP_DESYNC, DRIVE_SCALE_DESYNC, BASE_DRIVE_BY_CELL_TYPE_DEFAULT,
)

scaled_drive = {ct: v * DRIVE_SCALE_DESYNC for ct, v in BASE_DRIVE_BY_CELL_TYPE_DEFAULT.items()}
cfg = HDPColumnConfig(n_neurons=500, duration_ms=2000.0, dt_ms=0.5, seed=0,
                       base_drive_by_cell_type=scaled_drive)
# ... build_model / apply_drive_correction / layer_size_scale_override as above ...
hdp_kwargs = dict(BASE_HDP_KWARGS_DEFAULT); hdp_kwargs.update(DEFAULT_HDP_DESYNC)
```

Verified, 5-seed stable (N=500, 2000ms): overall rate 7.3 → 12.6 Hz, per-neuron rate std
0.99 → 6.0 Hz, `H` fluctuates genuinely (`H=1.028±0.023`, range `[0.953, 1.227]`) while
every neuron stays clear of the `H_min`/`H_max` clamp rails, and `kappa_synchrony` stays
at 0.044 (still async-irregular). This is a second-pass refinement of an earlier, wider
and more skewed candidate (`H=[0.96, 1.47]`) -- raising `gamma` from 0.3 to 0.5 tightened
both tails since the rate-drain term is rate-bounded rather than weight-multiplicative,
so it does not amplify the H>1 weight-growth feedback loop the way `alpha` does;
`gamma>=0.55` hits a stability cliff (weight runaway) at every `K_ctrl`/`alpha` tried.
`H`'s range has yet to symmetrically span `[0.8, 1.2]` — the floor (~0.95) appears
structurally bottlenecked by E neurons' large `tau_i` (size=5 for E), not by these three
gains, so treat this as the current best candidate, a still-open point unlike the frozen
`DEFAULT_HDP`.

### Two distinct, intentionally-separate operating modes

`DEFAULT_HDP` and `DEFAULT_HDP_DESYNC` answer different questions and should stay
separate rather than collapse into one "better" default — pick the preset that
matches the question:

| Parameter / outcome | `DEFAULT_HDP` (Stable) | `DEFAULT_HDP_DESYNC` (Desync) |
|---|---|---|
| Purpose | Maximum stability, long-duration validation, canonical freeze candidate | Increased H dynamics, larger per-neuron variability, exploration/research mode |
| `K_HDP` | `0.01` | `0.01` |
| `tau_0_ms` | `200.0` | `5.0` |
| `K_ctrl` | `5.0` | `0.15` |
| `alpha` | `0.01` (from `BASE_HDP_KWARGS_DEFAULT`) | `0.05` |
| `gamma` | `0.0` | `0.5` |
| `barrier_c` / `barrier_d` | `0.01` / `0.01` | `0.01` / `0.01` |
| Drive scale | `1.0` (`BASE_DRIVE_BY_CELL_TYPE_DEFAULT` as-is) | `1.2` (`DRIVE_SCALE_DESYNC`) |
| Validated overall rate | ~7.3-7.8 Hz | ~12.6 Hz |
| Per-neuron rate std | 0.99 Hz | 6.0 Hz |
| `H` behavior | Pinned (`H_mean≈1.001-1.003`, `H_std≈0.0006`) | Fluctuating (`H=1.028±0.023`, range `[0.953, 1.227]`) |
| `kappa_synchrony` | ~0.04-0.05 | ~0.044 |
| Validation evidence | 5-seed × 20s stability gate | 5-seed × 2s grid-search refinement, N=500 |
| Status | Frozen (0.4.6 candidate) | Best-of-two-pass; still open |

## Figures

Core figures are generated by reusable visualization functions:

- `jtfne.vis.raster(..., sort_by="z")`
- `jtfne.vis.lfp_traces(...)`
- `jtfne.vis.csd_traces(...)`
- `jtfne.vis.eeg(...)`
- `jtfne.vis.meg(...)`
- `jtfne.vis.emm(...)`
- `jtfne.vis.spectrolaminar_suite(...)` — single-run readout overview
- `jtfne.vis.tutorial_panels.spectrolaminar_suite_3panel(...)` — canonical depth × frequency motif (spectra | heatmap | crossover), LFP-based
- `jtfne.vis.circuit3d(...)`

---

## Configuration Reference

All Suite No. 2 workflows are specified through the chainable `Configuration` API.
Every field listed here stores in `cfg.metadata`; none are physical statuss.

### Smoke config vs full config

```python
import jaxfne as jtfne

# ── Smoke config (fast, 10 neurons, 100 ms) ─────────────────────────────────
cfg_smoke = jtfne.suite2_net1_config(seed=7, n=10, duration_ms=100.0, dt_ms=0.1)

# ── Full config (100 neurons, 1000 ms) ───────────────────────────────────────
cfg_full = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=1000.0, dt_ms=0.1)

# ── V1-V4 multi-area scaffold ────────────────────────────────────────────────
cfg_v1v4 = jtfne.suite2_v1_v4_config(seed=7, n_per_area=400, duration_ms=1000.0, dt_ms=0.1)
```

The same 10-domain grammar applies to both:

```python
cfg = (
    jtfne.Configuration()
    .runtime(seed=7, duration_ms=1000.0, dt_ms=0.1, dtype="float32")
    .column("V1", layers=["L1", "L2/3", "L4", "L5", "L6"], n=100)
    .cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
    .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45)
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
)
```

### Runtime domain (`cfg.runtime`)

| Parameter | Default | Unit / status | Description |
|---|---|---|---|
| `seed` | `42` | integer | PRNG seed for all random operations |
| `duration_ms` | `1000.0` | ms | Total simulation duration |
| `dt_ms` | `0.1` | ms | Fixed timestep (forward Euler) |
| `dtype` | `"float32"` | string | JAX array dtype (`"float32"` or `"float64"`) |
| `backend` | `"cpu"` | string | JAX backend (`"cpu"`, `"gpu"`) |
| `jit` | `True` | bool | Enable `jax.jit` compilation for main kernels |
| `vmap` | `True` | bool | Enable `jax.vmap` over trial batches |
| `n_trials` | `1` | integer | Number of stochastic trials |

### Column domain (`cfg.column`)

| Parameter | Default | Unit / status | Description |
|---|---|---|---|
| `column_name` | `"single_column"` | string | Column/area identifier |
| `layers` | `["L1","L2/3","L4","L5","L6"]` | list | Layer labels (declarative) |
| `n` | `100` | integer | Total neuron count per column |
| `x_size_mm` | `0.5` | mm | Lateral x extent (proxy geometry) |
| `y_size_mm` | `0.5` | mm | Lateral y extent (proxy geometry) |
| `z_size_mm` | `1.6` | mm | Laminar depth extent (proxy geometry) |
| `geometry_status` | `"declared_metadata"` | string | Always `declared_metadata`; no 3D PDE grid |

### Cell-type domain (`cfg.cell_types`)

| Cell type | Global fraction | Sign | Izhikevich preset | Drive default (a.u.) |
|---|---|---|---|---|
| `E` (excitatory) | `0.75` | `+1` | `cortical_eig` | `5.0` |
| `PV` (fast-spiking) | `0.10` | `−1` | `cortical_eig` | `3.0` |
| `SST` (Martinotti) | `0.08` | `−1` | `cortical_eig` | `3.5` |
| `VIP` (disinhibitory) | `0.07` | `−1` | `cortical_eig` | `3.0` |

### Izhikevich parameters by cell type (`cortical_eig` preset)

| Parameter | E | PV | SST | VIP | Unit |
|---|---|---|---|---|---|
| **a** | 0.02 | 0.10 | 0.02 | 0.02 | 1/ms |
| **b** | 0.20 | 0.20 | 0.25 | −0.10 | dimensionless |
| **c** | −65.0 | −65.0 | −65.0 | −55.0 | mV |
| **d** | 8.0 | 2.0 | 2.0 | 6.0 | mV/ms |
| **Spike threshold** | 30 | 30 | 30 | 30 | mV |
| **Reset V** | −65 | −65 | −65 | −55 | mV |

PV has high `a` (fast recovery) and low `d` (weak AHP) → fast-spiking.
VIP has negative `b` (bistable subthreshold) and higher reset `c` → irregular firing.

### Connectivity domain (`cfg.connectivity`)

| Parameter | Default | Unit / status | Description |
|---|---|---|---|
| `within_area` | `"all_to_all_uniform_random"` | string | Recurrent topology mode |
| `within_gain` | `0.45` | dimensionless | Global weight scale (net1); `0.35` for V1-V4 |
| `E_weight_range` | `(0.5, 2.0)` | a.u. | Excitatory weight range |
| `I_weight_range` | `(-1.5, -0.5)` | a.u. | Inhibitory weight range |
| `synaptic_tau_ms` | `5.0` | ms | Exponential synapse decay constant |
| `sign_policy` | `"intrinsic"` | string | Sign from emitter type, not overridden |
| `seed` | `None` | integer | Edge-weight PRNG seed |

### Drive domain (`cfg.drive`)

| Parameter | Default | Unit / status | Description |
|---|---|---|---|
| `baseline_drive_by_cell_type` | `{"E": 5.0, "PV": 3.0, "SST": 3.5, "VIP": 3.0}` | a.u. | Constant external drive per cell type |
| `noise_policy` | `"additive_poisson"` | string | Stochastic drive mode |
| `noise_amplitude` | `0.5` | a.u. | Noise scale (tuned by `suite2_tune_noise_agsdr_adam`) |
| `time_schedule` | `"constant"` | string | Drive profile (`"constant"`, `"pulse"`, `"ramp"`) |
| `trial_variability` | `False` | bool | Vary seed across trials |

### Inter-column connectivity domain (`cfg.inter_column_connectivity`)

For V1-V4 only. Declarative metadata; connectivity is stored as parameter dicts, resolved into sparse matrices only at `simulate()` time.

| Parameter | Default | Unit / status | Description |
|---|---|---|---|
| `source_area` | `"V1"` | string | Feedforward origin |
| `target_area` | `"V4"` | string | Feedforward target |
| `mode` | `"sparse"` | string | Topology (`"sparse"`, `"all_to_all"`) |
| `p_feedforward` | `0.3` | probability | V1→V4 connection probability |
| `p_feedback` | `0.2` | probability | V4→V1 connection probability |
| `feedforward_weight_range` | `(0.5, 2.0)` | a.u. | V1→V4 weight range |
| `feedback_weight_range` | `(0.3, 1.5)` | a.u. | V4→V1 weight range |
| `layer_to_layer_map` | `{"L2/3": "L4", "L5": "L1"}` | dict | Declared source→target layer routing |

### Field domain (`cfg.field`)

| Parameter | Value | Status |
|---|---|---|
| `domain` | `"laminar_column"` | Proxy geometry |
| `conductivity` | `"proxy"` | Relative-value |
| `boundary` | `"mean_zero_neumann"` | Boundary convention |
| `field_solver_status` | `"linear_solver"` | **Immutable** |
| `field_claim_level` | `"proxy_readout"` | **Immutable** |
| `physical_amplitude_calibrated` | `False` | **Immutable** |
| `source_projection_mode` | `"proxy_no_field_solve"` | Proxy only |

### Probe domain (`cfg.probes`)

| Probe mode | Default contacts | Returns | Scope |
|---|---|---|---|
| `"spikes"` | N neurons | `bool [N, T]` | Spike indicator |
| `"V_m"` | N neurons | `float32 [N, T]` | Voltage-proxy |
| `"source"` | N neurons | `float32 [N, T]` | Source current proxy |
| `"LFP"` | 16 | `float32 [Z, T]` | LFP-proxy |
| `"CSD"` | 16 | `float32 [Z, T]` | CSD-proxy (2nd spatial derivative) |
| `"EEG"` | 4 | `float32 [4, T]` | EEG proxy (declared leadfield) |
| `"MEG"` | 4 | `float32 [4, T]` | MEG proxy (declared leadfield) |
| `"EMM"` | 1 | `float32 [1, T]` | Signaling-energy proxy |

All probes carry `units_or_status: "proxy_relative_units"` and
`physical_amplitude_calibrated: false`.

### Objective domain (`cfg.objective`)

| Parameter | Default | Description |
|---|---|---|
| `firing_rate_target` | `{"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0}` | Target Hz per cell type |
| `band_definitions` | `{"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)}` | Hz ranges |
| `rejection_gates` | `{"max_firing_rate": (0.0, 200.0)}` | Hard acceptance bounds |

### Optimizer domain (`cfg.optimizer`)

| Parameter | Default | Description |
|---|---|---|
| `optimizer_family` | `"AGSDR"` | Optimizer type (`"AGSDR"`, `"random_search"`) |
| `surrogate_status` | `"soft_rate_surrogate"` | Inner-loop approximation; not for statements |
| `budget` | `50` | Optimization iterations |
| `hard_gates` | `{"model_status": "computational_scaffold"}` | Cannot be overridden |

---

## Simulation Execution

### Core pipeline

```python
import jaxfne as jtfne

cfg = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=1000.0, dt_ms=0.1)
model = jtfne.construct(cfg)

# Run bundle: keys are simulation, signals, readout, manifest, neuron_table
bundle = jtfne.suite2_run_bundle(model, seed=7, duration_ms=1000.0, dt_ms=0.1)

signals  = bundle["signals"]    # Signals object
manifest = bundle["manifest"]   # JSON-safe, truth-gated run manifest
readout  = bundle["readout"]    # readout results
```

### Solver semantics (per timestep)

The integration uses forward Euler with fixed `dt_ms`:

1. Evaluate recurrent input: `I_rec = W @ S_prev` (dense or sparse edge-list)
2. Evaluate external drive: `I_ext = baseline_drive + I_noise(noise_amplitude, key)`
3. Update `v` and `u` via Izhikevich equations
4. Detect threshold: if `v >= 30`, record spike, reset `v = c`, `u = u + d`
5. Update synaptic traces: exponential decay `tau = synaptic_tau_ms`
6. Record all probe outputs for this timestep

### Numerical bounds

| Quantity | Clamp / reset | Value |
|---|---|---|
| Membrane potential `v` | Hard clip | `[-100, +50]` mV |
| Spike threshold | Hard | 30 mV |
| Reset value | Cell-type specific | `c` parameter |
| Synaptic trace | Decay only | Never negative |

### Output structure

`suite2_run_bundle()` returns a dict with keys
`simulation`, `signals`, `readout`, `manifest`, `neuron_table`:

```python
{
    "simulation":   Simulation(...),            # the resolved Simulation spec
    "signals":      Signals(                     # proxy signals; access via .get(key)
        # spikes, V_m, source_proxy, lfp_proxy, csd_proxy, (eeg_proxy/meg_proxy/emm_proxy if requested)
    ),
    "readout":      [ReadoutResult(...), ...],   # requested readouts
    "neuron_table": [ {...}, ... ],              # per-neuron rows (list of dicts)
    "manifest": {                                # JSON-safe, truth-gated
        "duration_ms": 1000.0, "dt_ms": 0.1,
        "claim_level": "computational_scaffold",
        "field_solver_status": "linear_solver",
        "field_claim_level": "proxy_readout",
        "physical_amplitude_calibrated": False,
    },
}
```

Signal keys are `*_proxy` (`lfp_proxy`, `csd_proxy`, `source_proxy`); the legacy
`*_like` names are retired. Read them with `signals.get("lfp_proxy")`.

### Noise tuning

Use `suite2_tune_noise_agsdr_adam` to tune `noise_amplitude` toward the
firing-rate target:

```python
tuned_model, tune_report = jtfne.suite2_tune_noise_agsdr_adam(
    model,
    seed=7,
    duration_ms=500.0,   # shorter for tuning runs
    dt_ms=0.1,
    n_steps=30,
)
# Re-run with tuned model:
bundle_tuned = jtfne.suite2_run_bundle(tuned_model, seed=7, duration_ms=1000.0, dt_ms=0.1)
```

The surrogate optimizer minimizes the squared deviation from the firing-rate
target. **The surrogate path is for inner-loop optimization only; it does not
produce physical amplitude statuss.**

---

## Manifest & Reproducibility

Every simulation run should be accompanied by a manifest capturing all inputs
and outputs for auditability.

### Manifest structure

```python
import jaxfne as jtfne

manifest = bundle["manifest"]          # already produced by suite2_run_bundle
# or build one directly (cfg is positional; optional signals/readout/objective/tuning):
manifest = jtfne.manifest(cfg, signals=bundle["signals"])

# Validate manifest is JSON-safe (no NaN/Inf)
import json
json.dumps(manifest, allow_nan=False)  # must not raise
```

The manifest includes:

```json
{
  "version":      "0.4.1",
  "duration_ms":  1000.0,
  "dt_ms":        0.1,
  "n_steps":      10000,
  "outputs": {
    "spikes":     {"shape": [100, 10000], "dtype": "bool"},
    "V_m":        {"shape": [100, 10000], "dtype": "float32"},
    "lfp_proxy":  {"shape": [16,  10000], "dtype": "float32"}
  },
  "claim_level":                   "computational_scaffold",
  "field_solver_status":           "linear_solver",
  "field_claim_level":             "proxy_readout",
  "physical_amplitude_calibrated": false
}
```

### Reproducibility

To reproduce a run exactly:

```python
cfg2 = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=1000.0, dt_ms=0.1)
model2 = jtfne.construct(cfg2)
bundle2 = jtfne.suite2_run_bundle(model2, seed=7, duration_ms=1000.0, dt_ms=0.1)

# Outputs are bit-identical to bundle["signals"] within float32 rounding
import jax.numpy as jnp
assert jnp.allclose(bundle["signals"].V_m, bundle2["signals"].V_m)
```

### Validation gates

Before accepting any output:

| Gate | Check | Remediation |
|---|---|---|
| Finiteness | `jnp.all(jnp.isfinite(signals.V_m))` | Reduce drive or dt_ms |
| Firing rate | Per-cell-type rate in `[0.1, 150]` Hz | Adjust within_gain or drive |
| **Layer-rate balance** | Per-layer rates within ~10 Hz ± 5 (no layer silent vs others) | Apply graded per-layer drive via `with_emitter_parameters` (reuse the built model) |
| **Synchrony** | `jtfne.kappa_synchrony(signals.get("spikes"), dt_ms)` ≈ 0 | If κ ≳ 0.2–0.3 the network is locked and the spectrolaminar readout is biased — add drive heterogeneity/noise to de-synchronize |
| Spectral positivity | All band powers ≥ 0 | Check LFP-proxy output |
| JSON-safety | `json.dumps(manifest, allow_nan=False)` | Never export NaN/Inf |
| Gate fields | `physical_amplitude_calibrated == False` | Never override |

**Simulation-only outputs:** The manifest records configuration and output statistics.
PDE residuals, convergence diagnostics, and calibration certificates are outside
scope — the proxy field operator requires none.

---

## Interpretation of Results

### Expected firing rates

| Cell type | Expected range | Target (tuned) |
|---|---|---|
| E (pyramidal) | 5–15 Hz | 8 Hz |
| PV (fast-spiking) | 10–30 Hz | 15 Hz |
| SST (Martinotti) | 2–10 Hz | 4 Hz |
| VIP (disinhibitory) | 1–5 Hz | 2 Hz |

If observed rates are >2× outside these ranges, the network is unstable or frozen.

### Expected spectrolaminar proxy power

**Compute the spectrolaminar crossover from the LFP proxy, not the CSD proxy.**
The depth × frequency crossover (deep alpha/beta vs superficial gamma) is an LFP
phenomenon. CSD-proxy is the second spatial derivative of the LFP, so it sharpens
local sinks/sources and *suppresses* the broad, slow, spatially-extended deep
alpha/beta component — measuring the crossover on CSD yields a false "deep alpha/beta
absent" result. Use `signal_key="lfp_contacts"` (the
`summarize_spectrolaminar_similarity` default) for the crossover; CSD-proxy remains a
useful complementary readout of local current sinks/sources, not the band crossover.

LFP-proxy power is in **proxy relative units** (not µV²):

- **Alpha/beta (8–25 Hz):** relatively higher in deep layers (L5/L6).
- **Gamma (40–150 Hz):** relatively higher in superficial layers (L2/3).

Two prerequisites for a trustworthy readout (both checked in the validation gates):

1. **Low synchrony** — `jtfne.kappa_synchrony(spikes, dt_ms)` ≈ 0. A synchronized
   network oscillates as a whole, so the spectrum reflects one global rhythm rather
   than per-layer band structure, masking the crossover.
2. **Balanced per-layer rates** — no layer silent relative to others (target ~10 Hz);
   uneven rates bias the readout toward the hotter layers.

The crossover is **scale-dependent**: it is weak at small populations and sharpens with
neuron count and trial averaging. Use the multi-trial
`tutorial_utils.spectrolaminar_from_trials(..., signal_key="lfp_contacts")` for the
canonical, L4-referenced relative-power profile.

### Cell-type roles

| Cell type | Role | Ablation effect |
|---|---|---|
| E | Primary driver; recurrent excitation | Network goes silent (too little) or unstable (too much) |
| PV | Fast feedback inhibition | Loss causes runaway firing and loss of gamma |
| SST | Dendritic inhibition | Loss reduces layer-specific selectivity |
| VIP | Disinhibition of E via PV suppression | Loss reduces gain modulation |

### Null comparison

To confirm spectrolaminar features are not artifactual:
1. Shuffle spike times within each trial (preserves firing rate)
2. Recompute LFP/CSD proxy from shuffled spikes
3. Real band power should exceed shuffled by >2× for meaningful signal

### When to discard a run

Discard if any of the following:
- Any signal contains `NaN` or `Inf`
- Any cell-type firing rate is outside `[0.1, 150]` Hz
- `physical_amplitude_calibrated` was overridden (not possible via public API)
- Manifest JSON serialization fails

---

## Failure Modes

| Symptom | Likely cause | Remediation |
|---|---|---|
| All neurons silent (rate ≈ 0 Hz) | Drive too low or inhibition too strong | Increase `baseline_drive_by_cell_type["E"]`; decrease `within_gain` |
| Runaway firing (>100 Hz sustained) | Drive too high or E dominance | Decrease drive; increase PV/SST drive |
| NaN or Inf in output | Numerical overflow; `dt_ms` too large | Reduce `dt_ms` to 0.05; reduce drive amplitude |
| Flat LFP (no oscillations) | Insufficient recurrent feedback | Increase `within_gain`; check synaptic tau |
| VIP neurons always silent | VIP drive below threshold | Increase `baseline_drive_by_cell_type["VIP"]` |
| CSD all zeros | Source projection degenerate | Check `n_contacts >= 3`; verify column height > 0 |
| Slow JIT compile (>60 s) | Shape instability / recompilation loop | Fix `n`, `n_steps`, `n_contacts` before first run |
| Suite runs differ across machines | x64 not enabled consistently | Call `jtfne.enable_x64()` before array construction |

### Dense recurrent matrix on small CPUs

For `n > 500`, the dense `W @ S` matrix multiply becomes expensive on CPU.
Use the `recurrent_backend="edge_list"` option in `RuntimeConfig` to switch to
a sparse edge-list kernel:

```python
from jaxfne.core import RuntimeConfig
rt = RuntimeConfig(recurrent_backend="edge_list")
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7, runtime=rt)
```

---

## Exercises

1. **Smoke first:** Run `suite2_net1_config(n=10, duration_ms=100.0)` before
   attempting `n=100` to confirm the full pipeline executes.

2. **PV ablation:** Set `baseline_drive_by_cell_type["PV"] = 0.0` and rerun.
   Observe how gamma-band power changes when fast inhibition is removed.

3. **SST ablation:** Repeat with `SST = 0.0`. Compare to PV ablation — SST loss
   tends to broaden layer selectivity rather than collapse gamma.

4. **Noise tuning convergence:** Run `suite2_tune_noise_agsdr_adam` for 10, 20,
   and 50 steps. Plot the objective value versus iteration count.

5. **V1-V4 feedforward dominance:** In `suite2_v1_v4_config`, set
   `p_feedback = 0.0` (feedforward only). Does V4 gamma power decrease?

6. **Contact density sweep:** Vary `n_contacts` from 4 to 32 and plot the
   spatial resolution of the CSD-proxy profile.

7. **Null check:** Shuffle spike times in `bundle["signals"].spikes` and
   recompute LFP/CSD. Confirm real band power > 2× shuffled power.

---

## Coverage boundary

This tutorial covers reduced emitters, declared source projection, relative proxy readouts, and package-level figure generation. Solver-tuned amplitudes, subject-specific head geometry, and empirical parameter fitting belong to later workflows.

---

## Extended Mathematical Glossary

### Source tensor construction

Formal equation:

$$S[t, n] = \begin{cases}
R[t] & \text{(spike proxy mode)} \\
I_{\text{decomposed}}[t, n] + I_{\text{synap}}[t, n] & \text{(decomposed mode)} \\
I_{\text{total}}[t, n] & \text{(total current mode)}
\end{cases}$$

Terms:
- $S[t, n]$: source proxy from emitter $n$ at timestep $t$
- $R[t]$: population spike rate (relative units)
- $I_{\text{decomposed}}$: capacitive + ionic current
- $I_{\text{synap}}$: synaptic input current
- $I_{\text{total}}$: membrane current (total of all sources)

Worded equation: source proxy is selected from available current decompositions or spike rate, validated for finite values and single-source consistency.

Implementation location: `jaxfne.fields.construct_source_tensor`

Scope boundary: Proxy relative units. Physical amplitude calibration is outside scope.

---

### Geometric projection tensor (laminar approximation)

**Package default:** `mode="density_preserving"` — raw Gaussian weights (no row sum constraint):

$$K_{cn} = e^{-\frac{1}{2} \left( \frac{z_c - z_n}{\sigma} \right)^2}$$

**Opt-in:** `mode="row_normalize"` — row-stochastic kernel:

$$\mathbf{M}_{cn} = \frac{e^{-\frac{1}{2} \left( \frac{z_c - z_n}{\sigma} \right)^2}}{\sum_{m=1}^{N} e^{-\frac{1}{2} \left( \frac{z_c - z_m}{\sigma} \right)^2}}$$

Terms:
- $K_{cn}$ / $\mathbf{M}_{cn}$: contact $c$ to emitter $n$ weight
- $z_c$, $z_n$: relative laminar depth coordinates
- $\sigma$: Gaussian width parameter

Worded equation: default projection preserves source density (SUM-like); row-normalized mode is explicit opt-in and can flatten depth structure for off-population contacts.

Implementation location: `jaxfne.fields.proxy._row_normalize` and `jaxfne.fields.proxy.project_laminar_sources`

Scope boundary: Homogeneous isotropic infinite-medium proxy operator. Calibrated field-solver evidence is outside scope; this is a proxy operator design only.

---

### LFP-proxy linear projection

Formal equation:

$$\text{LFP}[t, c] = \sum_{n=1}^{N} K_{cn} \cdot S[t, n]$$

Terms:
- $\text{LFP}[t, c]$: LFP-proxy at contact $c$
- $K_{cn}$: Gaussian projection kernel (default `density_preserving`; optional `row_normalize`)
- $S[t, n]$: source proxy from emitter $n$

Worded equation: LFP-proxy is a weighted sum of source-proxy traces. Default kernel preserves density; row-normalized mode is opt-in.

Implementation location: `jaxfne.fields.project_laminar_sources`

Scope boundary: Laminar proxy only. PDE solution and physical calibration are outside scope.

---

### CSD-proxy (second spatial derivative)

Formal equation:

$$\text{CSD}[t, c] = -\frac{\partial^2 \Phi_e}{\partial z_c^2}$$

Terms:
- $\text{CSD}[t, c]$: current-source-density proxy at contact $c$
- $\Phi_e$: extracellular potential-proxy (LFP-proxy)
- $z_c$: laminar depth coordinate

Worded equation: CSD-proxy is the negative second spatial derivative of the potential-proxy along the contact axis, computed via finite differences.

Implementation location: `jaxfne.fields.project_laminar_sources`, lines ~130–133

Scope boundary: Proxy approximation only. Resistive-field PDE computation is outside scope.

---

### EEG/MEG linear proxy readout

Formal equation:

$$Y_{\text{EEG/MEG}}[t, c] = \sum_{n=1}^{N} L_c[n] \cdot S[t, n]$$

Terms:
- $Y_{\text{EEG/MEG}}$: EEG or MEG proxy trace
- $L_c$: declared EEG/MEG leadfield for channel $c$ (toy deterministic basis)
- $S[t, n]$: source proxy

Worded equation: EEG and MEG proxies are linear projections of source activity through declared (not empirically measured) leadfields.

Implementation location: `jaxfne.fields.eeg_proxy_transform`, `jaxfne.fields.meg_proxy_transform`, `jaxfne.vis.eeg`, `jaxfne.vis.meg`

Scope boundary: Declared geometry only. Relative proxy units; head geometry and real sensor placement are outside scope.

---

### EMM-proxy (signaling-energy summary)

Formal equation:

$$E_{\text{proxy}}[t] = \lambda_1 R[t] + \lambda_2 \|S[t]\|_1 + \lambda_3 \|\Phi_e[t]\|_2^2$$

Terms:
- $E_{\text{proxy}}$: relative signaling-energy proxy (normalized)
- $R[t]$: spike rate proxy
- $S[t]$: source-proxy vector
- $\Phi_e[t]$: potential-proxy vector
- $\lambda_1, \lambda_2, \lambda_3$: weighting factors

Worded equation: EMM-proxy is a normalized weighted combination of spike rate, source L1-norm, and field L2-norm.

Implementation location: `jaxfne.fields.emm_proxy_transform`, `jaxfne.vis.emm`

Scope boundary: Relative proxy units only. Biophysical metabolism is outside scope.

---

### Spectrolaminar band power summary

Formal equation:

$$P_{\text{band}}[c] = \frac{1}{|B|} \int_{f \in B} \text{PSD}[f, c] \, df$$

where $B \in \{\text{alpha/beta} = [8, 25]\text{ Hz}, \text{gamma} = [40, 150]\text{ Hz}\}$

Terms:
- $P_{\text{band}}[c]$: mean power in frequency band for contact $c$
- $\text{PSD}[f, c]$: power spectral density (Welch estimate)
- $B$: frequency band (alpha/beta or gamma)

Worded equation: Spectrolaminar profile is the mean PSD in each named frequency band, computed per contact.

Implementation location: `jaxfne.fields.spectrolaminar_bandpower`

Scope boundary: Relative power, proxy signal. Amplitude calibration is outside scope.

---

### V1/V4 feedforward-feedback routing metadata

Formal equation:

$$W_{\text{ff/fb}}[i, j] = \begin{cases}
> 0 & \text{if } \text{source} = V1, \text{target} = V4 \quad \text{(feedforward)} \\
> 0 & \text{if } \text{source} = V4, \text{target} = V1 \quad \text{(feedback)} \\
0 & \text{otherwise}
\end{cases}$$

Terms:
- $W_{\text{ff/fb}}$: connectivity matrix mask (declarative)
- source/target: area (V1 or V4) and layer assignment

Worded equation: V1-to-V4 feedforward and V4-to-V1 feedback connectivity are declared as separate sparse weight matrices.

Implementation location: `jaxfne.core.suite2_v1_v4_config`, connectiv ity metadata section

Scope boundary: Declared metadata only. Fitted weights and anatomical validation are outside scope.

---

## Figure Placeholders

The following figures are referenced in the notebook but not yet committed as PNG files. They will be generated by running the notebook:

- **Figure placeholder: spectrolaminar_suite_panel** — Six-panel figure (raster, LFP-proxy, CSD-proxy, PSD, EEG-proxy, EMM-proxy). Generated by `jtfne.vis.spectrolaminar_suite()`. Available until generated by tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb.

- **Figure placeholder: circuit3d_layout_scatter** — 3D scatter plot of V1-V4 emitter positions. Generated by `jtfne.vis.circuit3d()`. Available until generated by tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb.

These placeholders mark where final PNG or PDF figures should be placed once the notebook is executed and validated.

For a current, corrected spectrolaminar readout (depth-distribution crossings, absolute-power-vs-1/f check, slow-deep-homeostasis suite) generated on the canonical column, see the [Showcases guide](../guides/showcases.md#spectrolaminar-motif-with-depth-graded-slow-deep-homeostasis) instead.
