# Showcases

Runnable demonstrations of jaxfne's structural, homeostatic, and plasticity
knobs. Every figure on this page comes from a real `construct()`/`simulate()`
run with the exact parameters quoted in its caption — nothing here is a mockup.

All outputs stay on the package's conservative truth gates: `claim_level=
computational_scaffold`, `field_solver_status=linear_solver`,
`physical_amplitude_calibrated=False`. Nothing on this page is a biological
validation or a calibrated physical measurement — it is a computational
diagnostic of the proxy/scaffold pipeline.

## Interactive 3D network: V1 → V4 → PFC hierarchy

A three-area canonical-column hierarchy (300 neurons per area, ground-truth
E:I gradient, bidirectional feedforward/feedback inter-area connectivity)
rendered with `jtfne.vis.visualize_network_3d`. Drag to rotate, scroll to
zoom, hover a node for its area/layer/cell-type metadata.

```python
cfg = jtfne.build_multi_area_columns(["V1", "V4", "PFC"], n_per_area=300, ei_profile="canonical")
cfg = cfg.set_emitter("izhikevich", "cortical_eig").probes(["spikes", "V_m"], n_contacts=16)
model = jtfne.construct(cfg)

jtfne.vis.visualize_network_3d(
    model,
    title="V1-V4-PFC multi-area canonical columns (300 neurons/area)",
    show_layers=True, show_column_shells=True, show_edges=True, max_edges=600,
    output_html="network3d.html",
)
```

<iframe
  src="../assets/showcases/v1_v4_pfc_network3d.html"
  width="100%"
  height="700px"
  style="border: 1px solid #2e2e3a; border-radius: 8px; background: #0d0d10;"
  title="V1-V4-PFC multi-area network 3D"
  loading="lazy"
></iframe>

## Homeostasis: firing-rate change and full 10 s raster

A 200-neuron canonical V1 column (`jtfne.build_laminar_column("V1", n=200,
ei_profile="canonical")`), simulated for the full 10,000 ms at `dt_ms=0.5`,
comparing homeostasis off vs `.homeostasis(relative_baseline=1.0, r_star=10.0,
k_gain=1.0)`.

![Firing-rate change with homeostasis](../assets/showcases/homeostasis_rate_change_10s.png)

With homeostasis off the population settles to its natural ≈10 Hz
asynchronous-irregular attractor (κ synchrony ≈ 0.016). Turning the kernel on
does **not** converge the rate onto `r*=10 Hz` — it kicks the population into
a higher, `r*`-scaled regime (≈44 Hz here) within the first few hundred ms and
holds it there for the rest of the 10 s run, with synchrony staying low
(κ ≈ 0.0038). This is the documented one-sided-damper behavior of the kernel:
`g = clip(k_gain*(r_star - r), g_min, g_max)` readily adds excitatory bias
while the rate estimate `r` is below `r*` (always true early in a run, since
`r` starts at 0), but cannot symmetrically suppress activity back down once
the network settles into the resulting higher-rate regime. Lowering `r*`
below the natural baseline does shift the elevated regime down
monotonically (e.g. `r*=5` → ≈28 Hz, `r*=3` → ≈20 Hz) — the controller's
effect is real and direction-correct, just not exact setpoint tracking.

![Full 10 s raster, homeostasis on](../assets/showcases/homeostasis_full_raster_10s.png)

Vm stays sane throughout both runs (rest ≈ −84…−88 mV, spike peak ≈ +30 mV) —
no float blow-up, no NaN.

## Plasticity: closed-loop STDP under purely random stimulation

`Configuration.plasticity()` is declaration-only — it records intent in the
manifest but `Model.simulate()` does not consume it. The real, wired
synaptic-plasticity kernel runs through the separate streaming entry point,
`jtfne.run_stdp_stream`, which updates the weight matrix `W` every timestep
and feeds it back into the dynamics (genuine closed-loop online STDP).

A 100-neuron E/I cloud network (`jtfne.make_ei_cloud_network(100, seed=42)`,
70 E / 30 I) driven by **pure Gaussian noise** (no structured stimulus —
amplitude calibrated to a rate-compliant ≈9.7 Hz baseline) for 10 s, with
`plasticity_scale=0.1` — the scale [verified stable](../STDP_CLOSED_LOOP_REPORT.md)
against runaway (`plasticity_scale>=0.5` is not):

```python
stdp_state = jtfne.STDPState(W=W0, trace_pre=jnp.zeros(n), trace_post=jnp.zeros(n))
plasticity_config = jtfne.STDPPlasticityConfig(A_plus=0.01, A_minus=0.012)
(v, u, s, final_state), traj = jtfne.run_stdp_stream(
    v_init=v0, u_init=u0, s_init=s0, stdp_state=stdp_state,
    stim_drive=jnp.zeros((n_steps, n)), noise=random_noise, solver_config=solver_config,
    plasticity_config=plasticity_config, plasticity_scale=0.1,
    exc_mask=exc_mask, inh_mask=inh_mask, a=a, b=b, c=c, d=d,
    chunk_size_ms=1000.0, downsample_factor=1,
)
```

![Activity stability and weight drift under random stimulation](../assets/showcases/plasticity_random_stim_stability.png)

Firing rate stays flat across all ten 1 s chunks (no runaway), while the mean
synaptic weight drifts measurably over the same window — purely from STDP
acting on uncorrelated, random drive.

![Excitatory weight distribution before vs after](../assets/showcases/plasticity_weight_distribution.png)

The excitatory weight distribution shifts down over the 10 s run (mean
0.0547 → 0.0510): with no temporal structure in the stimulus there is little
causal pre-before-post pairing, so net synaptic depression dominates — a
real, measured STDP effect, not a no-op.

## Spectrolaminar motif with depth-graded ("slow-deep") homeostasis

The deep-α/β vs superficial-γ spectrolaminar crossover is a scale-emergent
regime property (`tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb`),
not a connectivity-weight effect — see the spectrolaminar-suite findings for
the full characterization. This run reproduces the canonical 10,000-neuron V1
column at the scale where the crossover has previously been observed, and
layers in a depth-graded homeostatic profile: deep layers (L5/L6) get a
**slower** rate-integration time constant and a **wider** excitability-bias
range than superficial layers, i.e. more homeostatic "capacity" and slower
reaction in deep cortex:

```python
cfg = jtfne.build_laminar_column("V1", n=10000, ei_profile="canonical")
# tau_r_ms, g_min, g_max graded per neuron by layer (deep = slow + wide capacity):
cfg = cfg.homeostasis(relative_baseline=1.0, k_gain=1.0, r_star=10.0,
                       tau_r_ms=tau_r_per_neuron,   # L5/L6: 1500 ms · others: 300 ms
                       g_min=g_min_per_neuron,      # L5/L6: -20  · others: -12
                       g_max=g_max_per_neuron)      # L5/L6:  14  · others:   8
model = jtfne.construct(cfg)
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

**What this run actually shows.** 10,000 neurons, 1000 ms, dt=0.5 ms: Vm
stays sane (rest ≈ −87 mV, peak ≈ +30 mV), global synchrony stays low
(κ ≈ 0.020, asynchronous-irregular), and overall rate is ≈36 Hz (elevated by
the same homeostasis-kick effect documented above).

![Spectrolaminar suite, slow-deep homeostasis](../assets/showcases/spectrolaminar_slow_homeostasis_suite.png)

**A methodology correction.** An earlier version of this page normalized each
*depth contact's* spectrum across frequency and compared α/β vs γ within that
normalization — which mostly measures the generic 1/f spectral shape, not
laminar structure, and wrongly read as "no crossover anywhere." The right
test treats each band's own power-by-depth as a distribution over depth
(normalized to sum to 1 across depth, separately per band) and asks where
*that* distribution sits relative to the other band's. Two distinct
distributions over the same domain with equal total mass cannot dominate
each other everywhere — they must cross at least once if they're not
identical. Re-run with that test, the 10k single-trial data shows **2 real
crossings** (depth 0.065 and 0.710): α/β leads at the very-superficial and
deep extremes, γ leads in between.

![Depth-distribution crossings, 10k vs 100-neuron](../assets/showcases/spectrolaminar_depth_distribution_crossings.png)

**A cleaner, trial-averaged check.** Single-trial spectral estimates are
noisy, so a second, simpler run isolates the question: 100-neuron canonical
V1 column, 32 LFP-proxy contacts with the top 2 and bottom 2 dropped before
analysis (boundary artifacts from the projection kernel), homeostasis on but
gentle (`k_gain=0.1`, chosen so rate stays contained — 14.3±0.06 Hz across 20
seeds vs. a 10.2 Hz no-homeostasis baseline, κ=0.0114±0.0003), averaged over
20 trials/seeds:

```python
cfg = (jtfne.build_laminar_column("V1", n=100, ei_profile="canonical")
       .homeostasis(relative_baseline=1.0, r_star=10.0, k_gain=0.1)
       .set_emitter("izhikevich", "cortical_eig")
       .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=32)
       .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
model = jtfne.construct(cfg)
lfp_trials = jnp.stack([jtfne.simulate(model, duration_ms=2000.0, dt_ms=0.5, seed=s)
                        .field.lfp_proxy for s in range(20)])
psd = jtfne.spectrolaminar_psd_jax(lfp_trials[:, :, 2:-2], fs=2000.0)  # drop edge contacts
```

This gives **3 real, reproducible crossings** (depth 0.401, 0.771, 0.932) —
not noise: the rate/κ spread across seeds is tiny (±0.4%/±2.6%), and one
crossing (depth ≈0.77) lands close to the single-crossing region the 10k
single-trial run shows at depth ≈0.71, despite the very different scale and
contact count. Both bands' depth-distributions still peak at the *same*
absolute depth (≈0.81, deep) — expected, since the canonical column's
deep-weighted dipole-size/density model (`_apply_canonical_biophysics`)
boosts every frequency equally at depth; the crossings are a second-order
*relative* effect riding on top of that shared envelope, alternating rather
than a single monotonic flip.

**The full 3-panel suite, in the standard layout.** Cell density by type
(left), the depth × frequency relative-power heatmap (center), and the
alpha-beta/gamma depth-distribution crossing (right, corrected methodology),
all referenced to cortical position relative to L4 (negative = superficial,
positive = deep) — 100-neuron canonical V1 column, homeostasis on
(`r_star=10, k_gain=0.1`), 32 contacts with the 2 edge contacts dropped each
side, 20-trial average:

![Spectrolaminar suite, 3-panel, corrected crossing methodology](../assets/showcases/spectrolaminar_suite_corrected.png)

The cell-density panel reproduces the canonical column's E:I gradient (E
density rising toward deep layers, I subtypes — PV/SST/VIP — concentrated
superficial) directly from `model.neuron_table()`, independent of the
spectral panels. The crossing panel shows the alpha/beta and gamma
depth-distributions tracking each other closely (97% similarity by total
overlap) while still crossing three times — consistent with the "occasionally
uncrossed, not globally dominant" picture rather than a clean superficial/deep
split.

**A second correction: the relative crossings are real, but there is no
absolute alpha/beta bump.** The crossing test above is entirely about
*relative* power — each band's own depth-distribution, separately
normalized to sum to 1. It says nothing about whether deep cortex actually
generates *more total power* in the alpha/beta band, or whether superficial
cortex actively *suppresses* it. Both are real, distinct mechanistic claims
in the literature (deep band-pass filter/resonator vs. superficial
absorption/notch), and the right way to tell them apart is to compare
**absolute** power, not relative power, against the **1/f background**
every layer shares:

![Absolute superficial vs deep power spectra, 1/f background check](../assets/showcases/spectrolaminar_absolute_power_1f_check.png)

Fitting each group's own background trend (log-log slope, alpha/beta
excluded from the fit) shows superficial and deep have **the same spectral
shape** — slopes differ by only 0.02 (100-neuron run) and 0.07 (10k run), an
order of magnitude smaller than the slopes themselves (≈−0.8 and ≈−0.6).
Deep sits uniformly above superficial by a roughly frequency-independent
gain factor (mean ratio 1.56× and 1.39×, std 0.17 and 0.24 — flat across the
whole spectrum, not peaked in-band). And critically: in **both** groups, the
alpha/beta band sits **at or below** that group's own extrapolated 1/f
trend (residuals −0.15 to −0.22 log10-units, i.e. 60-70% of trend, in both
superficial and deep) — not above it. There is no absolute deep bump and no
absolute superficial notch isolated to 10-25 Hz in either run.

Read against the decision tree this finding is checked against: deep is
**not** a band-pass filter or resonator (no band-selective gain, no
narrowband departure from the shared 1/f trend), and superficial is **not**
isolating an absorption notch either (it dips by about the same amount as
deep does, not more). What the model currently produces is closer to "flat
depth-dependent gain on top of an ordinary 1/f spectrum, same shape at every
depth" — exactly what's expected from an asynchronous-irregular (broadband,
κ≈0) regime with a depth-weighted dipole-size/density readout, and exactly
why the project's standing finding is that real band-limited laminar
structure needs deliberately-engineered layer-localized oscillations (a fast
superficial PV↔E gamma-PING loop, a slower resonant deep E-I loop), not
parameter grading alone. The small relative crossings documented above are
real in the strict mathematical sense, but they are second-order ripples
riding on a spectrum with no genuine absolute band-selectivity — treat them
as exactly that, not as evidence of a deep resonator or superficial filter.

**Honest summary.** Crossings are real and reproducible — the earlier "no
crossover" claim was a methodology error, not a finding about the model. But
the absolute-power test above shows those crossings are not backed by any
genuine band-selective mechanism: no absolute deep bump, no absolute
superficial notch, same 1/f shape at every depth. What's **not** yet
reproduced is the literature's specific textbook pattern (a single flip,
γ-dominant superficial / α/β-dominant deep, backed by a real resonance or
filter): both runs show an *alternating*, multi-crossing structure riding on
a flat-gain broadband background instead. The deliberately-engineered
oscillatory loops this project's prior work says the clean dichotomy needs
remain the open, unimplemented piece — homeostatic/connectivity-weight
grading clearly does *something* real to the relative depth structure, but
not (yet) anything with absolute spectral selectivity.

[STDP_CLOSED_LOOP_REPORT](../STDP_CLOSED_LOOP_REPORT.md) ·
[Homeostasis guide](homeostasis.md) ·
[Configuration Grammar](configuration_grammar.md)
