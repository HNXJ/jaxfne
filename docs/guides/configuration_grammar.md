# Configuration Grammar

`Configuration` is the primary abstraction layer in jaxfne. Everything else —
emitters, source tensors, field proxies, probes, objectives, optimizers, and
manifests — is *compiled from it*. A `Configuration` is not a bag of settings: it
is a **declarative specification** of a model that `construct()` and `simulate()`
turn into a runnable TFNE graph.

```text
        Configuration  (declarative specification — the dial you turn)
              │
         construct()       ← the compiler
              ▼
   Emitter → Source → Field → Probe → Objective → Optimizer → Manifest
   (the TFNE operator chain — emergent from the specification)
```

This is the central idea of the package: **jaxfne is the mathematical backend,
and the configuration is the biophysical specification.** How biophysical the
output is — calibrated amplitudes, real morphology, channel detail — is
determined by how much detail you put into the `Configuration` (and any
[Jaxley](jaxley_interop.md) models you bridge in), not by a fixed ceiling in the
backend. Every method below adds biophysical specificity to the same chain.

Each method returns a `Configuration`, so the whole model reads as one fluent
sequence:

```python
import jaxfne as jtfne

cfg = (
    jtfne.Configuration()
      .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5)
      .column(name="V1", layers=["L1", "L2/3", "L4", "L5", "L6"], n=1000)
      .geometry(layer_thickness={"L1": 0.1, "L2/3": 0.3, "L4": 0.2, "L5": 0.3, "L6": 0.3})
      .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
      .connectivity(mode="sparse")
      .set_emitter("izhikevich", "cortical_eig")
      .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
      .field(domain="laminar_column", conductivity="proxy")
)
model = jtfne.construct(cfg)
```

The sections below follow the operator chain top to bottom. Each says **what it
specifies** and **how it raises biophysical specificity**.

---

## Runtime

`.runtime(...)` / `.set_runtime(...)` — execution substrate: `seed`, `dtype`,
`backend`, `jit`, `vmap`, `duration_ms`, `dt_ms`.

Specificity dial: a stable, explicit runtime is the precondition for everything
else. Enable x64 (`jtfne.enable_x64()`) **before** building arrays, keep `dt_ms`
small enough for the dynamics you specify, and fix `seed` for reproducibility.
The runtime does not change the biology — it determines whether the biology you
specified is computed faithfully.

## Geometry

`.geometry(layer_thickness=, layer_cell_types=)`, `.layer_fractions(...)`,
`.column(name, layers, n)` / `.add_column(...)`, `.areas([...])`,
`.uniform3d(radius_mm=, height_mm=)`.

Specificity dial: geometry is where laminar structure enters. `.column()` names
the layers and total count; `.geometry(layer_thickness=...)` turns thicknesses
into cumulative depth bands; `.layer_fractions(...)` makes per-layer neuron counts
proportional to thickness (or an explicit fraction). Real depth bands are what
let LFP/CSD and spectrolaminar readouts express depth structure — `.uniform3d()`
placement collapses layer identity and should be used only for non-laminar models.

## Cell types

`.cell_types({...})` / `.set_cell_types(...)`, `.area_layer_cell_types(area, {...})`,
`.cell_params(selector, params)`.

Specificity dial: the E:I composition and its laminar gradient. A single global
`.cell_types({"E": 0.8, ...})` is the coarse setting; `.area_layer_cell_types(...)`
expresses the verified ground-truth gradient (E rises with depth, inhibition
peaks superficially, PV concentrates at L4). `.cell_params(...)` overrides
per-selector neuron parameters. The more layer- and type-resolved the
composition, the more the model can reproduce real laminar physiology.

!!! tip "Canonical prior"
    `jtfne.build_laminar_column(n=1000, ei_profile="canonical")` applies the
    verified per-layer E:I gradient and laminar placement for you — a good
    starting point you then refine with the methods above.

## Connectivity

`.connectivity(**)` / `.set_connectivity(**)`, `.connections(name=, source=, target=,
probability=, weight=, sign=, mechanism=, plasticity=)`,
`.inter_column_connectivity(source_area=, target_area=, layer_to_layer_map=, ...)`,
`.mechanisms(name=, kind=, params=)`.

Specificity dial: the circuit. `.connections(...)` declares explicit
source→target rules with probability, weight, sign, and synaptic mechanism;
`.inter_column_connectivity(...)` adds laminar-aware inter-area edges
(feedforward L2/3→L4, feedback L6→L1/L5). Prefer sparse construction at scale.
Richer, sign- and mechanism-resolved connectivity is what produces emergent
oscillations and the band-localized structure spectrolaminar readouts depend on.

**Mechanism resolution now drives simulated tau when fully declared.**
`construct()` compiles `.connections()` rules through one of two compilers: a
sign-only fallback (receptor inferred from weight sign, tau hardcoded
exc=2 ms/inh=5 ms — the only path that ever ran before this switch) or a
mechanism-aware path that resolves each rule's `mechanism=` against a
declared `.mechanisms(name=, kind=, tau_ms=, sign=)` entry for real
per-edge tau. The mechanism-aware path is selected **only when every**
declared connection rule has a resolvable mechanism reference; a model with
no mechanisms, or a mixed rule set where even one rule omits `mechanism=`,
runs entirely on the unchanged sign-only path. See
`tests/test_mechanism_aware_connection_compiler.py` for the parity and
divergence proof.

## Emitters

`.set_emitter(family="izhikevich", preset="cortical_eig")` / `.emitter(**)`,
`.drive(baseline_drive_by_cell_type=, drive_by_layer=, drive_by_area=,
time_schedule=, evoked_windows=, noise_policy=, ...)`,
`.cell_type_drives({...})`.

Specificity dial: the neuron model and its input. The built-in Izhikevich emitter
is tunable and float32-stable; `.drive(...)` sets baseline/laminar/evoked input
and noise. For real channel biophysics and morphology, bridge a **Jaxley** model
in as the emitter (see [Jaxley Interoperability](jaxley_interop.md)) — a Jaxley
HH network exports real transmembrane ionic current, the physical generator of
the extracellular field.

## Sources

The source tensor is *emergent* — `construct()`/`simulate()` build it from the
emitter output according to the emitter family and field settings (built-in
spike/current proxy, or reconstructed HH ionic current via the Jaxley bridge).

Specificity dial: a voltage/spike proxy source is coarse; a real transmembrane
**current** source (Jaxley HH) is the physically meaningful generator. Choosing
the emitter therefore chooses the source's fidelity. See
`JaxleyBridge.simulate_laminar_field` in the [Bridges API](../api/bridges.md).

## Fields

`.field(domain=, conductivity=, boundary=, **)`.

Specificity dial: how the source becomes an extracellular field readout. The
laminar proxy (Gaussian projection + finite-difference CSD) is a structural
`linear_solver` readout; its outputs carry the `*_proxy` suffix and remain
uncalibrated unless you supply calibration. Density-preserving projection
preserves laminar depth structure (row-normalization erases it). Finer geometry +
calibration is the path from proxy to physical amplitude.

## Probes

`.probes([...])` / `.set_probes(modes, n_contacts=)` / `.probe(**)`.

Specificity dial: what you measure and at what resolution — `spikes`, `V_m`,
`LFP`, `CSD`, `EEG`, `MEG`, spectrolaminar, with contact geometry (`n_contacts`).
More contacts and more modalities expose more of the model's structure; probes do
not change the model, they read it.

## Objectives

`.objective(firing_rate_target=, spectrolaminar_profile_target=, band_definitions=,
synchrony_metrics=, null_controls=, ablations=, rejection_gates=)`,
`.objective_outputs(name=, dtype=, shape=)`.

Specificity dial: what "correct" means. Targets (rate, synchrony, spectrolaminar
profile), and — importantly — the **null controls and ablations** that make a
result interpretable rather than a fit. A specified null/ablation turns a metric
into evidence.

## Optimizers

`.optimizer(optimizer_family="AGSDR", differentiability_status=, surrogate_status=,
search_space=, budget=, seed=, hard_gates=)`, `.trainables(name=, path=, selector=,
bounds=)`.

Specificity dial: how parameters are searched toward the objective. Declare the
trainable parameters (`.trainables(...)`), the search space and budget, and the
differentiability/surrogate status. Hard spike reset is non-differentiable, so the
surrogate status governs whether gradient-based tuning is admissible.

## Manifests

`jtfne.manifest(cfg, signals=...)`, `.validate()`, `.update_metadata(**)`.

Specificity dial: the receipt. The manifest binds the configuration, runtime
report, artifact hashes, and truth gates into a strict JSON-safe record.
`.validate()` checks the specification before you run. The manifest is what makes
a result reproducible and auditable — the closing operator of the chain.

---

## Why this matters

Reading `Configuration` as a compiler reframes the whole package: the object
grammar (the fluent `cfg` chain) and the TFNE operator grammar
(Emitter→Source→Field→Probe→Objective→Optimizer→Manifest) are two views of the
same system. You specify a model declaratively; jaxfne compiles and computes it.
The fidelity of the result is a property of the specification you wrote — which is
why `Configuration` is the deepest, most important surface in the package.

## See also

- [Tensor-Field Workflows](tensor_field_workflows.md) — the operator chain in depth.
- [Jaxley Interoperability](jaxley_interop.md) — real channels/morphology as emitters.
- [Bridges API](../api/bridges.md) · [Fields API](../api/fields.md) · [Objectives API](../api/objectives.md)
