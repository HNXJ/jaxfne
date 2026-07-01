---
name: jaxfne-spectrolaminar-suite
description: Build and interpret the jaxfne laminar/spectrolaminar suite and the Etude No.3 (V1 1k) and TCM 6-population etudes. USE when a task involves a spectrolaminar suite, depth x frequency relative power, the deep-alpha/beta vs superficial-gamma crossover, multi-trial laminar simulation at scale (>=1k neurons), or the laminar LFP/CSD proxy readout. Encodes the verified scalable pipeline and the proxy/readout caveats so they are not re-derived.
---

# jaxfne Spectrolaminar Suite & Etudes

USE FIRST: `catalog-glossary-jaxfne` (API), `jaxfne-config` (canonical column), `jaxfne-vis-modules` (plots).

## Scalable pipeline (>=1k neurons) — use the CORE path, not the dense tutorial pipeline
`jaxfne.tutorial_utils` spectrolaminar pipeline builds FOUR dense NxN weight
matrices -> impractical above ~1k (≈790 s / 2 trials at 10k). For scale, use the
core path with the SPARSE edge backend:

```python
cfg = (jtfne.build_laminar_column(n=10000, ei_profile="canonical", within_gain=0.45)
       .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45, p_connect=0.1)
       .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5, recurrent_backend="edge_list")  # <- sparse
       .set_emitter("izhikevich","cortical_eig")
       .probes(["spikes","V_m","LFP","CSD","source"], n_contacts=32)
       .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
model = jtfne.construct(cfg)                                   # ~40 s @10k (build ONCE, reuse)
sig   = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=i)   # ~19 s/trial @10k sparse
```
For explicit cell-type x layer motifs, drop to the public kernel:
`izhikevich_params_from_labels(labels, layer_labels=...)` + signed dense W
(rows=post, cols=pre; negative weight = inhibitory) -> `make_edge_list_from_dense(W)`
-> `simulate_edge_recurrent_izhikevich(params, edges, n_steps, dt_ms, key, noise_scale=)`.
Memory: run trials in chunks; keep only `lfp_contacts`/`source`+`spikes`.

## The spectrolaminar readout — proxy caveats (do not get these wrong)
- **Signal = LFP proxy** (`signal_key="lfp_contacts"` / `sig.field.lfp_proxy`), NOT CSD
  (CSD is a 2nd spatial derivative that suppresses broad deep alpha/beta).
- **Projection mode (default fixed):** `project_laminar_sources` now defaults to
  **`mode="density_preserving"`** (SUM-like; dense superficial layers contribute more).
  **`mode="row_normalize"`** is explicit opt-in only — each contact's weights sum to 1,
  which erases neuron-density and can flatten depth power (or invert attenuation for
  off-population contacts). Do not assume row-normalize is still the default; see
  `skills/FRICTIONS_STACK.md` F-003.
- **Size (dipole):** weight each neuron's contribution by size (deep E pyramidal >>
  superficial >> interneurons); deep cells dominate magnitude (power ~ a*f^2).
- **kappa gate:** report `jtfne.kappa_synchrony` every run; the readout is only
  trustworthy at kappa≈0 (asynchronous-irregular). Balance per-layer rates with
  homeostatic per-layer drive to avoid one-layer hyperactivity.

## The deep-alpha/beta vs superficial-gamma CROSSOVER (hard-won result)
The crossover is a REGIME property, NOT a connectivity-weight property. Verified
NEGATIVES (10k, 10 trials, kappa≈0.02): uniform-random connectivity, E-E sub-block
topology sweeps (supE/deepE x off/50/100), per-neuron size+rate gradients — NONE
produce it. In an asynchronous-irregular regime the LFP is BROADBAND at every depth,
so there is nothing to cross. The crossover requires band-limited OSCILLATIONS
localized by layer: a synchronous superficial gamma-PING (PV<->E fast loop) and a
resonant deep alpha/beta E-I loop, while GLOBAL kappa stays low. Magnitude balance
(sup vs deep power) is set by density + dipole size, not by superficial E<->PV scaling
(which caps ~0.5 and over-inhibits/destabilizes at high gain).

## Etudes
- `tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb` — V1 1k spectrolaminar; artifacts in gitignored `local/etude3/` (run notebook first; `test_etude3_v1_spectrolaminar_1k.py` skips artifact checks if absent).
- `tutorials/etudes/jaxfne_etude_tcm_v1_6pop.ipynb` — 6-population TCM column (`test_tcm_v1_6pop.py`).
- Docs: `docs/tutorials/07_jaxfne_suite_no_2_spectrolaminar_motif.md`.

Full project memory: `spectrolaminar-crossover-requires-oscillations`, `jaxfne-laminar-sim-recipe`, `jaxfne-perf-10k`.
