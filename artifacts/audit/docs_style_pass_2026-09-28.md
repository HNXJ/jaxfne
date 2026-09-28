# Docs style pass, 2026-09-28: review list

Pass: `match-writing-style` over `docs/` and README (human decision
2026-09-28: cut padding only; soften overclaims and list them; no
Version/Status headers on public pages). Commits b278f70 (README), eddfa1b
(pass A), 42e0e2f (pass B). Line numbers refer to the page before the pass.
Two cuts in pass B were reverted before commit: the `jdna.md` genetics
disclaimer and the G/D/H_D/Z_D role paragraph, and the protocol identifier
in `etudes/experiment_a.md`.

## 1. Softened overclaims (author review)

| File:line | Before | After | Why |
|---|---|---|---|
| faq.md:15 | CPU-first examples validate correctness | Examples are CPU-first | examples demonstrate; they do not validate |
| tutorial_figures.md:80 | Laminar field solution from source | Laminar field proxy from source | no PDE solve on the shipped path |
| api/bridges.md:85 | from a physically meaningful generator | reconstructed from channel currents; readouts stay proxy | "physically meaningful" is ungraded |
| api/emitters.md:11 | balances cost and realism | trading biophysical detail for speed | "realism" ungraded for a reduced model |
| api/tensor_operators.md:146 | the composability proof | the composability evidence | a passing sweep is evidence |
| notes/brian2_benchmark_receipt.md:3 | first real, quantitative … small, real, honest smoke comparison | First quantitative … small smoke comparison | credibility adjectives removed |
| tensor_network_ancestry.md:145 | Validate status checks | Status checks | a bullet does not validate |
| guides/configuration_grammar.md:88 | reproduce real laminar physiology | resolve more laminar structure | proxy outputs |
| guides/configuration_grammar.md:106 | produces emergent oscillations | can produce emergent oscillations | not shown for every case |
| guides/configuration_grammar.md:120 | parity and divergence proof | parity and divergence checks | tests check cases |
| guides/homeostasis.md:8, :9 | Eliminates hyper-/hypoactivity | Counters hyper-/hypoactivity | bounds activity; does not remove it |
| guides/homeostasis.md:99 | finite, stable regime | finite regime | finiteness is tested; stability is not |
| guides/homeostasis.md:113 | numerical stabilizer | keeps the numerics finite | bounded is not stable |
| guides/operator_composition.md:104, :167 | composability / executable proof | … check | single-run allclose |
| guides/poisson_admissibility.md:16 | numerically accurate, and physically consistent | passes the five numerical gates; not a physical-consistency claim | gates check the contract |
| guides/poisson_admissibility.md:162 | Ensures solution is accurate | Checks the solution against the tolerance | a gate checks |
| guides/jax_fem_interop.md:16 | proving that … is feasible | showing that … | existence is not proof |
| tutorials/08_v038_lfp_csd_readout.md:189 | due to larger somatic currents | from the depth-weighted source gain in the proxy readout | point neurons have no soma |
| tutorials/10_v0313_omission_oddball.md:110 | extracellular-like profiles | proxy profiles | resemblance not shown |
| tutorials/06_v036_100_neuron_ei_population.md:147 | Calibrate to real data | Prepare for calibration | the guide is proxy-only |
| protocols/protocol_d_biological_rbs.md:63 | First biological realization | Realized in D1 | priority claim unsupported |

## 2. Flagged, not edited (author decision)

| File:line | Issue |
|---|---|
| guides/objective_grammar.md:44 | "tens of seconds at 10k neurons": no traceable artifact |
| tutorials/13_canonical_column_etude.md:266; tutorials/04_simulate_tensor.md:5 | "~2 s at 1k, ~40 s at 10k": no traceable artifact |
| tutorials/05_v1_pfc_dual_column.md:139 | "~26 s on CPU": no traceable artifact |
| tutorials/06_v036_100_neuron_ei_population.md:119 | "~2–3 minutes": no traceable artifact |
| tutorials/03_network_100_ei.md:100 | "run efficiently on CPU with JAX vmap": no artifact |
| guides/plotly_visualization.md:205, 210, 214, 314 | file sizes disagree (~10 KB, 10–100 KB, 10–200 KB, ~100 KB) |
| guides/probe_operators.md:197, 223, 335 | "in v0.2.x" beside a v0.4.8 header |
| guides/tensor_field_workflows.md:25 | "(not PDE solvers in v0.3.x)": stale version |
| tutorials/08_v038_lfp_csd_readout.md:3 | "Version: 0.3.8" footer |
| guides/zenodo_doi.md:40 | "planned for 0.4.7": stale plan |
| protocols/protocol_c_wave.md:86 | "JAX 0.11 compatibility": stale version |
| api/sharding.md:22 | "planned for v0.3.20+" beside "single-device as of 0.4.17" |
| guides/atlas_suite.md:17 | "Microsecond spike events" with ms-scale dt |
| tutorials/07_v037_source_bookkeeping.md:62, 200 | µm depth labels vs relative-fraction geometry |
| BASELINE_DRIVE_REFERENCE.md:130 | "Unblocked (pending final validation)" |
| colab.md, interactive_visualizations.md | date headers removed with the status headers; no date remains on the page |
| gallery.md | generated page, skipped |
