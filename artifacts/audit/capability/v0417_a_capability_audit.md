# 0.4.17-A Capability Audit — public summary

**Status:** FROZEN (read-only)  
**Baseline:** `jaxfne==0.4.16` @ `15f32b3`  
**Frozen receipt:** `artifacts/audit/v0417_a_capability_audit.json`  
**Private matrix:** `scratch/figure_requirements_matrix_v2_0417.md` (gitignored)

## Question

What can published jaxfne 0.4.16 actually demonstrate today?

## Panel readiness map (counts)

| Figure | READY | ANALYSIS_ONLY | NEEDS_EXT | NEEDS_SCIENCE | BLOCKED |
|--------|------:|--------------:|----------:|--------------:|--------:|
| 1 TFNE language | 5 | 1 | 0 | 0 | 0 |
| 2 Emitter→source | 4 | 3 | 0 | 1 | 0 |
| 3 LFP/CSD | 5 | 2 | 1 | 1 | 1 |
| 4 Multiscale obs | 3 | 3 | 0 | 2 | 1 |
| 5 Oscillations/waves | 3 | 2 | 1 | 3 | 0 |
| 6 Adaptive biophysics | 6 | 2 | 0 | 0 | 2 |
| 7 Hierarchical TFNE | 2 | 2 | 1 | 2 | 0 |
| **Total (59 panels)** | **28** | **15** | **3** | **9** | **4** |

## Strict audits (user-requested)

### Figure 4 — EEG/MEG operator status

All EEG/MEG/EMM surfaces are **ANALYSIS_ONLY** at `proxy_readout` / `computational_scaffold`.  
`Y = Q @ L^T` (MEG identical kernel; `orientation_claim: none`).  
`simulate()` does **not** auto-produce EEG/MEG (`signals.get` fails by design).  
**No panel is READY for calibrated macroscopic EEG/MEG claims.**

### Figure 5 — delay vs wave vs feedback

| Capability | Status |
|------------|--------|
| Protocol D forward delays | **READY** (emitter kernel) |
| Delay continuation in `Model.simulate` | **NEEDS_GENERAL_EXTENSION** |
| Spectrolaminar PSD readouts | **READY** (proxy analysis) |
| Wave/phase/traveling-wave estimator | **NEEDS_NEW_SCIENCE** |
| Field→neuron / ephaptic | **NEEDS_NEW_SCIENCE** (reserved) |
| H4 memory-extension via topology/delay | **negative** (Protocol H; do not conflate with D) |

### Figure 6 — frozen evidence ladder

| Step | Status |
|------|--------|
| RBS / H→X | demonstrated |
| H4 topology/delay extension | **negative** |
| H→ω, ω→W→X | demonstrated |
| Closed HDP loop | **unresolved** (W3b) |
| W3 closed-loop memory | **BLOCKED** |
| New F_W required | **BLOCKED** (prohibited inference) |

### Figure 7 — capability decomposition

Treat as **capability checklist**, not permission for a large integrated model:

- Config + NeuronalTensor compile: **READY**
- Inter-area runtime receipt: **NEEDS_GENERAL_EXTENSION**
- Integrated Experiment B: **NEEDS_NEW_SCIENCE** (respects W3b blocks)

## Minimal 0.4.17 capability delta (ranked)

Set difference between publication requirements and existing general capabilities:

1. **0.4.17-B** — Canonical Multiscale Observation Etude packaging (Figs 2–4)
2. **0.4.17-B/C** — Independent probe `P` at frozen `(Q,𝒢,ℳ)`; fix `lfp_proxy_probe` contact depths
3. **0.4.17-C** — Wave estimator protocol + no-wave falsification (not Protocol D)
4. **0.4.17-C** — Unified `delay_state` continuation in runtime
5. **0.4.17-D** — One biological RBS phenotype under frozen protocol
6. **0.4.17-E** — Multi-area integration receipt
7. **0.4.17-E** — Experiment B (RBD path; closed-loop HDP memory blocked)

**Explicitly excluded from minimal delta:** calibrated EEG/MEG, cross-emitter physical Q equivalence, ephaptic feedback, W3/W3c closed-loop work.

## Operational debts (not scientific architecture)

| ID | Class |
|----|-------|
| `jax_lt_0_11_pin` | compatibility_debt |
| `upload_pypi_sh_stale_version` | release_tooling_debt |

## Next step

**0.4.17-B** — Multiscale Observation Etude (no implementation during 0.4.17-A).
