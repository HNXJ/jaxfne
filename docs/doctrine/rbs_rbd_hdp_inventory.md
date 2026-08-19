# RBS/RBD/HDP semantic inventory (read-only)

**Checkout:** `dev` @ `724aa32` (post D₀/D₁)  
**Inventory date:** 2026-08-13  
**Migration executed:** 2026-08-13 (semantic refactor commit pending)  
**Status:** inventory record — see `docs/doctrine/rbs_rbd_hdp.md` for current doctrine

Target doctrine: [`rbs_rbd_hdp.md`](rbs_rbd_hdp.md)  
Upstream authority (pending revision): `artifacts/project_sources/*.md`

---

## Executive contradictions (doctrine vs live)

| Location | Current claim | Target doctrine | Class |
|----------|---------------|-----------------|-------|
| `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` §8 title | "HDP: homeostasis-dependent plasticity" | Hidden-state Dependent Plasticity | **A** (upstream) |
| Same §8 opening | "slow **homeostatic state**" defines H | RBS; homeostasis is regime not definition | **A** |
| Same §8 nulls | "homeostatic plasticity-term null", "homeostatic-state null" | Rename null *descriptions* to RBS/HDP; keep math | **A** |
| `docs/guides/hdp.md` title/body | "H-state and HDP (Homeostasis-Dependent Plasticity)" | RBS + HDP (hidden-state dependent) | **A** |
| `jaxfne/emitters.py` `simulate_edge_recurrent_izhikevich_hdp` docstring | "Homeostasis-Dependent Plasticity" | Hidden-state Dependent Plasticity | **A** |
| `jaxfne/_config.py` `Configuration.hdp()` | "HDP (Homeostasis-Dependent Plasticity)" | **A** (docstring) + **D** (method name) |
| `jaxfne/emitters_homeostatic_ei.py` | homeostatic EI triple dynamics | **B** — genuine homeostatic kernel | **B** |
| `simulate_edge_recurrent_izhikevich_homeostatic` | per-neuron homeostatic trace `r_i`, `g_i` | **B** | **B** |
| `docs/guides/homeostasis.md` | legacy homeostasis guide | **B** + cross-link to RBD | **B/A** |
| `public_surface.py` comment | "H-state is the latent representation; not identical to HDP theory" | Already partially aligned; refine to RBS | **A** |
| Project sources README | "v0.4.8" bundle snapshot | Version drift vs live `v0.4.15` — **do not treat inventory as current implementation** | **C** note |

**No implementation contradiction** found that blocks the doctrine (e.g. HDP kernel math matches generalized \(F_H, F_W\)); conflict is **terminological**, not numerical.

---

## Classification key

| Code | Meaning |
|------|---------|
| **A** | Generic doctrine → migrate to RBS/RBD/HDP language |
| **B** | Kernel-specific homeostasis → **keep** homeostatic where mechanism is homeostatic |
| **C** | Historical / changelog / receipts → preserve or archive |
| **D** | Public API identifier → compatibility surface; document migration only if justified |

---

## A — Generic doctrine (migration required)

### Upstream project sources (synchronized revision block)

| File | Notes |
|------|-------|
| `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` | **Principal** — §8 HDP, H as homeostatic state, null naming |
| `artifacts/project_sources/1_global_rules_and_restrictions.md` | Continuation lists "H state" generically → RBS |
| `artifacts/project_sources/2_jaxfne_objective_grammar.md` | HDP nulls, "H-state null" |
| `artifacts/project_sources/5_docs_tutorials_etudes_and_suites.md` | HDP/adaptation tutorial types |
| `artifacts/project_sources/6_other_important_notes.md` | HDP continuation/stability notes |
| `artifacts/project_sources/3_jaxfne_visualization_rules.md` | HDP H trajectories (label → RBS) |
| `artifacts/project_sources/README.md` | Index prose only |

### Public / conceptual docs

| File | Occurrences (approx.) | Notes |
|------|----------------------|-------|
| `docs/guides/hdp.md` | 45+ | Primary user-facing doctrine page — full rewrite after upstream |
| `docs/api/runtime.md` | 5+ | `enable_hdp`, `h_state_locality` descriptions |
| `docs/api/index.md` | 10+ | H-state/HDP index links |
| `docs/api/neuronal_tensor.md` | 11+ | HDP on tensor path |
| `docs/api/emitters.md` | 12+ | Mixed HDP + homeostatic_ei — split B vs A carefully |
| `docs/etudes/hdp_controllability_reachability.md` | 7+ | Generalized-H evidence — map to RBS |
| `docs/public_surface_contract.md` | 10+ | `hdp_param_groups` semantics |
| `docs/quickstart.md`, `docs/index.md` | few | Navigation strings |
| `docs/HDP_REPORT.md` | 19+ | Internal report — relocate or mark historical (**C** candidate) |
| `AGENTS.md` | if present | H grammar — align with RBS/RBD |

### Implementation docstrings (non-API behavior)

| File | Notes |
|------|-------|
| `jaxfne/emitters.py` | HDP kernel docstring, comments "Homeostasis-Dependent" |
| `jaxfne/_model_simulate.py` | HDP dispatch comments |
| `jaxfne/_hdp_adaptive.py` | population restoring — RBD controller language |
| `jaxfne/_pipeline.py` | DynamicState "H" slot — document as RBS carry |
| `jaxfne/public_surface.py` | `hdp_param_groups` docstrings |
| `jaxfne/neuronal_tensor.py` | HDP metadata fields |
| `jaxfne/vis/hdp_diagnostics.py` | "homeostatic-plasticity" in module doc — split HDP vs homeostatic |

### Skills

| File | Notes |
|------|-------|
| `skills/jaxfne-science/SKILL.md` | F_H/F_Theta grammar — add RBS/RBD labels |
| `.cursor/skills/jaxfne-core/SKILL.md` | dynamics/H routing |

---

## B — Kernel-specific homeostasis (preserve terminology)

| File / symbol | Mechanism | Action |
|---------------|-----------|--------|
| `jaxfne/emitters_homeostatic_ei.py` | `dG/dt`, `dH/dt` homeostatic rules | Keep **homeostatic_ei** naming |
| `simulate_edge_recurrent_izhikevich_homeostatic` | `r_i`, `tau_r_ms`, `k_gain` excitability bias | Keep **homeostatic** in kernel docs |
| `RuntimeConfig.enable_homeostasis` | mutually exclusive with HDP | Keep flag name; clarify vs RBD |
| `docs/guides/homeostasis.md` | legacy homeostasis path | Keep; link to RBD property |
| `tests/test_homeostasis_dispatch.py` | homeostatic kernel tests | No rename |
| `tests/test_homeostatic_ei_*.py` | Phase B homeostatic_ei | No rename |
| `Configuration` homeostasis-related builders | if distinct from HDP | **B** |

**Rule:** `homeostatic_ei ≠ generic RBS doctrine`.

---

## C — Historical / archive

| Location | Notes |
|----------|-------|
| `docs/changelog.md` | Version history — do not rewrite past releases |
| `artifacts/project_sources/README.md` | v0.4.8 snapshot label |
| `docs/v047_refactor_audit.md` | prior audit |
| `artifacts/hdp_v2_rho_sweep/*` | measured receipts |
| `artifacts/.lab/*` | generated API snapshots |
| `docs/HDP_REPORT.md`, `STDP_*` | legacy internal reports |
| Committed étude metrics at `v0.4.15` | frozen evidence |

---

## D — API compatibility surfaces (no gratuitous rename)

| Identifier | Tier | Migration note |
|------------|------|----------------|
| `enable_hdp` | RuntimeConfig / public | Keep; document as HDP enable |
| `hdp_params` | public transport dict | Keep; `hdp_param_groups` metadata |
| `DEFAULT_HDP`, `DEFAULT_HDP_DESYNC`, `DEFAULT_HDP_V1_PFC_AAAB` | root / hdp_network | Keep preset names |
| `h_state_locality`, `h_state_dim`, `h_state_readout` | public | Keep; **doc** maps to RBS locality |
| `with_hdp_initial_state` | Model method | Keep |
| `compile_step_fn`, `scan_network`, `ContinuationState` | pipeline | Keep; document RBS in carry |
| `simulate_edge_recurrent_izhikevich_hdp` | ADVANCED namespace | Keep symbol |
| `HDPColumnConfig`, `hdp_network.build_model` | public builders | Keep; optional alias later |
| `K_HDP`, `K_ctrl`, `hdp_rule` | parameter keys | Keep — HDP still valid acronym |
| `EdgeList.delay_steps` | Protocol D (uncommitted) | **orthogonal** — RBD delay, not RBS |

**Evidence:** `artifacts/public_surface_contract_v0413.json`, `tests/test_public_surface_contract_v0413.py`

---

## Orthogonal: Protocol D₀/D₁ (not part of RBS migration)

| File | Role |
|------|------|
| `jaxfne/emitters.py` (local) | `delay_steps`, finite edge delay — **RBD** recurrent coupling |
| `tests/test_edge_delay_protocol_d016.py` | D₀/D₁ falsification |

Commit **separately** from doctrine migration.

---

## Search counts (repository-wide, indicative)

Phrases searched per handout §11 (includes tests, scripts, artifacts):

| Pattern | ~matches | Primary classes |
|---------|----------|-----------------|
| `HDP` | 200+ files | A + D + C |
| `H-state` / `H state` | 50+ | A + D |
| `homeostasis-dependent` | 10+ | A |
| `homeostatic state` | 5+ (mostly upstream + hdp.md) | A |
| `enable_hdp` / `hdp_params` / `DEFAULT_HDP` | 40+ files | D |

Full mechanical grep available via:

```bash
rg -n 'homeostasis-dependent plasticity|homeostatic state|HDP|H-state' jaxfne docs artifacts/project_sources tests --glob '*.py' --glob '*.md'
```

---

## Proposed migration order (post D checkpoint — **not executed**)

1. Revise `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` (upstream)
2. Propagate to `1_`…`6_` project sources
3. Rewrite `docs/guides/hdp.md` → RBS/RBD/HDP (keep filename or add redirect)
4. Targeted docstrings in `emitters.py`, `_config.py`, `public_surface.py`
5. `AGENTS.md` + skills cross-links
6. **Do not** rename `enable_hdp` / `DEFAULT_HDP` without versioned deprecation plan

---

## Stop-rule triggers to watch

| Risk | Status |
|------|--------|
| Rename breaks `__all__` / surface contract | D-tier freeze |
| Two competing canonical definitions | **Active** until project sources revised |
| Erase homeostatic_ei terminology | **Blocked** by B-class rule |
| Partial H continuation presented as full | Documented debt (`reject_population_continuation`) |

---

## Next authorized step

**Await review** after Protocol D₀/D₁ commit. Then execute doctrine migration per order above.

---

## 0.4.17 per-state semantic inventory (campaign Block B, executed)

Six-way classification applied to every H-carrying state in the live code.
Authority: `docs/doctrine/rbs_rbd_hdp.md`; campaign definitions:

- **RBS_GENERAL** — generic RBS container (`H ∈ R^{d_H}`, identity-preserving).
- **RBS_SPECIALIZATION** — a specialized RBS with declared semantics.
- **HOMEOSTATIC_TRACE** — homeostatic traces (B-class terminology; do not rename).
- **HDP_STATE** — HDP controller/plasticity dynamics state (`dH/dt = F_H`, `dW/dt = F_W(H,…)`).
- **NOT_RBS** — configuration plumbing, caches, readouts, derived quantities.
- **AMBIGUOUS** — semantics not yet declared; requires a typed coupling map.

| State (symbol) | Where it lives | Dynamics / semantics | Class |
|---|---|---|---|
| Node-local HDP trace `H_i(t) ∈ R^{d_H}` | `emitters.simulate_edge_recurrent_izhikevich_hdp` | `τ_i dH_i/dt = α·I_syn + β − γ·H_i·r_i − δ·W_i + ρ/H_i² − dC/dH_i + K_ctrl·(1 − H_i)` (live restoring term, emitters.py `K_ctrl`); generalized-state coupling `H@coupling.T` (D1 map, d_H > 1); spike event `H_i ← H_i − C_spike`; `τ_i = τ_0·size_i³`; bounds `H_min`/`H_max`; carries through DynamicState.H | **HDP_STATE** |
| Weight plasticity `W` under H | same emitter (w_rules `signed_linear`/`signed_quadratic`/`hebbian_product`) | `dW/dt = F_W(H, …)` per declared rule; typed coupling to H | **HDP_STATE** |
| Population-restoring H | `_hdp_adaptive.py` | `dH = (−e_vec − λ·H)/τ_H`, `dθ = (B@H)/τ_θ`, deviation convention `H* = 0`; population locality; continuation rejected | **HDP_STATE** |
| Homeostatic EI traces `x`, `G`, `H` | `emitters_homeostatic_ei.py` | three-timescale homeostatic traces (B-class naming preserved) | **HOMEOSTATIC_TRACE** |
| DynamicState continuation carrier (`state.dynamic.H`) | `_pipeline.py` | runtime carry of per-neuron HDP state across segment boundaries | **HDP_STATE** (carrier) |
| `h_state_locality` / `h_state_dim` / `h_state_readout` config | runtime configuration | configuration plumbing controlling HDP state shape/observation | **NOT_RBS** |
| Runtime aggregate `hdp_initial_H` param | `construct` (PlasticParams aggregation) | initial RBS values for runtime HDP state | **HDP_STATE** (initial values) |
| Canonical cell-type homeostatic config (`enable_homeostasis`) | cell-type configuration | enables HOMEOSTATIC_TRACE machinery | **NOT_RBS** (config) |
| Developmental state `H_D` (JDNA) | `jaxfne.jdna` (framework possibility) | no canonical genome declares `H_D`; no developmental state crosses into runtime | **RBS_GENERAL** (declared possible; unused) |

**Result.** No `AMBIGUOUS` H-state remains: every H-carrying state is
classified with declared dynamics and a typed coupling (or explicit absence of
one). JDNA development declares no runtime RBS (`d_D ≠ d_R`, no projection).
