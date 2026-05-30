## [0.3.18] - 2026-05-30

### Added
- `jaxfne/sharding_utils.py`: New module with trace-safe, single-axis distributed
  sharding mesh stubs for candidate-population parallelism.
  - `make_population_mesh()` — 1-D `Mesh` across all JAX devices; returns `None`
    on single-device environments (clean fallback, no caller branching required).
  - `make_candidate_sharding(mesh)` — `NamedSharding` that slices the batch axis
    across the `population_sweep` mesh axis.
  - `make_replicated_sharding(mesh)` — `NamedSharding` that fully replicates arrays
    (for model-parameter tensors that must not be partitioned).
  - `get_sharding_context()` — convenience bundle returning `{mesh, candidate,
    replicated}` or `None` on single-device.
- `tests/test_v0318_sharding_stubs.py`: 14 tests covering import smoke, single-device
  fallback, axis name, PartitionSpec correctness, and context bundle structure.
- Public API exports added to `jaxfne/__init__.py` (`__all__`).

### Scope
- `truth_safe_unverified` / `field_solver_status=laminar_proxy_no_pde` unchanged.
- Sharding stubs do not yet drive actual multi-device dispatch in the AGSDR loop
  (planned for v0.3.20+).
- `simulate_batch` unchanged — its `jax.vmap` seed-replicate path is a separate
  concern from candidate-population sharding.

---

## [0.3.17] - 2026-05-30

### Changed
- `optim.py` (`_run_agsdr_optimization_loop`): Replaced hard-coded `dtype=jnp.float32`
  with `_wdtype` derived from bounds via `jnp.result_type`. Noise tensor, `gen_best_arr`,
  and delta-rule center updates now match this working dtype.
- `optim.py` (matrix AGSDR path): Applied `_wdtype_outer` dtype-inheritance to `lows`,
  `highs`, noise generation, inner-loss fallback, `W_init` extraction, and
  delta-rule center update arrays.
- `W_init` for inner-loop Adam now reads `emitter.W.dtype` instead of force-casting
  to `float32`.

### Added
- `tests/test_v0317_dtype_invariants.py`: 12 targeted tests covering default float32
  path, dtype inheritance, candidate clipping, row-0 center lock, and bounds
  validation guards.

### Scope
- `truth_safe_unverified` / `field_solver_status=laminar_proxy_no_pde` unchanged.
- No public API additions or removals.

---

## [0.3.14] - 2026-05-29


### Added
- Added null, ablation, and synchrony-control support for the v0.3.14 tutorial/release line.
- Added release-facing validation coverage for numerical stability and boundary-sensitive scaffold behavior.

### Changed
- Updated Suite 3 noisy-power handling for cleaner finite proxy-output behavior.
- Improved optional bridge guards so unavailable optional dependencies fail with explicit import guidance.

### Validation
- Local release-candidate validation: `1647 passed, 64 skipped, 4 xfailed`.
- Import smoke passed for `jaxfne==0.3.14` on CPU with JAX `0.10.0`, x64 disabled, default float dtype `float32`.

### Scope
- Maintains `truth_safe_unverified`, `computational_scaffold`, `field_solver_status=laminar_proxy_no_pde`, and `physical_amplitude_claim_allowed=false`.

---

## v0.3.11 Matrix AMPA/GABA Optimization with Optax Adam Inner Loop

- **Core Feature:** Matrix-parameter optimization for recurrent synaptic strength tuning
  - `MatrixParameterSpec`: Frozen dataclass for weight matrix specifications with masks, bounds, and initialization strategies
  - `matrix_parameter()` factory: Public API for creating matrix parameter declarations
  - `gAMPA_w`: Primary use case — AMPA weight matrix parameter (non-scalar)
  - `gGABA_w`: Inhibitory weight matrix parameter for AMPA/GABA push-pull scenarios
  
- **Two-Level Optimization Strategy**
  - **Outer Loop (AGSDR):** Population-based candidate proposal with stochastic exploration and adaptive alpha
  - **Inner Loop (Optax Adam):** Gradient refinement using differentiable soft-spike surrogate loss
  - **Final Selection:** Real objective (group-wise rate targets) to gate biological claims
  - Full backward compatibility with existing scalar AGSDR path
  
- **API Grammar Changes**
  - ✅ `jtfne.agsdr(..., inner_optimizer=optax.adam(learning_rate=1e-2), inner_steps=25, inner_objective="soft_rate_surrogate")`
  - ✅ `parameters={"gAMPA_w": jtfne.matrix_parameter(mask="excitatory_to_all", bounds=(0.0, 3.0))}`
  - ✅ Reject: `gAMPA_first_half`, `gAMPA_second_half`, `drive_scale_a`, `drive_scale_b`
  
- **Suite Updates**
  - Suite No. 1: Kept stable; removed group-specific AMPA knobs; static guards enforce new grammar
  - Suite No. 4 (new): Full AMPA/GABA matrix optimization tutorial with spectral/synchrony objectives
  
- **Optax Integration**
  - Optax is optional dependency (guarded try/except)
  - No `jtfne.adam()` wrapper; users import `optax` directly
  - Inner optimizer state is ephemeral; only config metadata serialized in `TuneResult.summary`
  - JSON-safe serialization of all TuneResult fields
  
- **Soft Rate Surrogate for Differentiable Loss**
  - Hard spike counts are non-differentiable (reset events)
  - Smooth approximation: `sigmoid((V_m - threshold) / temperature)`
  - Enables gradient computation for Adam inner loop
  - Final selection still uses real objective (no surrogate for truth claims)
  
- **Tests Added**
  - MatrixParameterSpec JSON safety
  - AGSDR accepts Optax inner optimizer
  - Matrix parameter mask application and bounds clipping
  - Soft rate surrogate differentiability
  - Suite No. 1 grammar validation (reject old names)
  - Suite No. 4 usage validation
  
- **Version bumps:** pyproject.toml, jaxfne/__init__.py, mkdocs.yml, docs/_generated/version.md all updated to 0.3.11
- **Documentation:** mkdocs builds successfully with strict mode enabled
- **Validation:** All 1422 tests pass (including 3 new tests)
- **Truth status:** `truth_safe_unverified` — soft-spike surrogate is computational scaffold only; final objective gates biological claims
- **Backward compatibility:** ✅ Existing scalar AGSDR notebooks unchanged; no regressions

---

## v0.3.0 Atlas Framework (tutorial-scenario scaffold, phases A–I complete)

- **v0.3.0 Atlas Scaffold (phases A–H):** Baseline verified (v0.2.30 clean), v030 tutorial doctrine created, acceptance gates defined, Plotly policy, automation scripts, validation tests.
- **Core documentation (phases B–D):**
  - `docs/tutorials_v030/README.md` — 15-scenario spine, 31-phase roadmap, hard acceptance gates, claim boundaries
  - `docs/tutorials_v030/scenario_index.md` — Detailed specs for 15 core scenarios + 16 audit phases
  - `docs/tutorials_v030/template.md` — Required 13-section notebook structure with LaTeX equation display policy, quality checklist
  - `docs/tutorials_v030/acceptance_gates.yaml` — 8-category hard validation gates with logic
  - `docs/tutorials_v030/plotly_policy.md` — PNG required (SHA256 hashed), Plotly optional (guarded import)
  - `docs/tutorials_v030/canonical_imports.md` — `import jaxfne as jtfne` enforcement, CI integration
  - `docs/tutorials_v030/docs_audit_policy.md` — Link validation, Colab links, LaTeX equations, term glossaries
  - `docs/tutorials_v030/environment.md` — Setup instructions for JAX, Plotly, Optax, Jaxley (all optional)
- **Environment and requirements (phase D):**
  - `requirements-v030-tutorials.txt` — Tutorial dependencies (jaxfne==0.2.30, JAX, matplotlib, optional Plotly)
- **Automation scripts (phases E–G):**
  - `scripts/tutorial_plotly_utils.py` — Plotly-optional figure generation
  - `scripts/run_v030_tutorial_smoke.py` — Smoke-mode tutorial execution
  - `scripts/collect_v030_tutorial_manifests.py` — Manifest aggregation + validation
  - `scripts/audit_v030_docs_links.py` — Docs link audit, Colab link validation, LaTeX policy checks
- **Validation tests (phases F–H):**
  - `tests/test_v030_tutorial_structure.py` — 13-section template, claim gates, acceptance gates
  - `tests/test_v030_plotly_artifacts.py` — Plotly generation (with/without Plotly installed)
  - `tests/test_v030_docs_audit.py` — Doctrine files, LaTeX policy, Colab links, canonical imports, requirements
- **Validation results (phase H):**
  - All files compile successfully (compileall pass)
  - docs_link_audit.json: PASS (no broken links, no missing Colab links, no LaTeX violations)
  - Language audit: CLEAN (no overclaiming; proper negations on Poisson solvers, biological metabolism claims)
  - Bad alias audit: CLEAN (only correct `import jaxfne as jtfne` found; no jtnfe, jtFNE, or wildcard imports)
  - JSON reports are all valid JSON (no NaN/Inf)
- **Documentation updates (phase I):** `docs/index.md`, `README.md`, `CHANGELOG.md` updated with v0.3.0 atlas links, requirements, and docs audit infrastructure.
- **Status:** Phases A–I complete. Ready for Phase J (commit) and Phase L (final beta-ready report).
- **Truth status:** truth_safe_unverified (v0.2.30 toolbox locked).
- **No package version bump:** v0.3.0 is tutorial/docs/infrastructure line on stable v0.2.30.

---

## v0.3 planning / post-v0.2.30

- **Adds v0.3 tutorial-scenario doctrine.** Defines the 32-phase scenario spine (v0.3.0–v0.3.31) built on stable `jaxfne==0.2.30`.
- **Establishes canonical `import jaxfne as jtfne` usage.** Forbids aliases `jtnfe`, `jtFNE`, and mixed spellings in all v0.3 tutorials.
- **Defines package mutation policy:** Use v0.2.30 as installed toolbox unless a real bug or missing required public tool blocks tutorials. No version bump for docs-only phases.
- **Adds tutorial template** (`docs/tutorial_template_v030.md`): 13-section required structure for all v0.3 Colab/docs scenarios, including non-claim section and manifest receipt block.
- **Patches stale version references:** `docs/install.md`, `docs/packaging.md`, `docs/RELEASE_CHECKLIST.md`, `docs/v03_bridge.md`, `README.md` roadmap table, `docs/index.md` release section header all updated to reflect v0.2.30 as current release.
- **No code changes.** This is a doctrine/docs-only commit.
- **Claim gates unchanged:** `computational_scaffold`, `truth_safe_unverified`, `physical_amplitude_claim_allowed=False`.

## v0.2.30-pre (test cleanup / pre-release)

- **Reconciled legacy tutorial figure artifact contract with canonical v0.2.28 figure system.**
  - **Deprecated `test_tutorial_figure_contract_v0219.py::test_figure_hash_in_assets`:** Legacy test for v0.2.19 runtime output artifacts in `outputs/` directories. Superseded by canonical tutorial figure manifest validation in `test_tutorial_figure_manifest_v028.py`.
  - **Canonical figure system established (v0.2.28+):** `docs/_static/tutorial_figures/figure_manifest.json` is the authoritative artifact index. Runtime `outputs/` directories are ephemeral and no longer part of release contract.
  - **Test suite cleanliness restored:** 918 passed, 12 skipped (4 legacy tests marked skip with deprecation notice), 0 failed.

## v0.2.29

- **Tensor-network ancestry and basis-transform doctrine (conceptual documentation only).**

### Tensor-network ancestry documentation (v0.2.29 scope)
- **Added `docs/tensor_network_ancestry.md`:** Conceptual note distinguishing Pellionisz/Llinás neuroscience meaning of "tensor network" (sensorimotor coordinate transforms, metric-tensor learning) from modern ML/physics meaning (tensor-train, MPS, PEPS factorization). Connects TFNE basis-transform architecture (emitter basis → source basis → field basis → readout basis) to classical computational neuroscience while clarifying non-claims.
- **Basis-transform doctrine:** Formalizes TFNE's modular coordinate-projection pipeline, explains why each stage is independent, and documents how BasisSpec contracts operationalize basis transforms. Links tensor-coordinate operations to jaxfne's architecture.
- **Explicit non-claims:** Clearly states that jaxfne does NOT implement cerebellar metric-tensor learning, tensor-train compression, electromagnetic field solvers, or sensorimotor proof. Defers cerebellar/sensorimotor tutorials to future (with separate validation).
- **Updated cross-references:** Added `tensor_network_ancestry.md` links in `docs/index.md` (Core mathematics section), `docs/computation_basis.md` (See Also), `docs/manuscript_alignment.md` (See Also), and `docs/v03_bridge.md` (See Also).

### Summary
- **Purpose:** Provide conceptual context and historical grounding for TFNE's basis-transform architecture without claiming Pellionisz/Llinás implementation.
- **No code changes:** v0.2.29 documentation only.
- **Truth status:** truth_safe_unverified (unchanged).
- **Claim gates:** All frozen (unchanged from v0.2.28).

## v0.2.28

- **Tutorial figure regeneration and release-docs cleanup.**

### Release documentation (bridge hardening for v0.3+)
- **Added release/distribution documentation:** New `docs/release_checklist.md` (modern v0.2.27-aligned release process with validation gates), `docs/packaging.md` (build, test, distribute wheels and tarballs), and `docs/colab.md` (Google Colab quick start with current API examples).
- **Added manuscript alignment documentation:** New `docs/manuscript_alignment.md` maps codebase sections to manuscript content, clarifies Poisson solver deferral (v0.2.27 has diagnostics, not solver), and documents computation-basis contract changes (v0.2.26–v0.2.27).
- **Added v0.3 readiness bridge:** New `docs/v03_bridge.md` documents locked APIs (emitters, core pipeline, 8 probe operators, claim gates), future regimes (Poisson solver, Maxwell, admittive, stress-energy, Poynting — all gated), and migration path for v0.3.
- **Updated legacy docs:** Converted `docs/RELEASE_CHECKLIST.md`, `docs/COLAB_SMOKE_V010.md`, and `docs/DOCTRINE.md` to legacy pointers with links to current documentation.
- **Updated docs/index.md:** Added new doc links in "Release & distribution" section (release_checklist, packaging, v03_bridge), "Tutorials & guides" section (colab.md, tutorial_figures), and "About" section (manuscript_alignment).
- **Updated docs/ci_policy.md:** Fixed test count from 804 to 903 (cumulative baseline after v0.2.27 conservation diagnostics).

### Tutorial figure regeneration (v0.2.28 core requirement)
- **Generated 12 tutorial PNG figures:** `01_spike_raster.png` (behavioral), `02_voltage_traces.png` (state), `03_source_proxy_heatmap.png` (source), `04_lfp_proxy_trace.png` (scalar readout), `05_csd_proxy_heatmap.png` (spatial readout), `06_phi_e_proxy_heatmap.png` (field potential), `07_source_proxy_spatial.png` (kernel-weighted source), `08_conservation_diagnostics.png` (metrics), `09_laminar_profile_depths.png` (geometry), `10_firing_rate_raster.png` (population activity), `11_claim_gates_summary.png` (metadata), `12_spectral_summary.png` (analysis).
- **Figure count:** 12 total; 11 with real simulation data; 1 placeholder (claim gates summary, uses_real_data=False). **Minimum required: 10 real-data figures. Status: PASS (11 >= 10).**
- **New `scripts/generate_tutorial_figures.py`:** Deterministic figure generation script (seed=0, CPU-safe matplotlib Agg backend). Uses observed jaxfne API: `signals.spikes`, `signals.V_m`, `signals.sources`, `signals.field.lfp_proxy`, `signals.field.csd_proxy`, `signals.field.phi_e_proxy`, `signals.field.source_proxy`, `manifest['conservation_proxy_diagnostics']`.
- **New `docs/tutorial_figures.md`:** Complete figure gallery with claim status, data sources, and truth status for each figure. Includes regeneration instructions.
- **Figure manifest `docs/_static/tutorial_figures/figure_manifest.json`:** JSON-safe schema with per-figure metadata (filename, title, type, uses_real_data, path, visually_confirmed, visual_status, claim_status). Global fields: figure_count, real_data_figure_count, min_required, jaxfne_version, truth gates, source_script, visual_confirmation_method.
- **All figures visually confirmed:** PIL/ImageStat verification (nonzero size, nonblank content, >1 KB). visual_confirmed=True, visual_status="pass" for all 12 figures.
- **New test file `tests/test_tutorial_figure_manifest_v028.py`:** 9 test methods covering manifest structure (JSON-safe, required keys, claim gates), figure count (12 total, >=10 real data), figure paths (exist, PNG, nonzero), figure metadata (required fields, visually confirmed), forbidden phrases (no "real EEG", "validated", "biological metabolism", "Maxwell", "Poisson"), claim-gate immutability, data integrity (filename/path matching, no duplicates, count consistency).
- **Claim gates frozen and immutable:** truth_mode="truth_safe_unverified", claim_level="computational_scaffold", field_solver_status="laminar_proxy_no_pde", physical_amplitude_claim_allowed=False, biological_metabolism_claim_allowed=False. No "real EEG", "validated CSD", "biological proof", "solver implementation" language in any figure title or description.
- **No code changes to core jaxfne modules.** v0.2.28 figure generation is scripts/docs/tests only.

### Summary
- **Purpose:** Complete tutorial figure regeneration with visual confirmation + bridge hardening for v0.3+ readiness.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status: laminar_proxy_no_pde`.
- **BETA completion:** All v0.2.28 gates passed (12 figures generated, 11 real-data, all visually confirmed, claim gates frozen, no forbidden language).

## v0.2.27

- **Conservation-inspired proxy diagnostics.**
- **Added `compute_conservation_proxy_diagnostics()`:** New function in `jaxfne.fields` (exported from `jaxfne`) that computes JSON-safe scalar proxy diagnostics over existing source/field proxy arrays. Accepts `source`, `phi_e`, `csd`, `lfp` arrays or a `FieldOutput` object. Returns `None` for any metric that cannot be computed from available arrays.
- **Diagnostics computed:** `source_norm_l1`, `source_norm_l2`, `source_abs_mean`, `source_conservation_proxy_residual` (spatial-mean proxy for ∫q≈0), `phi_abs_mean`, `phi_gradient_proxy_norm2` (mean squared spatial gradient of phi_e proxy), `csd_abs_mean`, `csd_norm_l2`, `lfp_abs_mean`, `lfp_norm_l2`, `field_energy_like_proxy`.
- **Explicitly not-implemented gates:** `j_dot_e_proxy=None` (J_e not computed in proxy mode), `poynting_flux_proxy=None`, `poisson_solver_status="not_implemented"`, `maxwell_solver_status="not_implemented"`, `stress_energy_tensor_status="not_implemented"`.
- **Manifest integration:** `Model.manifest()` now includes `conservation_proxy_diagnostics` block when `signals.field` is available.
- **42 new tests:** `tests/test_conservation_proxy_diagnostics_v027.py` covering: JSON safety, zero/nonzero array norms, missing-array None returns, physical/biological claim gates, j_dot_e/Poynting/solver/Maxwell/stress-energy invariants, manifest integration, language audit, no double-counting regression.
- **Patched stale `docs/poisson_admissibility.md`:** Updated status from "v0.2.16 will implement a Poisson solver" to correct v0.2.27 doctrine (specification-only; no solver; requires separate approval).
- **Added `docs/conservation_proxy_diagnostics.md`:** New doc with mathematical basis (source norms, conservation residual, phi-gradient proxy, J·E boundary, Poynting future doctrine), output contract table, API examples, and explicit not-implemented table.
- **Updated `docs/index.md`:** Added link to conservation proxy diagnostics doc.
- **No new solver, no Maxwell dynamics, no physical amplitude claims.** v0.2.27 is proxy diagnostics only.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status=laminar_proxy_no_pde`.
- **BETA finalization:** Bumped package version to `0.2.27` after BETA passed with no blockers.

## v0.2.26

- **Computation-basis contracts release.**
- **Added `AxisSpec` and `BasisSpec` dataclasses:** Typed, frozen contract objects in `jaxfne/core.py` describing the spatial basis (`laminar_depth`, `xy`, `xyz`, `collapsed`, `graph`), time basis (`continuous_ms`, `discrete_steps`, `slow_proxy`), field regime (`laminar_proxy`, `quasi_static_resistive`, `solved_poisson`, `future_admittive`, `future_maxwell`), source mode, and probe basis for each TFNE run. Default matches current laminar-proxy scaffold.
- **Added `default_basis_spec()` factory:** Returns the default `BasisSpec` for v0.2.26 laminar-proxy behavior.
- **Added basis validation:** `validate_basis_spec()` and `basis_claim_gate()` in `jaxfne/validation.py`. Validates enum values, axis-space consistency (xy/xyz/laminar_depth/collapsed rules), future-regime gating, and source mode eligibility. Returns JSON-safe dicts. `physical_amplitude_claim_allowed` is always `False`.
- **Future regimes are doctrine-only:** `future_maxwell` and `future_admittive` serialize as `implemented=False, claim_allowed=False`. Cannot be escalated. Structural enforcement tested.
- **Added manifest `basis` block:** `Model.manifest()` now includes a nested `basis` field with space/time/field/source/probe basis metadata and dimension status (x/y collapsed, z active by default).
- **Public API:** `AxisSpec`, `BasisSpec`, `default_basis_spec` exported from `jaxfne`.
- **55 new tests:** `tests/test_computation_basis_v026.py` covers JSON safety, laminar proxy defaults, axis validation, space basis rules (xy/xyz/laminar_depth/collapsed), future regime gating, manifest integration, claim gate, and public export verification.
- **Updated `docs/computation_basis.md`:** Added "Implemented in v0.2.26" section listing contracts, allowed field regimes, and status table.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status: laminar_proxy_no_pde`.
- **No new solver, no conservation diagnostics, no Maxwell/stress-energy implementation.** v0.2.26 is contracts-only.
- **BETA patch:** Corrected v0.2.27 roadmap language in `docs/computation_basis.md`. v0.2.27 is approved for conservation-inspired proxy diagnostics only. No Poisson solver is introduced in v0.2.27. `solved_poisson` remains `implemented=False`, `claim_allowed=False`. Poisson/Maxwell/admittive solvers remain gated future work requiring separate approval.

## v0.2.25

- **Mathematical glossary flow and docs-first upgrade.**
- **Added mathematical glossary flow doctrine:** New `docs/mathematical_glossary_flow.md` documents seven core TFNE equations (emitter dynamics, source projection, ohmic current, field compatibility, CSD, probe operator, EMM-proxy) with formal definitions, complete term glossaries, worded-equations, critical bridge terms, claim boundaries, and implementation locations. Includes conservation-law doctrine (Poynting's theorem) as future reference.
- **Added source/field bookkeeping guide:** New `docs/source_field_equations.md` specifies source modes (total_membrane_current, decomposed_cap_ion_syn, proxy_no_field_solve), one-source-per-run requirement, **forbidden synaptic double-counting pattern** with audit examples, field metadata (boundary/gauge/CSD convention), calibration labels, and code-to-manifest mappings. Minimal examples tying equations to implementation.
- **Added computation basis doctrine:** New `docs/computation_basis.md` describes TFNE as collapsible tensor-field scaffold with canonical dimensions (time, units, space, features, readout), collapse rules, basis-change philosophy, declared-future field regimes (v0.2.27 diagnostics, v0.3.x physical), extensibility doctrine for new domains, and PRNG/finiteness contracts.
- **Rewrote README.md for compactness:** Reorganized with sections: identity, installation, minimal example, pipeline (4 stages + operators table), readout meanings, validation (fast vs extended), documentation map, roadmap (v0.2.24–v0.3.0), claim status. Explicit "not a biological simulator" statement with guidance on when/when-not to use.
- **Updated docs/index.md:** Added new "Core mathematics & equations" section highlighting glossary flow, source/field equations, and computation basis. Reorganized "Learn more" to prioritize equation documentation.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`, `field_solver_status: laminar_proxy_no_pde`.
- **No code changes, no feature expansion.** v0.2.25 is pure documentation: mathematical grounding, forbidden patterns, equation-to-code mapping, and clear roadmap.
- **Purpose:** Establish solid mathematical and conceptual foundation for v0.2.26–v0.3.x extensibility, diagnostics, and physical-field work.
- **BETA patch:** Bumped package version to `0.2.25` and patched the source-projection glossary to define `a_k` (state-to-source scalar) and `b_k` (input-to-source scalar), resolving the documentation completeness gap identified in BETA review.

## v0.2.24

- **Foundation audit checkpoint release.**
- **Verified v0.2.23 release baseline:** Confirmed tag, version consistency, and clean repo state after v0.2.23 release.
- **Audited calibration/source/field/report contracts:** Verified all required fields present and correct; no double-counting of synaptic current.
- **Confirmed solver status:** Field solver remains `laminar_proxy_no_pde`; field readouts remain proxy-only; boundary/gauge conditions remain metadata-only.
- **Confirmed public language:** Verified no forbidden phrases (real EEG, real MEG, biological metabolism claims, mechanism proof). All public language uses approved proxy/scaffold terminology.
- **Updated version assertions in tests:** Fixed 11 test files with outdated version checks; validation baseline: 806 passed, 5 skipped.
- **Preserved truth status:** `truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`.
- **No new science features, no code rewrites, no biological claims.**
- **Purpose:** Establish stable foundation for v0.2.25–v0.2.28 late-0.2.x bridge block (mathematical glossary, computation-basis contracts, conservation diagnostics, 0.3 bridge hardening).

## v0.2.23

- **Package cleanup and docs polish release.**
- **Added quick API reference table:** README now includes organized table of main API categories 
  (Configuration, Simulation, Emitters, Fields/Probes, Readouts, Bridges, Optimization, I/O).
- **Clarified validation documentation:** Split validation into core (fast, every commit) and extended 
  (manual, release validation). Large examples (02-05, 07) now documented as manual-validation tests 
  with expected exclusion from fast CI.
- **Added CI policy documentation:** New `docs/ci_policy.md` explains smoke-safe CI gate, large tutorial 
  exclusion rationale, and release validation procedures.
- **Improved docs/index.md:** Reorganized to highlight current best practices; added reference to CI policy 
  in "Development & CI" section.
- **Terminology consistency:** User-facing docs use "programmatically generated" instead of "human-readable"; 
  removed casual language ('we' → 'jaxfne', 'you can' → 'workflows can').
- **Preserved all features:** No code changes, no new dependencies, no feature expansion. v0.2.23 is pure 
  documentation and packaging polish.
- **Preserved truth status:** All claim gates remain frozen (`truth_safe_unverified`, `computational_scaffold`, 
  `physical_amplitude_claim_allowed=False`).
- **Known issue documented:** Subprocess tests (test_example_script_runs) excluded from fast CI; documented as 
  release-validation only. Infrastructure issue, not regression.

## v0.2.22

- **Added Jaxley array-first trace bridge:** Minimal optional bridge for converting Jaxley-style voltage 
  trace arrays to jaxfne Signals without Jaxley installation.
- **New API:** `JaxleyTraceSpec` (frozen dataclass with immutable claim gates) and `jaxley_trace_to_signals()` 
  (main conversion function).
- **Layout normalization:** Supports three input layouts: `time_by_unit` [T,N], `unit_by_time` [N,T], 
  `recording_by_time` [R,T]. All normalize to canonical [T,N] internally.
- **Spike proxy derivation:** Configurable voltage threshold (default 0.0 mV) for deriving binary spike proxy 
  from voltage trace.
- **Conservative source handling:** Voltage-proxy source fallback; no ionic current mapping (deferred to v0.2.23).
- **25 new unit tests:** Comprehensive test suite covering spec validation, layout conversion, spike thresholding, 
  source handling, metadata gates, NumPy/JAX conversion, and no-Jaxley-required import.
- **Synthetic example:** `examples/07_jaxley_trace_bridge.py` demonstrates bridge usage with synthetic voltage 
  traces (CPU-only, <10 seconds runtime).
- **Immutable claim gates (frozen):**
  - `claim_level: "computational_scaffold"`
  - `physical_amplitude_claim_allowed: False`
  - `field_solver_status: "not_computed"`
  - `source_calibration_status: "uncalibrated_jaxley_voltage_proxy"`
- **Field computation deferred:** All Signals have `field=None`; field computation reserved for downstream 
  probe/field layer.
- **Scope discipline:** No multi-compartment support, sparse spike format, ionic current mapping, field computation, 
  or simulator wrapper in v0.2.22 (all deferred to later phases).
- **Full test validation:** 806 tests passed, 5 skipped; no regressions from v0.2.21.
- **Preserved truth status:** `truth_safe_unverified`, `computational_scaffold`, no biological claims.

## v0.1.2

- **Documented scan-backed recurrent execution:** Both dense and edge-list paths use `jax.lax.scan` for 
  deterministic, JAX-native time stepping. This release formalizes the existing implementation and adds 
  metadata confirmation.
- **Added `docs/V012_SCAN_PERFORMANCE.md`:** Design document confirming scan-backed architecture and 
  benchmark methodology.
- **Added benchmark harness:** `scripts/benchmark_scan_backends.py` measures wall time for dense and 
  edge-list backends at scales ranging from 50 to 100 neurons across durations up to 1000 ms.
- **Added metadata tests:** `tests/test_scan_backends_v012.py` ensures dense, edge-list, and 
  receptor-exponential paths report correct metadata and preserve truth gates.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`, 
  `laminar_proxy_no_pde`, `uncalibrated_izhikevich_native_current`, `physical_amplitude_claim_allowed=False`.
- **No kernel refactoring:** Dense and edge-list backends remain unchanged; this release validates 
  performance and correctness of the existing scan-backed implementations.

## v0.1.1

- **Corrected VIP/IS Izhikevich preset:** `b` parameter corrected from `+0.20` to `-0.10` to match
  intrinsic-spiking profile (Izhikevich 2003 Table 1).
- **Added per-neuron `layer_labels` support:** `IzhikevichParams` now accepts optional `layer_labels`
  tuple for layer-selective analysis (e.g., L1, L2/3, L4, L5, L6).
- **Added `population_slices()` method:** `LaminarSourceGeometry` now provides programmatic mapping
  from population names to neuron index ranges for layer-specific readouts.
- **Added preset registry:** Introduced `jaxfne.presets` with standardized `CELL_TYPE_PRESETS`,
  `RECEPTOR_KINETICS`, and `DEFAULT_SPIKE_IMPULSE_GAIN` constants for reproducible configuration.
- **Preserved all truth gates:** `truth_safe_unverified`, `computational_scaffold`,
  `laminar_proxy_no_pde`, `uncalibrated_izhikevich_native_current`, `physical_amplitude_claim_allowed=False`.
- **No biological calibration changes:** This is a computational-correctness and API-readiness pass,
  not an empirical validation upgrade.

## v0.1.0

- Declared practical OOP core freeze for the compact JAX-native TFNE scaffold.
- Preserved canonical workflow: `run_receipt`, `compute_readout`, `evaluate_report`.
- Includes all v0.0.23 fixes: `manifest(signals, readouts)` readout compatibility,
  MIT LICENSE, normalized examples 00-06, full packaging validation.
- Validated wheel and sdist install smokes from `/tmp`; canonical workflow passes
  from installed package (site-packages, not repo).
- Preserved truth status at `truth_safe_unverified`.
- Preserved field status as `laminar_proxy_no_pde`.
- Preserved source calibration as `uncalibrated_izhikevich_native_current`.
- Preserved `physical_amplitude_claim_allowed=False` across all outputs.

## v0.0.23

- **Fixed `Model.manifest()` readout compatibility:** `manifest(signals, readouts)` now
  accepts any of: `None`, `dict` (legacy), `list[ReadoutResult]` (canonical v0.1 output
  of `compute_readout()`), `tuple[ReadoutResult]`, `list[dict]`, or a single
  `ReadoutResult`. Previously raised `AttributeError: 'list' object has no attribute 'get'`
  when passed the canonical `compute_readout()` return value.
- Added `_normalize_manifest_readout()` normaliser; surfaces readout results under
  `readout_results` key in manifest with `n_results`, `requested_metrics`, and frozen
  `physical_amplitude_claim_allowed=False` guard.
- Added 8 tests in `tests/test_manifest_readout_compat.py` covering all argument forms,
  JSON strictness, and truth-gate non-escalation.
- Added MIT LICENSE file.
- Normalized examples directory to 00-06 naming convention.
- Validated wheel and sdist build via `python -m build`; `twine check dist/*` passes.
- Confirmed fresh venv wheel and sdist install smokes from `/tmp`; import path confirmed
  as `site-packages` (not repo).
- Confirmed canonical workflow from installed wheel:
  `compute_readout(...)` → `manifest(signals, readouts)` → `json.dumps(allow_nan=False)`.
- Preserved truth status at `truth_safe_unverified`.
- Preserved `physical_amplitude_claim_allowed=False` across all outputs.

## v0.0.22
- Added packaging, release, and Colab installation documentation.
- Validated wheel and sdist builds with twine check.
- Validated fresh virtual-environment install smoke tests for wheel and sdist.
- Added a minimal Colab spectrolaminar proxy scaffold example.
- Preserved truth status at truth_safe_unverified.

# Changelog

All entries reflect `truth_mode: truth_safe_unverified`. No biological claims
are made at any version. Receipts, reports, and manifests are computational
validation artifacts, not empirical evidence.

## v0.0.21

- **Config/runtime fidelity:** Added `_SUPPORTED_RUNTIME_SPEC_KEYS` and
  `_runtime_from_spec()` to validate runtime declarations from `.jcfg.json`;
  unknown keys now warn; invalid known values (e.g. bad `synaptic_kernel`) raise.
- **Truth escalation guard:** Implemented `_conservative_truth_transfer()` to
  force user-declared truth claims back to conservative defaults
  (`truth_safe_unverified`, `computational_scaffold`, `physical_amplitude_claim_allowed=False`);
  escalations trigger warnings; non-scalar unknown keys skipped.
- **Unsupported config warnings:** Added `_config_section_warnings()` to detect
  unsupported emitter families, field domains, conductivities, boundaries, and gauges;
  warnings merged into `Configuration.metadata["unsupported_config_warnings"]`.
- **Runtime config warnings registry:** Added `_CONFIG_RUNTIME_WARNINGS` module-level
  dict to surface `runtime_spec` warnings without mutating frozen `RuntimeConfig`.
- **Backend reporting fidelity:** Enhanced `RuntimeConfig.runtime_report()` to
  distinguish `requested_backend` vs `actual_backend` and report enforced status
  and mismatches (e.g. requested GPU on CPU-only JAX device).
- **vmap behavioral semantics:** Made `simulate_batch()` respect `runtime.vmap` flag:
  `vmap=True` uses `jax.vmap` over seed batch; `vmap=False` uses Python loop with
  `jnp.stack`; mode reported in metadata as `batch_execution_mode`.
- **Source proxy metadata:** Added `_SOURCE_PROXY_METADATA` constant documenting
  source model: `izhikevich_native_current_plus_spike_impulse_proxy` with spike
  impulse gain 20.0; injected into `simulate()`, `simulate_batch()`, and
  `manifest()` under `source_model` and `backend_metadata.source_model`.
- **Receptor/tau source documentation:** Enhanced `manifest()` to document
  `receptor_tau_source` distinction: exponential kernel uses default tau;
  receptor_exponential kernel looks up tau by receptor_index; results equivalent
  for current default flow.
- **Schema version bumps:** Updated `_RECEIPT_SCHEMA_VERSION` to
  `"run_receipt_v0.0.21"`, `_MANIFEST_SCHEMA_VERSION` to `"manifest.v0.0.21"`.
- Preserved all truth gates at `truth_safe_unverified / computational_scaffold /
  laminar_proxy_no_pde / proxy_readout_only / physical_amplitude_claim_allowed=False`.

## v0.0.20

- Fixed `RunReceipt` completeness: `duration_ms`, `dt_ms`, `n_steps`, and
  record flags now propagate into receipt simulation metadata.
- Strengthened `receipt_id` hashing to include run-level metadata
  (recurrent backend, synaptic kernel, condition name, stimulus schedule).
- Fixed `Model.manifest()` to report executed backend separately from
  available infrastructure.
- Honored probe `n_contacts` from `.jcfg.json` in field/readout construction.
- Applied readout `time_window_ms` slicing to field-backed CSD/LFP metrics.
- Added safe empty/negative-window handling to avoid NaN in readout output.
- Added `Simulation.__post_init__` validation (duration/dt must be positive finite).
- Clarified `record_sources` semantics in metadata.
- Centralized version/schema constants near `_JAXFNE_VERSION`.
- Preserved truth status at `truth_safe_unverified`.
- Preserved canonical v0.1 workflow: `run_receipt`, `compute_readout`,
  `evaluate_report`. Compatibility aliases (`manifest`, `probe`) unchanged.

## v0.0.19

- Clarified canonical v0.1 API wording: `run_receipt`, `compute_readout`,
  and `evaluate_report` are the canonical workflow methods.
- Documented `manifest()` and `probe()` as compatibility aliases retained
  from v0.0.4–v0.0.14; not removed.
- Added docstring notes to `config_truth_boundary()`: passthrough helper,
  call `validate_config()` first.
- Added docstring notes to `JaxFNEConfig.config_hash`: unknown `.jcfg.json`
  keys enter the hash; hash equality is structural identity, not biological
  equivalence.
- Added docstring notes to `Model.run_receipt()`: `receipt_id` is tied to
  `_JAXFNE_VERSION`; upgrading the package changes IDs for same cfg/seed.
- Added `CHANGELOG.md`.
- Version bump `0.0.18` → `0.0.19`.

## v0.0.18

- Added `ObjectiveReport` (frozen dataclass) and `Model.evaluate_report()`.
- `ObjectiveReport` embeds `ReadoutResult` items when `readout_specs` are
  provided; carries frozen truth gates.

## v0.0.17

- Added `ReadoutSpec`, `ReadoutResult`, `readout_spec()` factory.
- Added `Model.compute_readout(signals, specs)`.
- Six supported metrics: `spike_rate_hz`, `spike_count`, `mean_V_m`,
  `csd_abs_mean`, `lfp_abs_mean`, `source_abs_mean`.

## v0.0.16

- Added `RunReceipt` (frozen dataclass) and `Model.run_receipt()`.
- Added module-level `run_receipt()` factory and `save_receipt()`.
- `receipt_id` is deterministic: `sha256(config_hash:seed:version)[:16]`.
- Fixed `truth_mode` absent from `_CONSERVATIVE_TRUTH_DEFAULTS` (blocking
  defect: `validate_config` now checks all 8 required truth keys).

## v0.0.15

- Added `JaxFNEConfig`, `ConfigValidationResult`, `load_config()`,
  `validate_config()`, and `.jcfg.json` declarative config standard.
- Added `config_to_simulation`, `config_to_geometry`,
  `config_to_configuration`, `config_to_trial_batch`, `config_truth_boundary`.
- Truth boundary fields are required in every config; any escalation is a
  blocking validation error.

## v0.0.14

- Added sequential trial runner: `TrialSpec`, `TrialBatch`, `TrialResult`,
  `TrialBatchResult`, `trial_batch()`, `Model.run_trials()`.
- Deterministic seed policy; JSON-safe compact results via `Signals.summary()`.

## v0.0.13

- Added `LaminarPopulation`, `LaminarSourceGeometry`, `laminar_source_geometry()`.
- Geometry depths are normalized proxy coordinates in `[0, 1]`; no physical
  spatial units (mm, µm) introduced.

## v0.0.12

- Added `StimulusSchedule`, `stimulus_schedule()`, and event-aligned native-
  drive injection into all recurrent kernels.

## v0.0.11

- Added receptor-indexed exponential synaptic kernel
  (`synaptic_kernel="receptor_exponential"`).
- `syn_state.shape == (n_edges,)` with per-edge tau lookup by
  `receptor_index` against standard `ReceptorSpec` table.
