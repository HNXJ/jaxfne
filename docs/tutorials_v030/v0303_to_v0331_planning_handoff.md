# v0.3.3–v0.3.31 Planning Handoff

**Date:** 2026-05-24  
**Status:** Ready for human review and v0.3.3 kick-off  
**truth_mode:** truth_safe_unverified  
**Package baseline:** jaxfne 0.2.30 (stable, no v0.3 release yet)

---

## 1. Current Accepted State

### v0.3.1: Single Izhikevich Neuron
- **Status:** Merged (commit 6c3bde8) + validation audit closed
- **Figures:** 2 PNG (voltage trace, spike raster) at 150 dpi
- **Truth gate:** computational_scaffold, proxy_readout_only, physical_amplitude_claim_disallowed
- **Key metric:** Voltage range, spike count, firing regularity
- **Manifest:** docs/tutorials_v030/manifests/v0301_single_izhikevich_neuron_manifest.json

### v0.3.2: Single-Neuron Parameter Sweep
- **Status:** Merged (commit b3c0f7f) + duration/rate-regime gates closed
- **Figures:** 2 PNG (parameter sweep heatmap, regime lines) at 150 dpi
- **Truth gate:** computational_scaffold, proxy_readout_only, physical_amplitude_claim_disallowed
- **Key metric:** Out-of-regime detection, sweep coverage, regime boundary validation
- **Manifest:** docs/tutorials_v030/manifests/v0302_single_neuron_parameter_sweep_manifest.json

### Visualization Doctrine Status
- Doctrine finalized at commit 41b71b73
- PNG requirement: minimum 2–4 figures per scenario, 150 dpi, SHA256-hashed, docs-stable paths
- Plotly: optional, guarded import, must not block PNG-only users
- Figure naming: `v030_XX_YY_description.png` (XX=scenario, YY=figure index)

### Package Version Constraint
- Current: **jaxfne 0.2.30** (stable)
- No v0.3 package release yet
- v0.3.x is tutorial atlas engineering on top of v0.2.30
- Version bump and PyPI release deferred until after v0.3.31

---

## 2. Standard Workflow for All Future Tutorials

Every tutorial **MUST** follow this workflow:

### Step 1: Implementation
- Copy `docs/tutorials_v030/template.md` to new scenario markdown
- Copy `examples/vXXX_template.py` stub to new scenario script
- Implement simulation, data collection, and PNG figure generation
- Update manifest generation code in script to populate metadata
- Write accompanying markdown with results summary

### Step 2: Self-Audit
- Run example script: `python examples/v03X_scenario.py`
- Verify all PNG files generated in docs-stable path
- Check manifest JSON validity: `python -m json.tool docs/tutorials_v030/manifests/v03X_manifest.json`
- Verify no overclaiming (use checklist in `canonical_imports.md`)
- Verify truth gates declared (use `acceptance_gates.yaml` template)

### Step 3: Deep THETA Review
- Invoke theta-validate prompt on tutorial markdown + example code
- Address all gate failures before proceeding
- Ensure all figures pass visibility checks
- Verify manifest completeness

### Step 4: Patch (if blocked)
- If theta fails on manifest/figure gate: generate missing asset
- If overclaim detected: redact or reclassify claim
- If truth gate fails: adjust truth metadata
- Re-run self-audit

### Step 5: PR Readiness
- Commit example script and docs markdown together
- Run collector: `PYTHONPATH=. python scripts/collect_v030_tutorial_manifests.py outputs --validate ...`
- Verify collector counts all scenarios PASS
- Push to feature branch, open PR with acceptance gates checklist

---

## 3. Phase V Visualization Rule for All Future Tutorials

Reference `/Users/hamednejat/workspace/main/jaxfne/docs/tutorials_v030/visualization_doctrine.md` and include **concrete figure requirements in every future tutorial prompt:**

**Mandatory in each tutorial scope doc or prompt:**
```
### Visualization Requirements (Phase V)
- Minimum figures: 2–4 PNG
- Minimum DPI: 150 (300 dpi if publication-intent)
- Figure naming: v030_XX_YY_description.png
- Storage: docs/tutorials_v030/_static/figures/
- Manifest: record all PNG SHA256 hashes
- Plotly: optional; guarded import; must not block PNG-only users
```

---

## 4. Scenario Map: v0.3.3–v0.3.15 (Core Scenarios)

### v0.3.3: Two-Neuron E/I Pair
- **Purpose:** Introduce multi-neuron connectivity, synaptic delay, postsynaptic conductance
- **Simulation:** 2 neurons (1 E excitatory, 1 I inhibitory), AMPA + GABA_A synapses
- **Minimum figures:** 3–4 (E voltage, I voltage, raster, phase plane)
- **Key manifest gates:**
  - Synaptic current conservation (single kernel mode)
  - Receptor-indexed synaptic delay metadata
  - Postsynaptic conductance unitless (proxy amplitude)
- **Likely blockers:**
  - Synaptic delay implementation; defer if complex
  - Postsynaptic conductance amplitude claim; mark proxy-only
- **Truth boundary:** Proxy connectivity; no biological synapse calibration

### v0.3.4: Small Recurrent E/I Network
- **Purpose:** Introduce recurrent feedback, population dynamics, asynchronous-irregular regime
- **Simulation:** 32 neurons (24 E, 8 I), all-to-all sparse recurrent, 2 synaptic receptors
- **Minimum figures:** 3–4 (population raster, population voltage distribution, spectral power, network diagram)
- **Key manifest gates:**
  - Edge count and receptor distribution
  - Recurrent loop detection
  - Asynchrony metric (CV of ISI)
- **Likely blockers:**
  - All-to-all connectivity scaling; limit to 32 neurons if needed
  - Sparse edge list generation
- **Truth boundary:** Computational demonstration; no circuit topology calibration

### v0.3.5: 100-Neuron E/I Population
- **Purpose:** Scale to realistic-size population, explore population statistics, laminar organization
- **Simulation:** 100 neurons (80 E, 20 I), sparse random connectivity, 3–4 synaptic layers (L2/3, L4, L5, L6 proxies)
- **Minimum figures:** 4 (population raster, firing rate distribution, spectral power, layer-wise CV ISI)
- **Key manifest gates:**
  - Sparseness ratio
  - Layer assignment metadata
  - Population-level metrics (mean firing rate, coefficient of variation)
- **Likely blockers:**
  - Memory scaling for 100-neuron recurrent simulation
  - Laminar assignment convention
- **Truth boundary:** Laminar proxy only; no anatomical calibration

### v0.3.6: Laminar Cortical Column
- **Purpose:** Introduce realistic laminar depth (L1–L6), within-layer and cross-layer connectivity patterns
- **Simulation:** 256 neurons (64 per layer L1–L6), layer-specific E/I ratios, laminar-dependent synapse types
- **Minimum figures:** 4–5 (laminar raster, depth-dependent firing rate, laminar connectivity matrix, CSD-like readout, spectral profile)
- **Key manifest gates:**
  - Layer assignment with depth coordinates
  - Within-layer vs. cross-layer synapse count
  - Laminar depth field readout mock (CSD proxy)
- **Likely blockers:**
  - Laminar coordinate system definition
  - CSD field proxy calculation
- **Truth boundary:** Simplified laminar proxy; no biological cell-type calibration

### v0.3.7: Source Bookkeeping
- **Purpose:** Validate internal sources (Izhikevich native current, recurrent synaptic current, external drive, spike impulse proxy)
- **Simulation:** Single 256-neuron column, decompose total source into components, track conservation
- **Minimum figures:** 4 (total source vs. time, source component breakdown, source conservation residual, field-source correlation)
- **Key manifest gates:**
  - Source decomposition: native + drive + recurrent + spike + noise
  - Source conservation proxy (should be ~0 in steady-state average)
  - Double-count guard status
- **Likely blockers:**
  - Source decomposition accounting (ensure no double-counting)
  - Spike impulse proxy amplitude scaling
- **Truth boundary:** Proxy bookkeeping; no physical current conservation proof

### v0.3.8: LFP/CSD-Like Readouts
- **Purpose:** Introduce laminar field summation, source-to-field projection, CSD-like derived readouts
- **Simulation:** 256-neuron column + 16-contact mock probe, laminar kernel projection, CSD numerical derivative
- **Minimum figures:** 4 (LFP vs. time, CSD heatmap, depth profile at peak, phase-amplitude coupling example)
- **Key manifest gates:**
  - Probe geometry and contact spacing
  - Kernel normalization (row-sum validity)
  - CSD sign convention (positive = extracellular source-like)
  - LFP/CSD finiteness checks
- **Likely blockers:**
  - Kernel derivation (use template from v0.3.0 scaffold)
  - CSD numerical stability
- **Truth boundary:** Laminar proxy kernel; no physical Poisson/Maxwell solver

### v0.3.9: EEG/MEG-Like Proxies
- **Purpose:** Introduce far-field approximations, spatial smoothing, band-limited signal readouts
- **Simulation:** 256-neuron column + simplified 2D head model (3 shells), source-to-scalp field, band-pass filtering (delta/theta/alpha/beta)
- **Minimum figures:** 4–5 (scalp potential topography, time-frequency spectrogram, band-limited time series, source-space dipole moment, atlas comparison)
- **Key manifest gates:**
  - Head model geometry (spherical or realistic BEM)
  - Source-to-scalp kernel derivation
  - Band-pass filter specifications
  - Far-field approximation validity
- **Likely blockers:**
  - Head model implementation (use standard spherical model if needed)
  - Band-limited filtering design
- **Truth boundary:** Simplified far-field proxy; no forward model validation vs. empirical EEG

### v0.3.10: EMM/Conservation Proxies
- **Purpose:** Diagnostic sanity checks: energy-like proxies, Poynting-flux analogues, stress-energy bounds
- **Simulation:** 256-neuron column + field + probe, compute energy-like diagnostics without physics solver
- **Minimum figures:** 3–4 (energy-like proxy vs. time, Poynting flux proxy, spectral coherence, conservation residual)
- **Key manifest gates:**
  - Energy proxy formula (source × field inner product)
  - Passivity check (energy trend monotonic-like)
  - Spectral moment conservation check
  - Stress-energy tensor proxy status
- **Likely blockers:**
  - Definition of "energy-like" proxy (decide on convention first)
  - Numerical stability of differencing
- **Truth boundary:** Diagnostic proxy only; no real energy/power claim

### v0.3.11: Spectrolaminar Baseline (FIRST PyPI CHECKPOINT CANDIDATE)
- **Purpose:** Consolidate all single-paradigm baselines; validate spectrolaminar stability across parameter sets
- **Simulation:** 256-neuron column, 5 stimulus conditions (increasing input drive), measure spectral profile per layer
- **Minimum figures:** 5 (spectrograms x5 conditions, layer-wise power curves, layer-wise coherence, comparison table, figure panel)
- **Key manifest gates:**
  - Parameter sweep metadata
  - Spectral stability check (same band structure across conditions)
  - Laminar dominance per condition
  - Reproducibility check (same seed = same output)
- **Likely blockers:**
  - Spectral estimation stability
  - Laminar averaging convention
- **Truth boundary:** Computational demonstration; no physiological proof
- **PyPI consideration:** If all prior scenarios PASS + manifest audit PASS, v0.3.11 is a natural checkpoint for package release with label "v0.3.11-tutorial-atlas-phase-1"

---

## 5. Scenario Map: v0.3.12–v0.3.15 (Paradigm Scenarios)

### v0.3.12: Evoked L4 Drive
- **Purpose:** Introduce sensory-like bottom-up drive to layer 4
- **Simulation:** 256-neuron column, time-varying input current to L4 excitatory neurons, evoked response
- **Minimum figures:** 3–4 (L4 voltage raster, population PSTH, layer-wise time course, spectral transient)
- **Key manifest gates:**
  - L4 stimulation parameters (amplitude, duration, latency)
  - PSTH reliability (trial count if simulated multiple times)
  - Latency jitter metadata
- **Likely blockers:** Stimulus timing convention
- **Truth boundary:** Input-driven benchmark; no sensory validation

### v0.3.13: Omission/Oddball Scaffold
- **Purpose:** Implement expectation violation paradigm; demo learning/plasticity scaffolds
- **Simulation:** 256-neuron column, sequence of standard and deviant stimuli, simple Hebbian plasticity gate (no learning enabled yet)
- **Minimum figures:** 4 (response to standard, response to deviant, learning trace, oddball index)
- **Key manifest gates:**
  - Stimulus sequence metadata
  - Plasticity gate status (learning disabled in v0.3.13)
  - Oddball effect size (deviant response - standard response)
- **Likely blockers:**
  - Oddball index definition
  - Plasticity gate implementation
- **Truth boundary:** Scaffold only; no biological learning calibration

### v0.3.14: Nulls/Ablations/Synchrony Gates
- **Purpose:** Validation by lesion: remove components and check robustness
- **Simulation:** 256-neuron column, 5 ablation conditions (no noise, no I neurons, no recurrent, no external drive, baseline)
- **Minimum figures:** 4 (ablation summary heatmap, firing rate sensitivity, spectral sensitivity, population synchrony index)
- **Key manifest gates:**
  - Ablation condition labels
  - Synchrony index definition (Kuramoto order parameter or Laguerre-Volterra)
  - Robustness criteria (mean response shift tolerance)
- **Likely blockers:**
  - Synchrony metric definition
  - Tolerance thresholds
- **Truth boundary:** Computational robustness check; no mechanism proof

### v0.3.15: Multi-Area Laminar Scaffold
- **Purpose:** Extend from single column to 3–5 interconnected columns (visual cortex-like), hierarchical feedforward/feedback
- **Simulation:** 3 areas (V1, V2, V3 proxies) × 256 neurons × layer = 2304 neurons total, feedforward L4→L2/3→L5, feedback L5→L1, Izhikevich + synaptic dynamics
- **Minimum figures:** 5 (hierarchical raster, layer-wise spectral profile per area, inter-area coherence, feedforward/feedback flow, dimensionality reduction)
- **Key manifest gates:**
  - Area-to-area connectivity matrix
  - Feedforward/feedback synapse count
  - Propagation latency metadata
  - Population dimensionality per area
- **Likely blockers:**
  - Multi-area coordination
  - Computational cost (2304 neurons recurrent)
- **Truth boundary:** Multi-area proxy framework; no anatomical circuit calibration

---

## 6. Audit/Consolidation Map: v0.3.16–v0.3.31

### Phase "Audit" (v0.3.16–v0.3.20)

#### v0.3.16: Tutorial Figure Audit
- **Goal:** Retroactive PNG validation and regeneration if needed
- **Scope:** All 15 tutorial PNG sets (60+ figures total)
- **Validation target:**
  - All PNG files exist and are readable
  - All PNG SHA256 hashes match manifest
  - All PNG DPI ≥ 150
  - All PNG file size ≥ 1 KB (no blank figures)
  - All PNG naming follows convention
- **Expected output:** Audit report JSON, re-hash list if regeneration needed

#### v0.3.17: Colab Execution Audit
- **Goal:** Validate notebook portability to Google Colab
- **Scope:** All 15 tutorial notebooks (if converted to .ipynb) + Colab-specific stubs
- **Validation target:**
  - All cells execute in Colab environment (no local file path issues)
  - All outputs regenerate correctly
  - All figures display in Colab
  - Execution time < 10 min per notebook
- **Expected output:** Colab smoke-test report, migration checklist

#### v0.3.18: Manifest/Schema Audit
- **Goal:** Validate manifest consistency across all tutorials
- **Scope:** 15 manifest JSON files
- **Validation target:**
  - All keys present per schema version
  - All required fields non-null
  - All enums valid (truth_mode, claim_level, etc.)
  - All SHA256 hashes properly formatted
  - No schema version drift
- **Expected output:** Schema validation report, required schema updates

#### v0.3.19: Equation Glossary Audit
- **Goal:** Build comprehensive glossary of all equations, variables, units used in v0.3
- **Scope:** All 15 tutorial markdown files + example scripts
- **Validation target:**
  - All equations have unique identifiers (e.g., EQ_003_izhikevich_voltage)
  - All variables defined with unit and biological meaning (if applicable)
  - No undefined symbols in equations
  - LaTeX rendering correct
- **Expected output:** Equation glossary markdown + LaTeX macro file

#### v0.3.20: Objective/Report Audit (SECOND PyPI CHECKPOINT CANDIDATE)
- **Goal:** Consolidate all objective functions, gates, and truth claims
- **Scope:** All acceptance_gates.yaml entries, all manifest claim_level values
- **Validation target:**
  - No conflicting truth claims across tutorials
  - All gates consistently applied
  - Objective functions documented and tested
  - Report generation code works end-to-end
- **Expected output:** Unified gates manifest, objective API documentation
- **PyPI consideration:** If all v0.3.12–v0.3.15 scenarios PASS + all audits PASS, v0.3.20 is checkpoint for label "v0.3.20-tutorial-atlas-phase-2-audit" or possible PyPI release v0.3.21

### Phase "Consolidation" (v0.3.21–v0.3.31)

#### v0.3.21: Performance Audit / Possible PyPI Checkpoint
- **Goal:** Benchmark all 15 tutorials for speed, memory, JAX compilation time
- **Scope:** Wall-clock time, peak memory, JIT recompilation events
- **Validation target:**
  - All scenarios complete in <5 min (v0.3.1–2) to <30 min (v0.3.15)
  - Peak memory < 2 GB for single-machine CPU
  - JIT compilation < 10% of total runtime
  - Reproducibility: same seed = same output within floating-point tolerance
- **Expected output:** Performance benchmark table, JAX profiling report
- **PyPI decision:** After v0.3.20 audit + performance validation, decide whether to release v0.3.21 as PyPI package or defer to v0.3.31

#### v0.3.22: Package-Gap Audit
- **Goal:** Identify missing public APIs, helper functions, or utilities needed for tutorials
- **Scope:** All 15 tutorials + supporting scripts
- **Validation target:**
  - No private imports (imports of underscore-prefixed modules)
  - All public API calls documented
  - Helper functions either exported or marked tutorial-local
  - Dependency imports all declared in pyproject.toml
- **Expected output:** Public API inventory, helper refactoring checklist

#### v0.3.23: Manuscript Alignment Pass
- **Goal:** Align tutorial narrative with planned publication manuscript sections
- **Scope:** All tutorials + planned paper outline
- **Validation target:**
  - Each tutorial maps to 1+ manuscript section
  - Figure numbering matches paper figure numbers (where applicable)
  - Terminology consistent with draft manuscript
  - Supplementary materials identified
- **Expected output:** Manuscript-to-tutorial mapping table

#### v0.3.24: Biophysics Tutorial Consolidation
- **Goal:** Write coherent pedagogical narrative across all 15 tutorials
- **Scope:** Tutorial docs + README
- **Validation target:**
  - Learning path clear: single-cell → networks → circuits → systems
  - Each tutorial includes "learning objectives" section
  - Prerequisite and follow-on links between tutorials documented
  - Glossary references consistent
- **Expected output:** Consolidated tutorial guide markdown

#### v0.3.25: Release-Notes/Docs Cleanup
- **Goal:** Write comprehensive v0.3 release notes and final docs polish
- **Scope:** CHANGELOG, docs/tutorials_v030/README.md, scenario_index.md
- **Validation target:**
  - All 15 scenarios documented
  - All 60+ figures captioned
  - All truth claims clearly marked
  - Installation/quickstart sections updated
  - Troubleshooting/FAQ section added
- **Expected output:** Release notes markdown, updated README, FAQ

#### v0.3.26: All-Notebooks Run-Through
- **Goal:** Full end-to-end execution of all tutorial notebooks (or scripts)
- **Scope:** All 15 examples (v031–v0315)
- **Validation target:**
  - All scripts run without error
  - All outputs match prior recorded values (within tolerance)
  - All figures regenerate
  - Collector validates all manifests
- **Expected output:** E2E smoke-test report

#### v0.3.27: All-Figures Regeneration
- **Goal:** Regenerate all PNG and Plotly figures from scratch
- **Scope:** All 60+ PNG + optional Plotly HTML
- **Validation target:**
  - All PNG SHA256 hashes match manifest (or updated with new hashes)
  - All Plotly HTML valid if generated
  - All figure DPI, size, color profiles consistent
- **Expected output:** Figure audit report, updated manifest hashes

#### v0.3.28: Truth-Claim Adversarial Audit
- **Goal:** Stress-test all truth claims and gates for inconsistencies or overreach
- **Scope:** All manifests + acceptance_gates.yaml + tutorial markdown
- **Validation target:**
  - No claim of "biological realism" without gate approval
  - No claim of "calibration" without validation data
  - All "computational_scaffold" claims justified
  - No hidden overclaiming in figure captions
  - All "experimental" or "exploratory" claims marked
- **Expected output:** Adversarial audit report, claim reclassification if needed

#### v0.3.29: Final Colab Smoke
- **Goal:** Last validation of Colab compatibility
- **Scope:** All 15 tutorial notebooks in Google Colab environment
- **Validation target:**
  - All notebooks execute in Colab without manual intervention
  - All outputs display correctly
  - No path or API compatibility issues
- **Expected output:** Colab compatibility checklist, final notes

#### v0.3.30: Final Atlas Bundle
- **Goal:** Create publishable v0.3 tutorial atlas artifact
- **Scope:** All 15 tutorials + manifests + figures + docs
- **Validation target:**
  - All files present and checksummed
  - Archive integrity valid
  - Documentation complete and navigable
  - License/attribution clarity
- **Expected output:** v0.3-atlas.tar.gz with manifest

#### v0.3.31: v0.3 Series Postmortem / Possible Final PyPI Checkpoint
- **Goal:** Document lessons learned, identify future roadmap items, final truth status
- **Scope:** All 30 prior scenarios + 16 audit phases
- **Validation target:**
  - All truth claims validated or explicitly deferred
  - All blockers identified and documented
  - Roadmap updated (future phases: v0.4, v0.5, etc.)
  - Package quality gates met
- **Expected output:** Postmortem markdown, final roadmap, packaging decision
- **PyPI decision:** After all audits PASS + manuscript alignment confirmed, decide final release label: v0.3.31 as "v0.3.31-complete-tutorial-atlas" or v0.3 (latest minor bump)

---

## 7. Proposed PyPI Release Checkpoints

**Important:** Do NOT execute these checkpoints without explicit approval. Decision gates:

### Checkpoint A: v0.3.11 (First phase-1 content checkpoint)
- **Triggers after:** v0.3.3 through v0.3.11 all PASS + manifests valid + collector validates
- **Label suggestion:** `jaxfne==0.3.11` with PyPI tag "v0.3.11-tutorial-atlas-phase-1"
- **Rationale:** First stable multi-scenario tutorial block with spectrolaminar baseline
- **Decision required:** Include or skip?

### Checkpoint B: v0.3.21 (After audit phase-2)
- **Triggers after:** v0.3.20 audit PASS + performance benchmarks PASS
- **Label suggestion:** `jaxfne==0.3.21` with PyPI tag "v0.3.21-tutorial-atlas-phase-2-audited"
- **Rationale:** Consolidated audit ensures quality and consistency
- **Decision required:** Include or skip? Skip v0.3.11 and release here instead?

### Checkpoint C: v0.3.31 (Final atlas completion)
- **Triggers after:** v0.3.30 final atlas bundle PASS + v0.3.31 postmortem PASS
- **Label suggestion:** `jaxfne==0.3.31` with PyPI tag "v0.3.31-complete-tutorial-atlas" OR release as `jaxfne==0.3` (if moving to stable minor version)
- **Rationale:** Complete v0.3 tutorial atlas with full validation
- **Decision required:** Release v0.3 or v0.3.31? Single release or three-phase (v0.3.11, v0.3.21, v0.3.31)?

**Default:** No releases without explicit user approval. v0.3 remains docs/tutorial engineering on top of v0.2.30 until decision.

---

## 8. Open Questions for Human Review

1. **PyPI release strategy:** Single v0.3 release after v0.3.31, or three-phase releases at v0.3.11, v0.3.21, v0.3.31?

2. **Tutorial asset distribution:** Should PNG figures and manifest JSON files be included in PyPI package distribution, or docs-only (committed to repo but not packaged)?

3. **Plotly HTML storage:** Should Plotly HTML be committed to repo or generated dynamically during docs build?

4. **Smoke runner path mismatch:** Current smoke runner in v0.2.30 expects certain paths. Should v0.3 unify path conventions before v0.3.3 starts?

5. **Public API for multi-neuron networks:** Should `with_emitter_parameters()` be treated as accepted public API for general use, or remain tutorial-local helper? If public, should we add to `__init__.py` exports?

6. **Two-neuron helper (v0.3.3):** Should we create a `build_two_neuron_ei_pair()` public helper in core library, or keep it tutorial-local?

7. **Manifest schema versioning:** Current manifest schema is v0.0.21. Should we freeze at v0.0.21 for all v0.3, or allow schema evolution (v0.0.22, v0.0.23)? If evolution, what's the compatibility story?

8. **Colab notebook portability:** For v0.3 notebooks, should we auto-generate `.colab.ipynb` variants with magic commands, or provide manual instructions?

9. **Truth gates standardization:** Should we formalize acceptance_gates.yaml as a durable schema artifact with version control, or treat it as tutorial-scoped documentation?

10. **Backwards compatibility:** If v0.3.x tutorials require API changes to core jaxfne (e.g., new receptor spec format), how do we handle v0.2.30 codebase compatibility? Separate v0.3 branch or same main?

---

## 9. Validation After Planning

**Repository state (pre-commit validation):**
- Compileall: ✓ PASS
- Pytest: ✓ 981 passed, 44 skipped
- Collector: ✓ 2 PASS (v0301, v0302), 0 FAIL
- Manifest validity: ✓ All JSON valid
- Figure count: v0301 (2 PNG), v0302 (2 PNG)
- Overclaim audit: ✓ No overclaiming found
- Canonical import audit: ✓ No typos (jtnfe, jtFNE) in examples
- Visualization doctrine: ✓ Linked, Phase V rule established

**This planning document:**
- 7 sections (current state, workflow, visualization rule, scenarios, audits, PyPI strategy, open questions)
- 30 scenario definitions (v0.3.3–v0.3.31)
- 3 proposed PyPI checkpoints (v0.3.11, v0.3.21, v0.3.31)
- 10 open questions for human decision

---

## 10. Next Safe Actions

1. **Immediate (human review):**
   - Read this handoff
   - Decide on PyPI checkpoint strategy (sections 7–8)
   - Address open questions (section 8)

2. **After approval:**
   - Create v0.3.3 implementation prompt based on section 4 (v0.3.3 template)
   - Kick off v0.3.3 two-neuron E/I pair scenario
   - Follow standard workflow (section 2) for every subsequent tutorial

3. **Ongoing:**
   - Reference this handoff throughout v0.3.3–v0.3.31 for consistency
   - Update this document if major decisions shift scope or timelines
   - Report any scenario blockers to GitHub Project gamma for resolution

---

**Document prepared:** 2026-05-24  
**Baseline validation:** All prior phases (v0.3.1, v0.3.2, visualization doctrine) confirmed
**Status:** Ready for v0.3.3 kick-off pending human review and decision responses
