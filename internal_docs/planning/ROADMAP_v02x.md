---
status: public
version_scope: v0.2.3-onwards
last_updated: 2026-05-20
next_review: 2026-06-30
---

# jaxfne Public Roadmap: v0.2.4–v0.2.21

## Executive Summary

**Current state:** jaxfne v0.2.3 (released 2026-05-19) is a stable computational scaffold providing eight proxy operators, standardized JSON-safe output contracts, and frozen claim-status metadata across all readouts. The package supports compact JAX-native TFNE (Tensor-Field Neural Equations) workflows for laminar proxy simulations with explicit scope limitations.

**Vision:** The v0.2.4–v0.2.21 roadmap expands jaxfne along three strategic pillars:

1. **Field/proxy mathematics and admissibility diagnostics** — Clearer source-field contracts, source-conservation diagnostic metadata, and field-solver status reporting.
2. **Calibration specification and reporting** — Explicit calibration specifications, validation workflows, and reference calibration examples for future empirical or physical-unit workflows.
3. **Colab-ready multimodal tutorials** — Five executable tutorial notebooks covering single-neuron models, two-neuron E/I networks, 100-neuron laminar E/I networks, V1 six-layer column, and V1-PFC dual-area hierarchies.

**Scope:** jaxfne remains a computational scaffold throughout v0.2.x. All outputs carry immutable validation metadata declaring computational intent: proxy readouts, no empirical calibration, no biological mechanism claims, no whole-brain PDE solver. Physical-amplitude or mechanism-level claims are reserved for future versions with empirical validation, calibration evidence, geometry, and ablation studies.

---

## Strategic Themes & Public Vocabulary

### Approved Public Vocabulary

jaxfne v0.2.x uses precise, computation-focused terminology:

- **validation metadata:** JSON-safe fields reporting operator status, assumptions, and claim limitations.
- **claim-status metadata:** Frozen immutable fields declaring truth mode, claim level, solver status, and physical amplitude eligibility.
- **proxy readout:** A declared computational operator (e.g., `lfp_proxy`, `csd_proxy`) that simulates readout behavior without empirical calibration or solver validation.
- **simulated proxy operator:** An operator exposing internal state or derived quantities without biological validation or physical calibration.
- **computational scaffold:** A package designed for reproducible model construction, testing, and report generation—not biological mechanism validation.
- **calibration specification:** A structured declaration of required empirical data, validation targets, and physical-unit mapping for future calibration workflows.
- **reproducible output bundle:** A JSON-safe manifest containing signals, metadata, receipts, and readouts suitable for audit trails and cross-model comparison.

### Operator Terminology Convention

All simulated field-like operators use the `-proxy` suffix to declare computational intent:
- `lfp_proxy` (declares a computational proxy, not an empirically validated extracellular potential)
- `csd_proxy` (declares a computational proxy, not an empirically validated current-source density)
- `eeg_proxy` (declares a computational proxy, not an empirically validated scalp-surface potential)
- `meg_proxy` (declares a computational proxy, not an empirically validated magnetometer readout)

This terminology explicitly prevents conflation with empirically validated readouts and clarifies operator status in all public documentation.

### Claim-Status Fields

All operators expose conservative default claim-status metadata:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
source_calibration_status: uncalibrated_izhikevich_native_current
physical_amplitude_claim_allowed: false
```

These fields declare the default v0.2.x scope (proxy workflows, no empirical calibration). When a run supplies calibration, source-conservation, geometry, solver residuals, and validation evidence, these fields may legitimately change to reflect stronger physical claims. Fields are visible in JSON-safe output metadata only, never in narrative documentation.

---

## Scientific Boundaries

### No Physical Amplitude Claims Without Evidence

jaxfne does not claim physical EEG, MEG, LFP, or CSD amplitudes without:
- Empirical calibration datasets
- Source conservation diagnostics and tolerance specifications
- Boundary condition validation (mean-zero gauge verification)
- Symmetric positive-definite (SPD) field tensor validation
- Solver residual reports
- Cross-model null tests and ablations

### Literature References as Technical Templates

The laminar profile roadmap uses literature-derived technical references, including **Lichtenfeld et al. (2024)** for macaque laminar cell-type organization and **Mendoza-Halliday et al. (2024)** for spectrolaminar LFP power motifs. These references guide declared template design and validation metadata; they are not presented as reproduction claims.

**Full citations:**

Lichtenfeld, M. J., Mulvey, A. G., Nejat, H., Xiong, Y. S., Carlson, B. M., Mitchell, B. A., Mendoza-Halliday, D., Westerberg, J. A., Desimone, R., Maier, A., Kaas, J. H., & Bastos, A. M. (2024). *The laminar organization of cell types in macaque cortex and its relationship to neuronal oscillations*. bioRxiv. https://doi.org/10.1101/2024.03.27.587084

Mendoza-Halliday, D., Major, A. J., Lee, N., et al. (2024). *A ubiquitous spectrolaminar motif of local field potential power across the primate cortex*. Nature Neuroscience, 27, 547–560. https://doi.org/10.1038/s41593-023-01554-7

### Conservative Default Claim-Status Across v0.2.x

All phases from v0.2.4 through v0.2.21 use the same conservative default claim-status metadata unless explicit calibration and validation evidence justify stronger claims. Physical amplitude eligibility, solver status, and scope remain at default levels throughout v0.2.x; v0.3.x may introduce optional calibration workflows that update these fields.

---

## Phase Breakdown: v0.2.4–v0.2.21

### v0.2.4: Field/proxy mathematics hardening I

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Clarify source-field interface contracts and add source-conservation diagnostic metadata.

**Tasks:**
- Document source projection modes (proxy_reduced_emitter, custom_declared_source, empirically_calibrated)
- Add source_conservation_status field to field metadata (not_applicable_proxy_mode, assumed_conserved_pending_validation, validated_within_tolerance_X)
- Define and export source-conservation test cases (constant-current validation, monopole/dipole checks)
- Add solver assumption declarations to field report

**Acceptance Criteria:**
- Source projection modes enumerated and exported in core.py
- Field metadata includes source_conservation_status declaration
- Test cases for source conservation validate and report tolerance specifications
- All tests pass (61 baseline + new conservation tests)
- No new imports required; no external solver dependencies

---

### v0.2.5: Calibration operator abstraction

**Strategic Pillar:** Calibration specification and reporting  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Define abstract calibration specification interface for future empirical workflows.

**Tasks:**
- Design CalibrationSpec class (target metrics, validation datasets, physical-unit mappings, tolerance specs)
- Add calibration_spec field to Configuration (optional, None by default)
- Implement calibration_status method on Model (returns uncalibrated | pending_validation | validated_within_spec)
- Create calibration report template (JSON schema, sample output)
- Document calibration workflow for future v0.3.x implementations

**Acceptance Criteria:**
- CalibrationSpec class defined with required fields
- Model.calibration_status() returns correct enum
- All current configs remain valid (backward compatible)
- Calibration report template is valid JSON schema
- Tests pass

---

### v0.2.6: Field/proxy mathematics hardening II

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Add boundary condition and gauge validation metadata.

**Tasks:**
- Add boundary_condition_status field to field metadata (declared_mean_zero_neumann, custom_declared, validated_residual_bound)
- Add gauge_status field (declared_mean_zero_gauge, custom_declared, validated_sink_free)
- Implement boundary/gauge validation tests (verify zero-mean property, sink-free verification)
- Export boundary/gauge assumptions in field report

**Acceptance Criteria:**
- Boundary and gauge fields appear in all field metadata
- Validation tests report tolerance and residual norms
- All proxy field configurations declare boundary/gauge status
- Tests pass

---

### v0.2.7: Tutorial notebook standard

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Establish consistent structure and metadata for all tutorial notebooks.

**Tasks:**
- Define tutorial template (title, learning objectives, prerequisites, scope limitations, setup, execution, outputs, closing remarks)
- Create shared utility library for tutorial imports and reproducibility (seed management, print helpers, result formatting)
- Document scope/truth statement template for all tutorials (e.g., "This tutorial is exploratory; outputs are proxy readouts without empirical validation")
- Establish nbconvert test harness (execute all tutorials, validate outputs, report execution time)

**Acceptance Criteria:**
- Tutorial template documented in docs/tutorial_template.md
- Utility library exports standard functions
- All templates include explicit scope limitation statement
- Tutorial test harness passes for existing notebooks

---

### v0.2.8: Tutorial 1 – single-neuron Colab

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create first executable tutorial: single-neuron Izhikevich model in Colab.

**Tasks:**
- Create `tutorials/01_single_neuron_izhikevich.ipynb`
- Show emitter configuration, simulation setup, spike output, voltage trace, source projection
- Include minimal field readout (lfp_proxy sampling)
- Generate and display readout report (JSON structure, claim-status fields)
- Colab-ready: include `%pip install jaxfne` cell, proper imports

**Acceptance Criteria:**
- Notebook executes end-to-end without errors
- Produces spikes, voltage, source, lfp_proxy outputs
- Reports claim-status metadata with immutable fields
- Includes explicit scope statement (proxy readouts, no calibration)
- nbconvert execution passes

---

### v0.2.9: Tutorial 2 – two-neuron E/I Colab

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create second tutorial: two-neuron excitatory-inhibitory circuit.

**Tasks:**
- Create `tutorials/02_two_neuron_ei_circuit.ipynb`
- Show excitatory and inhibitory neuron configuration (E: RS, PV: FS)
- Define synaptic connectivity matrix (AMPA, GABA_A)
- Run simulation, display spike rasters and voltage traces
- Extract and compare source projections for E vs. PV neurons
- Generate combined readout report

**Acceptance Criteria:**
- Notebook executes end-to-end
- Shows E/I spike patterns and voltage differences
- Source projections are extracted and compared
- Readout report includes both neurons
- Colab-compatible

---

### v0.2.10: Tutorial 3 – 100-neuron E/I network

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create third tutorial: balanced E/I network with population metrics.

**Tasks:**
- Create `tutorials/03_100_neuron_ei_network.ipynb`
- Build 100-neuron network (80 E, 10 PV, 10 SST; random connectivity)
- Run simulation, compute population spike rates and synchrony metrics
- Extract and display layer-resolved lfp_proxy and csd_proxy
- Generate multi-metric readout report (spike_rate_hz, lfp_abs_mean, csd_abs_mean)
- Include network visualization and population-level statistics

**Acceptance Criteria:**
- Notebook executes (may require slightly longer runtime ~30–60s)
- Population metrics computed (mean rate, rate SD, sync)
- Field readouts extracted at multiple depths
- Multi-metric readout report generated
- Colab-compatible (may suggest GPU)

---

### v0.2.11: Tutorial 4 – V1 six-layer column

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create fourth tutorial: realistic V1 laminar column with layer-specific populations.

**Tasks:**
- Create `tutorials/04_v1_six_layer_column.ipynb`
- Define six laminar populations (L1, L2/3 E+PV, L4 E+PV, L5 E+PV, L6 E+PV) with ~400 neurons
- Use Lichtenfeld/Mendoza-Halliday profile templates for cell-type distribution
- Define laminar-specific connectivity (feedforward L4→L2/3, feedback L5/L6, local E→PV/SST)
- Run simulation with visual stimulus proxy (thalamic drive)
- Extract depth-resolved lfp_proxy, csd_proxy, and per-layer spike rates
- Generate layer-resolved readout report with claim-status metadata

**Acceptance Criteria:**
- Notebook executes (may require 1–2 min on CPU)
- Layer populations defined and executed
- Laminar readouts extracted and visualized
- Per-layer readout metrics computed
- References to Lichtenfeld/Mendoza-Halliday included as technical template declarations
- Includes explicit scope statement (proxy simulation, no validation)

---

### v0.2.12: Field/probe figure utilities

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create reusable visualization utilities for field and probe readouts.

**Tasks:**
- Create `jaxfne/viz.py` with functions: plot_laminar_profile, plot_csd_contour, plot_lfp_proxy_heatmap, plot_readout_report
- Add matplotlib helper functions for figure formatting, color palettes, axis labeling
- Document scope limitations in plot titles (e.g., "proxy readout, not empirically validated")
- Add examples to docstrings
- Create `docs/figure_gallery.md` with sample plots

**Acceptance Criteria:**
- All plot functions accept signals, field, metadata inputs
- Generated figures include scope disclaimers
- Functions tested with tutorial data
- Docstrings include usage examples

---

### v0.2.13: Lichtenfeld/Mendoza-Halliday profile data abstraction

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create profile template module abstracting cell-type organization from literature references.

**Tasks:**
- Create `jaxfne/profiles.py` with CorticalProfile dataclass (layer_names, cell_types, fractions_by_layer, depth_bounds_um, reference_citations)
- Implement ProfileTemplate class for Lichtenfeld/Mendoza-Halliday macaque V1 template (L1–L6, E/PV/SST/VIP distributions)
- Add ProfileTemplate.to_population_list() returning list of LaminarPopulation objects compatible with LaminarSourceGeometry
- Include reference citations as metadata fields (author, year, DOI)
- Add validation tests (fractions sum to 1, layer depths are valid)

**Acceptance Criteria:**
- ProfileTemplate class defined and documented
- V1 Lichtenfeld/Mendoza-Halliday template is complete and validated
- Template citations are included and formatted correctly
- Can generate LaminarPopulation lists for use in Configuration
- Tests pass

---

### v0.2.14: Tutorial 5 – V1-PFC dual-column

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create fifth and most complex tutorial: two interconnected columns (V1 and PFC) with inter-area connections.

**Tasks:**
- Create `tutorials/05_v1_pfc_dual_column.ipynb`
- Build V1 (six-layer, ~400 neurons) and PFC (three-layer simplified, ~300 neurons) columns
- Define inter-area connections (V1 L5 → PFC, PFC L5 → V1 L1)
- Simulate oddball-like task with V1 visual drive and PFC top-down modulation
- Extract and compare field readouts across areas and layers
- Generate dual-area readout report
- Demonstrate readout_spec filtering and per-area metrics

**Acceptance Criteria:**
- Notebook executes (2–3 min runtime expected)
- Dual columns simulated with inter-area connectivity
- Per-area readout metrics extracted
- Report demonstrates readout_spec usage and filtering
- Scope statement included

---

### v0.2.15: Tutorial smoke runner

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create automated test harness for executing all tutorials and validating outputs.

**Tasks:**
- Create `scripts/tutorial_smoke_test.py`
- Iterate all tutorial notebooks, execute via nbconvert, capture outputs
- Validate: no errors, all outputs present, readout reports are JSON-safe, execution times logged
- Generate smoke-test report (pass/fail per tutorial, execution times, warnings)
- Add CI integration hook (optional: can be run manually or in GitHub Actions)

**Acceptance Criteria:**
- Script executes all five tutorials
- Reports pass/fail status
- Execution times logged
- Report is machine-readable JSON
- All tutorials pass smoke test

---

### v0.2.16: Documentation site MVP

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 2–3 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create lightweight public documentation site (README pointers + docs/ markdown + generated API reference).

**Tasks:**
- Organize `docs/` structure: overview.md, installation.md, tutorials.md, api_reference.md, roadmap.md
- Generate API reference from docstrings (Configuration, Model, Simulation, ReadoutSpec, ReadoutResult)
- Create quick-start guide (minimal example, explanation, scope statement)
- Add FAQ section addressing claim-status questions (what's proxy readout, when can I claim physical amplitude, etc.)
- Create docs/ROADMAP.md (this document)

**Acceptance Criteria:**
- docs/ structure is organized and navigable
- API reference is complete and linked in README
- Quick-start guide is executable
- FAQ addresses common claim-status questions
- All markdown syntax is valid

---

### v0.2.17: Calibration examples & reference workflow

**Strategic Pillar:** Calibration specification and reporting  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Provide concrete calibration examples and reference workflows for future empirical validation.

**Tasks:**
- Create `examples/calibration_reference_workflow.py` demonstrating:
  - Load v0.2.x uncalibrated simulation results
  - Define CalibrationSpec with target dataset, physical-unit mapping, tolerance specs
  - Implement mock calibration procedure (scale factors, sign corrections)
  - Generate before/after calibration reports
- Document calibration workflow in `docs/calibration_guide.md`
- Create calibration_status_examples.md with sample manifest fragments

**Acceptance Criteria:**
- Reference workflow example is complete and documented
- CalibrationSpec usage is clear
- Before/after reports are generated
- Calibration guide explains workflow steps

---

### v0.2.18: Operator status export & audit CLI

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Create CLI tool and export functions for auditing operator status across models.

**Tasks:**
- Implement Model.export_operator_status() returning dict of all operators + claim-status metadata
- Create CLI: `jaxfne audit-operators config.yaml` reporting all operator status fields
- Add `jaxfne audit-manifest manifest.json` validating manifest schema and claim-status consistency
- Export as JSON and CSV for spreadsheet import

**Acceptance Criteria:**
- export_operator_status() returns complete operator inventory
- CLI is functional and documented
- Output is JSON-safe and includes all claim-status fields
- Validation catches schema violations

---

### v0.2.19: Package consistency audit

**Strategic Pillar:** Field/proxy mathematics and admissibility diagnostics  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Verify consistency of claim-status metadata, terminology, and scope statements across codebase and documentation.

**Tasks:**
- Run vocabulary audit (grep for forbidden internal terms in public docs)
- Verify all operator reports include immutable claim-status fields
- Check Configuration/Model docstrings for consistent scope language
- Audit example code for compliant readout report handling
- Generate consistency report (pass/fail, findings summary)

**Acceptance Criteria:**
- No forbidden internal terms found in public documentation
- All operators report immutable claim-status fields correctly
- Docstring language is consistent
- Examples follow scope statement conventions
- Consistency report shows pass status

---

### v0.2.20: Release candidate

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1–2 weeks  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Prepare v0.2.20 release candidate with final validation.

**Tasks:**
- Update CHANGELOG.md with all v0.2.4–v0.2.20 features and improvements
- Run full test suite: unit tests (61 baseline + new tests), tutorial smoke tests, validation audit
- Generate release notes emphasizing roadmap progress (three pillars achieved)
- Bump version to 0.2.20 in pyproject.toml and jaxfne/core.py
- Create git tag v0.2.20 (candidate, not yet released)

**Acceptance Criteria:**
- All tests pass
- Tutorial smoke tests pass
- Consistency audit shows pass status
- Release notes document all changes
- Version is bumped correctly

---

### v0.2.21: Consolidated practical scaffold release

**Strategic Pillar:** Colab-ready multimodal tutorials  
**Estimated Duration:** 1 week  
**GitHub Tracking:** TODO: create issue/card

**Goal:** Final release of v0.2.x line as a mature, documented, reproducible computational scaffold.

**Tasks:**
- Finalize release notes and API documentation
- Publish PyPI release (v0.2.21)
- Create GitHub Release with v0.2.21 tag
- Update README.md with current version and roadmap link
- Archive v0.2.x branch (keep main clean for v0.3.x development)
- Announce release with brief summary of three strategic pillars completed

**Acceptance Criteria:**
- v0.2.21 published to PyPI
- GitHub Release created with release notes
- README reflects v0.2.21 release
- All documentation is current and linked
- v0.2.x development closes; v0.3.x planning begins

---

## Appendix A: Claim-Status Metadata Reference

All jaxfne v0.2.x outputs include these conservative default claim-status fields:

| Field | Default Value | Meaning |
|-------|-------|---------|
| `truth_mode` | `truth_safe_unverified` | Computational scaffold; no biological mechanism validation |
| `claim_level` | `computational_scaffold` | Package is designed for model construction and testing, without mechanism validation |
| `source_calibration_status` | `uncalibrated_izhikevich_native_current` | Sources are declared emitter state; no physical current units |
| `field_solver_status` | `laminar_proxy_no_pde` | Field is a proxy projection, not a PDE solution |
| `field_claim_level` | `proxy_readout_only` | Field readouts (LFP, CSD) are computational proxies |
| `physical_amplitude_claim_allowed` | `false` | No amplitude claims without empirical calibration and validation |

These conservative defaults appear in all operator reports, receipts, and manifests throughout v0.2.x. Future runs with calibration evidence may legitimately update these fields when warranted.

---

## Appendix B: Public Vocabulary Quick Reference

### Approved Terminology

| Term | Usage | Example |
|------|-------|---------|
| proxy readout | Computational operator output | "lfp_proxy is a proxy readout without PDE validation" |
| simulated proxy operator | Declared operator without empirical validation | "All eight operators are simulated proxy operators in v0.2.x" |
| computational scaffold | Framework for model construction | "jaxfne is a computational scaffold for TFNE workflows" |
| validation metadata | JSON-safe operator status | "The probe report includes validation metadata" |
| claim-status metadata | Immutable scope fields | "All operators freeze claim-status metadata" |
| source-conservation diagnostics | Validation checks | "v0.2.4 adds source-conservation diagnostics" |
| calibration specification | Abstract interface for future calibration | "CalibrationSpec defines target datasets and tolerances" |
| Lichtenfeld et al. (2024) | Technical reference for laminar cell types | "Tutorial 4 uses Lichtenfeld et al. templates" |
| Mendoza-Halliday et al. (2024) | Technical reference for spectrolaminar LFP | "v0.2.13 implements Mendoza-Halliday profile abstraction" |

jaxfne development maintains strict public vocabulary discipline. All internal documentation and development discussions use approved terminology. This ensures consistent language across public-facing materials, user examples, and contributor guidelines.

---

**Document version:** v0.2.3-onwards  
**Status:** Public roadmap for v0.2.4–v0.2.21 development  
**Last updated:** 2026-05-20  
**Next review:** 2026-06-30  
**Feedback:** Issues and comments welcome via GitHub Project gamma
