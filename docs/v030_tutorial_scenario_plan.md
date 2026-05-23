# v0.3 Tutorial-Scenario Plan

**Status:** v0.3_TUTORIAL_SCENARIO_DOCTRINE_READY  
**Stable toolbox:** jaxfne == 0.2.30  
**truth_mode:** truth_safe_unverified  
**claim_level:** computational_scaffold  

---

## Mission

The v0.3 line is a **tutorial-scenario implementation line** built on the stable `jaxfne==0.2.30`
toolbox. v0.3.x phases primarily create Colab notebooks, documentation, equations, figures,
proofs, and scenario guides. The package version is **not automatically bumped** during
tutorial work.

**This line is not a constant package-mutation line.**

---

## Canonical Import and Usage

Every v0.3 tutorial must use exactly this alias, everywhere, without exception:

```python
import jaxfne as jtfne
```

**Forbidden aliases:** `jtnfe`, `jtFNE`, `jaxFNE`, `jfne`, or any mixed casing.
Reject all pull requests and tutorials that introduce inconsistent aliases.

### Canonical configuration pattern

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        name="<tutorial_name>",
        kind="cortical_column",
        n=<n_neurons>,
        layers=["L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(name="laminar_probe", modes=["spk", "vm", "source", "lfp_like", "csd_like"], n_contacts=16)
)

model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=<ms>, dt_ms=0.1, seed=0)
signals = model.simulate(sim)
manifest = model.manifest(signals)
```

---

## Package Mutation Policy

| Mode | Trigger | Action |
|------|---------|--------|
| **Default (SCENARIO_DOC_ONLY)** | Tutorial is docs/notebooks/figures | Use `jaxfne==0.2.30`; no package change |
| **SCENARIO_EXECUTION_BLOCKED_BY_PACKAGE_BUG** | Tutorial execution raises an error in a public API | Stop; file minimal bug report; fix only the bug |
| **SCENARIO_REQUIRES_PACKAGE_PATCH** | Required public tool is missing or API/docs mismatch blocks reproducible use | Scope-minimal patch; bump patch version; re-verify |
| **SCENARIO_READY_FOR_COLAB** | Tutorial runs end-to-end on PyPI-installed package | Green for Colab publication |
| **SCENARIO_READY_FOR_BETA** | Tutorial passes BETA audit (independent read-only check) | Green for docs release |

### What does NOT require a package bump

- Adding tutorial notebooks
- Adding or updating documentation files
- Adding figures, equations, or prose explanations
- Adding test scripts that use the public API as-is
- Fixing documentation errors (typos, stale version refs)
- Adding new example scripts that work with the existing API

### What DOES require a package bump

- A public API call raises an unexpected exception in a tutorial
- A documented parameter does not exist or has the wrong signature
- A required manifest field is missing or malformed
- A JSON-safe guarantee is violated

---

## v0.3.0–v0.3.31 Tutorial Spine

Each phase is a scenario unit. Most phases produce:
- One Colab notebook (or draft)
- One or more documentation patches
- Figures from real simulation data (not manually created)
- A manifest receipt with frozen claim gates

| Phase | Name | Focus |
|-------|------|-------|
| **v0.3.0** | Scenario Doctrine and Tutorial Template | Doctrine doc, template, canonical import, package mutation policy |
| **v0.3.1** | Single Izhikevich Neuron | Single-neuron dynamics, spiking, parameter table, claim gates |
| **v0.3.2** | Single Neuron Parameter Sweep | Sweep a/b/c/d over grid; show regime transitions; no overclaims |
| **v0.3.3** | E/I Two-Neuron | E→I inhibition, mutual coupling, synchrony |
| **v0.3.4** | Small Recurrent E/I Network | 20–50 neurons; recurrent dynamics; spike rasters |
| **v0.3.5** | 100-Neuron E/I Cortical Population | Full population dynamics; cell-type breakdown |
| **v0.3.6** | Laminar Column Basis | Depth axis, contact layout, what laminar means in this framework |
| **v0.3.7** | Source Projection and Source Bookkeeping | Source modes, double-count guard, `source_bookkeeping` manifest fields |
| **v0.3.8** | LFP-like and CSD-like Readout | Proxy readouts, kernel normalization, what they do NOT claim |
| **v0.3.9** | EEG-like and MEG-like Proxy | Forward-projection proxy; explicit non-claims for real EEG/MEG |
| **v0.3.10** | EMM-proxy and Conservation Diagnostics | EMM operator, source norms, field-gradient proxy, diagnostic reports |
| **v0.3.11** | Spectrolaminar Baseline Motif | Laminar depth × frequency motif; alpha/beta/gamma proxy |
| **v0.3.12** | Evoked L4 Drive | Thalamocortical drive scaffold; L4 excitatory burst; propagation |
| **v0.3.13** | Omission / Oddball Scaffold | Global oddball paradigm; omission-response scaffold |
| **v0.3.14** | Null Models and Shuffled Controls | Spike-shuffled, phase-shuffled baselines; what controls test |
| **v0.3.15** | Ablation Tutorial | Cell-type knockout; lesion scaffold; gain manipulation |
| **v0.3.16** | Synchrony Anti-Seizure Gates | Synchrony metrics; synchrony-gating patterns; what is not seizure proof |
| **v0.3.17** | Objective Report | Smoke, rhythm, synchrony objectives; manifest scoring |
| **v0.3.18** | Optimizer / GSDR-AGSDR | Fitness landscape; custom optimizer loop; Optax optional |
| **v0.3.19** | Runtime, dtype, Seed Reproducibility | Deterministic seeds; dtype effects; backend consistency |
| **v0.3.20** | Manifest / JSON Safety / Asset Hash | Full manifest walkthrough; JSON-safety audit; figure hashing |
| **v0.3.21** | Basis Collapse: 1D, 2D, 3D, Laminar | BasisSpec; collapse rules; declared vs implemented field regimes |
| **v0.3.22** | Tensor-Network Ancestry / Basis Transform | Pellionisz/Llinás context; covariant tensor basis; distinction from ML |
| **v0.3.23** | Jaxley Bridge Conceptual Tutorial | Array-first bridge; voltage proxy doctrine; no scope drift |
| **v0.3.24** | Multi-Area Cortical Scaffold | Two-area E/I hierarchy; feedforward and feedback pathways |
| **v0.3.25** | Dense Laminar Multi-Area | Combined laminar × multi-area; spectrolaminar across areas |
| **v0.3.26** | Performance Benchmark | Benchmark script; wall-clock model; claim boundaries |
| **v0.3.27** | Tutorial Figure Regeneration Audit | Re-run figure generation; hash verification; visual confirmation |
| **v0.3.28** | Colab Compatibility Audit | All tutorials runnable from Colab; first-cell install; portability |
| **v0.3.29** | Manuscript-Equation Alignment Audit | Map every public equation to code; identify gaps; patch or gate |
| **v0.3.30** | Biophysics Tutorial Complete Pass | End-to-end from Izhikevich dynamics to laminar field proxy; full chain |
| **v0.3.31** | v0.3 Series Post-Audit and Release-Readiness Report | BETA audit of all v0.3 tutorials; decision gate for v0.4 planning |

---

## Scope Constraints

### No new solvers during tutorial phases

The following are **not** in scope for any v0.3.x phase:

- Poisson field solver
- Maxwell field solver
- Stress-energy tensor
- Poynting flux
- Conductivity tensor (physical, calibrated)
- Physical amplitude claims
- Biological proof-of-mechanism claims

These require a separate approved phase with design review, benchmark plan, and
calibration evidence.

### No new probe operators or readout modalities

All readout operators already exist in v0.2.30. v0.3 tutorial phases document and
demonstrate them; they do not add new ones without a dedicated package-patch phase.

### No matplotlib in core modules

Figures are generated in examples, scripts, and tutorial notebooks only.
No `matplotlib` or `plotly` import belongs in `jaxfne/core.py` or any
other core module.

---

## Claim Gates (Frozen for all v0.3 tutorials)

Every v0.3 tutorial manifest must satisfy:

```python
manifest["truth_mode"] == "truth_safe_unverified"
manifest["claim_level"] == "computational_scaffold"
manifest["physical_amplitude_claim_allowed"] == False
manifest["field_solver_status"] == "laminar_proxy_no_pde"
manifest["field_claim_level"] == "proxy_readout_only"
```

The `json.dumps(manifest, allow_nan=False)` call must succeed without error.

---

## GitHub Releases Note

After every completed tutorial-scenario phase:

- Commit to `main` with message `docs: v0.3.N <short description>`
- Do **not** create a git tag unless the package version itself changed
- Do **not** create a PyPI release for docs-only phases

If a package patch is required:
1. Bump version in `pyproject.toml` and `jaxfne/core.py`
2. Update all version-asserted tests
3. Build and upload to PyPI
4. Tag the commit `vX.Y.Z`
5. Proceed with the tutorial after verification

---

## Related Documents

- [Tutorial Template](tutorial_template_v030.md) — required structure for every v0.3 notebook
- [v0.3 Readiness Bridge](v03_bridge.md) — locked APIs and migration from v0.2.27
- [Computation Basis](computation_basis.md) — collapsible tensor-field scaffold
- [Performance Baseline](performance_baseline.md) — v0.2.30 benchmark receipts
- [Release Checklist](RELEASE_CHECKLIST.md) — build, tag, upload workflow
- [Packaging Guide](packaging.md) — wheel/sdist build and PyPI procedures
