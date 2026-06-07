# Nature Methods publication roadmap — jaxfne / TFNE

**Branch policy:** Cursor publication track on `cur`. This document is planning and evidence infrastructure only — not a manuscript draft.

**Baseline SHA (synced):** `f77822e` — `main = dev = agy = cur` at time of roadmap creation.

**Canonical import:** `import jaxfne as jtfne`

---

## 1. Target (one paragraph)

jaxfne is a reproducible, JAX-native software method for Tensor-Field Neural Equations (TFNE): an explicit source-to-field/readout scaffold for computational neurophysiology. The Nature Methods submission target is a **software/methods paper** that makes emitter dynamics, source bookkeeping, proxy field operators, probe readouts, validation gates, and artifact manifests **executable, auditable, and reproducible** — not a claim of validated biological simulation or calibrated EEG/MEG forward modeling.

---

## 2. Manuscript thesis

jaxfne makes source-to-field assumptions explicit, executable, auditable, and reproducible. The contribution is **method transparency**: typed configuration, deterministic seeds, JSON-safe manifests, probe operator contracts, and tutorial-scale evidence pipelines that let reviewers inspect what was computed, what was *not* solved, and what would be required to escalate claims.

---

## 3. Currently supported (evidence-backed scope)

| Capability | Status | Evidence location |
|------------|--------|-------------------|
| Deterministic JAX simulations | supported | `jtfne.simulate`, `tests/test_api_smoke.py` |
| Typed / chainable configuration | supported | `jtfne.Configuration`, `tests/test_config_*` |
| Source and probe JSON reports | supported | `jaxfne/fields/probes.py`, `tests/test_probe_operators_v021.py` |
| Proxy readouts: spk, vm, source | supported | `Signals.get`, probe operators |
| Proxy readouts: lfp_like, csd_like | supported | `FieldOutput`, laminar projection |
| Proxy readouts: eeg_like, meg_like, emm_proxy | supported (probe path) | `Model.probe`, `fields/probes.py`; not on core `FieldOutput` |
| Manifests and validation reports | supported | `jaxfne/io.py`, `jaxfne/validation.py` |
| Tutorial notebooks and examples | supported | `tutorials/`, `examples/` |
| Optional lazy Jaxley bridge | supported | `jaxfne/bridges.py`, `tests/test_jaxley_optional_dependency.py` |
| Connectivity compiler (v0.3.30) | supported | `jaxfne/connectivity.py` |
| Population analysis metrics | supported | `jaxfne/analysis/metrics.py` |

**Immutable truth gates (every public run report):**

```text
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

---

## 4. Not yet supported (do not claim in manuscript)

- Calibrated EEG/MEG forward modeling (toy leadfields only; `physical_amplitude_claim_allowed: false`)
- Solved 3D PDE / Maxwell / Poisson field dynamics (laminar row-normalized proxy projection only)
- Physical amplitude interpretation or SI-unit calibrated readouts
- Biological mechanism proof or empirical validation against held-out observed datasets
- Metabolic / EMM biological claims (EMM-proxy is normalized within-run activity cost)
- Release-tagged reproducibility bundle (no fixed publication tag/SHA yet)
- Final Nature Methods figure asset pipeline (inventory only in this phase)

---

## 5. Figure plan

### Main figures (Nature Methods style)

| Figure | Purpose | Generator (planned) | Artifacts | Validation gate | Claim boundary |
|--------|---------|---------------------|-----------|-----------------|----------------|
| **Fig 1** | TFNE operator grammar (Emitter→Source→Field→Probe→Objective) | `scripts/publication/fig01_architecture.py` (TBD) | `figures/publication/fig01_tfne_architecture.png`, manifest | compileall + diagram hash | computational scaffold only |
| **Fig 2** | Source/field contracts and truth gates | `scripts/publication/fig02_contracts.py` (TBD) | `fig02_source_field_contracts.png`, JSON schema excerpt | `validate_config`, probe report tests | no PDE solve claim |
| **Fig 3** | jaxfne backend: Config→Net→simulate | `examples/00_minimal_column.py` + panel script | `fig03_jaxfne_backend.png`, run receipt | `test_api_smoke.py` | package-native API only |
| **Fig 4** | Minimal install + 10 s smoke run | `docs/quickstart.md` command block | `fig04_minimal_install_run.png`, stdout log | fresh venv install receipt | no GPU requirement |
| **Fig 5** | Runtime scaling (neurons × contacts × seeds) | `scripts/benchmark_jaxfne.py` (extend) | `fig05_runtime_scaling.png`, CSV table | benchmark receipt w/ hardware | no unsupported speedup claims |
| **Fig 6** | Readout family panel (8 probes) | `examples/04_two_neuron_ei_multimodal.py` | `fig06_readout_family_panel.png`, probe_report.json | `test_probe_operators_v021.py` | all proxy readouts |
| **Fig 7** | Reproducibility artifacts (manifest/hash) | `scripts/publication_inventory.py` | `fig07_reproducibility_artifacts.png`, inventory.json | SHA256 round-trip | fixed tag/SHA required for final |
| **Fig 8** | Adjacent tools comparison (positioning) | manual diagram + citation table | `fig08_adjacent_tools_comparison.png` | literature review checklist | compare capabilities, not superiority |

### Extended Data (1–10)

| ED | Purpose | Generator | Artifact | Gate |
|----|---------|-----------|----------|------|
| ED1 | API stability / public surface | `scripts/snapshot_public_api.py` | `ed01_api_stability.png` | snapshot diff test |
| ED2 | JSON schema / config validation | `validate_config` fixtures | `ed02_json_schema_validation.png` | `test_config_schema_*` |
| ED3 | Notebook execution receipts | `scripts/run_all_tutorials.py` | `ed03_notebook_execution_receipts.png` | notebook CI markers |
| ED4 | Manifest hashes | example output bundles | `ed04_manifest_hashes.png` | `test_artifact_json_safety_*` |
| ED5 | Optional dependency laziness | import-cost tests | `ed05_optional_dependency_laziness.png` | `test_jaxley_optional_*`, `test_root_import_*` |
| ED6 | Benchmark scaling tables | benchmark scripts | `ed06_benchmark_scaling_tables.png` | hardware receipt |
| ED7 | Probe operator contracts | probe unit tests | `ed07_probe_operator_contracts.png` | eight-operator suite |
| ED8 | Tutorial atlas coverage | tutorial manifest | `ed08_tutorial_atlas_coverage.png` | `tutorial_manifest.json` |
| ED9 | Failure modes and nulls | negative tests | `ed09_failure_modes_and_nulls.png` | emitter validation tests |
| ED10 | Release archive receipt | release scripts (approval-gated) | `ed10_release_archive_receipt.png` | tag + sdist/wheel hash |

**Manuscript draft note:** No `.tex` / PDF source found in-repo at roadmap time. Existing alignment doc: `docs/manuscript_alignment.md` (v0.2.27-era; requires update for v0.3.29 gates). Assumed external draft contains TFNE operator grammar, jaxfne v0.3.x surface, TBD slots for version/SHA/tests/figures, and proxy-scope gates.

---

## 6. Evidence matrix

| Row | Current status | Missing evidence | Command to generate | Acceptance criterion |
|-----|----------------|------------------|---------------------|----------------------|
| Package/API | green | publication API snapshot pinned to tag | `python scripts/snapshot_public_api.py` | diff empty vs tag |
| Install | partial | clean-room install log (Linux/macOS) | `pip install jaxfne && python -c "import jaxfne"` | import + version match |
| Minimal run | green | figure panel for Fig 4 | `python examples/00_minimal_column.py` | exit 0, finite outputs |
| Tutorials | partial | executed notebook receipts for all atlas entries | `python scripts/run_all_tutorials.py` | all marked PASS |
| Figures | red | 18 PNG assets (see inventory) | `python scripts/publication_inventory.py` | all `exists: true` |
| Manifests | partial | publication output bundle at fixed SHA | example scripts → `outputs/publication/` | JSON-safe + SHA256 |
| Tests | green | pinned count in manuscript | `pytest tests/ -q` | 2111+ passed, 0 failed |
| Docs | green | publication nav + roadmap | `mkdocs build --strict` | strict pass |
| Benchmarks | red | hardware-tagged tables | `python scripts/benchmark_jaxfne.py` | CSV + env receipt |
| Optional deps | green | lazy-import audit | `python scripts/audit_notebooks_and_assets.py --check` | Jaxley GREEN |
| Examples | green | publication figure stubs | `python scripts/publication_inventory.py` | inventory JSON valid |
| Release archive | red | signed tag, PyPI, Zenodo DOI | approval-gated release scripts | human approval |

---

## 7. Benchmark plan

**Axes (no performance claims until receipts exist):**

| Axis | Levels (planned) | Metric | Receipt field |
|------|------------------|--------|---------------|
| Neuron count N | 1, 2, 100, 1k | wall time (s) | `hardware_cpu`, `jax_device` |
| Laminar contacts Z | 4, 16, 64 | compile count | `compilation_registry` dump |
| Seeds / batch B | 1, 4, 16 | throughput (steps/s) | `duration_ms`, `dt_ms` |
| Readout families | spk, vm, source, lfp, csd, eeg, meg, emm | marginal cost | per-readout timing |
| Runtime mode | eager, jit | speedup ratio | label as **proxy scaffold** only |
| Memory | peak RSS | MB | `psutil` or `/usr/bin/time -v` |
| CPU baseline | x86_64 Linux | reference table | required for all tables |
| Accelerator | optional GPU | optional appendix | never default claim |

**Stop rule:** No manuscript sentence may cite speedup factors until `outputs/publication/benchmark_receipt.json` exists with hardware, package version, and git SHA.

---

## 8. Reviewer-risk register

| Risk | Evidence needed | Safe response |
|------|-----------------|---------------|
| "Is this just toy simulation?" | Tutorial atlas, 100-neuron example, connectivity compiler, analysis metrics | jaxfne is a **reproducible computational scaffold** for explicit source-to-field/readout workflows; scope is tutorial-to-medium scale with auditable metadata. |
| "How is this different from NEURON/Brian/NEST/Jaxley/LFPy/MNE/TVB?" | Fig 8 positioning table, API comparison matrix | jaxfne foregrounds **TFNE source bookkeeping + proxy field/probe contracts + JSON manifests**; does not replace biophysical simulators or empirical forward models. |
| "Are EEG/MEG claims overreaching?" | Probe reports with `physical_amplitude_claim_allowed: false` | All EEG/MEG outputs are **linear-projection proxies** with toy leadfields; not validated against measured data. |
| "Where is empirical validation?" | Explicit "not supported" section + null/failure tests | Empirical validation is **out of scope** for v0.3.x; paper claims reproducible method artifacts, not biological truth. |
| "Can a new user reproduce figures?" | Install docs, quickstart, inventory SHA, fixed tag (TBD) | Reproducibility commands will pin `git checkout <tag>` + `pip install .[dev]` + `scripts/publication_inventory.py`. |
| "Are APIs stable?" | `artifacts/public_api_before.json`, snapshot tests | Public API guarded by snapshot tests; breaking changes require semver + migration notes. |

---

## 9. Submission gap checklist

| Item | Status | Owner / next action |
|------|--------|---------------------|
| Code availability | partial | GitHub public; add exact tag/SHA in manuscript |
| Data availability | N/A (software paper) | State "no empirical dataset required"; synthetic demo data only |
| Software dependencies | partial | Pin `pyproject.toml` + lockfile appendix |
| Install instructions | green | `docs/install.md`, Colab badge |
| Test/demo data | partial | Bundle minimal `.jcfg.json` + seed manifests |
| Reproducibility commands | partial | Add `docs/publication/reproduce.md` (TBD) |
| LLM-use statement | TBD | Placeholder in checklist JSON |
| Author/contribution | TBD | CRediT placeholder |
| Competing interests | TBD | Statement placeholder |
| License | green | MIT — confirm in manuscript |
| DOI / archive | red | Zenodo release after tag approval |
| Methods concision | partial | Map to Nature software reporting expectations |
| Figure source files | red | 0/18 PNG assets present at roadmap time |

---

## 10. Stop rules

1. **No physical amplitude upgrade** without solver, geometry, calibration, boundary/gauge, residual, units, and external comparison evidence.
2. **No release claim** without fixed git tag and SHA recorded in `publication_checklist.json`.
3. **No manuscript performance number** without benchmark command + hardware receipt.
4. **No notebook execution claim** without executed notebook log in `outputs/publication/`.
5. **No figure citation** without `figures/publication/<name>.png` hash in inventory JSON.
6. **No PyPI/TestPyPI/GitHub Release** without explicit human approval.
7. **No branch deletion or force-push** on permanent branches (`main`, `dev`, `agy`, `cur`).

---

## Related files

- Machine-readable checklist: `publication_checklist.json`
- Artifact inventory script: `../../scripts/publication_inventory.py`
- Legacy alignment (needs refresh): `../manuscript_alignment.md`
- Scope doctrine: `../scope_and_limitations.md`
