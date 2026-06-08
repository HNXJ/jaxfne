# jaxfne Alpha Checkpoint Handout

**Checkpoint label:** `alpha-publication-artifact-stack`  
**Intended use:** attach this handout plus a repo zip to an external reviewer (GPT or human) for deep, critical, scoreboard-precise review against Nature Methods–style **software/methods** publication criteria.  
**Not intended use:** biological mechanism proof, calibrated EEG/MEG validation, or release authorization.

---

## 1. Freeze receipt

| Item | Value |
|---|---|
| Repository | `https://github.com/HNXJ/jaxfne` |
| Branch | `cur` |
| Commit SHA | verify live: `git rev-parse HEAD` on `cur` (do not cite handout SHA without re-freeze) |
| Package version | `0.3.29` |
| Checkpoint date (UTC) | `2026-06-07` |
| Main figures | `8/8` |
| Extended Data | `10/10` |
| Publication artifact stack | **complete** (ED1–ED10) |
| Release/tag/publish/archive | **not executed** (approval-gated) |

**Canonical import:**

```python
import jaxfne as jtfne
```

---

## 2. Core thesis (what jaxfne is)

`jaxfne` makes **emitter → source → field → probe → objective → optimizer** assumptions **explicit, executable, auditable, and hashable** in a JAX-native computational scaffold.

It is **not** claiming:

- real EEG/MEG forward modeling with calibrated amplitude
- PDE field solving with residual-validated electromagnetics
- metabolic or mechanism proof from objective success alone
- empirical validation against held-out neural recordings in this package paper

---

## 3. Truth gates (must remain unchanged in review)

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Any reviewer score that assumes physical amplitude, solver validation, or mechanism proof **without cited repo evidence** should be marked **reject** or downgraded.

---

## 4. Publication artifact inventory

### Main figures (8/8)

| ID | Script | PNG | Claim boundary |
|---|---|---|---|
| fig01 | `scripts/publication/fig01_architecture.py` | `figures/publication/fig01_tfne_architecture.png` | computational_scaffold_only |
| fig02 | `scripts/publication/fig02_contracts.py` | `figures/publication/fig02_source_field_contracts.png` | no_pde_solver_claim |
| fig03 | `scripts/publication/fig03_backend.py` | `figures/publication/fig03_jaxfne_backend.png` | package_native_api |
| fig04 | `scripts/publication/fig04_minimal_install_run.py` | `figures/publication/fig04_minimal_install_run.png` | cpu_safe_tutorial |
| fig05 | `scripts/publication/fig05_runtime_scaling.py` | `figures/publication/fig05_runtime_scaling.png` | benchmark_receipt_required |
| fig06 | `scripts/publication/fig06_readout_family_panel.py` | `figures/publication/fig06_readout_family_panel.png` | proxy_readouts_only |
| fig07 | `scripts/publication/fig07_reproducibility_artifacts.py` | `figures/publication/fig07_reproducibility_artifacts.png` | fixed_tag_sha_required |
| fig08 | `scripts/publication/fig08_adjacent_tools_comparison.py` | `figures/publication/fig08_adjacent_tools_comparison.png` | capability_comparison_not_superiority |

### Extended Data (10/10)

| ID | Script | PNG | Claim boundary |
|---|---|---|---|
| ed01 | `scripts/publication/ed01_api_stability_snapshot.py` | `ed01_api_stability_snapshot.png` | local_api_snapshot_only |
| ed02 | `scripts/publication/ed02_json_schema_validation.py` | `ed02_json_schema_validation.png` | config_schema_validation_receipt |
| ed03 | `scripts/publication/ed03_notebook_execution_receipts.py` | `ed03_notebook_execution_receipts.png` | receipt_driven_structural_scan |
| ed04 | `scripts/publication/ed04_optional_dependency_laziness.py` | `ed04_optional_dependency_laziness.png` | local_subprocess_optional_dep_receipt |
| ed05 | `scripts/publication/ed05_manifest_hashes.py` | `ed05_manifest_hashes.png` | local_artifact_integrity_receipt_only |
| ed06 | `scripts/publication/ed06_benchmark_scaling_tables.py` | `ed06_benchmark_scaling_tables.png` | local_cpu_runtime_receipt_only |
| ed07 | `scripts/publication/ed07_probe_operator_contracts.py` | `ed07_probe_operator_contracts.png` | proxy_readout_operator_contracts_only |
| ed08 | `scripts/publication/ed08_tutorial_atlas_coverage.py` | `ed08_tutorial_atlas_coverage.png` | tutorial_atlas_coverage_receipt_only |
| ed09 | `scripts/publication/ed09_failure_modes_and_nulls.py` | `ed09_failure_modes_and_nulls.png` | failure_null_local_receipt_only |
| ed10 | `scripts/publication/ed10_release_archive_receipt.py` | `ed10_release_archive_receipt.png` | release_archive_evidence_receipt_only |

PNG paths: `figures/publication/<name>.png`  
Manifest/receipt paths: `outputs/publication/*` (gitignored; regenerate on live checkout)

---

## 5. Alpha validation receipt (live checkout)

Commands run at alpha checkpoint:

```bash
git switch cur
git rev-parse HEAD
python3 scripts/publication_inventory.py
# main figures: 8/8 present
# extended data: 10/10 present

python3 -m compileall -q jaxfne scripts/publication tests
# exit 0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py \
  tests/test_root_import_lightweight.py \
  tests/test_signals_get_v0329.py \
  tests/test_objective_null_reproducibility_v0330.py \
  tests/test_artifact_json_safety_v0330.py \
  tests/test_public_docs_hygiene.py -q --tb=line
# 277 passed, 6 skipped

python3 -m venv .venv-docs
. .venv-docs/bin/activate
pip install -e . -r docs/requirements.txt
python -m mkdocs build --strict
# exit 0
```

Root-import laziness (isolated subprocess): **pass** (`optional_loaded []`).

---

## 6. Approval-gated items (explicitly pending)

ED10 records these as `pending_approval`, `not_created`, or `not_applicable_without_release_approval` — **not failures**:

| Item | Alpha status |
|---|---|
| Version tag at current HEAD | pending approval (`v0.3.29` tag exists but may not match HEAD) |
| Wheel/sdist build | not approved |
| PyPI/TestPyPI publish | not approved |
| GitHub release | not approved |
| Archive/DOI | not approved |
| Full clean-room pip install | pending approval (import smoke only) |

---

## 7. Scoreboard for external review (27 factors)

**Instructions to reviewer:** For each factor, assign:

- **Score now** (0–100)
- **Target** (reference)
- **Evidence found in zip/repo** (file paths, test names, manifest IDs)
- **Gap** (what is missing or overstated)
- **Classification:** `accept` | `follow-up` | `patch` | `reproduce` | `reject`
- **Stop condition** (what would block submission)

Do **not** treat proxy outputs as physical measurements. Do **not** credit empirical validation without datasets and held-out tests.

| # | Factor | Hamm score now | Target | Primary evidence to inspect |
|---:|---|---:|---:|---|
| 1 | Branch/release hygiene | 90 | 100 | `cur` @ live SHA (re-freeze), clean tree, no force-push |
| 2 | Main figure stack | 88 | 100 | `figures/publication/fig01–08`, scripts, ED5 hashes |
| 3 | Extended Data stack | 98 | 100 | ED1–ED10 PNGs + scripts |
| 4 | Manifest/hash closure | 72 | 100 | `ed05_manifest_hashes.py`, `outputs/publication/*` |
| 5 | Notebook execution evidence | 45 | 95 | ED3, ED8; `notebook_execution_completeness_claim_allowed: false` |
| 6 | JSON/schema validation | 65 | 95 | ED2, `test_artifact_json_safety_v0330.py` |
| 7 | API stability | 65 | 95 | ED1, `test_api_smoke.py`, `jaxfne/__init__.py` |
| 8 | Optional dependency laziness | 70 | 95 | ED4, `test_root_import_lightweight.py` |
| 9 | JAX numerical discipline | 65 | 95 | runtime rules, scan/vmap/jit tests |
| 10 | Source bookkeeping | 55 | 95 | source modes, double-count guards |
| 11 | Probe/readout contracts | 78 | 95 | ED7, `docs/probe_operators.md` |
| 12 | Electromagnetic admissibility | 35 | 90 | P0–P5 ladder, `docs/poisson_admissibility.md` |
| 13 | Physical amplitude discipline | 80 | 100 | truth gates in all manifests |
| 14 | Mechanism-claim discipline | 82 | 95 | ED9 nulls, `test_objective_null_reproducibility_v0330.py` |
| 15 | Benchmark evidence | 65 | 85 | ED6 local CPU receipt only |
| 16 | Adjacent-tool positioning | 70 | 95 | fig08, no superiority claims |
| 17 | Tutorial-to-package discipline | 45 | 90 | ED8 atlas, tutorial thinness tests |
| 18 | Release archive readiness | 72 | 100 | ED10 receipt; release actions pending |
| 19 | Manuscript alignment | 50 | 95 | `docs/manuscript_alignment.md`, roadmap |
| 20 | Empirical validation readiness | 10 | 70 | intentionally out of scope for alpha |
| 21 | Config-first backbone | 35 | 95 | `Configuration`, 0.3.28+ ladder |
| 22 | Identity/selectors | 30 | 95 | selector tests v0.3.29 |
| 23 | Connectivity rules | 25 | 95 | connection-rule compiler tests |
| 24 | Weld/reconstruct/flatten | 15 | 90 | experimental HPC contracts |
| 25 | Solver-readiness | 20 | 90 | schemas before solver promotion |
| 26 | Jaxley bridge | 60 | 90 | optional lazy bridge, guarded tests |
| 27 | PyNWB bridge | 25 | 85 | doctrine/plan; no empirical NWB claim |

**Reviewer deliverable:** recalibrated 27-row table + top 10 blockers + top 10 strengths + overall publication posture (`accept` / `follow-up` / `patch` / `reject`).

---

## 8. Key doctrine files (read first)

| File | Role |
|---|---|
| `internal_docs/loop_context/AGENT_QUICKREF.md` | primary agent anchor |
| `internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md` | deep biophysics reference |
| `internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md` | live posture |
| `docs/publication/publication_checklist.json` | machine-readable artifact list |
| `docs/publication/nature_methods_roadmap.md` | evidence matrix |
| `AGENTS.md` | worker contracts |
| `.cursor/rules/00-jaxfne-baseline.mdc` | baseline gates |
| `.cursor/rules/10-publication-track.mdc` | publication workflow |

---

## 9. Known alpha limitations (do not penalize as surprise)

1. `outputs/publication/*` is gitignored — zip may lack manifests unless regenerated locally.
2. Notebook full-execution receipts are structural/receipt-driven, not universal PASS execution.
3. Benchmark panel is local CPU receipt; no cross-hardware superiority claim.
4. Physical solver and empirical validation are **future scope**, not alpha defects.
5. `PUBLICATION_READINESS_SCOREBOARD.md` in `internal_docs/loop_context/` may lag glossary; prefer glossary §11 and this handout.

---

## 10. Zip packaging note for sender

Exclude `.git`, `.venv*`, `__pycache__`, large `outputs/` except if regenerated manifests are needed for review.

Recommended regeneration before zipping:

```bash
git archive --format=zip --prefix=jaxfne/ -o jaxfne-alpha-cur.zip cur
# then on extracted copy, run fig01–08 and ed01–10 scripts to populate outputs/publication/
```

Or zip live checkout after `outputs/publication/` regeneration for manifest-aware review.

---

## 11. Alpha verdict (internal, pre-external review)

| Dimension | Status |
|---|---|
| Publication artifact stack | **complete** (8/8 + 10/10) |
| Scientific claim scope | **controlled** (computational scaffold) |
| Release readiness | **pending approval** |
| External critical review | **requested** |

**Classification:** `accept` for alpha checkpoint handoff; `follow-up` on manuscript alignment, notebook execution depth, manifest bundle in zip, and release/archive execution after approval.
