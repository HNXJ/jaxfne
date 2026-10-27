# C9 and C2 reproducibility audit

Date: 2026-10-26 (`date -u +%Y-%m-%d`). Base: 5b6694a2.

Scope: lane L9 of `artifacts/programme/design/plans/arxiv_paper_plan.md`. Read-only. No figure generator, test or gate was run. Live reads: doi.org and zenodo.org (public API, no token) and `git ls-files`. Status words: observed (read from a file or a live response), inferred, unknown.

## 1. Lock file and pinned environment

| Item | State | Evidence |
|---|---|---|
| Lock file (uv.lock, poetry.lock, Pipfile.lock, conda file, requirements lock, constraints) | none found | Glob over the worktree; only `docs/requirements.txt` matched |
| Build backend | exact: `hatchling==1.29.0` | `pyproject.toml:10` |
| Runtime dependencies | ranges: jax and jaxlib >=0.4.25,<0.12; numpy >=1.24; scipy >=1.10 | `pyproject.toml:44-47` |
| Exact pins in extras | `scipy==1.17.1` (viz and dev, Python below 3.15); `ruff==0.17.1` (dev) | `pyproject.toml:63`, `:74` |
| viz extra (used by Read the Docs and the dashboard figures job) | range `matplotlib>=3.10.9,<3.12`; plotly, kaleido, pandas, Pillow, reportlab unpinned | `pyproject.toml:63`; `.readthedocs.yaml:14`; `.github/workflows/dashboard.yml:92` |
| dev extra | `matplotlib>=3.10.9,<3.11` (Python below 3.15); pytest, build, nbclient unpinned | `pyproject.toml:74` |
| Docs requirements | ranges: mkdocs >=1.6.1, mkdocs-material >=9.7.7, pymdown-extensions >=12.3 | `docs/requirements.txt:1-3` |
| Python used for builds | Read the Docs 3.13; dashboard figures job 3.12 | `.readthedocs.yaml:9`; `.github/workflows/dashboard.yml:89` |
| Renderer conflict | frozen manifest requires `matplotlib>=3.10.9,<3.11` and records 3.11.1 as incompatible; the viz extra admits 3.11 | `artifacts/publication/frozen_manifest.json:36`, `:41-43`; `pyproject.toml:63` |

## 2. Zenodo DOI

Repo sources of the concept DOI 10.5281/zenodo.21448155:

| Source | Lines | Agrees with concept DOI |
|---|---|---|
| `CITATION.cff` | 10, 29 | yes |
| `docs/reference/citation.md` | 10, 19 | yes |
| `README.md` | 9, 64 | yes |
| `tests/test_citation_cff.py` | 29 | yes |
| `docs/reference/project/zenodo_doi.md` | 24-26 | lists version DOIs v0.4.7 (10.5281/zenodo.21448156) and v0.5.6 (10.5281/zenodo.23404707); the list is dated 2026-10-16 (line 21) |

Resolution (curl, no token):

| DOI | Route | Final URL | Final HTTP |
|---|---|---|---|
| 10.5281/zenodo.21448155 (concept) | first hop 302 to `https://zenodo.org/doi/10.5281/zenodo.21448155`, then 2 redirects | `https://zenodo.org/records/23566508` | 200 |
| 10.5281/zenodo.23404707 (v0.5.6) | 2 redirects | `https://zenodo.org/records/23404707` | 200 |
| 10.5281/zenodo.21448156 (v0.4.7) | 2 redirects | `https://zenodo.org/records/21448156` | 200 |

Zenodo API `GET https://zenodo.org/api/records/23566508` returned HTTP 200 (saved to `C:/workspace/pytest_temp/l9/zenodo_record_23566508.json`): version v0.5.7, DOI 10.5281/zenodo.23566508, conceptdoi 10.5281/zenodo.21448155, publication_date 2026-10-22.

| Finding | Evidence |
|---|---|
| Concept DOI and per-release version DOIs both exist; concept DOI agrees across all repo sources | rows above |
| Version DOI for v0.5.7 (10.5281/zenodo.23566508) is in no repo file | grep for 23566508 over the worktree: no match |
| Version list is not complete after v0.5.6 | `zenodo_doi.md:21`; Zenodo versions query returned HTTP 400 on two parameter forms, full list not retrieved |
| Release date disagrees | `artifacts/release/current_release_authorities.json:4` says published 2026-10-21; Zenodo publication_date 2026-10-22 |
| Release target disagrees | `current_release_authorities.json:3` says 0.5.7; `pyproject.toml:15` and `CITATION.cff:5` say 0.5.8 (candidate, unreleased) |

## 3. License

| Item | Value | Evidence |
|---|---|---|
| LICENSE file | MIT License; "Copyright (c) 2024-2026 H N" | `LICENSE:1`, `:3` |
| pyproject.toml | `license = {text = "MIT"}` (table form, not an SPDX expression); MIT classifier | `pyproject.toml:19`, `:37` |
| CITATION.cff | `license: MIT` | `CITATION.cff:7` |
| README badge | license-MIT, linking LICENSE | `README.md:8` |
| Author name strings | "Hamed Nejat" in pyproject; "Nejat, Hamed" in CITATION; "H N" in the copyright line | `pyproject.toml:20`; `CITATION.cff:17-18`; `LICENSE:3` |
| Paper licence | CC BY 4.0 planned; no manuscript source in the repo carries it | `artifacts/programme/design/plans/arxiv_paper_plan.md:3` |

Software licence is MIT in all four places. The name strings differ across the three sources.

## 4. Data and code statement

| Where | Line | Text (short) | Covers |
|---|---|---|---|
| `README.md` | 64 | "Concept DOI (all versions): ..." | citation only |
| `README.md` | 8 | license badge | licence |
| `docs/reference/citation.md` | 9 | "Every published GitHub Release is archived on Zenodo." | code archive |
| `README.md`, `docs/**/*.md`, `artifacts/programme/paper/` | none | no "data availability" or "code availability" text found | nothing |
| `artifacts/etudes/papers/pcl/SOURCE_P1.md` | 44-45 | PCL trained networks and data on Zenodo 10.5281/zenodo.16813454; "That archive's licence is unknown here." | external data, licence unknown |
| `artifacts/etudes/papers/caminos/SOURCE_P1.md` | 39 | mean-field code on Zenodo 10.5281/zenodo.20391404; "Its licence is unknown" | external code, licence unknown |
| `pyproject.toml` | 233, 247 | sdist excludes `/receipts` and `artifacts/programme/receipts` | sdist only; git tracking of receipts not checked |

## 5. C2: figure regeneration

Status key. Command exists: a generator in the repo matches the figure content by name or docstring. Partial: only an API, a test, a lane run or a predecessor-numbered figure covers part of the content. None: nothing found. No command was run, so runnability is unverified. Plan row numbers are in `arxiv_paper_plan.md`.

| Fig | Plan content | Plan line | Status | Script or file | Note |
|---|---|---|---|---|---|
| F1 | TFNE language and grammar | 22 | command exists (predecessor numbering) | `scripts/publication_figures/fig01_grammar.py` | Docstring: "Figure 1: TFNE ... grammar"; outputs the frozen `artifacts/figures/publication/fig01_tfne_grammar.png`; manifest marks it write-once (`frozen_manifest.json:2`); rerun behaviour not read |
| F2 | Geometry and containment, AT-10-N20 | 23 | partial | `scripts/generate_atlas_figures.py` (`--at AT-10`, lines 22-24) | Atlas outputs tracked under `artifacts/publication/atlas/AT-10-N20` (309 tracked files under `artifacts/publication/atlas`); the id AT-10-N20 appears in generator source only at `scripts/generate_docs_visuals.py:13` |
| F3 | H-state, RBS, RBD, HDP, neutral-H recovery | 24 | partial | `scripts/publication_figures/fig06_hwd_evidence.py`; `tests/test_hdp_reduction_identity.py` | fig06 draws the frozen H/W/D ladder (docstring); no neutral-H or typed-coupling figure |
| F4 | Fast compute, baselines, augmentation | 25 | partial | `scripts/benchmarks/scaling_benchmark.py`; `scripts/evidence_figures/fig05_runtime_scaling.py` | Brian2 comparison at `scripts/benchmarks/brian2_izhikevich_comparison.py`; no NEST or Jaxley baseline script found by name |
| F5 | Brunel 2000, Izhikevich 2003 | 26 | partial | `artifacts/etudes/papers/brunel2000/brunel2000_api.py`; `artifacts/etudes/papers/izhikevich2003/run_izh.py` | Brunel lane has API and test, no run script |
| F6 | Potjans 2014, CAMINOS | 27 | partial | `artifacts/etudes/papers/potjans2014/potjans2014_api.py`; `artifacts/etudes/papers/caminos/run_caminos.py` | Potjans lane has API and test, no run script |
| F7 | Vogels 2011, PCL, GSDR fits | 28 | partial | `artifacts/etudes/papers/vogels2011/vogels2011_api.py`; `artifacts/etudes/papers/pcl/pcl_report.py`; `artifacts/etudes/papers/gsdr2026/run/run_gsdr.py` | Lane-level pieces; no panel generator |
| F8 | Claim coverage by status tier; fit summary | 29 | partial | `scripts/build_dashboard_figures.py`; `scripts/check_claim_ledger_055.py` | Dashboard figure builder and ledger check; no status-tier figure generator found |
| S1 | Notation table | 35 | command exists (2026-10-26) | `scripts/publication_figures/supplement.py` | table from `definition_map.md`; `tests/test_supplement_generators.py` |
| S2 | Relative quantities, calibration | 36 | partial (tests only) | `tests/test_calibration_contracts_v025.py` | |
| S3 | Geometry property tests | 37 | partial (tests only) | `tests/test_geometry_invariants.py` | |
| S4 | NeuronalTensor schema, flattening | 38 | partial (tests only) | `tests/test_neuronal_tensor_geometry.py` | |
| S5 | HDP reduction identity per rule | 39 | command exists (2026-10-26) | `scripts/publication_figures/supplement.py`; `tests/test_hdp_reduction_identity.py` | table and panel per registered rule |
| S6 | Benchmark detail: CPU, GPU, Brian2, NEST, Jaxley | 40 | partial | `scripts/benchmarks/scaling_benchmark.py`; `scripts/benchmarks/brian2_izhikevich_comparison.py` | no NEST or Jaxley script found by name |
| S7 | Augmentation determinism, provenance | 41 | partial (tests only) | `tests/test_augment_determinism.py`; `jaxfne/augment.py` | |
| S8 | Brunel sensitivity (dt, seeds) | 42 | partial | `artifacts/etudes/papers/brunel2000/brunel2000_api.py` | no sensitivity run script found |
| S9 | Izhikevich Fig. 2 cell classes | 43 | partial | `artifacts/etudes/papers/izhikevich2003/run_izh.py`; `artifacts/etudes/papers/izhikevich2003/calibrate_detector.py` | |
| S10 | Potjans per-population rates | 44 | partial | `artifacts/etudes/papers/potjans2014/potjans2014_api.py` | |
| S11 | CAMINOS PV-SOM lag | 45 | partial | `artifacts/etudes/papers/caminos/caminos_analysis.py` | |
| S12 | Vogels segment B, memory recall | 46 | partial | `artifacts/etudes/papers/vogels2011/vogels2011_api.py` | |
| S13 | PCL, signed T2 failure | 47 | partial | `artifacts/etudes/papers/pcl/port/fig3_v5_t2.py` | |
| S14 | GSDR fit detail, held-out seeds | 48 | partial | `artifacts/etudes/papers/gsdr2026/fit/fit_gsdr.py` | |
| S15 | Negative results: traveling waves, NO_ADAPTATION, closed loop | 49 | partial | `scripts/publication_figures/fig05_protocol_c.py` | traveling-wave frozen figure only; NO_ADAPTATION appears in publication audit scripts such as `scripts/publication_figures/fig06_hwd_evidence.py`; no closed-loop generator found by name |
| S16 | Extras: Mejias, Destexhe, kernel LFP, Jansen-Rit | 50 | partial | `artifacts/etudes/papers/mejias2016/mejias2016_api.py`; `artifacts/etudes/papers/destexhe2009/run_destexhe.py` | no Jansen-Rit or kernel-LFP generator; "Jansen" appears only in references and dashboard metadata |

Counts: command exists 3 (F1, S1, S5). Partial 21 (F2 to F8, S2 to S4, S6 to S16). None 0.

Frozen figures: the 28 files in `frozen_manifest.json` (21 JSON, 7 PNG) are all git-tracked (`git ls-files --error-unmatch`, exit 0).

## 6. CI coverage

| Workflow | Trigger and branches | Docs build | Figure work | Publishes |
|---|---|---|---|---|
| `ci.yml` (CI Fast) | push to dev and main; PR to dev (`ci.yml:4-7`) | `mkdocs build --strict` (`ci.yml:90`) | broad gate runs audit steps including docs-figures check (`run_test_gate.py:584-587`, `:734`; `ci.yml:130`) and the Figures 1-7 test (below) | nothing; wheel and sdist build, twine check (`ci.yml:159-163`) |
| `release_ci.yml` (CI Release and Scheduled) | push and PR to main; nightly cron `0 0 * * *` (`:9`); dispatch | installs docs deps (`:47-48`); no direct mkdocs step | full pytest including slow (`:59`), which includes the Figures 1-7 test | dist artifacts (`:115-116`) |
| `dashboard.yml` (Dashboard) | push to dev with path filter (`:9-22`); dispatch | none | `figures` job (needs `publish`): `build_docs_assets.py --site _site_figures` (`:99`) | gh-pages `/dashboard/` (`:75`) and `/figures/` (`:117`); dev only |
| `publish.yml` (Publish) | release published; dispatch (`:3-6`) | none | none | TestPyPI and PyPI, SHA-checked against the release manifest (`:43-48`, `:75-81`) |
| `notebook_execution.yml` | nightly cron `0 3 * * *` (`:6`); dispatch | none | none | none; `pytest -m notebook` (`:33`) |
| Read the Docs | configured outside the repo; branch and trigger not readable here | `.readthedocs.yaml:13-15` runs `pip install ".[viz]"` and `build_docs_assets.py --build`; mkdocs `fail_on_warning: true` (`:24`) | atlas HTML regenerated at build | hosting only |
| Publication gate | `run_test_gate.py publication` (`run_test_gate.py:916-938`) renders Figures 1-7 and compares with frozen PNGs in scratch | not in any workflow (no `.github/workflows` file names `equivalence_gate` or `publication_figures`); not called by `gate_release` (`:761-797`) | | |

Figures 1-7 test: `tests/test_equivalence_gate_v20260815.py:63-81` has no slow marker and `BROAD_PYTEST_IGNORE` is empty (`run_test_gate.py:98`), so it runs in the broad sweep on dev pushes and in release CI on main. On Linux it asserts dimension equality only; pixel and byte identity are recorded as informational (lines 74-81).

## 7. C9 and C2 gaps

Closed by 2026-10-27: the figure lock `.github/figure_env_lock.txt` (82 pins, matplotlib 3.10.9) and the
`publication` job of `release_ci.yml`, which installs under the lock on a fresh checkout and runs
`run_test_gate.py publication`; the v0.5.7 version DOI in `zenodo_doi.md`; release target 0.5.8 and the 0.5.7
date in the authority file; the LICENSE name and the F-to-fig mapping (H5, H1, Hamm 2026-10-26);
`docs/requirements.txt` pins mkdocs 1.6.1, mkdocs-material 9.8.0 and pymdown-extensions 12.3, the versions
Release CI 40444082095 installed (their own dependencies stay unpinned).

| Gap | Evidence | What would close it |
|---|---|---|
| Frozen figures are write-once, and their generators list them as outputs | `frozen_manifest.json:2`; `scripts/publication_figures/fig01_grammar.py:1-10` | regenerate into scratch and compare, as the publication gate does, rather than overwrite |
| Figures 1-7 test checks dimensions only off macOS | `tests/test_equivalence_gate_v20260815.py:74-81` | owner decision on the binding renderer and criterion off macOS |
| Only F1 has a matching generator (predecessor numbering); the other 23 rows are partial or none | section 5 counts | build figure generators in the lanes the plan sequences (0.5.9 to 0.6.0), per `arxiv_paper_plan.md:78-80` |
| No data and code availability statement | section 4 | owner writes the statement for the manuscript and README |
| Paper licence CC BY 4.0 declared nowhere in the repo | `arxiv_paper_plan.md:3` | declare it with the manuscript source |
| Licences of PCL's natural-image set, DVS128 and N-MNIST unknown (PCL code and data and the CAMINOS mean-field code are CC-BY 4.0) | `artifacts/etudes/papers/pcl/SOURCE_P1.md`; `artifacts/etudes/papers/caminos/SOURCE_P1.md` | confirm the licences before the statement names them |
| No generator for Jansen-Rit or kernel-LFP (S16) | section 5, row S16 | owner builds a generator or drops them from S16 |
| No NEST or Jaxley baseline script found (F4, S6) | section 5, rows F4 and S6 | build the baselines in the benchmark lane |

## Not checked

| Item | Why open |
|---|---|
| `ci.yml` fast-test Python matrix (`ci.yml:16-31`) | not read |
| Git tracking of `artifacts/programme/receipts` and of other generator inputs | not checked; a fresh clone needs them |
| Runnability of any generator | not run, per the packet |
| Read the Docs branch and trigger | configured outside the repo |
| Full Zenodo version list | versions API query returned HTTP 400 twice |
