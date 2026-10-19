# artifacts/

Development state and evidence for jaxfne. Excluded from the wheel and sdist (`pyproject.toml`).
Every tracked folder is listed here (`tests/test_artifacts_root_layout.py`). Frozen means hashed in
`publication/frozen_manifest.json`, `scripts/harness/HARNESS_MANIFEST.json` or
`release/current_release_authorities.json`. Each folder holds at most 7 direct entries where it
can (Hamm 2026-10-15); new work goes into the folder of its kind. Past records (old release
receipts, dated audits, `archive/`, `legacy/`) keep the paths they were written with.

## State (root files)

`AGENTS.md` (operating manual), `context.md` (skill router), `open_stack.md`, `goal_stack.md`,
`todo_stack.md`, `fact_stack.md` (Hamm edits), `state.md`, `memory.md`. Defects live in one
registry, `assurance/issue_log/ISSUE_LOG.md` (high severity is the problem view; Hamm 2026-10-18).

## Folders

| Folder | Holds | Frozen |
|---|---|---|
| `assurance/` | `issue_log/` open issues P-NNN, `private_acceptance/` 0.4.x acceptance scorecards, `api_contracts/` public API snapshot `public_api_before.json` and `public_surface_contract_v0413.json` (pinned by their tests) | release authorities |
| `viewers/` | standalone column and pseudogenome HTML viewers (default output path of `jaxfne.vis`) | |
| `runs/` | `benchmark/` agent benchmark (0.5.5), `mcc3_10s_checkpoint/` MCC3 10 s scientific checkpoint (cited by the etude manifest), `perf/` performance baselines, matrices, profiles, scaling and workflow benchmark | |
| `e2/` | E2 preregistration, execution, confirmatory run and audit (stays: its preregistered records cite their own paths and hashes; manuscript-cited) | cited by publication |
| `atlas/` | Atlas AT-01…AT-10 scripts, results and bundle tooling (stays: the frozen 0.5.5 agent benchmark `artifacts/runs/benchmark/agent_tasks_055.json` cites `artifacts/atlas/...` paths and its `spec_digest`s are frozen) | cited by benchmark |
| `protocols/` | protocol families; `protocols/hdp/` holds `hdp_v2_rho_sweep/`, `hdp_k_w_ctrl_sweep/` HDP sweep outputs (cited by `protocol_c/` P2V-A3) and `hdp_presets/` realized `H` per preset (`scripts/measure_hdp_presets.py`) | |
| `release/` | release receipts and `current_release_authorities.json` | release authorities |
| `changelog.d/` | release-note fragments assembled into `docs/reference/changelog.md` | |
| `publication/` | manuscript figure specs, receipts, claim ledger, `frozen_manifest.json` | frozen + harness |
| `figures/` | publication figure renders | `publication/` fig01–fig07 frozen |
| `etudes/` | `experiment_a/` frozen canonical bundle (stays: cited frozen path, `jaxfne/` defaults), paper replications (`papers/`: caminos, gsdr2026, izhikevich2003, pcl, pcl_h, qlearning), standalone bundles (`bundles/`), theory checks (`theory/`), compact-grammar lane (`jdna/`), lane records (`meta/`) | cited by publication |
| `programme/` | plans (`design/plans/`), gap reports (`gaps/`), capabilities and inventories (`design/capabilities/`), live coverage triple (`coverage/`: `atlas_coverage.json`, `claim_ledger_055.md`, `methods_055.md`), receipts by version and family (`receipts/`), 3 dated files at root (kept paths) | release authorities |
| `protocol_c/` | Protocol C (waves) specs and receipts (stays: `jaxfne/` runtime defaults, FROZEN supplement rows) | cited by publication |
| `protocol_d_biological_rbs/` | Protocol D (biological RBS) specs and receipts | cited by publication |
| `protocol_e_integration/` | Protocol E (integration) specs and receipts | cited by publication |
| `protocol_h_rbd/` | Protocol H (RBD, H4 matrix) receipts | cited by publication |
| `protocol_w/` | Protocol W (closed-loop W) contract and receipts (stays: `jaxfne/` runtime receipt paths, FROZEN supplement rows) | cited by publication |
| `audit/` | audits and deep reviews, split by kind: `reviews/` prose reviews, `studies/` architecture studies, `probe_results/` probe-result JSONs, `capability/` v0417 capability inputs (the audit JSON itself stays at `audit/` root: the frozen `publication/publication_evidence_index.json` cites it), `baselines/` W-series baselines and tallies | cited by publication |
| `project_sources/` | mathematical specification (the project-source set) | |
| `control/` | agent control: `skills/` canonical agent skills (mirrored by `scripts/harness/sync_skills.py`), `harness/` evidence and workflow rules and review packet template, `subagents/` subagent procedures (vocabulary critic, developer, pool), `vocabulary/` `JAXFNE_VOCABULARY.md` | harness |
| `tutorials/` | tutorial notebooks by kind (`suite/` 5, `intro/` 5, `etudes/` lane notebooks, `templates/`), versioned series (`versioned/` 12), status records (`status/`) | |
| `archive/` | sealed stacks and decisions, byte-for-byte | |
| `legacy/` | pre-0.4.x notebooks and docs (kept by Hamm 2026-10-11) | |
| `.lab/` | Labyrinth API graph nodes in `bundle.json`, read by `scripts/lab_balance.py` | |

Untracked: `attestations/` (gitignored test XML), `developer/` (local notes).

Tree degree (Hamm 2026-10-15): at most 7 direct folders and 7 direct files per folder, dot
entries exempt. `tests/test_artifacts_tree_degree.py` enforces it and holds every exception with
its reason. A folder that a frozen record, the package (`jaxfne/`), the harness or an external
tool addresses by path stays where it is; so do the record class roots `archive/` and `legacy/`
and generated series kept whole. At the root, the 10 control files and 16 such folders are
exempt and the rest (`assurance/`, `changelog.d/`, `control/`, `programme/`, `protocols/`,
`runs/`, `tutorials/`) fit the limit. `.lab/` stays at the root: the Labyrinth tool reads
`<project>/artifacts/.lab`.

`archive/scripts/` holds scripts with no reference in the tree, archived 2026-10-13 (T-17):

| script | lines | last commit before the move | what it did |
|---|---|---|---|
| `audit_semantic_negatives.py` | 114 | 19aef19a 2026-09-19 | semantic-negative docs gate (Batch B7), never wired |
| `generate_mechanism_tutorials.py` | 524 | 9958a725 2026-09-28 | generated the three mechanism notebooks 01–03 |
| `make_delta_test_01_report.py` | 629 | baf43d49 2026-09-30 | built the v0.3.31 Delta-Test 01 companion bundle from receipts |
| `repair_notebooks.py` | 198 | 0635f3a8 2026-08-26 | split long notebook code cells into chunks of at most 10 lines |
