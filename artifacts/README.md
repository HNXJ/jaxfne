# artifacts/

Development state and evidence for jaxfne. Excluded from the wheel and sdist (`pyproject.toml`).
Every tracked folder is listed here (`tests/test_artifacts_root_layout.py`). Frozen means hashed in
`publication/frozen_manifest.json`, `scripts/harness/HARNESS_MANIFEST.json` or
`release/current_release_authorities.json`; receipts point at each other by path, so folders
stay where they are and new work goes into the folder of its kind.

## State (root files)

`AGENTS.md` (operating manual), `context.md` (skill router), `open_stack.md`, `goal_stack.md`,
`todo_stack.md`, `problem_stack.md`, `fact_stack.md` (Hamm edits), `state.md`, `memory.md`.
`public_api_before.json` and `public_surface_contract_v0413.json` are pinned by their tests.

## Folders

| Folder | Holds | Frozen |
|---|---|---|
| `issue_log/` | open issues P-NNN | release authorities |
| `release/` | release receipts and `current_release_authorities.json` | release authorities |
| `changelog.d/` | release-note fragments assembled into `docs/changelog.md` | |
| `publication/` | manuscript figure specs, receipts, claim ledger, `frozen_manifest.json` | frozen + harness |
| `figures/` | publication figure renders | `publication/` fig01–fig07 frozen |
| `etudes/` | paper replications (`REPLICATIONS.md`, one folder per paper) and etude bundles | cited by publication |
| `programme/` | plans, Atlas gap reports, `atlas_coverage.json`, jnwb inventory | release authorities |
| `atlas/` | Atlas AT-01…AT-10 scripts, results and bundle tooling | cited by publication |
| `protocol_c/` | Protocol C (waves) specs and receipts | cited by publication |
| `protocol_d_biological_rbs/` | Protocol D (biological RBS) specs and receipts | cited by publication |
| `protocol_e_integration/` | Protocol E (integration) specs and receipts | cited by publication |
| `protocol_h_rbd/` | Protocol H (RBD, H4 matrix) receipts | cited by publication |
| `protocol_w/` | Protocol W (closed-loop W) contract and receipts | cited by publication |
| `e2/` | E2 preregistration, execution, confirmatory run, audit | |
| `hdp_v2_rho_sweep/`, `hdp_k_w_ctrl_sweep/` | HDP sweep outputs (cited by `protocol_c/` P2V-A3) | |
| `mcc3_10s_checkpoint/` | MCC3 10 s scientific checkpoint (cited by the etude manifest) | |
| `hdp_presets/` | realized `H` per preset (`scripts/measure_hdp_presets.py`) | |
| `perf/` | performance baselines, matrices, profiles | |
| `benchmark/` | agent benchmark (0.5.5) | |
| `audit/` | audits and deep reviews | cited by publication |
| `project_sources/` | mathematical specification (the project-source set) | |
| `vocabulary/` | `JAXFNE_VOCABULARY.md` | |
| `skills/` | canonical agent skills (mirrored by `scripts/harness/sync_skills.py`) | harness |
| `subagents/` | subagent procedures (vocabulary critic, developer, pool) | |
| `harness/` | evidence and workflow rules, review packet template | |
| `tutorials/` | tutorial notebooks and their status | |
| `viewers/` | standalone column and pseudogenome HTML viewers | |
| `private_acceptance/` | 0.4.x acceptance scorecards | release authorities |
| `archive/` | sealed stacks and decisions, byte-for-byte | |
| `legacy/` | pre-0.4.x notebooks and docs (kept by Hamm 2026-10-11) | |
| `.lab/` | Labyrinth API graph nodes in `bundle.json`, read by `scripts/lab_balance.py` | |

Untracked: `attestations/` (gitignored test XML), `developer/` (local notes).

`archive/scripts/` holds scripts with no reference in the tree, archived 2026-10-13 (T-17):

| script | lines | last commit before the move | what it did |
|---|---|---|---|
| `audit_semantic_negatives.py` | 114 | 19aef19a 2026-09-19 | semantic-negative docs gate (Batch B7), never wired |
| `generate_mechanism_tutorials.py` | 524 | 9958a725 2026-09-28 | generated the three mechanism notebooks 01–03 |
| `make_delta_test_01_report.py` | 629 | baf43d49 2026-09-30 | built the v0.3.31 Delta-Test 01 companion bundle from receipts |
| `repair_notebooks.py` | 198 | 0635f3a8 2026-08-26 | split long notebook code cells into chunks of at most 10 lines |
