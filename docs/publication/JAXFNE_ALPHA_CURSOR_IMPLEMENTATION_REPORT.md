# jaxfne alpha Cursor implementation report

## Status

**accept** — Tasks 01–09 executed on live `cur`. Publication artifact stack complete. Release/archive remains approval-gated.

## Repo state

| Field | Value |
|---|---|
| branch | `cur` |
| HEAD | `c34a2c8ddebdfe2702f6c7b223efe6c0541963bf` |
| origin/cur | sync at push |
| version | `0.3.29` |
| working tree | clean except gitignored `outputs/publication/*` |
| inventory | `8/8 main + 10/10 ED` |

## Changed files by task

| Task | Tracked changes |
|---|---|
| 01 | `docs/publication/JAXFNE_ALPHA_HANDOUT.md`, `internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md` |
| 02 | `figures/publication/*.png` (12 refreshed at regen SHA) |
| 03 | `internal_docs/receipts/context_review/{MAIN_ALIGNMENT_FINAL_RECEIPT,README_UPDATED_CONTEXT,SCORECARD}.md` |
| 04 | `docs/publication/JAXFNE_ALPHA_EXTERNAL_REVIEW_SCOREBOARD.md`, `internal_docs/loop_context/PUBLICATION_READINESS_SCOREBOARD.md` |
| 05 | `docs/publication/MANUSCRIPT_TBD_REPLACEMENT_PLAN.md` |
| 06 | no source patch (clean venv laziness pass) |
| 07 | gitignored `outputs/publication/qa_logs/*`, `alpha_runtime_environment.txt` |
| 08 | `docs/publication/RELEASE_ARCHIVE_APPROVAL_GATE.md` |
| 09 | this report |

Bundle files **not** imported or committed (execution instructions only).

## Commands run

| Command | Result |
|---|---|
| Task 01 freeze + doc SHA reconcile | `fc80a0e` commit |
| Task 02 all fig/ed generators (18 scripts) | exit 0 |
| `python3 scripts/publication_inventory.py` | 8/8 + 10/10 |
| strict JSON on `outputs/publication/*.json` | pass |
| finite JSON gate | `finite_json_values_pass` |
| Task 06 `.venv-lazy` import smoke | `forbidden_after []` |
| `pytest tests/test_root_import_lightweight.py` (clean venv) | 1 passed |
| targeted pytest (6 modules) | 277 passed, 6 skipped |
| `mkdocs build --strict` (.venv-docs) | exit 0 |
| Task 03 path tests | 254 passed, 5 skipped |

## Publication outputs

| Asset family | Count/status |
|---|---|
| main PNGs | 8/8 |
| ED PNGs | 10/10 |
| output JSON strict parse | pass |
| finite JSON | pass |
| ED10 release actions | none (`release_action_taken: false`) |

## Root organization

**Moved to `internal_docs/receipts/context_review/`:**

- `MAIN_ALIGNMENT_FINAL_RECEIPT.md`
- `README_UPDATED_CONTEXT.md`
- `SCORECARD.md`

**Kept at root (functional):** `README.md`, `pyproject.toml`, `mkdocs.yml`, `AGENTS.md`, `LICENSE`, `.gitignore`, `.readthedocs.yaml`, `.github/`, `.cursor/`, `jaxfne/`, `tests/`, `docs/`, `scripts/`, `figures/`, `tutorials/`, `examples/`, `internal_docs/`, `artifacts/`, `benchmarks/`.

**Why not 3 folders + 3 files:** tests, mkdocs, publication scripts, CI, and release tooling hard-code root paths. Deeper reorg is a separate migration branch.

Local receipt: `outputs/publication/root_inventory_after_cleanup.md` (gitignored).

## Truth/evidence status

- Truth gates preserved in all regenerated manifests.
- Proxy readouts remain proxy readouts (`laminar_proxy_no_pde`).
- ED10 tag/wheel/publish/archive: `pending_approval` / `not_applicable_without_release_approval`.
- No tag, build, publish, or archive action performed.

## Blockers

1. `outputs/publication/*` gitignored — include in external review zip manually.
2. Manuscript PDF TBD slots need replacement per `MANUSCRIPT_TBD_REPLACEMENT_PLAN.md`.
3. Release/tag/PyPI/archive require separate explicit approval.
4. Notebook execution remains structural receipt, not universal PASS.

## P1 follow-up (post P0 provenance closure)

| Item | Status |
|---|---|
| Proxy-safe API/docs wording | patched (`docs/api/probes.md`, `docs/api/core.md`, tensor workflow guides, `probe_operators.md`) |
| Two-part review bundle note | `docs/publication/TWO_PART_REVIEW_BUNDLE.md` |
| Handout §10 two-part pins | updated with `a0292503...` outputs SHA |
| Notebook execution scope | preserved as structural (ED3/ED8); no universal PASS claim |
| `test_public_docs_hygiene.py` | 218 passed, 5 skipped |

Frozen review target unchanged: `source_head=9f831bd`, `jaxfne_version=0.3.29`, `outputs_zip_sha256=a0292503...7107c4`.

## Next safe action

1. Send **two-part bundle** per `TWO_PART_REVIEW_BUNDLE.md` (repo at `9f831bd` + `outputs-provenance-patched.zip`).
2. Manuscript TBD replacement and release/tag/archive remain approval-gated.
3. No tag, build, publish, GitHub release, archive, or DOI until explicit approval.
