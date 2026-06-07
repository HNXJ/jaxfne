<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# JAXFNE Backlog and Worker Prompts

## Status classification

**accept with follow-up**: the inspected zip is publication-track current through ED8, but output manifests are absent from the zip and ED9/ED10 remain open.

## Immediate backlog

| ID | Classification | Work | Why |
|---|---|---|---|
| P-ED9 | patch | Add ED9 failure modes and null controls | Reviewers need proof that objectives/proxies cannot be overinterpreted silently. |
| P-ED10 | approval-gated patch | Add ED10 release archive receipt | Strong journal submission needs exact tag/SHA/wheel/archive/DOI receipts. |
| P-OUT | reproduce | Regenerate all `outputs/publication/*` manifests/receipts | Attached zip has PNGs/scripts/checklist but missing output manifests. |
| P-LAZY | reproduce/follow-up | Re-run root import laziness in clean venv | Local container had pandas loaded before import, contaminating test. |
| P-API | follow-up | Reconcile `Net` object grammar with root exports | README names Net; root inspection found Config/Model but not root Net. |
| P-MANUSCRIPT | follow-up | Convert roadmap/checklist into manuscript claims table | Needed for Nature Methods-style reviewer scope questions. |
| P-PYNWB | plan | Design optional PyNWB export bridge | Future computational-biophysics strength; not required before ED10. |
| P-SOLVER | plan | Physical solver experimental namespace design | Must wait for source/field schema and admissibility gates. |

## Worker prompt: ED9 failure modes and null controls

```text
Resume jaxfne publication work from live cur.

Start:
- git fetch --all --prune
- git switch cur
- git pull --ff-only origin cur
- git status --short
- git rev-parse HEAD
- python3 scripts/publication_inventory.py

Expected:
- jaxfne version 0.3.29 or later
- main figures 8/8
- Extended Data 8/10 if ED8 has merged
- clean tree

Create branch:
- pub/ed09-failure-modes-null-controls

Create:
- scripts/publication/ed09_failure_modes_and_nulls.py
- figures/publication/ed09_failure_modes_and_nulls.png
- outputs/publication/ed09_failure_modes_and_nulls_manifest.json
- outputs/publication/ed09_failure_modes_and_nulls_receipt.json

Scope:
- failure/null controls only
- no package API changes unless a real failing public contract blocks ED9
- no physical-amplitude claims
- no tags/releases/packages
- preserve physical_amplitude_claim_allowed=false

ED9 must cover at least:
1. objective null controls: layer shuffle, band-label shuffle, uniform gain, no-field projection, phase randomized, source polarity flip where available;
2. failure-mode taxonomy: NaN/Inf JSON rejection, missing seed, unsupported knob, physical-claim escalation, proxy-solver confusion, optional dependency eager import, notebook-local scientific engine;
3. gate outcomes: pass/fail/not_applicable with exact validator or test name;
4. reviewer question: what prevents objective success from being presented as mechanism proof;
5. explicit status: local receipt, not empirical validation.

Validation:
- python3 -m compileall -q scripts/publication jaxfne tests
- python3 scripts/publication/ed09_failure_modes_and_nulls.py
- python3 scripts/publication_inventory.py
- python3 -m json.tool outputs/publication/ed09_failure_modes_and_nulls_manifest.json >/dev/null
- python3 -m json.tool outputs/publication/ed09_failure_modes_and_nulls_receipt.json >/dev/null
- python3 -m json.tool docs/publication/publication_checklist.json >/dev/null
- python3 -m mkdocs build --strict
- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_objective_null_reproducibility_v0330.py tests/test_artifact_json_safety_v0330.py tests/test_public_docs_hygiene.py -q --tb=line

Acceptance:
- inventory becomes 8/8 main + 9/10 ED
- ED9 figure is script-generated
- ED9 manifest and receipt strict JSON pass
- truth gates preserved
- no package API changes unless explicitly justified and tested
- no physical solver/amplitude/mechanism wording

Report:
Status, repo state, changed files, commands run, exact results, evidence/truth status, blockers, next safe action.
```

## Worker prompt: ED10 release archive receipt

```text
Do not start ED10 without explicit approval for release/archive work.

After approval, resume from live cur and create branch pub/ed10-release-archive-receipt.

Create:
- scripts/publication/ed10_release_archive_receipt.py
- figures/publication/ed10_release_archive_receipt.png
- outputs/publication/ed10_release_archive_receipt_manifest.json
- outputs/publication/ed10_release_archive_receipt_receipt.json

Scope:
- exact release/archive receipt only
- tag/SHA/wheel/sdist/archive/DOI status
- clean install smoke
- no scientific claim escalation

Must record branch, commit SHA, clean status, package version, wheel/sdist SHA256 if built, release/archive status if approved, clean install/import smoke, checklist hash, artifact inventory hash, and remaining limitations.

Validation:
- python3 -m compileall -q jaxfne scripts/publication tests
- python3 scripts/publication/ed10_release_archive_receipt.py
- python3 scripts/publication_inventory.py
- python3 -m json.tool outputs/publication/ed10_release_archive_receipt_manifest.json >/dev/null
- python3 -m json.tool outputs/publication/ed10_release_archive_receipt_receipt.json >/dev/null
- python3 -m mkdocs build --strict
- clean-env install/import smoke

Acceptance:
- inventory becomes 8/8 main + 10/10 ED
- all release/archive entries are evidence-backed or explicitly pending/not_applicable
- no publication/release action occurs without approval
- all truth gates preserved
```

## Worker prompt: root import laziness follow-up

```text
Reproduce optional dependency laziness in a clean virtual environment.

Start from live cur. Do not patch until provenance is known.

Commands:
- python3 -m venv .venv-lazy
- . .venv-lazy/bin/activate
- python -m pip install -U pip
- python -m pip install -e .
- python -c "import sys; before={m.split('.')[0] for m in sys.modules}; print('forbidden_before', sorted({'matplotlib','plotly','pandas','optax','jaxley'} & before)); import jaxfne as jtfne; loaded={m.split('.')[0] for m in sys.modules}; print('forbidden_after', sorted({'matplotlib','plotly','pandas','optax','jaxley'} & loaded)); print('version', jtfne.__version__)"
- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_root_import_lightweight.py -q --tb=short

If clean venv passes, classify attached-container failure as environment contamination.
If clean venv fails, patch root import laziness without removing public wrappers.

Likely files if patch is needed:
- jaxfne/__init__.py
- jaxfne/tutorial_utils.py
- tests/test_root_import_lightweight.py only if the test itself has a proven false-positive issue

Acceptance:
- root import passes in clean venv
- optional helpers still import lazily when used
- canonical `import jaxfne as jtfne` works
- no public API removal without wrapper
```
