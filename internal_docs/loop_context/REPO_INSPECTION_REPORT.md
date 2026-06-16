<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# Repo Inspection Report

> **Superseded (2026-06-08).** Historical record from `jaxfne-pub-ed08-tutorial-atlas-coverage.zip` inspection. For live state run `python3 scripts/evidence_figures_inventory.py` on `cur` and read `CURRENT_PUBLICATION_STATE.md`. Live inventory: **10/10 Extended Data**; findings below about missing ED9/ED10 and stale `.cursor/rules` are obsolete.

## Input

| Item | Value |
|---|---|
| Uploaded zip | `/mnt/data/jaxfne-pub-ed08-tutorial-atlas-coverage.zip` |
| SHA256 | `ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761` |
| Extracted root | `/mnt/data/jaxfne_repo_inspect/jaxfne-pub-ed08-tutorial-atlas-coverage` |

## Repo facts from inspection

| Fact | Value |
|---|---|
| Package name | `jaxfne` |
| Package version | `0.3.29` |
| Checklist branch | `pub/ed08-tutorial-atlas-coverage` |
| Checklist SHA | `9a8c7db58f588bde9f5e8c31b664d56c4982958e` |
| Checklist generated | `2026-06-07T22:25:39Z` |
| Python files under `jaxfne` | 39 |
| Test files | 138 |
| Tutorial files | 24 |
| Tutorial notebooks | 18 |
| Example Python files | 19 |
| Docs files | 105 |

## Publication inventory

```text
main figures: 8/8 present
extended data: 8/10 present
missing evidence figures: ed09_failure_modes_and_nulls.png; ed10_release_archive_receipt.png
outputs/evidence in zip: missing
```

## Commands run

```bash
unzip -q /mnt/data/jaxfne-pub-ed08-tutorial-atlas-coverage.zip -d /mnt/data/jaxfne_repo_inspect
python3 scripts/evidence_figures_inventory.py
python3 -m json.tool docs/evidence_artifacts/evidence_checklist.json >/dev/null
python3 -m json.tool outputs/evidence/inventory.json >/dev/null
python3 -m compileall -q jaxfne scripts/evidence_figures
PYTHONPATH=. python3 -c "import jaxfne as jtfne; print(jtfne.__version__, hasattr(jtfne, 'Config'), hasattr(jtfne, 'Net'), hasattr(jtfne, 'Paradigm'), hasattr(jtfne.vis, 'visualize_network_3d'))"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py -q --tb=short -x
```

## Results

| Check | Result |
|---|---|
| `evidence_inventory.py` | passed; generated local `outputs/evidence/inventory.json`; reported 8/8 main and 8/10 ED |
| `evidence_checklist.json` strict JSON | passed |
| generated `inventory.json` strict JSON | passed |
| `compileall jaxfne scripts/evidence_figures` | passed |
| local import smoke | passed; version `0.3.29`; `Config=True`; `Net=False`; `Paradigm=True`; `visualize_network_3d=True` |
| targeted pytest | failed at `test_root_import_does_not_load_heavy_optional_modules` because `pandas` was already loaded in the container before `import jaxfne` |

## Interpretation of pytest failure

The container environment preloaded `pandas` before package import, so the root-import laziness failure is inconclusive as a package regression. The exact failing assertion reported `Forbidden heavy packages loaded by root import: ['pandas']`. A direct subprocess check showed `pandas before True` before `import jaxfne`. Re-run this gate in a clean venv.

## Findings

1. The publication source stack is current through ED8.
2. ED9 and ED10 are the only missing planned Extended Data figures in the checklist.
3. The attached zip contains figure PNGs and generator scripts but not committed `outputs/evidence/*` manifests/receipts expected by the checklist.
4. The active `.cursor/rules/10-publication-track.mdc` in the repo is stale; it mentions 6/10 or 4/10 ED expectations and ED5 as next. Replace its state with 8/10 ED and ED9 as next.
5. README object grammar names `Net`, but root API inspection did not expose `jtfne.Net`; avoid public examples that instantiate `jtfne.Net` until live API confirms it or wrapper is added.
6. Current scientific posture remains proxy/readout scaffold with physical amplitude disabled.

## Next safe action

Sync live `cur`, confirm ED8 merge, regenerate outputs/manifests, run clean optional-dependency laziness check, then implement ED9.
