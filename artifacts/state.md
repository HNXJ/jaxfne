# state

Stamp: jaxfne@dev@0c8ca0f0 · 2026-10-15 (written 2026-10-15T20:22:33Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- 0.5.6 published 2026-10-15: tag v0.5.6 on main b24e09ef (Release CI 38791659454 green), GitHub
  Release, PyPI via Publish run 38801056076; receipt `artifacts/release/v0_5_6_release_receipt.json`.
  Release merged back into dev; install and Colab docs name 0.5.6 as published.
- dev 0c8ca0f0: Muse cleanup lane (unused imports and bindings in scripts/, dead `tfne._PN_TOL`);
  Muse review PASS; dev CI green.
- dev 95324945 (PCL lane: LIF tensor route with event input, per-position local inhibition,
  cover-table feed, simple-to-complex pooling, `resolution="event_scan"`): Stage C exact on 20
  clips through the tensor route; Muse adversarial review PASS. Dev Release CI run 38764130367:
  success.
- NeuronalTensor on dev: emitter registry and AdEx route, posed geometry for wiring and delays,
  HDP on AdEx, `project_model_sources`, parameter handles. `NeuronalTensor.to_dict` omits the
  0.5.6 optional fields at their defaults, so pinned identity and phenotype hashes are unchanged.
- CAMINOS Fig. 1 against PLoS CB (primary, Hamm 2026-10-14; `artifacts/etudes/papers/caminos/CALIBRATION.md`
  §10): f_pk 33.80 Hz, rate order and R4 (v7) reproduced on the core route; PV→SOM lag 6.47 ms
  against 8.5 ms by the ref. 42 method (off, a finding that needs X). Dashboard row updated.
- Tracked tree: 2382 files (budget 2400, `budget_files` in `scripts/tree_size_allowlist.json`);
  87628423 bytes (budget 89128960), measured 2026-10-16.
- main a98e6149: 0.5.6 merged from dev 4842b87c by PR 108 (2026-10-15). Main CI green: CI Fast
  run 38757281037, Release CI with slow tests run 38757281031. rc gate PASS on 4842b87c with
  JUnit parity against Release CI run 38743501399 (3.11 and 3.14 each 6030/6030, 0 unjustified).
- Version state: 0.5.6 published (tag, GitHub Release, PyPI); main b24e09ef carries the receipt
  commit a5fae28d via PR 109.
- Dependabot PR 106 (ruff 0.17.0 in the dev extra) and PR 107 (docs pymdown-extensions >= 12.2)
  merged into dev and closed on GitHub.

## Broken or in flight

- PCL lane (lane 2b, rebuilt as a NeuronalTensor on the core): Stage C passed through the tensor
  route on 20 clips (CALIBRATION.md Step 9). Open in the record: the `no construct route (OPEN)`
  marker (CALIBRATION.md:24) is not marked closed.
- Lane 4f (GSDR fit, 350 generations): running; at generation 200 best loss 4082 (flat since
  about generation 150), κ ≈ 0.95 against 0.0156 in the data, active fraction 1.0 (inferred: a
  regime where every neuron fires every cycle).
- Lane 5d (Destexhe 2009 on the core): migration and fidelity done in its clone (lane tests
  28 passed); S2 seeds 1–8 queued on the GPU lock behind 4f.
- Open items: T-20 (lint leftovers, ignored `--check` flags), T-21 (Destexhe on core); T-17 waits
  on Hamm.

## Last gate results

- Broad gate on 95324945 (WSL, self-contained clone): 6020 passed; the 2 tag-receipt tests that
  failed there (clone origin set to a Windows path) pass alone after the origin fix.
- rc gate on 4842b87c: PASS (broad 5895, notebook 33, slow 30; JUnit parity; twine check).
- Dashboard: `build_dashboard_data.py --check`, `build_dashboard_figures.py --check`,
  `generate_replications_page.py --check` fresh after the CAMINOS row update.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). WSL: `~/venv_jaxfne_gate`
  (broad gate, CPU), `~/venv_jaxfne` (CUDA; one job at a time under `flock /tmp/jaxfne_gpu.lock`),
  `~/venv_rtd313` (Read the Docs probes).

## Next action

Agents: integrate the Muse fix lane (F1-F3 review findings) after a second review; integrate lanes as they land (review, broad gate, dashboard checks, push dev, CI on
the exact SHA): 5d (Destexhe) after its S2 scoring, 4f (GSDR) on Hamm's next-step choice; T-26.
