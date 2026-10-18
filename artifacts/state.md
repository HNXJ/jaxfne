# state

Stamp: jaxfne@dev@36f65d8b · 2026-10-18 (written 2026-10-18T01:06:55Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- 2026-10-18 on dev: declared input sources (T-24, `XSource`, `Model.x_sources`); replication
  checker `scripts/check_replications.py` in every gate including dev (25 listed findings); T-31
  fixes (silent fallbacks, narrowed handlers, real publication gate); one sign table (T-18); AdEx
  zero-delay edges as a declared intra-step fixed point (T-22 phase 1). Broad gate on 36f65d8b:
  6407 passed, 1 timing flake (passes alone 3 of 3; the guard now
  interleaves its runs).

- 2026-10-17 on dev: multi-compartment HH route (jaxley oracle 8.3e-12 mV, Muse review PASS);
  run manifests carry activity regimes (T-23); tensor identity digest fails closed; JUnit
  download per SHA (T-26); harness integrity re-hashed and run as gate family
  `harness_integrity`; replication checklist in `etudes/meta/RESIDUAL_PROTOCOL.md`. Broad gate
  on d8e486bb: 6266 passed, 0 failed.
- Whole-repo audit 2026-10-17 (Muse and Claude, verified by a third pass): open items in todo
  T-30 (replication conformance) and T-31 (silent fallbacks, gates); owner calls O-24 to O-29.

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
- Tracked tree: 2415 files (budget 2428, `budget_files` in `scripts/tree_size_allowlist.json`);
  87985676 bytes (budget 89128960), measured 2026-10-17.
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
- GSDR: the one-compartment fit saturates (κ ≈ 1 against 0.0156; CALIBRATION §6.4–6.10). The
  refit on the reference 4-compartment cell runs (O-30, about 17 h on the GPU).
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
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). WSL: `~/venv_jaxfne_gate`
  (broad gate, CPU), `~/venv_jaxfne` (CUDA; one job at a time under `flock /tmp/jaxfne_gpu.lock`),
  `~/venv_rtd313` (Read the Docs probes).

## Next action

Agents: integrate lanes as they land (review, broad gate, dashboard checks, push dev, CI on the
exact SHA): T-24, the replication checker and the T-31 fixes (O-31, O-32); the GSDR regime
diagnostic when the refit's manifest exists (O-30); then T-30 per paper.
