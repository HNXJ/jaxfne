# state

Stamp: jaxfne@dev@6f85f5c3 · 2026-10-14 (written 2026-10-14T16:36:48Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- dev 6f85f5c3 (core AdEx for Destexhe 2009: `Vpeak_mV`, `ext_lambda` schedules and drive
  window, `sampling="capped_bernoulli"`, one harvest for both tensor bridges): CI (Fast) success.
  Previous push 25b7c1d5 (integration of lanes 1, 3, 6, 7, 8): CI (Fast) success.
- NeuronalTensor on dev: emitter registry and AdEx route, posed geometry for wiring and delays,
  HDP on AdEx, `project_model_sources`, parameter handles. `NeuronalTensor.to_dict` omits the
  0.5.6 optional fields at their defaults, so pinned identity and phenotype hashes are unchanged.
- CAMINOS Fig. 1 against PLoS CB (primary, Hamm 2026-10-14; `artifacts/etudes/caminos/CALIBRATION.md`
  §10): f_pk 33.80 Hz, rate order and R4 (v7) reproduced on the core route; PV→SOM lag 6.47 ms
  against 8.5 ms by the ref. 42 method (off, a finding that needs X). Dashboard row updated.
- Tracked tree: 2353 files (budget 2353, raised from 2337 for the lane files), 86796770 bytes
  (budget 89128960).
- main 413fb4a2.

## Broken or in flight

- Lane 2b (PCL rebuilt as a NeuronalTensor on the core; LIF route, footprint rule, pooling,
  same-step inhibition, oracle gate): running.
- Lane 4f (GSDR fit, 350 generations): running; at generation 200 best loss 4082 (flat since
  about generation 150), κ ≈ 0.95 against 0.0156 in the data, active fraction 1.0 (inferred: a
  regime where every neuron fires every cycle).
- Lane 5d (Destexhe 2009 on the core): migration and fidelity done in its clone (lane tests
  28 passed); S2 seeds 1–8 queued on the GPU lock behind 4f.
- Open items: T-20 (lint leftovers, ignored `--check` flags), T-21 (Destexhe on core), T-9 and
  T-17 wait on Hamm.

## Last gate results

- Broad gate on 6f85f5c3 (WSL, self-contained clone): 5850 passed, 0 failed.
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
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). WSL: `~/venv_jaxfne_gate`
  (broad gate, CPU), `~/venv_jaxfne` (CUDA; one job at a time under `flock /tmp/jaxfne_gpu.lock`),
  `~/venv_rtd313` (Read the Docs probes).

## Next action

Integrate lanes as they land (review, broad gate, dashboard checks, push dev, CI on the exact
SHA): 5d after its S2 scoring, 4f with a diagnosis of the all-fire regime, 2b stage by stage.
