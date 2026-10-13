# state

Stamp: jaxfne@dev@8138a5c3 · 2026-10-12 (written 2026-10-12T16:50:14Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- dev 2d5dc284: CI (Fast) green on 44d42ea9 and 902d75e7; Dashboard workflow green.
- Replication dashboard live at https://hnxj.github.io/jaxfne/dashboard/ (gh-pages `/dashboard/`
  only; the old MkDocs root is untouched).
- CAMINOS Fig. 1: R1–R3 PASS (v6), R4 FAIL (v6), R4v7 PASS (drive p·NE, onset p = 0.04);
  p = 0.07 counts equal v6 in seeds 1–5. Lane `l2-caminos` 1e768f9d.
- PCL Fig. 3: v4 FAIL (T2, T4) stands; diagnostic at `pcl-port` 5640ce18.
- `poisson_drive` targets other than "all" are refused since 7d4eaa8c.
- Agent context: root `AGENTS.md` loads `artifacts/AGENTS.md` (3c255d03); root `MEMORY.md`
  merged into `artifacts/memory.md` and `todo_stack.md` cut to 13 KB (325751d2); open stack
  adopted (c7b3a056).
- `connections(sampling="bernoulli")` (G-2 F1) since 5ddfd8f2; default path and config hashes
  unchanged.
- AdEx conductance kernel `jaxfne.emitters.simulate_adex` (ADVANCED, 8138a5c3): matches the
  CAMINOS reference cell and recurrent step within rtol 1e-6 (float64). Public route
  `emitter(family="adex")` (d10c93f2); external Poisson conductance drive through
  `Simulation(poisson_drive=)`, bit for bit with a transcription of the reference (b7581aed).
- Dashboard v2 (423c6554, d92064d5): four classes, model pages, validation and roadmap tabs,
  static figures (`scripts/build_dashboard_figures.py --check`).

## Broken or in flight

In `artifacts/open_stack.md`.

## Last gate results

- Broad gate, AdEx kernel (76955729 tree): 5093 passed, 0 failed, 75 skipped. Second review PASS.
- `tests/test_agent_context_hygiene.py`, `test_artifacts_root_layout.py`, `test_memory_brief.py`:
  39 passed at c7b3a056; `report_hygiene_check.py` clean.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). GPU runs: WSL2 CUDA
  `~/venv_jaxfne`, one job at a time after a jchat claim.

## Next action

G-2 F6 (SOM in the AdEx table and the field proxy sign), then the CAMINOS P6 prereg;
Izhikevich 2003 P3 prereg; narrowing step 2.
