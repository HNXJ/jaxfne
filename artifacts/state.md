# state

Stamp: jaxfne@dev@325751d2 · 2026-10-12 (written 2026-10-12T15:23:01Z)

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
  merged into `artifacts/memory.md` and `todo_stack.md` cut to 13 KB (325751d2).

## Broken or in flight

- G-2 F1 `connections(sampling="bernoulli")` on branch `f1-bernoulli` (clone `C:/workspace/oc_f1`):
  review FAIL, storing `sampling` changes `config_hash` (README atlas pin
  `91f037cf383e0bdb`); fix: store the key only for "bernoulli". Broad gate: 1 failed (that pin),
  5038 passed.

## Last gate results

- Broad gate, dev at 7d4eaa8c (poisson fix): 5039 passed, 0 failed, 75 skipped.
- `tests/test_agent_context_hygiene.py`, `test_artifacts_root_layout.py`, `test_memory_brief.py`:
  39 passed at 325751d2; `report_hygiene_check.py` clean.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). GPU runs: WSL2 CUDA
  `~/venv_jaxfne`, one job at a time after a jchat claim.

## Next action

Fix F1's stored key, re-run the gate and review, push; then F2–F6.
