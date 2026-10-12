# state

Stamp: jaxfne@dev@2d5dc284 · 2026-10-12 (written 2026-10-12T14:37:21Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- dev 2d5dc284: CI (Fast) green on 44d42ea9 and 902d75e7; Dashboard workflow green.
- Replication dashboard live at https://hnxj.github.io/jaxfne/dashboard/ (gh-pages `/dashboard/`
  only; the old MkDocs root is untouched).
- CAMINOS Fig. 1: R1–R3 PASS (v6), R4 FAIL (v6), R4v7 PASS (drive p·NE, onset p = 0.04);
  p = 0.07 counts equal v6 in seeds 1–5. Lane `l2-caminos` 1e768f9d.
- PCL Fig. 3: v4 FAIL (T2, T4) stands; diagnostic at `pcl-port` 5640ce18.

## Broken or in flight

- P-1 fix on branch `fix-poisson-target` (worktree `C:/workspace/wt_poisson`): broad gate 5029
  passed, 0 failed, 85 skipped; second review PASS; skip count differs from 902d75e7 (75), rerun
  with `-rs` in progress. Not pushed.
- G-2 features: design map done, nothing built.

## Last gate results

- Broad gate, dashboard branch at 902d75e7: 5032 passed, 75 skipped.
- `build_dashboard_data.py --check`, `audit_vocabulary.py --check`, `report_hygiene_check.py`:
  pass at 2d5dc284.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `artifacts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14, JAX 0.11). GPU runs: WSL2 CUDA
  `~/venv_jaxfne`, one job at a time after a jchat claim.

## Next action

Resolve the skip-count difference, push P-1, then build the G-2 features one at a time.
