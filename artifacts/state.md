# state

Stamp: jaxfne@dev@0108594b · 2026-10-27

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26: tag v0.5.8 on main 11957afa, GitHub Release, PyPI; receipt
  `artifacts/release/v0_5_8_release_receipt.json`. 0.5.9 needs Hamm's go.
- 2026-10-27 on dev: P-052 `sigmoid_beta` surrogate on the dense Izhikevich default route (forward
  bit-identical, other routes refuse); the Muse audit at cfc839ab (six lanes) and its verified fixes;
  `run_stdp_stream` consumes `inh_mask`; PCL core lanes L1-L4 (multi-spike event scan, interval updates,
  F_W in the scan, `EventScanBoundary`), each oracle-exact; 9B, 9F, 9G closed; thin stacks (c88be711);
  worker lanes R, D, T (0108594b); lane C: `simulate_batch` realizes the surrogate, HH `dt` refuses
  strings, relative units in `visualize_network_3d`.
- Broad gate on lane C's tree: 7575 passed, 0 failed; all paper lanes pass. Report gate fail_count 0.
- CI Fast green through c88be711.

## Broken or in flight

- PCL R2 on real files is next (T-3); the C++ reference binary needs WSL installs (O-58).
- The P-052 surrogate's off-diagonal gradient disagrees in sign with the finite difference (corr -0.2 to
  -0.4); its fitness for training is Hamm's call (`artifacts/runs/perf/profiles/precision_058.md`).
- Lane C2 (T-41 refusals) is running in its clone.

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

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14) runs the broad gate, one suite
  at a time on this machine. WSL: `~/venv_jaxfne` (CUDA, one GPU job at a time, claimed in jchat
  `p-jaxfne`), `~/venv_jaxfne_gate` (CPU gate), `~/venv_rtd313` (Read the Docs probes).
- Tracked tree: `python scripts/check_tree_size.py --check`; budget in `scripts/tree_size_allowlist.json`.

## Next action

Fast tier of `todo_stack.md` in lanes R, D, T, C (at most 3 Haiku workers at once), then T-41 and T-42. The `artifacts/` layout map (2026-10-27): 21 of 23 root folders are pinned
by frozen records, record pins or runtime paths, the root sits at its 7-folder limit, and no move is warranted.
