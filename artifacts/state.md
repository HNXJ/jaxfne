# state

Stamp: jaxfne@dev@9a1efe20 · 2026-10-28

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26: tag v0.5.8 on main 11957afa, GitHub Release, PyPI; receipt
  `artifacts/release/v0_5_8_release_receipt.json`. 0.5.9 needs Hamm's go.
- 2026-10-27 on dev: P-052 `sigmoid_beta` surrogate on the dense Izhikevich default route (forward
  bit-identical, other routes refuse); the Muse audit at cfc839ab (six lanes) and its verified fixes;
  `run_stdp_stream` consumes `inh_mask`; PCL core lanes L1-L4 (multi-spike event scan, interval updates,
  F_W in the scan, `EventScanBoundary`), each oracle-exact; 9B, 9F, 9G closed; thin stacks (c88be711);
  worker lanes R, D, T (0108594b); lane C: `simulate_batch` realizes the surrogate, HH `dt` refuses
  strings, relative units in `visualize_network_3d` (6f41515c); lanes C2 (T-41 refusals), E (S1 reasons),
  T2 (`paper-lanes` CI job), the nightly skip test, dead example branches.
- 2026-10-28 on dev (a95a40bb to 760bcec3): T-55 items 1-9 (matrix tune bounds scale the masked W
  entries and keep signs; refusals and seeds in optim, jdna, granger); broad-gate stamp and rc reuse of
  Release CI JUnit with an executed case per family (T-42 D, F); S1 follows `_repo_root()` (T-44, five
  allowlist entries cleared); canary chain 7, jdna develop (T-42 B); HH full-kernel gradient test and HH
  under x64; the compile guard counts per cache key (T-38 closed); protocol D refuses a negative drive
  onset (T-41); `fast-build` needs `paper-lanes` (T-42 C).
- 2026-10-28 on dev (d3fbad69 to 9a1efe20): matrix masks follow `W[post, pre]` with `I_to_E`; exact
  float32 PSD phase; F3 check `scripts/check_fidelity_fix.py` over `Fidelity-fix:` trailers, a CI
  `paper-lanes` step (T-44 F3).
- Broad gate on 9a1efe20's tree: 7743 passed, 0 failed; all paper lanes pass.
- CI Fast on 9a1efe20 green (run 40697196089); its F3 step accepted both tagged fixes (2 of 3 commits).

## Broken or in flight

- PCL R2 on real files is next (T-3); the C++ reference binary needs WSL installs (O-58).
- The P-052 surrogate's off-diagonal gradient disagrees in sign with the finite difference (corr -0.2 to
  -0.4); its fitness for training is Hamm's call (`artifacts/runs/perf/profiles/precision_058.md`).
- Hamm ruled "fix" on both 2026-10-28: matrix masks follow `W[post, pre]` (new `I_to_E`; Suite No. 4
  tunes `gGABA_w` with it); `spectrolaminar_psd_jax` takes its phase from exact float32 remainders
  (broadband noise against float64: 2.4e-4 to 1e-6 of peak at 10 s, 2e-2 to 4e-6 at 1000 s). The T-55
  sinusoid figure (11.5% at 1000 s) was relative to an off-grid leakage peak and overstated the error.

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

T-45 (D31, D32 public names wait on Hamm) and T-46 (licences, owner statement) in worker lanes (at
most 3 at once); tag each fidelity fix with `Fidelity-fix: <node id>`; T-3 R2 waits on the WSL installs
(O-58). The `artifacts/` layout map (2026-10-27): 21 of 23 root folders are pinned by frozen records,
record pins or runtime paths, the root sits at its 7-folder limit, and no move is warranted.
