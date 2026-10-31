# state

Stamp: jaxfne@dev@4b0a6042 · 2026-10-31

Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

## Works (with evidence)

- 0.5.8 published 2026-10-26: tag v0.5.8 on main 11957afa, GitHub Release, PyPI; receipt
  `artifacts/release/v0_5_8_release_receipt.json`. 0.5.9 needs Hamm's go.
- 2026-10-27 to 2026-10-28 on dev: P-052 `sigmoid_beta` surrogate on the dense Izhikevich default route;
  PCL core lanes L1-L4, each oracle-exact; T-55 items 1-9; T-R28 waves 1 and 2 (Hamm's 53 rulings of
  2026-10-28); matrix masks follow `W[post, pre]` (`I_to_E`); bfloat16 is storage only (float32 compute,
  Hamm 2026-10-28); P-048 default `u0 = b * v0`; P-051 named `instantaneous` receptor; C1 closed; D28 public
  `QuantityKind`; `fast-build` needs `paper-lanes`. Broad gate on 8f3ce57e: 7819 passed, 0 failed; on
  752c38ec: 7835 passed, 0 failed.
- `c3e1035a` (qlearning precondition over seeds, T-58) is on origin/dev (`git branch -r --contains`, 2026-10-30).
- T-56 landed (periodic Hann, off-grid bin refusal, tune seeds; ba78f7ee to 66f32038).

## Broken or in flight

- PCL R2 on real files is next (T-3). The C++ reference builds headless in WSL and its test binary passes
  7 of 7 (T-3 in the working-tree `todo_stack.md`); the oracle comparison is not run yet.
- O-58 (WSL DNS) is closed: `pip download jaxfne==0.5.7 --no-deps` returned rc 0 in WSL on 2026-10-30.
- P-052 surrogate: the off-diagonal gradient disagrees in sign with the finite difference (corr -0.2 to
  -0.4); its fitness for training is Hamm's call (`artifacts/runs/perf/profiles/precision_058.md`).
- Spectrolaminar PSD: Hamm ruled "fix" on 2026-10-28; `spectrolaminar_psd_jax` takes its phase from exact
  float32 remainders. The T-55 sinusoid figure (11.5% at 1000 s) was relative to an off-grid leakage peak.

## Cell Reports replication (Tahvili et al.)

- FIG-A runners for Figs 3, 4, S3 are on local dev and not on origin/dev: `d58c3f70` and `2add03c9`
  (identical content to scratch commits `54b13c7c` and `fa40047b`, which sit on no branch), then `5fc43f10`
  (tree-size budget), `d777526f` (audit fixes) and `4b0a6042` (checkout assertion).
- Dev is 20 commits ahead of origin/dev (`git log --oneline origin/dev..dev`).
- Figs 5 and S1 runner packet: pending integration; `cellrep_figures.py` has no runner for them yet.
- Figure status: Fig 1 panels B, C and D checked (owner's brief, 2026-10-30; no record in the repo). Fig 1D
  is blocked on Hamm's permission to download the Onorato data: file name, source and size needed (owner's
  brief; no record in the repo). Figs 2, 5, 6 and S1 runners, all paper-scale runs and per-panel verdicts
  are pending.

## Open decision for the owner

- S1 drive (O-73): the design under review holds the drive fixed (`ext_sources="fixed"`) while p varies; the
  S1 runner is not written. Sources differ on the scaling: the design note says the Methods do not give it;
  `SOURCE_P1.md` line 18 and commit `b8b38aea` give p·NE Poisson sources at 4 Hz. The open point is then
  private vs shared sources (G3). Hamm to rule.

## Generated files and their checks

| File | Generator | Check |
|---|---|---|
| `dashboard/data/replications.json` | `scripts/build_dashboard_data.py` | `--check` (CI `Dashboard` workflow) |
| `dashboard/figures/*.png`, `manifest.json` | `scripts/build_dashboard_figures.py` | `--check` (CI `Dashboard` workflow) |
| `docs/start/replications.md` | `scripts/generate_replications_page.py` | `--check` (CI `Dashboard` workflow) |
| `docs/_static/atlas*/**.html`, two etude dirs' HTML (untracked) | `scripts/build_docs_assets.py --build` | `--check`, `tests/test_build_docs_assets.py` |
| `docs/start/gallery.md` atlas index | `scripts/generate_gallery.py` | `--check`, `tests/test_generate_gallery.py` |
| `docs/reference/errors.md` raise-site index | `scripts/generate_errors_index.py` | `--check`, `tests/test_generate_errors_index.py` |
| `artifacts/assurance/api_contracts/public_surface_contract_v0413.json` | `scripts/generate_public_surface_contract.py` | `tests/test_public_surface_contract_v0413.py` |

## Environment

- Windows: `C:/workspace/venv_py314_jax011` (Python 3.14) runs the broad gate, one suite at a time on this
  machine. WSL: `~/venv_jaxfne` (CUDA, one GPU job at a time, claimed in jchat `p-jaxfne`),
  `~/venv_jaxfne_gate` (CPU gate), `~/venv_rtd313` (Read the Docs probes).
- Tracked tree: `python scripts/check_tree_size.py --check`; budget in `scripts/tree_size_allowlist.json`.
- `artifacts/` layout map (2026-10-27): 21 of 23 root folders are pinned by frozen records, record pins or
  runtime paths; the unexempted root folders sit at the limit of 7 (`tests/test_artifacts_tree_degree.py`).

## Next action

Work goes to the CLI workers (opencode Muse, Cursor GLM 5.3 flash, agy Gemini; Hamm 2026-10-28), at most 3
lanes, cloned from a still base at `E:/repos/_wt/jaxfne/base`. Next: T-3 R2 (the WSL oracle run), C4 (radius
cutoff), C6 (laminar benchmark), the P-052 beta scan, the Figs 5 and S1 runners. Tag each fidelity fix with
`Fidelity-fix: <node id>`.
