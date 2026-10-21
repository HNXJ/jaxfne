# state

Stamp: jaxfne@dev@2bdc1701 · 2026-10-21 (written 2026-10-21T12:07:39Z)

Overwritten each session. Where this file and git or a gate disagree, git and the gate win.

## Works (with evidence)

- 0.5.7 published 2026-10-21: tag v0.5.7 on main 03a86014 (PR 113; CI Fast 39585668154, Release
  CI 39585668167 green), GitHub Release, PyPI via Publish run 39595006450; receipt
  `artifacts/release/v0_5_7_release_receipt.json`. rc gate PASS on dev 508dffa9 (broad 6917
  passed, 0 failed; JUnit parity 7053 = 7053 nodes against Release CI 39565883872, 6 explained
  differences). Install and Colab docs name 0.5.7 as published. Release body edited
  2026-10-21T14:09:15Z (Hamm: "Name it, fix in 0.5.8"): a Known issue section names the
  `istdp_vogels2011` presynaptic trace scaling with out-degree; the fix is f57aae0f in the kernel
  LFP lane (todo section 2). Tag, PyPI files and receipt unchanged.

- 2026-10-20: Brunel 2000 Fig. 8 reproduced (conditional) under signed P3 v4 (seeds 4-8, dt 0.1 ms,
  CPU float32; float64 at C agrees): `papers/brunel2000/RESULT_P3.md`, REPLICATIONS, dashboard.
  0.5.7 lanes A1-A7 integrated (A1 optimizer pins, A2 one checklist path, A3 records as run, A4 one
  sign table, A5 one drive-default table, A6 ruff E741/E402 gated, A7 P-035/P-036 archived);
  Brunel P5 ledger; references page split into core references and 58 research studies. Left for
  0.5.7: A1b lane tests, P-044, Brunel S2/S3, A9 (todo section 2); Destexhe/GSDR pilots to 0.5.8.
  Batuev and Sukhov 2026 (Dynamics 6(4):42) queued at P0 (Hamm 2026-10-20).
- 2026-10-19 evening: release plan 0.5.7-0.5.10 (`etudes/meta/release_plan_0.5.7-0.5.10.md`):
  0.5.7 truthful core, 0.5.8 core-compliant replications, 0.5.9 scale, 0.5.10 fields; D1-D3 for Hamm.
- 2026-10-19 evening: Hamm signed brunel2000 P3 v4 (`PREREG_P3_v4.md`: band (b), the paper's
  sim-theory gaps; Q3 (ii), stationary global plus CV > 0.1); O-52 closed. Scored run seeds
  4-8 at A-D, dt 0.1 ms on the CPU, launched on the commit carrying v4.
- 2026-10-19 later on dev: Brunel Fig. 8 A-D at full scale (N=12500, seed 0, B8 pilot,
  unscored; brunel2000 CALIBRATION): cell rates 56.3, 37.2, 5.84 Hz at B, C, D against
  60.7, 37.7, 5.5; global 197 Hz (B) and 22.5 Hz (D) against ~180 and 22; A synchronous
  regular at 333 Hz; C less irregular than AI (CV 0.39). Core: vectorized `fixed_indegree`
  sampler and host-side construct (full-scale construct from over 15 min to 1-4 min,
  outputs bit-identical), `activity_regime` reports `spectral_peak_hz`. Potjans 2014 P2
  pilot. Broad gate on cl-int11 (dev 32997f0e + 5): 6786 passed, 0 failed; dev 7bbb5f7c.

- 2026-10-19 on dev: sol's control-surface audit and the scan's keeper fixes O-40 to O-43
  (x64 test marker, stale pointers and flags, GSDR MCDP in the core with corrected fit
  manifests, REPLICATIONS records); Hamm's 2026-10-18 rulings (T-33 records, refusals of
  non-zero `Simulation.plasticity` and absent gpu/tpu, reserved lowercase rule names,
  tree-degree scope, tooling triage, foil sweep); `problem_stack.md` retired into the open
  stack; dashboard class "mixed" (CAMINOS); T-32a lane kit, T-32b `simulate_hh_batch`;
  four new lanes at P1 (Brunel 2000, Potjans and Diesmann 2014, Vogels 2011, Mejias 2016);
  RTD strict. Broad gate on fc5d13f4: 6574 passed, 0 failed.

- 2026-10-18 later on dev: T-22 intra-step fixed point runs to quiescence with
  `intra_step_rounds` as a refusing cap (Hamm 2026-10-18); CAMINOS true zero delay scored on it
  (CAMINOS CALIBRATION §11: 3 rounds used per step, lag about 7 ms, f_pk 43–46 Hz, as the 0.1 ms
  run); GSDR refit on the reference morphology stays saturated and the excess traces to per-cell
  OU drive (GSDR CALIBRATION §6.12–6.13; CPU and GPU agree). Broad gate on 1764f4b0: 6501
  passed, 0 failed; on f554a7e2: 6503 passed. CI Fast green on 1764f4b0 and c63e1abf.

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
- Whole-repo audit 2026-10-17 (two agents, verified by a third pass): open items in todo
  T-30 (replication conformance) and T-31 (silent fallbacks, gates); owner calls since ruled.

- 0.5.6 published 2026-10-15: tag v0.5.6 on main b24e09ef (Release CI 38791659454 green), GitHub
  Release, PyPI via Publish run 38801056076; receipt `artifacts/release/v0_5_6_release_receipt.json`.
  Release merged back into dev.
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
- Tracked tree: 2520 files (budget 2520, `budget_files` in `scripts/tree_size_allowlist.json`);
  89225818 bytes (budget 90177536, 86 MiB since 2026-10-19), measured 2026-10-19 by `check_tree_size.py`.
- main a98e6149: 0.5.6 merged from dev 4842b87c by PR 108 (2026-10-15). Main CI green: CI Fast
  run 38757281037, Release CI with slow tests run 38757281031. rc gate PASS on 4842b87c with
  JUnit parity against Release CI run 38743501399 (3.11 and 3.14 each 6030/6030, 0 unjustified).
- Dependabot PR 106 (ruff 0.17.0 in the dev extra) and PR 107 (docs pymdown-extensions >= 12.2)
  merged into dev and closed on GitHub.

## Broken or in flight

- PCL lane (lane 2b, rebuilt as a NeuronalTensor on the core): Stage C passed through the tensor
  route on 20 clips (CALIBRATION.md Step 9). Open in the record: the `no construct route (OPEN)`
  marker (CALIBRATION.md:24) is not marked closed.
- 2026-10-19: Read the Docs `latest` green again on main cf46d6d3 (PR 110, fix-only port of dev
  617c09c0; P-035 closed). Left of Hamm's 2026-10-18 rulings: T-33 GSDR/Destexhe B8 runs and
  the PCL Fig. 3 core rescoring; T-32 pilots for the four new lanes (O-50).
- GSDR: saturated at both g* (κ_fit 0.60 against 0.0156); the rate estimand waits on
  `PREREG_P3.md`.
- Whole-project scan of 2026-10-18 done: verified findings P-036 to P-042 in the issue log,
  open-stack items; all closed by 2026-10-19 except the RTD étude stand-ins (O-41 remainder).

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

Agents: T-32 toward one replication a day (GSDR fit on `simulate_hh_batch`, pilots for the
four P1 lanes, T-32c–e), the rest of T-33; then T-30 per paper.
