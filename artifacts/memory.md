# `artifacts/memory.md` — jaxfne agent orientation (durable facts only)

<!-- Every backticked repo path and `jaxfne.*` symbol here is resolved by
tests/test_memory_brief.py; keep SHAs, versions, counts and status out. -->

> Read first. Mutable state: `artifacts/todo_stack.md`. Stable facts: `artifacts/fact_stack.md`. Procedure: `artifacts/control/skills/`. Rules: `artifacts/AGENTS.md`. Verified failure lessons: §10. Import: `import jaxfne as jtfne` (`README.md`).

## 1. What jaxfne is

- JAX package for Tensor-Field Neural Equations (TFNE): a containment and composition model for neural models at several resolutions, not one prescribed biophysical equation (`artifacts/AGENTS.md`, `README.md`).
- Scientific grammar: `Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest`. Execution grammar: `CircuitSpec -> construct -> Model -> simulate -> Signals`.
- Factor at specification time, flatten at execution time; biological identity, topology, signs and parameter ownership survive compilation.
- Internal quantities stay relative; absolute units come only from explicit calibration transforms (`docs/theory/doctrine/relative_quantity_grammar.md`).
- Evidence vocabulary: SPECIFIED / IMPLEMENTED / TESTED / OBSERVED. Failed prospective receipts are preserved, never tuned post hoc.

## 2. Repository map (paths present in this tree)

| Path | Purpose |
|---|---|
| `jaxfne/` | Package (§3) |
| `tests/` | Verification; `_numeric_gates.py` holds shared numeric helpers |
| `scripts/` | Gates, audits, generators, runners (§6) |
| `artifacts/` | Project state, evidence, programme records (§7) |
| `docs/` | Published docs (mkdocs strict); `docs/theory/doctrine/` holds canonical math |
| `examples/` | Release-smoked scripts |
| `.github/workflows/` | `ci.yml` (Fast), `release_ci.yml`, `notebook_execution.yml`, `publish.yml` |
| `artifacts/programme/`, `artifacts/atlas/` | Receipts, `atlas_coverage.json`, `claim_ledger_055.md`; Atlas assays `at*.py`, results in `artifacts/atlas/results/` |
| `artifacts/control/skills/`, `artifacts/release/` | Canonical `jaxfne-*` skills; release receipts, `current_release_authorities.json` |
| `scripts/run_test_gate.py` | Source of truth for gates `dev`, `broad`, `release`, `rc`, `publication` |
| `scripts/harness/sync_skills.py --update --manifest` | Canonical skills → mirrors (the only sanctioned edit→sync flow) |

- Markers (`pyproject.toml`): `slow`, `notebook`, `release`. Dev gate modules: `scripts/run_test_gate.py:DEV_PYTEST_TARGETS`.
- Surface pins: `tests/test_public_surface_contract_v0413.py`, `tests/test_public_api_snapshot_v034.py`. Gate hierarchy: `tests/test_release_gate_hierarchy.py`.

## 3. Package map

Tiers per `jaxfne/public_surface.py`, which owns the full name list: CANONICAL, ADVANCED (reachable, not in `__all__`), COMPATIBILITY, EXPERIMENTAL_INTERNAL.

| Module | Responsibility | Main entry points |
|---|---|---|
| `jaxfne` (root, CANONICAL) | CircuitSpec → Model → Signals | `configuration`, `construct`, `simulate`, `connect`, `compute_fields`, `manifest`, `run_continuation` |
| `jaxfne.neuronal_tensor` | Preferred CircuitSpec form | `NeuronalTensor`, `Area`, `Layer`, `NeuronType`, `load_canonical_neuronal_tensor` |
| `jaxfne.tfne` (ADVANCED) | Spec-language compiler | module `tfne` (parse, resolve, realize inside) |
| `jaxfne.jdna` | Development genome → tensor | `PseudoGenome`, `develop`, `load_canonical_pseudogenome` |
| `jaxfne.emitters` (kernels ADVANCED) | Neural dynamics | `Emitter`, `IzhikevichEmitter`, `standard_receptor_specs` |
| `jaxfne.fields` | Sources → fields → probes | `construct_source_tensor`, `project_laminar_sources`, `probe_laminar_modes` |
| `jaxfne.connectivity` | Connection compilation | `compile_connection_rules`, `ConnectionCompileResult` |
| `jaxfne.builders` + `jaxfne.presets` | Column and area construction | `build_laminar_column`, `build_multi_area_columns`, `CELL_TYPE_PRESETS` |
| `jaxfne.optim` | Tuning optimizers | `agsdr`, `gsgd`, `random_search`, `OptimizerSpec` |
| `jaxfne.paradigm` | Stimulus paradigms | `Paradigm`, `standard_visual_omission`, `rate_synchrony_targets` |
| `jaxfne.io` | Evidence I/O | `save_receipt`, `config_hash`, `sha256_file` |
| `jaxfne._pipeline` | State and continuation | `DynamicState`, `checkpoint_state`, `restore_state` |
| `jaxfne.vis` (lazy) | View-only panels | `plot_raster`; `jaxfne.vis.atlas_suite` is the 7-panel Atlas contract |

## 4. Core workflow

```python
import jaxfne as jtfne
jtfne.enable_x64()  # before array construction, for float64
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)
model   = jtfne.construct(tensor, runtime)
signals = jtfne.simulate(model)  # inherits duration/dt/seed from runtime
```

- `Configuration` is the compatibility path (`docs/start/quickstart.md`). HDP: `RuntimeConfig(enable_hdp=True, hdp_params={...})` to `simulate`. STDP is separate: `run_stdp_stream`.
- Continuation: `checkpoint_state` / `restore_state` / `run_continuation`; chunked ≡ continuous, bit-exact, delays in flight included.

## 5. Key concepts

- **NeuronalTensor**: preferred CircuitSpec (`Area → Layer → NeuronType`, `InterConnection`, `AreaConnection`); canonical genomes in `jaxfne/configs/`. **Configuration**: fluent compatibility builder.
- **source** `Q` is derived from executed activity `X`. **field** `Φ(r,t)` is a proxy solve (`conductivity="proxy"`), never calibrated unless validated. **probe** refuses invented contacts unless opted in.
- **H / RBS, RBD, HDP, W**: relative biophysical state is a dependency container, not one homeostatic scalar; RBD is dynamics of H (fixed W is valid); HDP is per-rule plasticity `dΘ/dt`. Sources: `artifacts/fact_stack.md`, `docs/theory/doctrine/rbs_rbd_hdp.md`.
- **connect / JDNA**: `connect()` composes areas; `develop(G, seed)` completes a PseudoGenome under the TFNE constraint.
- **Atlas AT-01…AT-10**: canonical simulations from `artifacts/project_sources/8_atlas.md`. AT-10 is the 20-area system; its registry id is `AT-10-N20`.
- **OMITTED / REFUSED**: absent or refused quantities are shown as sentinels, never synthesized (`jaxfne/vis/atlas_suite.py`).
- **Calibration levels**: CALIBRATED ≠ REDUCED_PHYSICAL ≠ RELATIVE_PROXY; live fields carry `physical_amplitude_calibrated=False`.

## 6. Verification

- Gates: `dev` (curated architecture), `broad` (`-m "not slow"`), `release` (broad + slow + notebook + strict docs + examples), `rc` (release + parity, build, twine, isolated smoke, attestation). Run them with `python scripts/run_test_gate.py <gate>` (aliases in `Makefile`).
- CI Fast (`ci.yml`, Ubuntu, py 3.11 and 3.14): compileall → ruff → docs audits → atlas firewall and coverage → docs strict build → doc↔code integrity → `run_test_gate.py dev` (dev) or `broad` (main) → examples → wheel smoke.
- Run environment (Hamm 2026-10-13: every agent, every jaxfne run): WSL Ubuntu, venv `~/venv_jaxfne` (Python 3.12, jax CUDA, pytest). From a Windows clone, in Git Bash: `wsl --cd "$(pwd -W)" -- bash -lc 'PYTHONPATH=. ~/venv_jaxfne/bin/python -m pytest -q -p no:cacheprovider <tests>'`; scripts the same way with `python <script>`. Tests and small runs set `JAX_PLATFORMS=cpu`; GPU only for a run claimed in jchat (B5), float32, backend and dtype in its manifest. PCL's numba port and the broad pre-push gate keep their Windows interpreters until a Linux gate run is recorded.
- Attestations and JUnit land gitignored in `artifacts/attestations/`; a tracked attestation changes the attested SHA. Docs policy: `docs/reference/project/ci_policy.md`.

## 7. Project state and control files

Roles and edit rules: `artifacts/AGENTS.md` (Project control). Coverage rows: `artifacts/programme/coverage/atlas_coverage.json`, checked by `scripts/check_atlas_coverage.py`.

## 8. Conventions and pitfalls

- Dtypes: `float32` default; `float64` only with x64 enabled before array construction (`jaxfne/_runtime_config.py:_ALLOWED_DTYPES`).
- Units: time in ms; doc atlases use binary-exact `dt_ms=0.5`. Delay realizes as `round(delay_ms / dt_ms)` steps; a positive delay rounding to 0 steps is refused.
- Configured ≠ realized ≠ executed ≠ effective: inspect each stage; equal counts or names are too weak; perturbations must be asymmetric.
- Frozen: `artifacts/publication/frozen_manifest.json` files, frozen etude bundles (`scripts/gate_etude_no_resim.py`), write-once Atlas outputs.
- Docs currency (Hamm 2026-10-12): at version N.M.x the docs nav carries N.M.x results; pages with results from N.(M−1).x and earlier are regenerated or listed under the top-level Archive section (`docs/archive/index.md`), files kept in place, titles without "Archive:" prefixes. Prose states what a thing is: no "X, not Y" or "X, never Y" (gate: `tests/test_public_docs_hygiene.py`).
- Public surface: tiers owned by `jaxfne/public_surface.py`; regenerate pins with `scripts/generate_public_surface_contract.py`; check CI on the pushed SHA after any export change.

## 9. Where to look first

| Task | Read first |
|---|---|
| Build and run a model | `docs/start/quickstart.md`, `docs/guides/grammar/configuration_grammar.md`, `docs/reference/api/core/core.md` |
| NeuronalTensor, canonical genomes | `jaxfne/neuronal_tensor.py`, `jaxfne/configs/`, `docs/guides/grammar/jdna.md` |
| TFNE program → execution | `jaxfne/tfne.py`, `docs/theory/doctrine/tfne_algebra.md`, `tests/test_tfne_execution.py` |
| Plasticity (HDP, RBD) | `docs/guides/state/hdp.md`, `jaxfne/hdp_rule.py`, `jaxfne/hdp_network.py` |
| Fields, probes, calibration | `docs/guides/state/calibration.md`, `jaxfne/fields/`, `docs/reference/api/readout/fields.md` |
| Visualize (never simulate in vis) | `docs/guides/interop/atlas_suite.md`, `jaxfne/vis/atlas_suite.py` |
| Atlas simulation or gap row | `artifacts/project_sources/8_atlas.md`, `artifacts/programme/coverage/atlas_coverage.json`, `artifacts/atlas/at*.py` |
| Numeric mismatch, dtype, determinism | `tests/_numeric_gates.py`, `jaxfne/util.py`, §10 |

## 10. Verified reusable lessons

Each entry: trigger → repair, then cause, evidence (by ID; SHAs and counts are in git history) and scope. Global rules are not repeated here.

- Gate passes locally, fails on a CI fresh clone (a skill doc cites the gitignored `scratch/CURRENT_TASK.md`) → allowlist it as `generated_skill_refs` in `scripts/doc_code_integrity_allowlist.json`; prove with the file absent. Evidence: CI red, then green. [harness]
- Untracking files that become build products (ignored copies stay on disk and keep every local gate green) → run the broad gate in a fresh `git clone` of the commit before pushing; a check of "the file exists" becomes a check against the tracked registry; mkdocs `--strict` does fail on markdown links to absent HTML (a raw-HTML-only probe misled). Evidence: R2 step 4, gallery collector and strict build, both caught before push. [docs/harness]
- A fresh-clone gate passes because a test module skips (`importorskip("mkdocs")`: the jax venv has no mkdocs) → run each skipping module under an interpreter that has its dependency, in the fresh clone. Evidence: a release CI run failed 2 `test_mkdocs_atlas_stubs.py` tests whose fixtures took "built" atlas files from disk; the fixtures now write them under `tmp_path`. [harness]
- Conjunctive abs+rel freeze is unsatisfiable (the abs leg demands sub-ulp agreement) → freeze in allclose form (abs governs near zero only) in `tests/_numeric_gates.py`; keep the original FAIL immutable; record the human adjudication. [harness]
- Shared helper upcasts float32 sim data to float64 (`np.asarray(x, dtype=float)`) and changes committed figures → preserve input dtype; verify by a normalized figure diff (expect stamp-only bytes). [vis]
- Large-file edit yields a diff far bigger than intended (exact-oldString edits; formatter churn) → re-read each edited region; check `git diff --stat`; unattributed reflow gets a P-ID until proven harmless. Evidence: P-004, P-008. [jaxfne]
- PowerShell ParserError or silent wrong results (no unix aliases; inline `$env:X='v';`, `?` in URLs) → Select-String, Select-Object, Get-ChildItem; single-purpose calls; complex quoting in a temp `.py` file. [windows host]
- `dt_ms=0.1` atlas builds fail `INVALID_TIME_GRID` (float32 grid drift) → binary-exact `dt_ms=0.5` for doc atlases; label the actual dt. [docs/vis]
- Regenerated Plotly HTML differs byte-wise (random div UUIDs) → treat UUID-only diffs as identical; verify via manifest sha and linkcheck. [docs/vis]
- Frozen etude trajectories do not reproduce on the tip tree (V_m and Q float hashes drift; JAX version unrecorded) → two tiers: hash, else claims (spikes and geometry exact; floats within 1e-3); record JAX and lib versions (P-003). [etudes]
- Local gates pass while CI goes red (local py3.14 Windows with no-viz split; CI py3.11 Ubuntu with viz; POSIX tests skip locally) → read each push's CI result on the exact SHA (public Actions API) before the next step; fix forward; never create a GitHub Release from a red commit. [release/CI]
- Docs build green, docs lie about the API → `scripts/audit_doc_code_integrity.py` is a permanent gate family in broad, release and rc, with an explicit ci.yml step. Evidence: P-006. [harness]
- Bounded subagent work leaves partial tree state (overlap, interruption) → disjoint file sets per worker; after an interruption run `git status` and the gate battery first; workers never commit or push. [jaxfne]
- A pre-existing `git stash list` entry → never bare-pop (it takes the top entry whatever its owner). A mistaken pop: `git reset --hard HEAD` (only on Hamm's instruction) recovers only if the tree is sealed and pushed (verify HEAD == origin/dev, clean status). Evidence: P-008. [git] OPEN, see report.
- Scripted patch over many files reports success per file (success proves only that the script ran) → each edit carries machine-checkable postconditions; a rerun reports zero changes; verify by `git diff --stat` and a whole-tree check. Evidence: P-025. [scripted edits]
- Bit-exact float hash pinned across CI legs flips on another runner (float32 reductions carry ~2e-6 relative rounding) → pin only exact passthroughs; for reduction and matmul leaves use twin-equality plus numpy-float64 agreement with a stated margin; name unproven leaves as watch items. [tests/CI]
- ruff E402 after a `sys.path` guard → ruff exempts only a bare `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`; parse gh JSON in pwsh with `ConvertFrom-Json`; pass `-R HNXJ/jaxfne` to gh outside the repo. Evidence: P-025. [windows host]
- jchat claim stays on the board after a release post (another item id, or the command's `[release]` prefix repeated) → start the release text with the bare item id, verbatim from the claim; verify with `claims`. [jchat]
- A refusal test passes while the guarded entry point stays callable (helper-only tests miss the call-site mutant) → ship an end-to-end test through the public entry point with registry-valid names; kill the check body and the call site. [tests]
- A measured compatibility table transposed on re-reading → re-read the axis labels; cross-check each cell against the tests that pin it; the suite is the arbiter. [tests]
- PR merge BLOCKED with all checks green (protect-main requires resolved review threads) → if the code does what the authorized task specified, rebut with the stack item and invariant, then resolve; verify via a GraphQL `reviewThreads` query (comment node ids are not thread ids). [git/CI]
- Assuming `gh` is unauthenticated, from an old note → run `gh auth status`; the live keyring login (HNXJ) lets PR create, merge and checks work; trust the live result. Evidence: PRs created, watched, merged. [git/CI] OPEN, see report.
- Pushing a merge to main as for dev → dev pushes bypass the rules; main is PR-only with required checks and no bypass for this identity (GH013); merge to main only through `gh pr create` and `gh pr merge`. [git/CI]
- Coverage `evidence` written as a note (the gate parses one existing path) → keep it a bare existing path; put runs, SHAs and reasoning in the stack decision. [harness]
- Null metadata in a generated Atlas manifest (`sha256` covers panels only) → fill only values certain from elsewhere (dev seed, write-once genome sha); verify a fields-only diff; never touch panels. [atlas]
- New docs page quotes paper titles with discouraged words; local check read through `| tail` (exit code of `tail`) → run every ci.yml pre-test gate alone with `rc=$?` before a docs push; allowlist exact title phrases only. Evidence: CI Fast red on the references page, fixed forward. [docs/CI]
- Wording sweep over docs passes its gate, then the suite fails (a test reruns a generator that rewrites a docs page with the old text) → grep `scripts/` for the swept phrases and fix the generator in the same commit; search line-final variants too (`with no$`). Evidence: figures_1_7 cross-audit summary. [docs]
- State-only push adds a tracked file under `artifacts/` (CI Fast skips the root-layout test; one push broke it locally) → run `tests/test_artifacts_root_layout.py` before pushing; a new root file needs an allowlist entry with a reason. [state]
- Chart data regenerated, broad gate green, Dashboard workflow red (it runs `build_dashboard_figures.py --check`; broad does not) → after any change under `dashboard/data/` run `python scripts/build_dashboard_figures.py --check` before pushing. Evidence: a Dashboard run went red on the chart-data commit. [dashboard/CI]
- A replication packet lets a worker read the authors' code → it imported and ran it (GSDR lane 4c, `compute_kappa`) → every replication packet says "read the authors' code as text only; never import or run it"; references are computed by hand in a test. Evidence: GSDR CALIBRATION.md breach record. [replications]
- A lane adds optional fields to tensor dataclasses and its packet gate passes → pinned identity and phenotype hashes still shift (fields serialized at defaults; caught only by the broad gate at integration, 2026-10-14) → `NeuronalTensor.to_dict` writes optional fields only when set; packets touching serialized classes run `tests/test_readme_atlas_pin.py` and `tests/test_jdna_area_rules_055.py`. [tensor]
- The CI-idle check and the push in one command (the push ran while CI (Release & Scheduled) on the previous SHA was in progress, 2026-10-14) → run `gh run list` alone, read every status, push in a separate command only when none is `in_progress`. [git/CI]
- A fix-forward after a broad gate changes a lane test and only that lane is rerun before the push → the replication audit (`check_replications.py --check`, inside the report gate and CI Fast) flagged a new S1 and dev went red (2026-10-20) → after any change under `artifacts/etudes/`, run `python scripts/run_test_gate.py report` (fail_count 0) before pushing. [replications/CI]
- Release CI's newest-Python leg resolves the newest jax (0.11.2 on 2026-10-26, against 0.10.x locally and on 3.11) and a scorer's stored float32 residual moves two ulps → a noise floor sits at or above the residual's rounding scale (for a central difference, eps32 * |loss| / h), never a fixed fraction under it; dispatch Release CI on dev after any new scorer, before the candidate day. Evidence: run 40262604219, `score_numerics_checks.py`. [release/CI]
- A worker sets `--basetemp` to its own scratch folder → pytest deletes the basetemp target at start, with the receipts in it (lane L8, 2026-10-26) → packets name a fresh `--basetemp <scratch>/pt_<run>` per run, beside the receipts folder. [tests/packets]
- Release version bump → the canonical atlas records the version that built it (`test_canonical_manifest_provenance`), the public surface contract states the version, and the new changelog section must hold zero ghost and adverb lint hits (older sections sit at the limits) → regenerate and promote the atlas, edit the contract line, and count lint hits per section before the broad gate. [release]
