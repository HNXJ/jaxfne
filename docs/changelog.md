# Changelog

All notable changes are documented here in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Changed
- Chapters P/Q: human docs de-parrot agent jargon; `clamp_truth_gate_metadata` blocks claim escalation on update_metadata/manifest/migrate_schema.
- Root declutter: contributing guide under `docs/` + `.github/CONTRIBUTING.md`; changelog only under `docs/changelog.md`.

## v0.4.6 (2026-07-12)

**Internal git tag** for the release-readiness polish wave (README/docs community files, citation metadata, scope page, CI coverage, root cleanup). Distribution to GitHub Release / TestPyPI / PyPI stays deferred until Hamm confirms **0.4.7**.

### Added
- `CITATION.cff`; contributing under `docs/contributing.md` + `.github/CONTRIBUTING.md`; changelog under `docs/changelog.md`
- `docs/scope_and_status.md`, `docs/for_ai_agents.md`, Zenodo wiring guide
- `jaxfne/py.typed`; pytest-cov coverage artifacts in CI

### Changed
- README landing page shortened; Jaxley comparison in docs only
- Lean `AGENTS.md`; `.legacy/` under `artifacts/legacy/`

### Removed
- Root `CODE_OF_CONDUCT.md`

## v0.4.5 (2026-07-03)

**HDP v2, NeuronalTensor as a first-class circuit representation, full-repo visualization isolation, and a large test/doc alignment pass.** The homeostasis-dependent-plasticity (HDP) kernel gained a real, stability-validated second generation (linear equilibrium controller + safety barrier, per-cell-type drive tuning, wired into `RuntimeConfig`), `NeuronalTensor` became a genuine second build path (Areas × Layers × NeuronTypes × AreaConnections, JSON round-trip, multi-area placement, a bridge into `Configuration`), and every `matplotlib`/`plotly` call in the installable package was consolidated under `jaxfne.vis` (verified zero leakage elsewhere). Alongside the features: all 89 public docs got a real content review against current source (average score 81→91/100, several docs that documented fabricated APIs were rewritten to match reality), the test suite was consolidated toward étude/suite-notebook execution as the primary coverage mechanism, and CI was fixed for real (a pre-existing failure and several stale exclusions were traced and genuinely resolved).

### Added
- **HDP v2** — passive resource income, H-taxed synaptic drain, `hdp_rule` families (`signed_linear`/`signed_quadratic`), a linear equilibrium controller (`K_ctrl`) restoring `H_i` toward 1.0, and an asymmetric safety barrier near `H_min`/`H_max`. Wired into `core.py`'s `RuntimeConfig`/dispatch and exposed via `Configuration.hdp()`. `K_ctrl=5.0` validated as the genuine stability-critical term after root-causing a prior instability (F-017).
- **`NeuronalTensor`** — the canonical `[Areas, AreaConnections]` circuit representation (`Area = [Layers × NeuronTypes, InterConnections]`), with `Pose3D` multi-area 3D placement, `merge_neuronal_tensors`, JSON save/load, and `neuronal_tensor_to_configuration()` bridging it into the existing `construct()`/`simulate()` pipeline. `construct()` now dispatches on input type (`Configuration` or `NeuronalTensor`) transparently. Canonical JSON configs (including `default_macaque_V1`) promoted into `jaxfne/configs/`.
- **`Configuration` declarative verbs** — `.plasticity()`, `.homeostasis()`, `.connectivity()`, `.drive()`, `.optimizer()` record structured intent into metadata for `manifest()`/inspection (none of these affect `simulate()` directly — each is honestly self-documented as declarative-only).
- `jtfne.connect(...)` — model-to-model ensemble operator; `Configuration.connections()` compiles declarative selector rules into real edges at `construct()` time via a mechanism-aware connection compiler.
- `general_sequential_oddball_paradigm` — a backbone for arbitrary sequential-task paradigms (local/global oddball, omission, DMS-style event lists), replacing several one-off paradigm builders.
- 4 new étude notebooks (5, 6, 7, 8), closing the last remaining placeholder (étude 8 — continuous HDP adaptation).
- `cable_filter_tensor` and `csd_tensor` — named, standard pipeline stages for LFP/EEG/MEG and CSD, replacing ad hoc inline math.
- `Synaptic Tensor` — an additive mechanism-correct tau lookup (AMPA/NMDA/GABA-A/GABA-B) usable independent of the recurrent-network path.

### Fixed
- **Visualization isolation completed** — every real `matplotlib`/`plotly` call in `jaxfne/` now lives under `jaxfne/vis/*`; the two remaining references outside it (`export.py`/`tutorial_utils.py`) are lazy imports inside deprecated shims that delegate to `jaxfne.vis.export_figure`, both already carrying `DeprecationWarning`.
- 4 dead-stub `jaxfne.vis` plotting functions (built a figure, plotted nothing, returned `None`) implemented for real.
- HDP JIT cache key now fingerprints all `homeostasis_params` (was silently reusing a stale compiled kernel on some parameter changes); homeostasis diagnostics now correctly forward `w_trace`/`w_final`.
- A real N=2 cell-type edge case, a silent bad-dtype fallthrough (now rejects loudly, with `bfloat16` support added), and several NeuronalTensor↔Configuration bridge fidelity gaps (per-area/per-layer cell-type fractions, `AreaConnection`/`InterConnection` wiring, `PlasticParams.H`/reversal potentials).
- `NeuronalTensor`/`Area` now validate element types at construction (`__post_init__`) — passing the wrong type (e.g. a `Configuration` where an `Area` belongs) raises `TypeError` immediately instead of silently corrupting state.
- CI: removed a stale `--ignore` exclusion for 2 test files that had been broken in May and were later fixed but never re-included; added the `kaleido` dependency (a pre-existing gap, present in every CI run checked back through history); fixed several doc-hygiene lint failures introduced by the doc-alignment pass itself.
- `project_laminar_sources` now defaults to `density_preserving` (was a silent behavior mismatch with its own documentation).

### Changed
- **Test suite consolidated** toward étude/suite-notebook execution as the primary coverage mechanism: of 210 test files reviewed, 2 deleted and 41 slimmed to keep only tests a notebook run can't replace (pure-function invariants, error paths, schema/contract checks); the rest were confirmed to already carry unique value and left untouched.
- **Full documentation review**: all 89 `docs/*.md` files given a real content review against current source (not just a scan) — average score 81→91/100, 88/89 fully resolved. Several `docs/api/*.md` files that documented entirely fabricated dataclass fields/function signatures (up to 65 points below their post-fix score) were rewritten to match reality.
- `pyproject.toml` gained `[project.urls]` (Homepage/Repository/Documentation/Issues/Changelog).

## v0.4.4 (2026-06-21)

**Multi-area études + a real fix to inter-area connectivity.** Études 6 and 7 are implemented (previously placeholders), built on the built-in Izhikevich emitter and the canonical E:I profile. Implementing them surfaced and fixed genuine bugs in the multi-area connectivity path.

### Fixed
- `inter_column_connectivity` now **accumulates** specs instead of overwriting — multiple calls (one per directed projection) all materialize. Previously every call replaced the single metadata key, so `build_multi_area_columns` wired only the last adjacent pair.
- `build_multi_area_columns` wires each adjacent pair in **both directions**: feedforward lo→hi (L2/3 → L4) and genuine top-down feedback hi→lo (L6 → L1/L5). Its `p_feedback` now produces real top-down edges.
- `_interarea_W` layer matching tolerates the merged 5-layer scheme (`"L2/3"`) as well as split `"L2"`/`"L3"` columns, so feedforward from superficial layers wires under the default `DEFAULT_LAYERS`.

### Added
- **Étude 6 — Multi-Area Network**: V1→V2→V4→PFC hierarchy with feedforward + feedback, inter-areal connectivity census, interactive 3D layout, per-area depth rasters, and an async-irregular (κ) gate.
- **Étude 7 — Multi-Trial Spectrolaminar Motif**: multi-trial depth × frequency relative power on a 1k canonical column, with the synchrony (κ) trust gate and the regime-based crossover caveat (oscillatory layers + low κ — not N alone).

## v0.4.3 (2026-06-21)

**Jaxley emitters reach the field stage, plus the flagship Configuration Grammar guide.** A Jaxley Hodgkin–Huxley network can now drive a laminar LFP/CSD readout from a physically meaningful generator — its reconstructed transmembrane ionic current — closing `Emitter → Source → Field → Probe` for the Jaxley bridge. jaxfne is the mathematical backend; the biophysical fidelity of the readout follows the model you provide.

### Added
- `JaxleyBridge.simulate_laminar_field(...)` — run a Jaxley HH model and return `Signals` with a laminar `FieldOutput` (`lfp_proxy`/`csd_proxy`/...). The source is the reconstructed HH ionic current `I_Na + I_K + I_Leak` (from recorded `HH_m/HH_h/HH_n` gating states + channel params), projected via `project_laminar_sources`. `signals.get("lfp_proxy")` and `jaxfne.vis.lfp`/`csd` work directly on the result. Izhikevich/Fire are non-capacitive (zero current) and raise a clear error.
- `normalize_depth` option (default on) min-max rescales the depth axis to `[0, 1]` so arbitrary-scale (µm) Jaxley geometry maps onto the projection's contact span instead of collapsing onto one contact; the raw range is preserved in metadata.
- **Configuration Grammar guide** — the flagship documentation: `Configuration` framed as the compiler whose declarative specification compiles into the Emitter→Source→Field→Probe→Objective→Optimizer→Manifest chain, with each of the eleven sections mapped to its real fluent method and described as a biophysical-specificity dial.
- **Homeostasis guide** — the homeostatic excitability controller (one parameter `k_gain`, `k_gain=0` null), covering both the built-in per-step kernel (`runtime(enable_homeostasis=True, homeostasis_params=...)`, diagnostics via `model.last_homeostasis_diagnostics()`) and the Jaxley outer-loop windowed controller.
- **Bridges API reference** (`api/bridges.md`) — `require_jaxley`/clip shim, `jaxley_to_signals`, and all three `JaxleyBridge` run modes.

## v0.4.2 (2026-06-20)

**Homeostatic numerical stability — hard-bounded, float32-safe emitters.** Enabling the homeostatic mode (one knob, gain `k≈1.0`) keeps each emitter in a stable operating regime AND hard-bounds its state, so a single neuron or simple component can never overflow or underflow in float32. The same mode eliminates hyperactivity and hypoactivity and provides short-term adaptation. Proxy/scaffold gates unchanged.

### Added
- Hard state bounds in the built-in Izhikevich homeostatic kernel (`v_floor`/`v_ceiling`/`u_abs_max`/`syn_abs_max`, tunable via `homeostasis_params`) — set far outside normal dynamics, so they never alter behaviour, only catch overflow/underflow. State stays finite in float32 under any finite (or even `±inf`) input current.
- `JaxleyBridge.simulate_homeostatic(..., current_clip_nA=, strict_finite=)` — the injected current is hard-bounded to keep the implicit solver away from overflow; output finiteness is verified (raises in strict mode rather than silently masking). Metadata records `state_hard_bounded` and `all_finite`.

### Notes
- The bounds are a safety net: in the normal regime the clamps stay inactive.

## v0.4.1 (2026-06-20)

**Jaxley emitters become a first-class path, with tfne homeostasis on top.** Proxy/scaffold gates unchanged.

### Added
- `jaxley_to_signals(module, recordings, dt_ms=...)` — convert a `jaxley.integrate` recordings array `(n_rec, n_time)` to a jaxfne `Signals`, pulling recorded-compartment xyz from `module.nodes` into metadata for downstream projection (exported at the root).
- `JaxleyBridge.simulate(...)` — run a Jaxley model end-to-end (stable `bwd_euler`) and return proxy `Signals` (previously a placeholder).
- `JaxleyBridge.simulate_homeostatic(...)` — outer-loop windowed homeostatic excitability controller around a Jaxley emitter, applying tfne's restoring-bias law `g = clip(k_gain·(target_rate_hz − r), g_min, g_max)` as a per-cell current via grad-safe `data_stimulate`, stitched with continuous state resume. Computational control proxy only (`biological_learning_claim=False`, `mechanism_claim_status="not_claimed"`).
- `hh_jaxley_reference_trace(...)` — real single-compartment Hodgkin–Huxley reference trace (previously a placeholder).
- ED10 release-archive receipt (`scripts/ed10_release_archive_receipt.py`) — binds release identity + truth gates to content hashes of the upstream evidence bundle (ED9), then self-hashes.

### Fixed
- `require_jaxley()` lazily installs a backward-compatible `jnp.clip(a_min=/a_max=)` shim so Jaxley (≤0.13) channel emitters integrate on current JAX (the shim self-disables when Jaxley adopts `min=/max=`). Metadata records `jax_clip_compat_installed`.

### Changed
- `jaxley` optional-dependency extra floored to `>=0.13.0` (tested version).
- CI now installs the `jaxley` extra so the Jaxley integration is exercised (drift-tested) on every push.

## v0.4.0 (2026-06-18)

**Fluent Configuration grammar + proxy-only probe vocabulary.**

### Added
- `Configuration.geometry(layer_thickness=...)` — declare laminar geometry from per-layer thickness (normalized to cumulative z-intervals).
- `Configuration.population(N, neurons={...})` — per-layer neuron budget decoupled from thickness (largest-remainder allocation; one `N` per area).
- Real inter-area edge wiring in `Configuration.inter_column_connectivity(...)`: materializes cross-area synapses with anatomical routing (feedforward L2/3→L4, feedback L6→L1/L5) and an explicit `layer_to_layer_map` override.

### Changed (breaking)
- The `*_like` probe vocabulary (`lfp_like`/`csd_like`/`eeg_like`/`meg_like`) is **fully retired with no aliases**. Use `*_proxy` names only; signal access and probe declarations reject `*_like` with a clear pointer to the `*_proxy` name.

### Fixed
- CI workflow `scope:`→`strategy:` (matrix jobs now run); retired-term doc guard; mkdocs strict-build links.
- Updated all tutorial notebooks to execute against the current API.

### Scope
- Outputs remain proxy readouts (`field_solver_status=linear_solver`, `physical_amplitude_calibrated=false`). No scientific-claim escalation.

## v0.3.42 (2026-06-14)

**Public context hardening release.**
- Added missing docstrings to stub emitters (`GLIFEmitter`, `LIFEmitter`).
- Enforced clean-venv lazy root imports to guarantee heavy optional packages are not loaded on import.
- Harmonized version references across documentation, package configurations, and tests.

## v0.3.41 (2026-06-14)

**Port of JAX computational kernels.** No scientific-claim escalation.

### Added
- JAX spectral analysis functions: `spectrolaminar_psd_jax`, `bandpower_jax`, `spectrolaminar_readout_kernel_jax`, `spectrolaminar_similarity_kernel_jax`, and vectorized batched variants `spectrolaminar_similarity_candidates_jax`, `spectrolaminar_similarity_candidates_seeds_jax`.
- Tensorized static-shape connectivity rule compilation kernel `compile_connection_rules_jax`.
- JAX-optimized activity-dependent STDP synaptic weight update kernel `update_stdp_weights_jax`.
- `StimulusSchedule.to_array_jax` for compiling event schedules into JAX arrays.
- Auto-JIT cache warming compile tracking under `Model._warmup_times` during JIT initialization.
- Experimental volume conductor loud-fail skeleton `solve_volume_conductor_experimental` that raises `NotImplementedError` pending boundary/gauge/calibration validation.

### Scope
- Outputs remain proxy readouts (`linear_solver`, `physical_amplitude_calibrated=false`).


## v0.3.40 (2026-06-13)

**Pre-0.4.0 hardening + device flexibility.** No scientific-claim escalation.

### Added
- `RuntimeConfig.backend` (cpu/gpu/tpu) is now **honored** for device placement —
  `simulate()` pins compile+execute via `jax.default_device`; `runtime_report()`
  reports `backend_enforced` by device availability (honest downgrade when
  absent). dtype (float32/float64) and jit remain per-`RuntimeConfig` adjustable.
- Accurate docstrings for all public `jaxfne.__all__` members; fail-loud
  pre-0.4 physical-solver placeholder in `experimental_hpc` (raises until a
  validated solver exists — the stable path stays `linear_solver`).

### Changed
- Repo hygiene: root thinned; non-public planning/roadmap material removed from
  the published repository.

### Scope
- Outputs remain proxy readouts (`linear_solver`,
  `physical_amplitude_calibrated=false`); computational scaffold only.


## v0.3.39 (2026-06-13)

**Stable release:** packaging/docs/PyPI consolidation of the v0.3.37–v0.3.38
quality line. No new APIs, no solver work, no scientific-claim escalation.

### Summary of the v0.3.37 → v0.3.39 quality line
- **v0.3.37** — docs comparison: strict notebook grammar, root export grammar,
  scope/truth-gate cells, `*_proxy` naming, API index regenerated.
- **v0.3.38** — root-export hardening: the six export helpers (`save_figure`,
  `save_figures`, `export_report`, `export_tutorial_artifacts`, `plot_raster`,
  `plot_spectrolaminar_suite`) promoted to formal `jaxfne.__all__` members
  (`len(__all__)` 179 → 185); clean-venv root-import laziness verified.
- **v0.3.39** — stable packaging: version bump to `0.3.39` across
  pyproject/core/mkdocs/docs; wheel + sdist rebuilt and `twine`-checked;
  published to PyPI, GitHub Release, and docs.

### Validation
- Targeted pytest, `mkdocs build --strict`, and `compileall` pass.
- Root import loads no `matplotlib`/`plotly`/`pandas`/`optax`/`jaxley`/`pynwb`.
- API index count equals runtime `len(jaxfne.__all__)` = 185.

### Scope
- Outputs remain proxy readouts (`linear_solver`,
  `physical_amplitude_calibrated=false`); computational scaffold, not a
  calibrated biological simulator.


## v0.3.37 (2026-06-13)

**Release:** Strict notebook grammar + truth gates. Published to PyPI (wheel +
sdist), GitHub Release (tag `v0.3.37`, commit `49aa025`), and docs.

### Added
- Root-level export grammar: `save_figure`, `save_figures`, `export_report`,
  `export_tutorial_artifacts`, `plot_raster`, `plot_spectrolaminar_suite` —
  the canonical replacement for direct `matplotlib`/`json` calls in
  release-facing notebooks.
- Scientific scope cells in all 15 release-facing tutorials documenting
  `computational_scaffold` / `proxy_readout` status, local nonlinearity, global
  linearity, and the `physical_amplitude_calibrated = False` truth gate.

### Changed
- Strict notebook call grammar: root-level `jtfne.<fn>()` only (no
  `jtfne.vis.*` / `jtfne.tutorial_utils.*` in public notebooks).
- Tensor-field naming standardized to `*_proxy` (e.g. `LFP-proxy`, `CSD-proxy`);
  public `*_like` wording removed.

### Validation
- 2284/2284 tests pass; `mkdocs build --strict` passes; `twine check` passes on
  both artifacts.
- Truth gates enforced: `field_solver_status = "linear_solver"`,
  `physical_amplitude_calibrated = False`; no physical EEG/MEG/LFP/CSD
  measurement wording.

### Scope
- Outputs remain proxy readouts (`linear_solver`,
  `physical_amplitude_calibrated=false`); the package is a computational
  scaffold, not a calibrated biological simulator.


## v0.3.26 (2026-06-02)

**Feature release:** Multi-area laminar workshop — inter-area connectivity, lesioning, AGSDR tuning, waveform explorer.

### Added
- Global multi-area network in `simulate_laminar_trials` with within-area recurrence + inter-area projections (`connectivity_spec`) and lesioning (`lesion_spec`); per-area depth leadfield. Defaults: feedforward V1 L2/3 (E) → V4 L4, feedback V4 L2/3 (E) → V1 L5/6.
- Per-cell-type Izhikevich overrides (`cell_type_izh_params`, wider "E-Wide" default), `single_cell_waveforms`, and `tune_laminar_agsdr` (AGSDR tuning of firing rate + kappa).
- Étude No. 1 rebuilt as a two-area Colab-ready customizable workshop notebook.

### Fixed
- Trial-averaged spectral power (per-trial PSD then mean); relative-power-density cross; uniform `dt_ms` kwarg across vis plotters; example API drift; GHA `checkout@v5` / `setup-python@v6`.

### Scope
- Proxy readouts only (`linear_solver`, `physical_amplitude_calibrated=false`); spectrolaminar motif is emergent, not imposed.


## v0.3.25 (2026-06-01)

**Feature release:** Cylindric scaffold, spectrolaminar motif, and 32-contact LFP artifacts.

### Added
- Cylindric scaffold geometry: per-area vertical cylinders (circular x-y cross-section, depth on z); `visualize_network_3d` gains `column_shape="cylinder"` shells and per-cell-type marker symbols.
- Genuine per-area three-panel spectrolaminar suite (cell density, relative power spectrum, alpha-beta/gamma crossing) with smoothed band profiles.
- 32-contact LFP/spectrolaminar artifacts: equal-spaced contacts, robust relative-power normalization excluding degenerate low-power channels, 32-channel stacked LFP waterfall.

### Fixed
- Notebook controls for trials, contacts, and per-layer cell-type fractions; CI dev-extra (`ipykernel`) for slow notebook-execution tests.

### Scope
- Outputs remain proxy readouts with `field_solver_status=linear_solver` and `physical_amplitude_calibrated=false`.


## v0.3.24 (2026-06-01)

**Release candidate:** Compact Etude No. 1 tutorial utilities and Colab-ready visualization flow.

### Added
- `jaxfne.tutorial_utils` helpers for laminar column configuration, adjustable model scale, cell-type distributions, stimuli, trial simulation, spectrolaminar summaries, and strict artifact export.
- `jaxfne.vis` tutorial panels for 3D scaffold, activity traces, and three-panel spectrolaminar figures.
- Thin Etude No. 1 notebook using `import jaxfne as jtfne` and package-level helper calls.

### Fixed
- Preserved root `build_laminar_column` while adding `build_tutorial_laminar_column` as an unambiguous tutorial alias.
- Restored optional dependency isolation: core import does not require pandas, matplotlib, plotly, or ipywidgets.
- Added per-area LFP/CSD tensor shape `(trials, areas, T, contacts)` and area-tagged spectrolaminar specs.

### Scope
- Outputs remain proxy readouts with `field_solver_status=linear_solver` and `physical_amplitude_calibrated=false`.


## v0.3.22 (2026-05-31)

**Patch release:** Fix `jtfne.vis.visualize_network_3d` missing from PyPI wheel.

### Fixed
- `jaxfne/vis/__init__.py` now exports `visualize_network_3d` in the published wheel.
  The v0.3.21 PyPI wheel was built before `visualize_network_3d` was added to the
  vis subpackage export list; v0.3.22 corrects this.
- Etude No. 1 notebook: install cell updated to `--force-reinstall --no-cache-dir`
  to prevent stale Colab runtime cache from shadowing the fix.
- Etude No. 1 notebook: install verification cell raises `RuntimeError` with clear
  path/version diagnostics if the installed `jtfne.vis` is still missing the function.
- Network visualization cell now catches both `ImportError` and `AttributeError` and
  falls back to a static `geometry3d` PNG so the notebook completes regardless.

### Added
- `tests/test_vis_network3d_public_api.py` — 5 regression tests that enforce the
  `jtfne.vis.visualize_network_3d` export contract. CI now fails loudly if this
  public API is ever dropped from the package.
## v0.3.21 (2026-05-30)

**Release:** Etude No. 1 completion and notebook template standardization.

### Added
- Added Etude No. 1 as an advanced multi-laminar cortical AGSDR workflow under `tutorials/etudes/`.
- Added a canonical notebook template under `tutorials/templates/` with unified setup, truth gates, and placeholder configuration.
- Added a template guide for Suites and Etudes.

### Changed
- Cleaned duplicated Etude notebook artifacts.
- Moved release/comparison receipts into `internal_docs/release_receipts/`.
- Updated release checklist and agent status metadata for the v0.3.21 release candidate.

### Validation status
- Package import and compile gates pass.
- Etude and template notebooks pass structural hygiene checks.
- Maintains `computational_scaffold`, `field_solver_status=linear_solver`, and `physical_amplitude_calibrated=false`.

---

## v0.3.19 (2026-05-30)

**Release:** Field proxy boundary handling improvements.

- Optimized `project_laminar_sources` boundary fallbacks for low contact counts.
- Added comprehensive boundary and stencil numerical parity tests.
- Maintains `linear_solver` status.

---

## v0.3.18 (2026-05-30)

**Release:** Sharding infrastructure for multi-device AGSDR.

- Added `jaxfne/sharding_utils.py` with distributed mesh and NamedSharding stubs.
- 14 new tests for sharding context and single-device fallbacks.
- Sharding stubs do not yet drive multi-device dispatch (reserved for v0.3.20+).

---

## v0.3.17 (2026-05-30)

**Release:** Dtype inheritance in AGSDR optimization.

- Updated AGSDR loop to inherit dtype from bounds, not force float32.
- Applied dtype-inheritance to noise generation, W_init, and delta-rule center updates.
- 12 new tests covering dtype invariants and candidate clipping.

---

## v0.2.3 (2026-05-19)

**Release:** Stable proxy operators and documentation infrastructure.

- Added MkDocs-based documentation site with Material theme
- Reorganized docs: tutorials, guides, API reference, about
- Added Jaxley interoperability documentation
- Cleaned public documentation (removed internal metadata)
- 492 tests passing

## v0.2.1 (2026-05-10)

- Introduced probe operator contracts
- Added status metadata
- Eight canonical readout channels

## v0.2.0 (2026-04-15)

- Initial release
- Izhikevich emitters
- Laminar field solver (proxy)
- Basic readout operators
