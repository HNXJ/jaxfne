# Changelog

All notable changes are documented here in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Scope note: entries are preserved verbatim as history. Early entries reference
internal governance documents (for example `AGENTS.md` review rules) that live
in maintainer-facing areas outside this public documentation tree; they record
the rules those releases were held to and are historical context for readers.

## [Unreleased]

## v0.4.18 (2026-08-28)

Structure/docs/CI/manuscript-tooling release — no scientific kernel change (Δscience=0). Hygiene guards only; 0.4.17 frozen evidence and public API contract unchanged.

### Added
- Manuscript: deterministic build pipeline `scripts/build_manuscript.py` (`M_submission` → `PDF_submission`) with DejaVu font, equation split, deterministic ID, and pagination; manuscript sources under `docs/publication/manuscript/`.
- CI: split curated dev gate (~90 s) vs broad main gate; viz capability boundary and `reportlab` guard for build tests.
- Docs inventory: `_generated/operator_inventory.md` regeneration; `artifacts/publication/final/` figure finalization (E2 ping/SSA combined figures).

### Changed
- Docs/nav: PseudoGenome (JDNA) labeling, subtractive compaction (C→R→E words-before-symbols, PseudoGenome G→D(K_D)→N), link consolidation, and `_generated` densification.
- Structure: tutorials/figures/skills/`AGENTS.md` consolidated under `artifacts/` with frozen manifest relocation; `.opencode` authorities migrated to neutral harness owners (`opencode.json`).
- JDNA docs: retitled PseudoGenome with compact `K_D` grammar and illustrative examples.
- Doctrine/metadata: `pyproject.toml` authors and citation tag alignment; `README`/`CITATION.cff`/`bibliography` sync (TRUE_DEFECT D1–D3).

### Fixed
- Build: `--out X.pdf` always produces PDF with deterministic backend and scientific captions (P1); `reportlab`-dependent build tests skip cleanly when viz extra absent.
- Hygiene guards (no trajectory change): `Model.checkpoint`/`restore_state` UTF-8 read consistency; `EdgeList.to_dict`/`from_dict` full-payload roundtrip with dtype preservation and backend version check; edge `delay_steps >= 0` validation and dense-backend nonzero-delay rejection; `json_safe` declared `NaN/Inf → None` with `UserWarning`; `spectrolaminar_psd_jax` `window="rectangular" | "hann"` taper (default rectangular, bit-identical).
- Path hygiene: stale `artifacts/` references corrected after relocation; JAX-native/package-native prose allowlist for vocab gate.

### Unchanged
- Public root contract: 189 symbols (176 CANONICAL + 13 COMPATIBILITY) from 0.4.17 — SurrogateConfig/surrogate_config reclassified to EXPERIMENTAL_INTERNAL (W4 scoping), otherwise frozen; frozen publication evidence, receipts, and TFNE computational scaffold semantics unchanged.
- Scientific kernels (`emitters`, `fields`, `_pipeline`, `_model_simulate` numerics): no trajectory or claim change beyond declared hygiene guards.

## v0.4.17 (2026-08-18)

JDNA reconciliation release: multi-model audit reconciliation (H1–H14),
bounded-simplex JDNA development with strict validation, serialization
schema/roundtrip guarantees, AGSDR surface reclassification, verified
scholarly references, and harness evidence discipline.

### Added
- JDNA: box-constrained simplex projection for cell-type fraction allocation
  (deterministic, bounds-respecting, sum-to-one); `validate_genome` strict
  checks — fraction domain, tolerance bands, joint feasibility, and
  inter-/area-connection reference integrity.
- JDNA serialization: `pseudogenome_from_dict` (exported) rejects unknown
  schemas; `save_pseudogenome` preserves schema version; tuple
  geometry/pose canonicalization restores exact roundtrip equality.
- PRNG separation: behavioral tests for \(K_D\) (development), \(K_S\)
  (runtime seed), \(K_A\) (optimizer seed) independence.
- Harness: H1–H14 review/evidence discipline rules in AGENTS.md.

### Changed
- AGSDR surface classification: `propose_blackbox_candidates` →
  COMPATIBILITY, `_tune_matrix_agsdr_optax` → HYBRID (documented in
  `jaxfne.optim.AGSDR_SURFACE_CLASSIFICATION` and objectives API).
- RBS inventory: node HDP kernel includes \(K_{\mathrm{ctrl}}(1-H)\) and
  \(H C^\top\) coupling terms.
- References: verified primary sources (Palm/Najarro/Risi 2021, Zador
  2019); removed unverifiable memo number.

### Fixed
- Fresh-clone test isolation: visual review self-generates into a
  temporary directory (H11).
- Dead imports (`asdict`, `_jdna_genome_module`, `REPO_ROOT`) removed.

### Unchanged
- Public root contract: 191 symbols (178 CANONICAL + 13 COMPATIBILITY) from
  0.4.13; frozen publication evidence untouched.

## v0.4.16 (2026-08-13)

RBS/RBD/HDP containment release: finite edge-delay recurrent kernel (Protocol D),
frozen Protocol H ladder through H4 falsification receipt, frozen Protocol W ladder
through W3b unresolved interpretation receipt, and doctrine alignment for TFNE
containment architecture.

### Added
- Protocol D0/D1: finite edge-delay recurrent kernel with grid-aligned delay
  quantization (`edge_delay_steps_from_ms`, `edge_list_with_delay_ms`).
- Protocol H: RBD state-memory ladder (H1–H4) with frozen prospective receipts;
  H4 falsification receipt (topology/delay extension).
- Protocol W: HDP parameter-memory ladder (W0–W3b) with frozen receipts;
  W3b interpretation frozen as **unresolved classification** (`N_S=0`, `N_X=1944`).
- Doctrine: TFNE containment architecture, unified RBS/RBD/HDP grammar,
  Protocol W HDP parameter-memory specification.

### Changed
- Emitters: Protocol D delay-step validation on recurrent edge lists.
- Scientific development boundary for Protocol W closed at W3b interpretation
  for 0.4.16; W3c remains parallel science off the release critical path.

### Fixed
- Test isolation: remove import-time `jax_enable_x64` side effects from Protocol W
  analysis modules; reset x64 in session `conftest.py`.
- Optional `kaleido` dependency: skip plotly PNG export vis smoke when absent.
- Pin JAX/jaxlib to `<0.11` pending homeostatic-EI cubic_penalty compatibility audit.

### Unchanged
- Public root export count and stable API contract from 0.4.15.
- Fused observation-operator execution and observation provenance semantics.

## v0.4.15 (2026-08-12)

Observation-composability release: explicit fused-operator provenance,
observation-authority tests, Multiscale and Heterogeneous Emitter Études.

### Added
- O1 observation provenance on eager diagnostics/reports (`execution_form`,
  `operator_chain`, orthogonal amplitude/validation/physical-claim axes).
- O3 observation authority regression (frozen \(Q\), distinct \(O\),
  identical-\(W\) control, linear superposition, `O(0)=0`).
- Études: [multiscale observation](etudes/multiscale_observation.md);
  [heterogeneous emitters](etudes/heterogeneous_emitters.md).

### Changed
- MEG-proxy receipts: `relative_linear_map_proxy`, `orientation_claim: none`.
- API/docs: HEI `drive_schedule` path; observation vocabulary in fields API.

### Unchanged
- 186 root exports; fused \(F/P\) execution; `Signals`/`FieldOutput` layouts;
  compatibility parameter names.

## v0.4.14 (2026-08-12)

Stable TFNE core release: frozen public API contract (186 root exports), compact
release-facing documentation, H-state/HDP evidence semantics (`param_groups`),
MCC-1/2/3 and HDP controllability/reachability Etude regression. Package
semantics match the 0.4.13 freeze at `4a0e887`.

### Added
- Structural API documentation map and Études top-level documentation group.
- Public surface contract (`artifacts/public_surface_contract_v0413.json`) and
  `jaxfne.public_surface` tier authority.
- HDP metadata `param_groups` (H-state | H-dynamics | adaptive parameter dynamics).

### Changed
- Root public exports: 173 CANONICAL + 13 COMPATIBILITY (186 total).
- Population-H public semantics via `h_state_locality="population"`; user-facing
  docs and metadata use locality labels.
- Agent context (`AGENTS.md`) and four router skills under `skills/jaxfne-*`.

### Fixed
- Release-facing documentation and tutorial index normalized by mathematical
  purpose; mechanical test alignment for Pass 2 doc contraction.

### Changed (0.4.13 development, included in 0.4.14)
- Generalized HDP execution documents `H_i(t) ∈ R^d_H` as a finite-dimensional
  hidden biophysical state. The scalar homeostatic-like HDP realization is the
  `d_H=1` case; vector coordinates, optional coupling, and adaptation-specific
  readouts are supported.
- Canonical HDP simulation paths use one explicit per-step PRNG sequence across
  ordinary, continuation, and batch execution.

## v0.4.8 (2026-07-22)

First release of the 0.4.8-0.4.48 long-term roadmap (`full_scorecard >= 80/100`
by 0.4.48). Ships Phases 1-3 (Baseline measurement, Defragmentation wave 1,
Computational efficiency wave 1). `full_scorecard` total: 58.5/80 as of this
version (Novelty 78.6, Computational efficiency 74, TFNE goals 30,
Generalization 50, Code defragmentation 60).

### Added
- Committed, repeatable scaling étude at N=100/1,000/10,000/100,000
  (`benchmarks/scaling_benchmark.py`) -- the N=100,000 case exercises the
  sparse-connectivity lever (`p_connect<1`) via `Configuration.uniform3d()`,
  since a dense (N,N) matrix at this N is infeasible (~40GB).

### Fixed
- `_apply_connectivity`'s dense-memory warning now covers
  `tcm_v1_6pop`/`suite2_interarea`/`inter_column_connectivity` configs; the
  warning used to skip these whenever `p_connect<1`.
- `_construct_build_network`'s dense fallback (reached when a `Configuration`
  doesn't set `columns`/`layer_cell_types`/`uniform_3d` metadata) now warns at
  scale -- previously, a plausible-looking `.network(kind=..., layers=[...])`
  recipe made any `connectivity(p_connect=...)` call inert, building
  a dense (N,N) matrix regardless (confirmed: ~50GB/194s at N=100,000 before
  the fix, ~1.1GB/11s after, at the identical N/p_connect).
- `eeg_proxy_transform`/`meg_proxy_transform` (`jaxfne/fields/probes.py`)
  merged into one internal `_leadfield_proxy_transform` implementation --
  both were byte-identical bodies, differing purely in a parameter name. Both
  public names kept as thin, signature-preserving wrappers.

### Changed
- `_model.py` (2,901 LOC) and `_construct.py` (2,946 LOC) split into cohesive
  submodules (`_model_simulate.py`, `_model_readout.py`, `_model_evaluate.py`,
  `_model_tune.py`, `_model_manifest.py`, `_construct_population.py`,
  `_construct_connectivity.py`, `_construct_presets.py`, `_construct_core.py`,
  `_construct_extras.py`) -- both files are now thin re-export aggregators,
  matching `core.py`'s existing pattern. Zero public API surface change
  (`jaxfne.__all__` count unchanged at 256 throughout).
- `_suite2_apply_connectivity`/`_suite2_neuron_population_from_config`/
  `_suite2_default_layer_cell_types` renamed to their general names
  (`_apply_connectivity`/`_neuron_population_from_config`/
  `_default_layer_cell_types`) -- all three were the sole implementation,
  called broadly, with behavior identical across every suite2 flag value (jaxfne-harden rule 10).

## v0.4.7 (2026-07-20)

Internal engineering scorecard: 100/100 (all 10 factors,
`artifacts/developer/release_0_4_7_scorecard.md`). First release published to
GitHub Release / TestPyPI / PyPI since v0.4.5.

### Added
- `homeostatic_ei` emitter: N-generalized cubic-penalty and cross-population
  coupling homeostasis, soft bounds, `bound_mode`, pairwise synaptic HDP,
  wired through `Configuration.set_emitter()`/`Model.simulate()`.
- `experimental_poisson_1d` bridged to `Model.neuron_table()`/`Signals.sources`;
  extended to piecewise (layered) conductivity with an analytic validation target.
- `jax-fem` optional bridge (schema-first, mirrors `JaxleyBridge`).
- `record_weight_trace` opt-out on the HDP kernel and `_pipeline.py`'s
  `compile_step_fn`/`scan_network` path (closes an 80GB-scale OOM at large
  N/step counts; additive, default unchanged).
- First real cross-tool benchmark (jaxfne vs Brian2, matched Izhikevich task),
  independently re-verified.
- Formal HDP barrier-potential equilibrium stability proof; significance
  testing added to the ED9 HDP evidence bundle.

### Fixed
- Dense-connectivity `construct()` OOM at N=100,000 (skips a wasted dense
  (N,N) allocation the suite2 path discards anyway); N=100,000 HDP
  long-duration étude executed end-to-end post-fix (H settles near
  equilibrium, memory stays bounded).
- `step_sdr_transform`/`step_gsdr_transform`/`step_agsdr_transform` docstrings
  corrected — `Model.tune()` never calls these functions.
- Version-alignment: `pyproject.toml`, `jaxfne.__version__`, `mkdocs.yml`, and
  `docs/_generated/version.md` now kept in lockstep (a stale-source gap was
  caught by the release CI's own regression suite this cycle).

### Changed
- `ruff` lint is now a hard CI gate (509 → 0 errors as of 2026-07-14, with
  every per-file suppression individually verified rather than blanket-applied).
- Pre-0.4.7 four-chapter polish (P–S): human docs de-parrot agent jargon; `clamp_truth_gate_metadata` blocks claim escalation on update_metadata/manifest/migrate_schema; dual-ask re-score leak 92 / overall 93.
- Root declutter: contributing guide under `docs/` + `.github/CONTRIBUTING.md`; changelog lives under `docs/changelog.md`.
- API docs export note links [Scope & status](scope_and_status.md) instead of repeating gate jargon.
- README: drop "Built for AI agents too" manifesto; quiet Documentation-table link to for_ai_agents (`docs/for_ai_agents.md` — repository-internal reference, excluded from the built site); states the Jaxley/jaxfne population-vs-compartment relationship.

### Removed
- Root `SECURITY.md` and `CODE_OF_CONDUCT.md`.

## v0.4.6 (2026-07-12)

**Internal git tag** for the release-readiness polish wave (README/docs community files, citation metadata, scope page, CI coverage, root cleanup). Distribution to GitHub Release / TestPyPI / PyPI stays deferred until Hamm confirms **0.4.7**.

### Added
- `CITATION.cff`; contributing under `docs/contributing.md` + `.github/CONTRIBUTING.md`; changelog under `docs/changelog.md`
- `docs/scope_and_status.md`, `docs/for_ai_agents.md`, Zenodo wiring guide
- `jaxfne/py.typed`; pytest-cov coverage artifacts in CI

### Changed
- README landing page shortened; Jaxley comparison lives in docs
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

## Earlier releases (v0.3.42 -- v0.2.0, 2026-04 -- 2026-06)

Condensed history: public context hardening (v0.3.42), JAX kernel ports including spectral/STDP/connectivity helpers (v0.3.41), device-flexibility and hardening (v0.3.40), stable packaging consolidation v0.3.37--v0.3.39, multi-area laminar workshop and cylindric scaffold/spectrolaminar features (v0.3.26--v0.3.24), Etude No.1 completion (v0.3.21), field-proxy boundary and sharding/dtype improvements (v0.3.19--v0.3.17), and initial stable proxy operators/docs infrastructure (v0.2.3--v0.2.0). Full per-release notes preserved in git history; no API or scientific claim changes beyond those summarized.

Reference: see git log and GitHub Releases for verbatim per-version notes.
