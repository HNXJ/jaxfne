## v0.3.26 (2026-06-02)

**Feature release:** Multi-area laminar workshop — inter-area connectivity, lesioning, AGSDR tuning, waveform explorer.

### Added
- Global multi-area network in `simulate_laminar_trials` with within-area recurrence + inter-area projections (`connectivity_spec`) and lesioning (`lesion_spec`); per-area depth leadfield. Defaults: feedforward V1 L2/3 (E) → V4 L4, feedback V4 L2/3 (E) → V1 L5/6.
- Per-cell-type Izhikevich overrides (`cell_type_izh_params`, wider "E-Wide" default), `single_cell_waveforms`, and `tune_laminar_agsdr` (AGSDR tuning of firing rate + kappa).
- Étude No. 1 rebuilt as a two-area Colab-ready customizable workshop notebook.

### Fixed
- Trial-averaged spectral power (per-trial PSD then mean); relative-power-density cross; uniform `dt_ms` kwarg across vis plotters; example API drift; GHA `checkout@v5` / `setup-python@v6`.

### Scope
- Proxy readouts only (`laminar_proxy_no_pde`, `physical_amplitude_claim_allowed=false`); spectrolaminar motif is emergent, not imposed.


## v0.3.25 (2026-06-01)

**Feature release:** Cylindric scaffold, spectrolaminar motif, and 32-contact LFP artifacts.

### Added
- Cylindric scaffold geometry: per-area vertical cylinders (circular x-y cross-section, depth on z); `visualize_network_3d` gains `column_shape="cylinder"` shells and per-cell-type marker symbols.
- Genuine per-area three-panel spectrolaminar suite (cell density, relative power spectrum, alpha-beta/gamma crossing) with smoothed band profiles.
- 32-contact LFP/spectrolaminar artifacts: equal-spaced contacts, robust relative-power normalization excluding degenerate low-power channels, 32-channel stacked LFP waterfall.

### Fixed
- Notebook controls for trials, contacts, and per-layer cell-type fractions; CI dev-extra (`ipykernel`) for slow notebook-execution tests.

### Scope
- Outputs remain proxy readouts with `field_solver_status=laminar_proxy_no_pde` and `physical_amplitude_claim_allowed=false`.


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
- Outputs remain proxy readouts with `field_solver_status=laminar_proxy_no_pde` and `physical_amplitude_claim_allowed=false`.


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
- Moved release/alignment receipts into `internal_docs/release_receipts/`.
- Updated release checklist and agent status metadata for the v0.3.21 release candidate.

### Validation status
- Package import and compile gates pass.
- Etude and template notebooks pass structural hygiene checks.
- Maintains `truth_safe_unverified`, `computational_scaffold`, `field_solver_status=laminar_proxy_no_pde`, and `physical_amplitude_claim_allowed=false`.

---

## v0.3.19 (2026-05-30)

**Release:** Field proxy boundary handling improvements.

- Optimized `project_laminar_sources` boundary fallbacks for low contact counts.
- Added comprehensive boundary and stencil numerical parity tests.
- Maintains `truth_safe_unverified`, `laminar_proxy_no_pde` status.

---

## v0.3.18 (2026-05-30)

**Release:** Sharding infrastructure for multi-device AGSDR.

- Added `jaxfne/sharding_utils.py` with distributed mesh and NamedSharding stubs.
- 14 new tests for sharding context and single-device fallbacks.
- Sharding stubs do not yet drive multi-device dispatch (planned for v0.3.20+).

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
