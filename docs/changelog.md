# Changelog

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
- Added claim-status metadata
- Eight canonical readout channels

## v0.2.0 (2026-04-15)

- Initial release
- Izhikevich emitters
- Laminar field solver (proxy)
- Basic readout operators
