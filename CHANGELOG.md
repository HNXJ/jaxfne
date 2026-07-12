# Changelog

All notable changes to this project are documented here in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

The canonical, detailed release notes live in [docs/changelog.md](docs/changelog.md).

## [Unreleased]

### Notes
- Public GitHub Release / TestPyPI / PyPI for **0.4.7** wait on Hamm confirmation after a 100/100 review gate.

## [0.4.6] - 2026-07-12

Internal git tag only (no GitHub Release / PyPI upload in this step).

### Added
- Root `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`
- `docs/scope_and_status.md`, `docs/for_ai_agents.md`, `docs/guides/zenodo_doi.md`
- `jaxfne/py.typed`; CI coverage reporting (`pytest-cov`)

### Changed
- README restructured; Jaxley comparison lives in docs
- `AGENTS.md` trimmed to a lean pointer
- `.legacy/` moved to `artifacts/legacy/`
- Docs guide openings rewritten for plain-language first impressions

### Removed
- Root `CODE_OF_CONDUCT.md`

## [0.4.5] - 2026-07-03

### Added
- HDP v2 homeostatic plasticity kernel (`RuntimeConfig`, `Configuration.hdp()`).
- `NeuronalTensor` as first-class circuit representation with JSON round-trip and multi-area placement.
- `general_sequential_oddball_paradigm` and expanded étude notebooks.
- Full visualization isolation under `jaxfne/vis/*`.

### Fixed
- HDP JIT cache fingerprinting; homeostasis diagnostics forwarding; NeuronalTensor bridge fidelity gaps.
- CI: kaleido dependency, stale pytest ignores removed; `project_laminar_sources` default `density_preserving`.

### Changed
- Test suite consolidated toward étude/notebook execution; documentation review (89 files, avg 81→91/100).
- `pyproject.toml` `[project.urls]` added.

[0.4.6]: https://github.com/HNXJ/jaxfne/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/HNXJ/jaxfne/compare/v0.4.4...v0.4.5
[Unreleased]: https://github.com/HNXJ/jaxfne/compare/v0.4.6...HEAD

Older releases: see [docs/changelog.md](docs/changelog.md) (v0.4.4 back to initial releases).
