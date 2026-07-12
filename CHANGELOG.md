# Changelog

All notable changes to this project are documented here in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

The canonical, detailed release notes live in [docs/changelog.md](docs/changelog.md).

## [Unreleased]

### Changed
- README restructured for jaxley-paced first impression; deep API branching moved to [docs/quickstart.md](docs/quickstart.md).
- `.legacy/` archived under `artifacts/legacy/` (root minimization).
- Root community files: `CONTRIBUTING.md`, lean `AGENTS.md`, [docs/for_ai_agents.md](docs/for_ai_agents.md).
- Removed root `CODE_OF_CONDUCT.md` (Contributor Covenant) — not retained for this project.

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

[0.4.5]: https://github.com/HNXJ/jaxfne/compare/v0.4.4...v0.4.5
[Unreleased]: https://github.com/HNXJ/jaxfne/compare/v0.4.5...HEAD

Older releases: see [docs/changelog.md](docs/changelog.md) (v0.4.4 back to initial releases).
