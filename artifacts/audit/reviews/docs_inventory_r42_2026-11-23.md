# Docs inventory (Solmind plan R42), 2026-11-23

Baseline 62059ce5. Scope: the 134 tracked `docs/**/*.md` files (0 notebooks). Read-only sweep by a Haiku worker,
criteria and the three suspect rows resolved by the lead.

## Rules, applied in order

1. archive: under `docs/archive/`, or listed in `mkdocs.yml` `exclude_docs` (not built).
2. stale: built, and (i) reachable from no nav page, or (ii) names a `jaxfne.X` that resolves in no public tier
   (`__all__`, the root-reachable tiers of `jaxfne/public_surface.py`, `jaxfne/__init__.py` bindings, existing
   submodules), or (iii) links a target that does not resolve on the built site (mkdocs `redirect_maps` count).
3. canonical: everything else.

## Result

| class | pages |
|---|---|
| archive | 53 (20 under `docs/archive/`, 33 by `exclude_docs`) |
| canonical | 81 |
| stale | 0 |

Suspects cleared:

| page:line | flag | resolution |
|---|---|---|
| `reference/changelog.md:414` | (ii) `jaxfne.units` | removal note in verbatim history; not a live reference |
| `reference/errors.md:500` | (ii) `jaxfne.emitter` | generated table truncates messages at 100 characters; the live message names `jaxfne.emitters_hh` |
| 9 image links on `etudes/{hdp_balance,hdp_controllability_reachability,heterogeneous_emitters,multiscale_observation,pcl}.md` | (iii) untracked `docs/assets/etudes/*.png` | build products (`.readthedocs.yaml` pre_build runs `scripts/build_docs_assets.py --build`); the live site serves `assets/etudes/pcl/c1_kernels.png` (200, image/png) and 404s a nonexistent sibling |
| `index.md:65`, `tutorials/index.md:19-29` | (iii) directory hrefs | resolve as built URLs or through `redirect_maps` (`mkdocs.yml:293-309`) |

Class (i): `scripts/check_docs_orphans.py` prints "No orphan doc pages found."; all 100 nav entries exist; 84 of
84 redirect targets exist.

## Duplicate candidates for R46 (2+ canonical pages, one subject)

- probes: `guides/readout/probe_operators.md`, `reference/api/readout/probes.md`
- visualization: `guides/interop/atlas_suite.md`, `reference/api/readout/vis.md`
- PseudoGenome: `guides/grammar/jdna.md`, `reference/api/model/jdna.md`, `tutorials/construct/01_define_genome.md`
- traveling waves: `theory/protocols/protocol_c_wave.md`, `theory/protocols/protocol_c_wave_evidence.md`
- RBS/HDP state (coarse label): `guides/state/hdp.md`, `tutorials/extend/06_add_state.md`

Guide vs API-reference pairs are a deliberate split; R46 decides per pair.
