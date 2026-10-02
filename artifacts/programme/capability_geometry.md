# Capability record: geometry (0.5.2 item 6)

Second record under agent-native step 4 (same shape as the delay worked
example): equation, code, docs, skill, tests and inspection for declared
relative geometry in one place.

## Equation

A declared range is a pair of fractions of the sampled (area, layer)
block's extent, within [0,1] (0.5.2 decision 0a). For an axis with
declared `[a, b]` over a block span `[span_lo, span_lo + span]`:

```text
bound_lo = span_lo + a * span
bound_hi = span_lo + b * span
pos = bound_lo + (bound_hi - bound_lo) * u,   u ~ Uniform[0,1)
```

Spans: `x`/`y` over `[-radius, +radius]` (`column_radius_mm`, default
0.25) plus the area offset; `z` over the layer `[z0, z1]` bounds. An
undeclared axis keeps its historical span; a declared full `[0, 1]` axis
is left on the historical bounds (rescaling by an exact full range would
still perturb floats). Outside `[0,1]`, half-declared, degenerate
(`hi <= lo`) and non-finite bounds are refused, not rescaled.

Limitations: relative fractions only — no mm/um, conductivity, or
distance semantics under [0,1]; one sampled block cannot honour two
different declared domains on one axis (`E_GEOMETRY_AMBIGUOUS`); JDNA
samples the same declaration upstream for completion (origins recorded).

## Code

- `jaxfne/tfne.py` — `_validate_geometry_body` (resolve-time refusal),
  `_tfne_area_of` / `_tfne_layer_of`, `_tfne_geometry_domains`
  (per-(area, layer) fractional domains, ambiguity refusal),
  `to_configuration` (`metadata["tfne_geometry"]`: declared + domains).
- `jaxfne/_construct_population.py` — samples the declared sub-range per
  block; absent declaration is exactly the historical call.
- `jaxfne/_model_manifest.py` — `tfne_geometry` (`value_tag: "relative"`,
  declared per-leaf `G`, realized domains); present only when a sub-range
  is declared.
- `jaxfne/jdna/completion.py` — JDNA completion samples declared `G`
  under `K_D` with origins (`complete_tfne`, `realize_geometry`).

## Docs

- `docs/doctrine/tfne_algebra.md` — Pipeline (relative geometry,
  executed) and the conformance entry (fractional domains, refusal).
- Receipt: `artifacts/programme/tfne_param04_receipt.md`.

## Skill

No new skill file (0.5.2 item 6 rule). Route through existing skills:
`jaxfne-science` for geometry experiments (positions are what field
observables are computed against — any LFP-style claim cites the
manifest-recorded fractional domains), `jaxfne-core` for routing.

## Procedure (text, not a skill file)

To confine a population: declare `G = [z0 = a; z1 = b]` (likewise `x`,
`y`) with `0 <= a < b <= 1` on the TFNE leaf, realize (pin the seed when
comparing textually-differing specs — the seed otherwise follows the
normalization digest), construct, and read executed positions from
`params["positions"]` / `neuron_table()`. To test it: sub-range changes
executed positions at equal seed and bounds them; bare == full-`[0,1]`
bit-identically; outside-`[0,1]` refuses; two domains on one sampled
block refuse. Never read physical lengths into fractional coordinates,
and never rescale-then-compare across different declared extents.

## Tests

- `tests/test_tfne_parameter_transfer.py` —
  `test_declared_geometry_reaches_the_executed_positions` (inverted pin),
  `test_conflicting_geometry_in_one_sampled_block_is_refused`,
  `test_geometry_inspection_configured_realized_executed_manifest`.
- `tests/test_tfne_ctx01.py` — integrated chain (fixture sub-range,
  outside-range refusal, JDNA origins).
- `tests/test_jdna_completion.py` — JDNA-side declared-domain sampling.

## Inspection (configured -> realized -> executed)

| Stage | Surface |
|---|---|
| configured | leaf `G` / `explicit.nodes[p].geometry` / `relation` origins |
| realized | `s["geometry"]` (per-leaf declaration) / `cfg.metadata["tfne_geometry"]` (per-block fractional domains) / JDNA `positions` + `value_origins` |
| executed | `model.params["positions"]` / `neuron_table()[*]["z"]` |
| manifest | `tfne_geometry` (declared + realized_domains, `value_tag: "relative"`) |

Epistemic status: RELATIVE_PROXY — fractional domains of the area extent,
not physical coordinates. Field claims built on these positions inherit
that status until an explicit calibration transform exists (0.5.2 item 4
owns the levels; this record does not relabel anything).
