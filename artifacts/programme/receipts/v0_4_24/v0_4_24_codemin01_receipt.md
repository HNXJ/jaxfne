# 24-CODEMIN-01 receipt — concepts-first source minimization

**Branch:** `dev`. No API, numerics, or semantics changed.

## Accepted (all REUSE into existing homes, equivalence-probed)
- `hdp_rule._segment_sum` → `emitters._segment_sum` import (cycle-safe;
  emitters never imports hdp_rule). −6.
- `vis/evidence_manifest.sha256_file` → `jaxfne.io.sha256_file` import
  (io is leaf-level; dropped now-unused hashlib import). −8.
- vis shared helpers into leaf-level `vis/core.py` (no new concepts):
  `is_E_cell_type`, `get_time_ms`, `neuron_rows`,
  `geometry3d_from_config`; 10 duplicate defs across spectra/traces/
  fields/network3d/column_viewer/pseudogenome_viewer replaced by alias
  imports; 2 dead defs deleted outright (spectra `_get_time_ms`,
  fields `_geometry3d_from_config`, both 0 uses). −~60 net.
- `_construct_population._empty_edge_list` → import from
  `_construct_connectivity` (documented leaf direction). −12.

## Rejected with cause (audited, kept)
- Frozen evidence-adjacent modules (`protocol/*`, `experiment_a/*`):
  receipt-frozen hashes forbid churn (`_git_head`×16 stays).
- Same-name/different-function pairs (`_to_numpy`, `population_rate_hz`,
  `build_laminar_column`, `export_tutorial_artifacts`,
  visualize.py's different-arity `_get_time_ms`/`_neuron_rows`): renaming
  is churn; behaviors differ.
- Public API pairs (`simulate`, `run_receipt`, `run_trials`,
  `with_emitter_parameters`, `manifest`, `get_signal`, plot facades):
  API-constrained → input to API-01, not touched here.
- Scheduled deprecations (LegacyMultiArea, save_png, TuneResult
  unpacking): still live in tests → HUMAN_DECISION, not unilateral.
- Universal kernel / JIT dispatch / recording variants: sealed NO_CHANGEs
  stand (SIMP-02/03/05).
- `_linear_proxy_from_sources` input-extraction delta: marginal, SKIP.

## Measures (MIN-01 baseline → now)
jaxfne/: 159 files (same), 62,475 → 62,394 lines (−81); defs −7 net;
public symbols 190 (same); kernels 14 (same). Concepts: −7 duplicated
helpers, +0 modules.

## Evidence
ruff clean; 158 tests green (vis smoke, viewer, atlas, HDP-01/LAW-01/
delayed qualification, tensor pipelines).
