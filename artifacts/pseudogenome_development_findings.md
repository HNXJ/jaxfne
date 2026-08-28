# PseudoGenome Development Viewer — Findings (HEAD 350730a, 0.4.18 candidate)

**Task:** Enhance PseudoGenome visualization to show **configured→realized development G→D(K_D)→N, not storage**, same-genome different K_D semantics. Use canonical 1000n: show genome JSON rules vs realized NeuronalTensor arrays (counts, positions, edges, weights, delays) and compare two developments with same genome different seed. Use interactive viewer or static panels, but ensure it visualizes development. Do not change kernels, Δscience=0. Return artifact-backed findings with HTML paths, verification.

## Artifact Simulation (canonical 1000n, 2026-08-28)

- **APIs used only (existing, no new kernels):** `PseudoGenome` (`jaxfne/jdna/genome.py:97`), `develop` (`jaxfne/jdna/genome.py:604`), `declared_constraints` (`jaxfne/jdna/genome.py:449`), `genome_rules_hash` / `phenotype_sha256` (`jaxfne/jdna/genome.py:131/144`), `NeuronalTensor` (`jaxfne/neuronal_tensor.py:247`), `construct` + `RuntimeConfiguration` (`jaxfne/neuronal_tensor.py:707`), `Model.params['edge_list']` / `params['positions']` / `neuron_table()` (`jaxfne/_model.py:params`), `EdgeList` (`jaxfne/emitters.py:587`).
- **Genome (G, configured):** `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` — 6 layers (L1 100, L2 250, L3 200, L4 100, L5 200, L6 150), per-layer E/PV/SST/VIP base fractions with tolerance bands (e.g. L1 E 0.45–0.55), depth bands, uniform_random geometry `x/y [0,1]`, `fraction_jitter_sigma=0.01`, 49 typed `inter_connections` (AMPA/GABA_A). **No** `positions`/`edge_list` stored — blob check via `rules={k:v for k,v in raw.items() if k!='description'}` has none (verified, see `tests/test_jdna_pseudogenome.py:62`).
- **Development D(K_D):** `develop(genome, seed=K_D)` — per-layer K_D via `jax.random.fold_in`, Gaussian jitter + box-simplex projection onto bands, largest-remainder integer counts, provenance `{genome_sha256, development_seed, phenotype_sha256}`. Same (G,K_D) → same N (deterministic), different K_D → different N within bands.
- **Realized (N/M) — two developments:** `K_D=0 → t0` phenotype `cbe6f7f96e12…`, `K_D=1 → t1` phenotype `90c1715c6666…`; then `construct(t, RuntimeConfiguration(seed=7))` (K_S held fixed) realizes `positions (1000,3)` and `EdgeList`:
  - Seed 0: L1 `E49 SST15 VIP36`, L2 `E162 PV49 SST29 VIP10`, … total E753; edges 215,190; weight mean 0.0228 σ0.010; delay unique `[0]`; τ `[0.1]`
  - Seed 1: L1 `E50 SST15 VIP35`, L2 `E162 PV48 SST28 VIP12`, … total E760; edges 215,079 (Δ=-111); weight mean 0.0234; 6/6 layers differ, all counts within declared `[floor(lo*N), ceil(hi*N)]` bands.
  - Positions: both `(1000,3)` finite, z in depth bands (0 superficial → 1 deep); per-layer geometry identical (x/y uniform), only cell-type labels per coordinate vary with K_D.
  - Edges/weights/delays: `pre/post (n_edges,)`, `weight (n_edges,)`, `tau_ms (n_edges,)`, `delay_steps (n_edges,)` — all realized arrays, not genome fields.

## Findings (artifact-backed)

| # | Path:Line | Evidence | Severity | Confidence |
|---|---|---|---|---|
| F1 | `jaxfne/jdna/genomes/canonical-v1-column-1000n.json` + `jaxfne/jdna/genome.py:604` | Genome file declares rules only (depth_band, cell_type_fractions, fraction_tolerance, geometry, inter_connections=49); `develop` maps `(G,K_D)`→`NeuronalTensor` with `phenotype_sha256` differing for K_D 0 vs 1 (`cbe6…` vs `90c1…`). No per-neuron positions/edges stored in genome — blob check passes (see summary JSON `genome_rules_hash 07282b09…`). | HIGH | 0.99 |
| F2 | `jaxfne/vis/column_viewer.py:5` (existing) | Prior viewer (`render_column_viewer`) started from `Model` (NeuronalTensor→Model) and showed configured vs realized at Model level (edge_list), but did not show **G→D** stage: no genome JSON panel, no `develop` provenance, no K_D comparison. Task requires G vs N, not just cfg.metadata vs edge_list. | HIGH | 0.99 |
| F3 | `jaxfne/vis/pseudogenome_viewer.py:66` `collect_pseudogenome_development_data` | Realized arrays collected explicitly: `NeuronalTensor` counts per layer/cell_type (int, bands checked), `Model.params['positions'] (1000,3)`, `EdgeList.pre/post/weight/tau_ms/delay_steps` plus derived `in/out degree`, per-category `E→E/E→I/I→E/I→I` counts/means. HTML renders genome JSON side-by-side with these arrays (counts, positions 3D, weight/delay/degree histograms, edge categories). | HIGH | 0.99 |
| F4 | `jaxfne/vis/pseudogenome_viewer.py:656` `render_pseudogenome_development_viewer` | Same-genome different K_D comparison: two developments `seeds=(0,1)` held at same `K_S=7` so differences isolate D(K_D); viewer shows phenotype hashes differ, 6/6 layers differ within bands, edges 215,190 vs 215,079, weight means differ, determinism re-develop seed 0 reproduces hash (verified in summary `deterministic_same_KD:true`). | HIGH | 0.99 |
| F5 | viewer HTML | Prior export `artifacts/column_viewer_canonical_1000n.html` (999k edges, all-to-all config) was Model-only; new `artifacts/pseudogenome_development_viewer.html` (267kB, sha256 `88e5e2fd…`) is the development viewer — G→D→N emphasis, not storage. Both coexist; task explicitly requires the latter. | MEDIUM | 0.98 |
| F6 | `jaxfne/vis/__init__.py:144` | Overhead guard: `import jaxfne` does not import `pseudogenome_viewer`/`matplotlib`/`plotly`; viewer is lazy (verified `sys.modules` check). No kernel/sampler/solver file touched (`git diff --stat HEAD` shows only `jaxfne/vis/*`, `jaxfne/util.py` (prior), docs, version). Δscience=0. | MEDIUM | 0.99 |

## Minimal Repair (Δscience=0, no overhead when unused)

**Location:** `jaxfne/vis/pseudogenome_viewer.py` (new file, ~660 lines, additive; kernels untouched)

```python
# jaxfne/vis/pseudogenome_viewer.py:66
def collect_pseudogenome_development_data(genome, seeds=(0,1), construct_seed=7) -> dict:
    """Genome rules (hash, constraints, layer fractions/bands, inter_connections)
    vs realized: develop(K_D) → NeuronalTensor counts + construct(K_S) → positions (N,3)
    + EdgeList(pre/post/weight/tau/delay) + degree/categories. Determinism + diff checks.
    """

# jaxfne/vis/pseudogenome_viewer.py:233
def render_pseudogenome_development_viewer(genome, seeds=(0,1), construct_seed=7,
                                           output_path="artifacts/pseudogenome_development_viewer.html") -> tuple[Path, dict]:
    """Standalone HTML via Plotly CDN: configured panel (genome JSON + layer/connection tables),
    D(K_D) panel (phenotype hashes, per-layer counts vs bands, edge/weight/delay summaries),
    two 3D position scatters (layer color, E/I symbol, hover x/y/z), weight/delay/degree histograms
    per seed. Held K_S fixed so diff is attributable to development.
    """
```

**Exposure:** `jaxfne/vis/__init__.py:144` re-exports `collect_pseudogenome_development_data`, `render_pseudogenome_development_viewer`; also importable as `jaxfne.vis.pseudogenome_viewer`. No `jaxfne/__init__.py` eager import — lazy via `vis` submodule import.

**No kernel change:** `jaxfne/emitters.py`, `jaxfne/_model_simulate.py`, `jaxfne/_construct_core.py`, solvers untouched; viewer reads arrays, never writes dynamics. Verified: `phenotype_sha256(develop(g,0))` identical before/after viewer render, `m0.params['edge_list'].n_edges` unchanged.

## Artifacts (canonical 1000n, G→D(K_D)→N)

- **Primary:** `artifacts/pseudogenome_development_viewer.html` (267,031 bytes, sha256 `88e5e2fd3084c9fe14f1ad5b6f62663daaa0bacaa8c549138d85a3316f87bfec`) — standalone, Plotly.js CDN, no server. Open in browser.
- **Summary:** `artifacts/pseudogenome_development_summary.json` (machine-readable: genome hash, phenotype hashes, per-layer counts+bands for both seeds, realized n_edges/weight/delay/tau, verification flags, Δscience 0).
- **Existing (Model-only, retained):** `artifacts/column_viewer_canonical_1000n.html` (561k, sha256 `fb117d59…`), `artifacts/column_viewer_canonical_1000n_summary.json`, `artifacts/visualize_bundle/` (8-panel post-hoc bundle) — complementary, not replaced.
- **Render:** `python -c "import jaxfne as jtfne; from jaxfne.vis.pseudogenome_viewer import render_pseudogenome_development_viewer; g=jtfne.load_canonical_pseudogenome('canonical-v1-column-1000n'); render_pseudogenome_development_viewer(g, seeds=(0,1), output_path='artifacts/pseudogenome_development_viewer.html')"`

### What the viewer shows (configured vs realized)

| Configured (G — genome JSON) | Realized (N — NeuronalTensor) | Realized (M — Model via construct) |
|---|---|---|
| `areas[0].layers[].n_neurons` (100,250,200,100,200,150) | `tensor.areas[0].layers[].n_neurons` same totals, but `NeuronType.fraction` realized as integer counts within bands (e.g. L1 E45–56 band → 49 vs 50) | `model.neuron_table()` 1000 rows with `x/y/z` (N,3) |
| `cell_type_fractions` + `fraction_tolerance` (e.g. L1 E0.5 [0.45,0.55]) | Integer counts `E:49 [45,56]` vs `E:50 [45,56]` — box-simplex + largest-remainder, all 6 layers diff but within bands | `EdgeList.weight` mean 0.0228 vs 0.0234 (finite) |
| `depth_band` (L1 [0,0.1] ... L6 [0.85,1]) | `Geometry3D.z_range` per layer (same declaration) | `positions[:,2]` z in [0,1] (depth) |
| `geometry.x_range [0,1] y_range [0,1] uniform_random` | Geometry declaration unchanged (genome stores ranges) | `positions[:,0:2]` realized uniform_random under K_S |
| `inter_connections` 49 rules (e.g. L1 E→SST AMPA) | Same 49 rules as `tensor.inter_connections` | `EdgeList.pre/post (n_edges,)`, `tau_ms (0.1)`, `delay_steps (0)`, `cat_counts E→E etc.` |

### Same-genome different K_D evidence (seed 0 vs 1, same G, K_S=7 fixed)

- Genome hash `07282b0928e9…` identical for both (description excluded).
- Phenotype hashes `cbe6f7f96e12…` vs `90c1715c6666…` differ (provenance `development_seed` 0 vs 1).
- Determinism: `phenotype_sha256(develop(g,0)) == phenotype_sha256(develop(g,0))` true (re-develop reproduces).
- 6/6 layers have different integer counts (e.g. L5 E172 vs 176, L2 VIP10 vs 12) but **all within bands** (`[166,186]` for L5 E etc.).
- Edges 215,190 vs 215,079 (Δ -111) — edge count changes because population sizes change (full bipartite per rule, p=1.0).
- Storage falsified: genome JSON has no `positions`/`edge_list` keys in rules blob; realized arrays exist only post-develop/construct.

## Verification (artifact-backed, run 2026-08-28, win32 py3.14)

```python
import jaxfne as jtfne, json, pathlib
from jaxfne.jdna import develop, genome_rules_hash, phenotype_sha256
g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
t0, t1 = develop(g, seed=0), develop(g, seed=1)
assert phenotype_sha256(t0) != phenotype_sha256(t1)          # same G different K_D → different N
assert phenotype_sha256(develop(g, seed=0)) == phenotype_sha256(t0)  # deterministic
m0 = jtfne.construct(t0, jtfne.neuronal_tensor.RuntimeConfiguration(seed=7))
m1 = jtfne.construct(t1, jtfne.neuronal_tensor.RuntimeConfiguration(seed=7))
assert m0.params["positions"].shape == (1000,3)
assert m1.params["positions"].shape == (1000,3)
assert m0.params["edge_list"].n_edges == 215190 and m1.params["edge_list"].n_edges == 215079
# genome never stores phenotype
raw = json.loads((pathlib.Path(jtfne.jdna.genomes_dir()) / "canonical-v1-column-1000n.json").read_text())
rules = {k:v for k,v in raw.items() if k!="description"}
assert "positions" not in json.dumps(rules) and "edge_list" not in json.dumps(rules)
# HTML exists and is standalone
import hashlib; p = pathlib.Path("artifacts/pseudogenome_development_viewer.html")
assert p.exists() and p.stat().st_size==267031
assert hashlib.sha256(p.read_bytes()).hexdigest()[:8]=="88e5e2fd"
```

- `pytest tests/test_jdna_pseudogenome.py` → 32 passed (phenotype variation, constraints, determinism, PRNG separation).
- `pytest tests/test_v0321_migration_boundaries.py::test_simulation_engine_has_zero_graphics_overhead` → passed (viewer not loaded on `import jaxfne`).
- `import jaxfne; sys.modules` contains no `vis`/`plotly`/`matplotlib` before explicit viewer import (verified).

## Overhead & Δscience

- **Overhead:** zero when unused — `render_pseudogenome_development_viewer` not imported in `jaxfne/_model_simulate.py`, `jaxfne/emitters.py`, `jaxfne/_construct_core.py`; Plotly/matplotlib imported lazily inside function only. `import jaxfne` time unchanged (verified no `jaxfne.vis.*` in `sys.modules` after import).
- **Δscience=0:** no change to kernels (`simulate_edge_recurrent_izhikevich`, `simulate_eig_izhikevich` untouched), no new dynamics, no parameter mutation; viewer is read-only view of `NeuronalTensor` + `Model` arrays. Summary field `"Δscience":0`, `"kernels_unchanged":true`.

## Severity/Confidence Summary

- F1 (genome is rules not storage + G→D→N) → severity HIGH, confidence 0.99, verified via hashes, blob check, determinism.
- F2 (prior viewer lacked G→D stage) → severity HIGH, confidence 0.99, gap closed by new viewer.
- F3 (realized arrays: counts/positions/edges/weights/delays shown) → severity HIGH, confidence 0.99, HTML renders all four array categories + 3D/histograms.
- F4 (same-genome different K_D semantics) → severity HIGH, confidence 0.99, 6 layers diff + phenotype hash diff + edge Δ, persisted in HTML+JSON.
- F5 (artifact coexistence) → severity MEDIUM, confidence 0.98, both viewers retained, paths distinct.
- F6 (overhead / Δscience) → severity MEDIUM, confidence 0.99, lazy import + no kernel diff, receipt in summary JSON.

## Paths

- `artifacts/pseudogenome_development_viewer.html` — primary interactive viewer (open in browser; Plotly CDN)
- `artifacts/pseudogenome_development_summary.json` — machine-readable receipt (hashes, counts, edges, verification)
- `artifacts/column_viewer_canonical_1000n.html` — complementary Model-level viewer (realized EdgeList only)
- `jaxfne/vis/pseudogenome_viewer.py:66` — `collect_pseudogenome_development_data` (data layer)
- `jaxfne/vis/pseudogenome_viewer.py:233` — `render_pseudogenome_development_viewer` (HTML renderer)
