# Tutorial cumulative audit — define → develop → inspect → simulate → observe → add state → add dynamics → compare

**HEAD:** `350730a` (docs: subtractive compaction) · **branch:** `dev` · **date:** 2026-08-28
**Constraints:** Δscience=0 · no new public API · all snippets use shipped `jaxfne` symbols only
**Canonical endpoint:** `canonical-v1-column-1000n` PseudoGenome → 1000-neuron NeuronalTensor (6 layers, 100/250/200/100/200/150)

---

## 1. Verdict

Current `docs/tutorials/index.md` at HEAD teaches **isolated, size-escalating demos** (n=1 → 2 → 100 → 600 → 100 ... → 1000). Each tutorial builds a *new* `Configuration` with a different `n`, different `.column/.cell_types/.connectivity` calls, and a different narrative. The canonical 1000-neuron PseudoGenome (`canonical-v1-column-1000n`) appears only at the *end* (tutorial 13) as an isolated étude — it is not reused earlier. Result: a reader cannot carry a variable `genome → tensor → model → signals` forward; every page resets.

**Proposed:** one cumulative 8-step arc that **reuses the same object chain** across all steps, endpoint 1000n by construction. Each step imports its predecessor's variables; no tutorial rebuilds from scratch.

---

## 2. Evidence: current order is isolated (HEAD vs realized canonical)

### 2.1 What HEAD declares vs what `canonical-v1-column-1000n` actually is

Receipt (executed 2026-08-28 on this tree, `python -c "import jaxfne..."`):

```
genome: canonical-v1-column-1000n
areas: 1
 area V1 6
  layer L1 100 {'E': 0.5, 'SST': 0.15, 'VIP': 0.35}
  layer L2 250 {'E': 0.648, 'PV': 0.2, 'SST': 0.1, 'VIP': 0.052}
  layer L3 200 {'E': 0.8, 'PV': 0.08, 'SST': 0.08, 'VIP': 0.04}
  layer L4 100 {'E': 0.75, 'PV': 0.18, 'SST': 0.04, 'VIP': 0.03}
  layer L5 200 {'E': 0.88, 'PV': 0.06, 'SST': 0.04, 'VIP': 0.02}
  layer L6 150 {'E': 0.9, 'PV': 0.0533, 'SST': 0.0267, 'VIP': 0.02}
  rules 48
area_connections 0
dev params {'fraction_jitter_sigma': 0.01}
tensor areas 1
 area V1 1000 [('L1', 100), ('L2', 250), ('L3', 200), ('L4', 100), ('L5', 200), ('L6', 150)]
```

Genome tolerance bands (from `load_canonical_pseudogenome`):

```
L1 {'E': (0.45, 0.55), 'SST': (0.1, 0.2), 'VIP': (0.3, 0.4)}
L2 {'E': (0.6, 0.7), 'PV': (0.15, 0.25), 'SST': (0.05, 0.15), 'VIP': (0.03, 0.08)}
...
```

Deterministic development (same genome): `seed=0` vs `seed=1` differ within those bands (verified above, e.g. L1 E 0.49 vs 0.50) and carry provenance `genome_sha256`, `phenotype_sha256`, `development_seed`.

No current tutorial before #13 loads this genome. Evidence from grep (`rg -n "canonical|PseudoGenome|develop"` on `docs/tutorials/*.md`):

| file | n= | mentions `canonical` | mentions `PseudoGenome`/`develop` | reuses prior variable? |
|------|----|----------------------|-----------------------------------|------------------------|
| `01_single_neuron_multimodal.md` | 1 | no | no | no — builds `jtfne.configuration().network(n=1)` |
| `02_two_neuron_ei.md` | 2 | no | no | no — new `n=2, cell_types={"E":1,"PV":1}` |
| `03_network_100_ei.md` | 100 | no | no | no — new `n=100, kind="balanced_ei_population"` |
| `04_v1_column.md` | 600 | yes (passing) | no | no — new `n=600, layers L1..L6` (note: *not* 1000, Colab badge points to unrelated 1000n etude) |
| `06_v036_100_neuron_ei_population.md` | 100 | yes (cross-ref) | no | no — fluent `Configuration().runtime().column(n=100)` |
| `07_v037_source_bookkeeping.md` | 48 | no | no | no — `n=48, n_contacts=16` |
| `08_v038_lfp_csd_readout.md` | 1 / 12 / 48 | no | no | no — `n=1` and `n=12` laminar examples |
| `09_v0310_eeg_meg_emm_proxy_bundle.md` | — | no | no | no — declarative sensor path, isolated |
| `10_v0313_omission_oddball.md` | 60 | no | no | no — `n=60, omission_oddball_paradigm()` |
| `05_v1_pfc_dual_column.md` | 2×100 | yes | no (tensor path, but V1 L4/L6 are pure-E hack) | no — script-driven, not a reused genome |
| `11_multi_laminar_cortical_agsdr.md` | V1+V4 | yes | no (via `tutorial_utils.make_laminar_column_config`) | no — synthetic `LAYER_CELL_TYPE_FRAC` not canonical |
| `12_izhikevich_single_emitter_explorer.md` | — | no | no | no — browser-only tool |
| `13_canonical_column_etude.md` | **1000** | **yes** | no (uses `laminar_cortex_config`/`jtfne.laminar_cortex_config`, not `PseudoGenome`) | **no** — builds via `laminar_cortex_config` builder, not `develop(G,K_D)`; isolated endpoint, no upstream dependency |

**Finding:** 12/13 tutorials ignore `PseudoGenome`; the one 1000n page does not use `develop`. The progression is size-based, not pipeline-based. The reader never sees `G → D(K_D) → N → construct → simulate → observe → H → HDP → compare` on one object.

### 2.2 Études are frozen demos, not tutorials — and they duplicate the problem

`docs/etudes/index.md` correctly states: "Études answer *what the grammar can demonstrate*; tutorials teach *how to use the grammar*." The four documented études (HDP controllability/reachability, multiscale observation, Experiment A, heterogeneous emitters) are frozen, single-seed, single-circuit publications with their own `N=10 / N=40 / N=2` choices. `docs/tutorials/index.md` § "Étude notebooks" then re-lists 12 legacy notebook études by theme (spectrolaminar family n=100/1000, homeostasis family, oddball/omission family, thalamocortical 6-pop) — all isolated, none reusing the canonical 1000n genome variable.

The two indexes together compound the fragmentation: tutorials escalate `n`, suites repeat the same escalation in notebook form, and legacy études re-list the same topics as standalone notebooks.

---

## 3. Proposed cumulative order (reuses same model, endpoint 1000n)

### 3.1 Pipeline invariant (from `docs/guides/jdna.md` at 0.4.17)

```
PseudoGenome --develop(K_D)--> NeuronalTensor --construct(K_S)--> Model --simulate(K_S)--> Signals
                                         │                     │
                                         └─── inspect ─────────┘
                                                        │
                                         observe (F,P,PSD) ← frozen X,Q
                                                        │
                                         add state (H / RBS)
                                                        │
                                         add dynamics (HDP: D_H)
                                                        │
                                         compare (null / lesion / shuffle / authority)
```

K_D ≠ K_S ≠ K_A (separate PRNG domains). Realized ≠ effective (edges inspectable via `model.params['edge_list']`; effectiveness is ΔX under intervention).

### 3.2 Eight steps, one variable chain

| # | verb | file (proposed) | title | what the cell does | reused variable | new concept | output shown |
|---|------|-----------------|-------|--------------------|-----------------|-------------|--------------|
| 1 | **define** | `01_define_genome.md` | Define the generative rules | `load_canonical_pseudogenome("canonical-v1-column-1000n")`, `validate_genome`, `genome_rules_hash`, `declared_constraints` | — (origin) | PseudoGenome stores *rules*, never phenotype; toleranced fractions; `fraction_jitter_sigma=0.01` | genome JSON excerpt, hash |
| 2 | **develop** | `02_develop_genome.md` | Develop genome → phenotype | `develop(genome, seed=0)` vs `develop(genome, seed=1)`, `phenotype_sha256`, determinism `develop(G,0)==develop(G,0)` | `genome` from 01 | K_D domain; jitter→box-simplex→largest-remainder counts; provenance | two tensors, count diff table (e.g. L2 PV 49 vs 48) |
| 3 | **inspect** | `03_inspect_tensor.md` | Inspect realized vs configured | `tensor.neuron_table()`, `to_dict()`, per-layer counts, rule count 48, `declared_constraints` bands, `save/load_neuronal_tensor` round-trip | `tensor` from 02 | Configured (`p_EI` prior) vs realized (48 rules, edges not yet realized) vs effective (not yet) | table: configured n, realized N=1000, cell-type bands |
| 4 | **simulate** | `04_simulate_tensor.md` | Compile and run one trajectory | `construct(tensor, RuntimeConfiguration(seed=1, duration_ms=1000, dt_ms=0.5))`, `simulate(model)`, `model.params['edge_list'].n_edges` (≈215k for canonical-v1-column-1000n, p=1.0 bipartite), `model.params['positions']` after `Pose3D` | `tensor` from 02 | Construction realizes positions+edges under K_S; cheap `simulate` on frozen model; `with_emitter_parameters` for per-layer drive | `signals.get("spikes")` shape (2000,1000), rate ~8-12 Hz |
| 5 | **observe** | `05_observe_fields.md` | Post-hoc observation operators | `project_laminar_sources`, `LinearReadout`, `spectrolaminar_psd_jax`, `kappa_synchrony` on *frozen* `signals` from 04; LFP/CSD/EEG/MEG toy leadfields; keep X,Q frozen, vary only O_k | `model, signals` from 04 | Observation authority: K_a≠K_b ⇒ Y_a≠Y_b at fixed X,Q; LFP (proxy) vs CSD (Dzz·LFP); spectral | multi-panel: raster, LFP-proxy, CSD-proxy, PSD |
| 6 | **add state** | `06_add_state.md` | Add H / RBS container | `PlasticParams(H=1.0)` aggregation → `Model.with_hdp_initial_state`, `h_state` dim, `DynamicState`, `checkpoint_state`/`restore_state` | `tensor, model` from 04 | H is stored but inert until dynamics enabled; per-neuron H0; H≠G, H≠D | H histogram, `model.last_hdp_diagnostics()==None` when HDP off |
| 7 | **add dynamics** | `07_add_dynamics.md` | Enable HDP adaptation | `RuntimeConfig(enable_hdp=True, hdp_params=DEFAULT_HDP)` or `DESYNC`, `diagnostics H_trace/w_trace`, `K_HDP/K_ctrl/K_w_ctrl` | `model` from 06 | `tau_i = tau0 * size_i^3`; rank vs reachability not in tutorial (see étude); long stationary run | H trace pinned vs fluctuating table (see §3.3) |
| 8 | **compare** | `08_compare_nulls.md` | Nulls, lesions, authority | Shuffled-time control, lesion `LESION_SPEC`, superficial-vs-deep crossover check (`signal_key="lfp_contacts"`), authority reading `metrics.json` from études | all prior | Effective = ΔX; null reframes metric into evidence | lesion Δrate, shuffled PSD baseline, `R_EI`/`E_P` table |

Every step's first code cell is `import jaxfne as jtfne; genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")` only in step 1; steps 2-8 begin with `# continued — genome/tensor/model/signals from previous step` and *do not* reload a different `n` or a different `Configuration` path. A smoke smoke variant (`n=100, duration_ms=100.0`) is offered as a 30-second preamble inside each notebook (`if SMOKE: ...`), but the documented main path is 1000n.

### 3.3 Concrete H presets carried, not re-described, in step 7

Step 7 imports the frozen presets verbatim; it does not invent new gains:

| preset | purpose | K_HDP | tau0 | K_ctrl | alpha | gamma | drive scale | realized H |
|--------|---------|-------|------|--------|-------|-------|-------------|------------|
| `DEFAULT_HDP` | stable long stationary | 0.01 | 200.0 | 5.0 | 0.01 | 0.0 | 1.0 | 1.001±0.0006 pinned |
| `DEFAULT_HDP_DESYNC` | desync/variability | 0.01 | 5.0 | 0.15 | 0.05 | 0.5 | 1.2 | 1.028±0.023 fluctuating |

Both rows are already validated (5-seed × 20s for stable, 5-seed × 2s for desync at N=500); step 7 merely *demonstrates* enablement on the same 1000n model.

---

## 4. Mapping table: current → proposed (fold / move / retire)

| current (HEAD) | n / object | disposition in cumulative design | rationale |
|----------------|-----------|----------------------------------|-----------|
| `01_single_neuron_multimodal.md` (n=1) | `configuration().network(n=1)` | **fold into 03 inspect as boxed contrast** "from one cell to one column" — not a step on the main chain | Isolated `n=1` rebuild contradicts reuse; keep as 5-line contrast, not a standalone stage |
| `02_two_neuron_ei.md` (n=2) | `n=2 E+PV coupling` | **same — 1-paragraph boxed contrast in 03** | Same as above |
| `03_network_100_ei.md` (100) | balanced EI pop, 8 probes | **smoke preamble for steps 2-5** (`if SMOKE: n=100`) | 100n remains useful as fast smoke, but not as the documented progression |
| `04_v1_column.md` (600) | `n=600, layers L1..L6` | **retire / redirect** → step 01 canonical genome (1000n). Keep a one-line deprecation note: "historical 600n example; canonical is 1000n via `load_canonical_pseudogenome`" | 600n is an ad-hoc Configuration example that predates the frozen genome; it claims 6-layer realism while differing from the frozen 1000n counts |
| `06_v036_100_neuron_ei_population.md` (100, fluent API) | `Configuration().runtime().column().cell_types(n=100)` | **fold into 04 simulate as "second on-ramp" box**: show `NeuronalTensor` vs `Configuration` converge on same `Model` (already documented in `docs/guides/configuration_grammar.md` § NeuronalTensor) | Fluent API remains first-class, but the tutorial main line uses the tensor-first path from `develop`; the Configuration path is shown as an equivalent box, not a separate progression |
| `07_v037_source_bookkeeping.md` (48) + `08_v038_lfp_csd_readout.md` (1/12/48) | source/field/probe tiers | **merge into 05 observe** (single chapter with three sub-panels: source bookkeeping → LFP/CSD → EEG/MEG/EMM) | Both pages describe the same operator stack at different toy n; cumulative version does it once on the frozen 1000n trajectory |
| `09_v0310_eeg_meg_emm_proxy_bundle.md` | multimodal sensors | **same — §3 of 05 observe** | Not a separate tutorial stage; one operator family |
| `10_v0313_omission_oddball.md` (60, oddball) | `omission_oddball_paradigm` | **fold into 08 compare as paradigm-variant box** + cross-link to `general_sequential_oddball_paradigm` | Paradigm is a compare/null variant, not a pipeline stage; belongs in step 8 |
| `05_v1_pfc_dual_column.md` (2×100, V1→PFC AAAB) | script-only, pure-E L4/L6 hack | **fold into 08 compare as multi-area variant** via `merge_neuronal_tensors` / `canonical-v1-v4-pfc-multiarea` (3000n) | Dual column is a compare/perturbation of the canonical column, not a prerequisite for understanding the single column |
| `11_multi_laminar_cortical_agsdr.md` (V1-V4, AGSDR) | `tutorial_utils.make_laminar_column_config` | **fold into 07 add dynamics + 08 compare**: AGSDR scatter is shown as one tuning call in step 7; knock-outs in step 8 | AGSDR on a synthetic V1-V4 config is not the canonical column; keep as tuning demo on the canonical model |
| `12_izhikevich_single_emitter_explorer.md` (browser) | browser Euler, 11 presets | **prerequisite appendix** (linked from steps 1 and 4, not in the 8) | Parameter intuition tool; browser solver ≠ JAX kernel; keep but exclude from linear progression |
| `13_canonical_column_etude.md` (1000n via `laminar_cortex_config`) | `laminar_cortex_config(... n=1000 ...)` | **replace body with cumulative steps 1-8** and keep as the *canonical reference page* (the 8 notebooks *realize* this page) | At HEAD it is already "Configuration→Construct→Simulate→Visualize→Tune→Post-tune" but via the builder path, not `PseudoGenome→develop`; rewrite its code blocks to `load_canonical_pseudogenome→develop→construct(...)` (3-line change, no science change) |
| Suites 1-3 (`06_suite_no_1 … 08_suite_no_3`) | accelerated arcs + notebooks | **retain as accelerated suites** — Suite 1 = steps 1-5 in one notebook (smoke n=10), Suite 2 = deep dive on step 5 (spectrolaminar), Suite 3 = deep dive on step 5/8 (scaling) | Suites are correctly scoped as "multi-part interactive courses" in `index.md`; they are not the linear progression and should stay separate |
| Legacy étude notebook table (12 entries) | notebooks `etudes/*.ipynb` | **remove from `tutorials/index.md`** — already indexed under `etudes/index.md`; replace with one sentence + link | The table duplicates the étude index and suggests those notebooks are tutorials; they are not (they are runnable artifacts) |
| NeuronalTensor box (bottom of `index.md`) | `Areas×Layers×NeuronTypes` | **promote to step 03-04 bridge** ("NeuronalTensor as the phenotype data model") | Already says the right thing — it *is* the path for H-state/HDP — but sits as an orphan; make it the explicit handoff between develop and construct |

No file is deleted without a redirect pointer; Δscience=0 (no new mechanism, no new `K_D`/`K_S` coupling, no new tuning target).

---

## 5. Changes to `docs/tutorials/index.md` (and one guide note)

### 5.1 `docs/tutorials/index.md` — replace "Single-topic progression" with "Cumulative (canonical 1000n) progression"

**Before (HEAD, 180 lines):** three tiers (Beginner 01-02, Intermediate 03/04/06/07/08/09/10, Advanced 05/11/12/13) + opaque suite table + legacy étude table + NeuronalTensor orphan box. Each row uses a different `n` and a different `Configuration` incantation.

**After (proposed, same file, no new files required):**

```markdown
# Tutorials

Tutorials teach **how to use the jaxfne grammar** as a cumulative arc.
Each tutorial builds on the same object chain; there are no isolated demos.

> **Canonical model:** every step reuses `canonical-v1-column-1000n`
> (`PseudoGenome → develop → NeuronalTensor → construct → simulate`).
> Same `genome`/`tensor`/`model`/`signals` variables throughout; change only
> `K_D` (development key), `K_S` (runtime key), or the observation operator.

## Notebook standard
[...unchanged...]

## Cumulative (canonical 1000n) progression

| # | verb | notebook | one-line | reuses |
|---|------|----------|----------|--------|
| 01 | define   | 01_define_genome.md   | The genome: rules, tolerance bands, `validate_genome`, `genome_rules_hash` | — |
| 02 | develop  | 02_develop_genome.md  | `develop(G, K_D)` → phenotype; determinism; `phenotype_sha256`; seed 0 vs 1 | `genome` from 01 |
| 03 | inspect  | 03_inspect_tensor.md  | `neuron_table`, realized vs configured, 48-rule count, provenance, round-trip | `tensor` from 02 |
| 04 | simulate | 04_simulate_tensor.md | `construct(tensor, RuntimeConfiguration)` → `simulate` → `Signals` | `tensor` from 02 |
| 05 | observe  | 05_observe_fields.md  | LFP/CSD/EEG/MEG/PSD post-hoc on frozen `X,Q`; authority `K_a≠K_b ⇒ Y_a≠Y_b` | `model,signals` from 04 |
| 06 | add state   | 06_add_state.md  | `PlasticParams.H` → `h_state`; `checkpoint/restore` | `tensor,model` from 04 |
| 07 | add dynamics| 07_add_dynamics.md | `enable_hdp` (`DEFAULT_HDP` vs `DESYNC`), `H_trace/w_trace` | `model` from 06 |
| 08 | compare  | 08_compare_nulls.md   | Shuffled, lesioned, multi-area 3000n via `merge_neuronal_tensors`; effective vs realized | all prior |

> Each page opens with the exact variable carried from the prior page.
> Smoke mode (`SMOKE=1`, `n≈100, duration_ms≈100`) runs in ~30s; the main path
> is the 1000n canonical (`duration_ms=1000, dt_ms=0.5`).

## Suites (accelerated arcs — same grammar, faster)

| Suite | Topic | Relation to cumulative |
|-------|-------|------------------------|
| Suite 1 | Computational Biophysics | steps 01-05 compressed into one notebook (smoke n=10→100) |
| Suite 2 | Corticospectrolaminar Motif | deep dive on step 05 (spectrolaminar) — same `project_laminar_sources`, same LFP-not-CSD rule |
| Suite 2 (Evoked L4) | Evoked L4 Drive | **variant of step 08** — `evoked_l4_drive_paradigm` as a compare condition |
| Suite 3 | Low-Frequency Scaling | **variant of step 08** — scale curve `N` as density-preserving compare |

## Box: Configuration vs NeuronalTensor (two on-ramps, one compiler)

[keep existing verbatim block + code fence; append one line]
> The cumulative path uses `develop(G,K_D)`. `Configuration` is the fluent
> builder for bespoke circuits; both converge on `construct → simulate`.
> See [Configuration Grammar](../guides/configuration_grammar.md).

## Legacy single-topic and étude notebooks (archive pointers)

The pre-cumulative pages (`01_single_neuron_multimodal` … `13_canonical_column_etude`)
and legacy `tutorials/etudes/*.ipynb` remain reachable under `artifacts/tutorials/`
and `docs/etudes/` respectively but are **not** the recommended path. The canonical
column formerly at `13_canonical_column_etude.md` is now realized by steps 01-08;
that file is rewritten to cross-link to them.

[...rest of file (Running tutorials, Quick example, Next steps) updated so the
quick example uses the canonical genome, not `suite2_four_celltype_config`: ...]
```

Quick-example patch:

```python
import jaxfne as jtfne

genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)
model  = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
print(f"neurons: {len(model.neuron_table())}")  # 1000
print(f"spike count: {int(signals.get('spikes').sum())}")
```

### 5.2 `mkdocs.yml` nav — add cumulative entries, keep legacy as archive

```yaml
  - Tutorials:
      - Overview: tutorials/index.md
      - Notebook standard: tutorials/notebook_standard.md
      - Tutorial Outputs: tutorials/tutorial_outputs.md
      - Cumulative (canonical 1000n):
          - Define (PseudoGenome): tutorials/01_define_genome.md
          - Develop (G→N): tutorials/02_develop_genome.md
          - Inspect (realized vs configured): tutorials/03_inspect_tensor.md
          - Simulate (construct → Signals): tutorials/04_simulate_tensor.md
          - Observe (field & probe): tutorials/05_observe_fields.md
          - Add state (H / RBS): tutorials/06_add_state.md
          - Add dynamics (HDP): tutorials/07_add_dynamics.md
          - Compare (nulls & lesions): tutorials/08_compare_nulls.md
      - Suites: [...]
      - Archive (pre-cumulative): [...]
      - Source, field, and readout: [...]
```

No new API symbols; only nav/docs additions.

### 5.3 Guide note — `docs/guides/jdna.md` already correct

No edit required. The guide already states the invariant that the tutorials must demonstrate: `G --[JDNA/develop; K_D]--> N --construct--> Model --simulate--> Signals`,Configured→Realized→Effective, and build-time scope. Tutorials now *exemplify* that guide line-by-line.

---

## 6. How to verify reuse (receipts, not prose)

Run the canonical chain end-to-end (Δscience=0, uses only shipped APIs):

```bash
python -c "
import jaxfne as jtfne
g = jtfne.load_canonical_pseudogenome('canonical-v1-column-1000n')
t0 = jtfne.develop(g, seed=0)
t1 = jtfne.develop(g, seed=1)
assert jtfne.jdna.genome.genome_rules_hash(g) == jtfne.jdna.genome.genome_rules_hash(g)
assert t0.provenance['development_seed']==0 and t1.provenance['development_seed']==1
assert len(t0.areas[0].layers)==6
# realized vs configured via declared_constraints
from jaxfne.jdna.genome import declared_constraints
dc = declared_constraints(g)
assert dc['areas']['V1']['layers']['L2']['n_neurons']==250
# same tensor through construct
m = jtfne.construct(t0, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
s = jtfne.simulate(m)
assert s.get('spikes').shape == (2000, 1000)
print('genome', g.name, 'tensor', t0.name, 'edges', m.params['edge_list'].n_edges)
"
```

Expected (from `artifacts/canonical_compact_summary_findings.md`): `edge_list.n_edges ≈ 215785` for canonical 1000n at `p=1.0` bipartite per rule (48 rules); not 79k (which implies sparser p). Keep the configured/realized/effective label from that artifact.

Idempotence:

```bash
python -c "import jaxfne as jtfne; g=jtfne.load_canonical_pseudogenome('canonical-v1-column-1000n'); assert jtfne.develop(g,seed=0).provenance['phenotype_sha256']==jtfne.develop(g,seed=0).provenance['phenotype_sha256']"
```

Observation authority (post-hoc, frozen X,Q):

```bash
python scripts/run_multiscale_observation_etude.py   # or Experiment A
# metrics.json must contain K_a != K_b => Y_a != Y_b and identical-W negative control
```

---

## 7. What this design does not do (and why)

* Does not add a compact `define/inherit/use` surface grammar — that is `artifacts/etudes/jdna-compact-grammar/` (preregistration) and is explicitly out of scope for tutorials (docs are data, never code generation).
* Does not make `PseudoGenome` mandatory — `Configuration` and direct `NeuronalTensor` construction remain first-class (per `docs/guides/configuration_grammar.md` and `docs/api/neuronal_tensor.md`). The cumulative arc *demonstrates* the genome path; it does not deprecate the fluent builder.
* Does not claim effectiveness — the tutorials show realization and measurement; authority and reachability remain in `docs/etudes/hdp_controllability_reachability.md` and `multiscale_observation.md`.
* Does not retune HDP gains — it imports `DEFAULT_HDP` / `DESYNC` frozen presets.

---

## 8. File change summary (proposed, additive)

* **Edit:** `docs/tutorials/index.md` — replace Single-topic progression table with Cumulative table (§5.1), patch quick example, demote legacy table to Archive pointers.
* **Edit:** `mkdocs.yml` nav — add `Cumulative (canonical 1000n)` section under Tutorials.
* **Add:** 8 new `docs/tutorials/0{1..8}_*.md` files (thin, 120-180 lines each) that reuse the variable chain above; or, minimally, rewrite the existing 13 files to follow the chain (additive, not destructive — keep old files under `artifacts/`).
* **No change:** `jaxfne/` package, public API, or étude bundles.
* **This audit:** `artifacts/tutorial_cumulative_audit.md` (this file).

```

