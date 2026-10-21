# Tutorials

One track, one model. Eight steps take the canonical 1000-neuron column
`canonical-v1-column-1000n` from its generative rules to a comparison against
nulls, and every step reuses the object the previous one produced.

<div class="mermaid">
flowchart LR
  G["PseudoGenome<br/>01 define"] -->|"develop(K_D)<br/>02"| N["NeuronalTensor<br/>03 inspect"]
  N -->|"construct(K_S)<br/>04"| M["Model"]
  M -->|"simulate<br/>04"| S["Signals"]
  S -->|"05 observe"| F["Fields & probes"]
  M -->|"06 add state"| H["H (RBS)"]
  H -->|"07 add dynamics"| W["HDP: dW/dt"]
  S -->|"08 compare"| C["Nulls & lesions"]
</div>

<div class="jx-cards">
  <a href="01_define_genome/"><img src="../assets/readme/schema.png" alt="Declared column structure"><strong>01 Define</strong><br/>PseudoGenome rules</a>
  <a href="02_develop_genome/"><img src="../assets/visuals/column_network.png" alt="Developed column network"><strong>02 Develop</strong><br/>Genome → tensor</a>
  <a href="03_inspect_tensor/"><img src="../assets/readme/network_3d.png" alt="Realized neuron positions"><strong>03 Inspect</strong><br/>Realized vs configured</a>
  <a href="04_simulate_tensor/"><img src="../assets/readme/raster.png" alt="Spike raster"><strong>04 Simulate</strong><br/>construct → Signals</a>
  <a href="05_observe_fields/"><img src="../assets/readme/lfp.png" alt="Laminar LFP proxy"><strong>05 Observe</strong><br/>Fields and probes</a>
  <a href="06_add_state/"><img src="../assets/visuals/hdp_h_dynamics.png" alt="Hidden state H over time"><strong>06 Add state</strong><br/>H / RBS container</a>
  <a href="07_add_dynamics/"><img src="../assets/visuals/hdp_weights.png" alt="Weight trajectories under HDP"><strong>07 Add dynamics</strong><br/>HDP adaptation</a>
  <a href="08_compare_nulls/"><img src="../assets/readme/oscillatory.png" alt="Power spectrum"><strong>08 Compare</strong><br/>Nulls and lesions</a>
</div>

```python
import jaxfne as jtfne

genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")  # 01 define
tensor = jtfne.develop(genome, seed=0)                                   # 02 develop (K_D)
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))  # 04 (K_S)
signals = jtfne.simulate(model)                                          # same compiler for every on-ramp
```

Smoke mode (`SMOKE=1`, `n≈100`, `duration_ms≈100`) runs in about 30 s; the
documented path is 1000 neurons, `duration_ms=1000.0`, `dt_ms=0.5`. Development
and runtime keys are distinct (`K_D ≠ K_S`; see the [PseudoGenome guide](../guides/grammar/jdna.md)).

| # | Step | Topic | Focus | Reuses |
|---|------|-------|-------|--------|
| [**01**](construct/01_define_genome.md) | define | PseudoGenome | `validate_genome`, `genome_rules_hash`, `declared_constraints`, tolerance bands | — |
| [**02**](construct/02_develop_genome.md) | develop | Genome → phenotype | `develop(G,K_D)` determinism, seed jitter within bands, `phenotype_sha256` | `genome` |
| [**03**](construct/03_inspect_tensor.md) | inspect | Realized vs configured | `neuron_table()`, per-layer counts, 48 rules, `save/load_neuronal_tensor` | `tensor` |
| [**04**](construct/04_simulate_tensor.md) | simulate | Construct → Signals | `construct(tensor, RuntimeConfiguration)` → `simulate`; `EdgeList.n_edges≈215k`, `Pose3D` positions | `tensor` |
| [**05**](extend/05_observe_fields.md) | observe | Fields and probes | LFP/CSD/EEG/MEG/PSD on the frozen `X,Q`; `K_a≠K_b ⇒ Y_a≠Y_b` | `model, signals` |
| [**06**](extend/06_add_state.md) | add state | H / RBS | `PlasticParams.H` → `h_state`, `with_hdp_initial_state`, checkpoint/restore | `tensor, model` |
| [**07**](extend/07_add_dynamics.md) | add dynamics | HDP | `enable_hdp` (`DEFAULT_HDP` vs `DESYNC`), `H_trace/w_trace`, `K_HDP/K_ctrl/K_w_ctrl` | `model` |
| [**08**](extend/08_compare_nulls.md) | compare | Nulls, lesions | shuffled, `LESION_SPEC`, multi-area 3000n via `merge_neuronal_tensors`, `kappa` | all |

The step table with receipts is audited in `artifacts/audit/reviews/tutorial_cumulative_audit.md`.
The fluent `Configuration` builder is the other on-ramp; both converge on
`construct → simulate` ([Configuration Grammar](../guides/grammar/configuration_grammar.md)).
`NeuronalTensor` is the declarative `Areas × Layers × NeuronTypes` object the track
carries; see the [API reference](../reference/api/model/neuronal_tensor.md) and the
[H-state / HDP guide](../guides/state/hdp.md).

## Scale: from one column to twenty areas

The same language builds the 20-area hierarchy used by Atlas AT-10-N20: areas in
hierarchy order, one chord per connected pair, chord width by edge count.

=== "Still"

    ![G_20 area graph](../assets/visuals/area_graph_n20.png)

=== "Interactive"

    <iframe class="jx-frame" src="../_static/visuals/area_graph_n20.html" loading="lazy" title="G_20 area graph"></iframe>

[Open full page](../_static/visuals/area_graph_n20.html). Reference atlas of the
1000 ms column run: [index](../_static/atlas/index.html) · [schema](../_static/atlas/schema.html) · [3D](../_static/atlas/network_3d.html) · [raster](../_static/atlas/raster.html) · [LFP](../_static/atlas/lfp.html) · [oscillatory](../_static/atlas/oscillatory.html).

All notebooks follow the [Colab notebook standard](notebook_standard.md).
Frozen scientific demonstrations are the **[Études](../etudes/index.md)**; the pages on this index are tutorials.
Earlier stand-alone tutorials and the suites, written for 0.4.x, are in the
[Archive](../archive/index.md).

## Running

Notebooks live under `tutorials/` (études under `tutorials/etudes/`); headless
runners under `examples/`:

```bash
python examples/v031_single_izhikevich_neuron.py
```

Regenerate the figures on these pages with `python scripts/generate_readme_atlas.py`,
`python scripts/generate_doc_page_atlases.py` and `python scripts/generate_docs_visuals.py`.

Next: [Guides](../guides/index.md) · [API reference](../reference/api/index.md) · [Jaxley interoperability](../guides/interop/jaxley_interop.md)
