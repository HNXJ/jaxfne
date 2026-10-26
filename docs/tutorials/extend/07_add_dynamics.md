# 07 — Add dynamics: HDP adaptation

> Continued from [06 — Add state](06_add_state.md). Same `model`; enable dynamics.

Two frozen presets (imported, rather than retuned) show the stability↔variability tradeoff on the same kernel:

```python
import jaxfne as jtfne
from jaxfne import RuntimeConfig
from jaxfne.hdp_network import BASE_HDP_KWARGS_DEFAULT, DEFAULT_HDP, DEFAULT_HDP_DESYNC

# continued — model from 06_add_state.md
hdp_kwargs = dict(BASE_HDP_KWARGS_DEFAULT); hdp_kwargs.update(DEFAULT_HDP)
out = jtfne.simulate(model, runtime=RuntimeConfig(enable_hdp=True, hdp_params=hdp_kwargs))
diag = model.last_hdp_diagnostics()
print(diag["H_trace"].shape, diag["w_trace"].shape)
```

| preset | `tau0` | `K_ctrl` | `alpha` | `gamma` | drive scale | realized `H` | status |
|--------|--------|----------|---------|---------|-------------|--------------|--------|
| `DEFAULT_HDP` | 200.0 | 5.0 | 0.01 | 0.0 | 1.0 | `1.001±0.001` pinned | frozen (freeze gate: 5 seeds × 20 s) |
| `DEFAULT_HDP_DESYNC` | 5.0 | 0.15 | 0.05 | 0.5 | 1.2 | `1.016±0.017` fluctuating | best-of-two-pass, still open |

Realized `H` is the per-neuron mean over the last 1 s of a 2 s run (N=500, seeds 0–4, jaxfne 0.5.5, per-cell-type kernel sizes without the deep-layer size inflation), as mean±std across the 2500 neurons pooled over seeds; population rate is 7.6 Hz and 12.6 Hz, and `kappa` (mean pairwise spike-train correlation) is 0.021 and 0.032, respectively; both are async-irregular. Receipt: `artifacts/protocols/hdp/hdp_presets/realized_h.json` (`scripts/measure_hdp_presets.py`). `H` restoration is `K_ctrl·(1-H_i)`; weight magnitude restoration `K_w_ctrl·(m0-m)` is independent. No new gains here.

<!-- jx-figure -->
## Figure

=== "Still"

    ![Weight trajectories W](../../assets/visuals/hdp_weights.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/hdp_10/hdp.html" loading="lazy" title="Weight trajectories W"></iframe>

Weight trajectories W(t) under HDP in the same small run. [Open full page](https://hnxj.github.io/jaxfne/figures/hdp_10/hdp.html).

Next: [08 — Compare](08_compare_nulls.md) — nulls, lesions, and authority.

## Interactive atlas (dark)

HDP dynamics change the trajectory and leave the circuit as it is: the pinned non-HDP reference panels are here ([Circuit schematic](https://hnxj.github.io/jaxfne/figures/readme/schema.html) · [Network 3D](https://hnxj.github.io/jaxfne/figures/readme/network_3d.html) · [Spike raster](https://hnxj.github.io/jaxfne/figures/readme/raster.html) · [LFP proxy](https://hnxj.github.io/jaxfne/figures/readme/lfp.html) · [Oscillatory response](https://hnxj.github.io/jaxfne/figures/readme/oscillatory.html)), and the small-mechanistic HDP run with full recording is here ([Circuit schematic](https://hnxj.github.io/jaxfne/figures/hdp_10/schema.html) · [Network 3D](https://hnxj.github.io/jaxfne/figures/hdp_10/network_3d.html) · [Spike raster](https://hnxj.github.io/jaxfne/figures/hdp_10/raster.html) · [LFP proxy](https://hnxj.github.io/jaxfne/figures/hdp_10/lfp.html) · [H dynamics](https://hnxj.github.io/jaxfne/figures/hdp_10/h_dynamics.html) · [HDP plasticity](https://hnxj.github.io/jaxfne/figures/hdp_10/hdp.html) · [Oscillatory response](https://hnxj.github.io/jaxfne/figures/hdp_10/oscillatory.html)). A short HDP run of the builder-path 1000-neuron column (weight trace off by recording budget) is here.

Regenerate: `python scripts/generate_readme_atlas.py --html-only` and `python scripts/generate_doc_page_atlases.py --slug hdp_10,hdp_1000`.
