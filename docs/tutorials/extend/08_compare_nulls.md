# 08 — Compare: nulls, lesions, and authority

> Continued from [07 — Add dynamics](07_add_dynamics.md). Same `model, signals`; now measure effectiveness as `ΔX`.

Realization does not imply effectiveness. Compare the same canonical trajectory against shuffled, lesioned, and multi-area controls.

```python
import jaxfne as jtfne, numpy as np
# continued — model, signals from 04_simulate_tensor.md
# shuffled-time control (preserves rate, destroys Q(t) structure)
spk = np.asarray(signals.get("spikes"))
np.random.seed(0)
spk_shuf = np.apply_along_axis(np.random.permutation, 0, spk)

# lesion a layer (knock-out) — e.g. silence L2/3 E cells via with_emitter_parameters or LESION_SPEC
# see tutorial_utils.simulate_laminar_trials — same API, same model

# multi-area control (3000n) via merge_neuronal_tensors:
t_v1 = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
merged = jtfne.merge_neuronal_tensors([t_v1, t_v1], name="v1_v1_compare")  # area collision → V1_1
# Wire V1→V1_1 AreaConnections explicitly; then construct merged as in step 04

# authority reading (from frozen etudes, not retuned):
# HDP reachability, spectral coupling, and kappa gates live in etudes/multiscale and hdp_controllability
print(jtfne.kappa_synchrony(spk, 0.5))
```

Rules retained from [13 — Canonical column](../../archive/tutorials/readouts/13_canonical_column_etude.md): reuse `construct` for drive sweeps and `with_emitter_parameters` for graded per-layer drive — construction stays the expensive step.

This is the endpoint: every earlier step's variable carries forward to a comparative judgment on one canonical model. For frozen publication claims, see [Études](../../etudes/index.md).

<!-- jx-figure -->
## Figure

=== "Still"

    ![Power spectrum of the reference run](../../assets/readme/oscillatory.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/readme/oscillatory.html" loading="lazy" title="Power spectrum of the reference run"></iframe>

Power spectrum of the reference run: the baseline that nulls and lesions are compared against. [Open full page](https://hnxj.github.io/jaxfne/figures/readme/oscillatory.html).

## Interactive atlas (dark)

Dark-theme Plotly panels from the pinned 1000 ms reference run of this same canonical column — the baseline the nulls/lesions compare against: [Circuit schematic](https://hnxj.github.io/jaxfne/figures/readme/schema.html) · [Network 3D](https://hnxj.github.io/jaxfne/figures/readme/network_3d.html) · [Spike raster](https://hnxj.github.io/jaxfne/figures/readme/raster.html) · [LFP proxy](https://hnxj.github.io/jaxfne/figures/readme/lfp.html) · [Oscillatory response](https://hnxj.github.io/jaxfne/figures/readme/oscillatory.html)

Regenerate: `python scripts/generate_readme_atlas.py --html-only`.
