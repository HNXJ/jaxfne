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

Rules retained from [13 — Canonical column](13_canonical_column_etude.md): reuse `construct` for drive sweeps and `with_emitter_parameters` for graded per-layer drive — construction stays the expensive step.

This is the endpoint: every earlier verb's variable has been carried forward to a comparative judgment on one canonical model. For frozen publication claims, see [Études](../etudes/index.md).
