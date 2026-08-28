# 07 — Add dynamics: HDP adaptation

> Continued from [06 — Add state](06_add_state.md). Same `model`; enable dynamics.

Two frozen presets (imported, not retuned) illustrate the stability↔variability tradeoff already validated on the same kernel:

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
| `DEFAULT_HDP` | 200.0 | 5.0 | 0.01 | 0.0 | 1.0 | `1.001±0.0006` pinned | frozen (5-seed×20s) |
| `DEFAULT_HDP_DESYNC` | 5.0 | 0.15 | 0.05 | 0.5 | 1.2 | `1.028±0.023` fluctuating | best-of-two-pass, still open |

Both keep `kappa ≈ 0.04` (async-irregular). `H` restoration is `K_ctrl·(1-H_i)`; weight magnitude restoration `K_w_ctrl·(m0-m)` is independent. No new gains are invented here.

Next: [08 — Compare](08_compare_nulls.md) — nulls, lesions, and authority.
