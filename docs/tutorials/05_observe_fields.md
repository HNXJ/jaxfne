# 05 — Observe: source → field → probe → spectra

> Continued from [04 — Simulate](04_simulate_tensor.md). Same frozen `model, signals` — observation varies only `O_k`.

One neural simulate; many observation operators post-hoc. This is the observation authority exercised in [Multiscale observation](../etudes/multiscale_observation.md) and [Experiment A](../etudes/experiment_a.md).

```python
import jaxfne as jtfne, numpy as np
# continued — model, signals from 04_simulate_tensor.md
from jaxfne.fields import project_laminar_sources

# field proxy (density-preserving Gaussian kernel, not PDE-solved)
lfp = signals.get("lfp_proxy")  # (2000, n_contacts)
csd = signals.get("csd_proxy")

# or explicitly on frozen Q:
# Y = project_laminar_sources(Q, positions, n_contacts=16, width=0.10)

# spectra on burn-in-trimmed traces
psd = jtfne.spectrolaminar_psd_jax(lfp, dt_ms=0.5)
kappa = jtfne.kappa_synchrony(np.asarray(signals.get("spikes")), dt_ms=0.5)
print(f"kappa={kappa:.3f}")  # expect ≈0 (async-irregular)
```

Rules:
- Freeze `X, Q`; then `K_a ≠ K_b ⇒ Y_a ≠ Y_b` (distinct operators give distinct readouts on same trajectory).
- CSD = `Dzz·LFP` suppresses the broad deep alpha/beta — compute the spectrolaminar crossover on **LFP**, not CSD (`signal_key="lfp_contacts"`).
- EEG/MEG/EMM are toy/declared linear leadfields (`analysis_only`); field amplitudes are `proxy_relative` / `physical_amplitude_calibrated=False`.

Next: [06 — Add state](06_add_state.md) — carry a per-neuron `H` container.
