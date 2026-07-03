"""Shared helpers for the ed9_*_evidence.py ablation-evidence scripts.

Extracted 2026-07-02 from byte-identical duplication between
ed9_homeostasis_evidence.py and ed9_hdp_evidence.py.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne


def build_imbalanced_model(n: int, hi_drive: float, lo_drive: float):
    """Canonical column with a hyperactive (first half) and quiet (second half) group."""
    cfg = (
        jtfne.build_laminar_column(n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model = jtfne.construct(cfg)
    n_real = int(model.params["emitter"].v0.shape[0])
    drive = np.where(np.arange(n_real) < n_real // 2, hi_drive, lo_drive).astype(np.float32)
    model = jtfne.with_emitter_parameters(model, drive_per_neuron=jnp.asarray(drive))
    return model, n_real


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(arr.mean()), "std": sd,
            "ci95_halfwidth": float(1.96 * sd / max(n, 1) ** 0.5), "n": n}
