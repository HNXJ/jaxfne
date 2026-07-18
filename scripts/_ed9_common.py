"""Shared helpers for the ed9_*_evidence.py ablation-evidence scripts.

Extracted 2026-07-02 from byte-identical duplication between
ed9_homeostasis_evidence.py and ed9_hdp_evidence.py.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy import stats as _scipy_stats
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


def _significance_test(a: list[float], b: list[float]) -> dict:
    """Two-sided Mann-Whitney U test (nonparametric, valid at small seed counts
    unlike a t-test's normality assumption) plus Cohen's d effect size, between
    two conditions' per-seed metric samples. Returns None fields (not a crash)
    if either sample has fewer than 2 points -- a test needs variance to test."""
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if arr_a.size < 2 or arr_b.size < 2:
        return {"test": "mannwhitneyu", "p_value": None, "cohens_d": None,
                "n_a": int(arr_a.size), "n_b": int(arr_b.size),
                "note": "fewer than 2 seeds in one condition -- no variance to test"}
    u_stat, p_value = _scipy_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    pooled_sd = np.sqrt(((arr_a.size - 1) * arr_a.var(ddof=1) + (arr_b.size - 1) * arr_b.var(ddof=1))
                        / max(arr_a.size + arr_b.size - 2, 1))
    cohens_d = float((arr_a.mean() - arr_b.mean()) / pooled_sd) if pooled_sd > 0 else None
    return {"test": "mannwhitneyu", "u_statistic": float(u_stat), "p_value": float(p_value),
            "cohens_d": cohens_d, "n_a": int(arr_a.size), "n_b": int(arr_b.size)}
