"""Protocol H4 — length × delay-heterogeneity factorial on activity-expressed RBD memory.

Primary endpoint (pre-registered):
    M_X = ∫ [M_X(Δ) - M_{X,shuffle}(Δ)]_+ dΔ

See ``docs/doctrine/protocol_h_rbd_memory.md`` §6.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.h3_decodability import (
    H3ProtocolConfig,
    area_above_shuffle,
    run_h3_decodability_study,
)

LengthClass = Literal["short", "long"]
DelayClass = Literal["uniform", "heterogeneous"]
CellKey = tuple[LengthClass, DelayClass]

CELL_KEYS: tuple[CellKey, ...] = (
    ("short", "uniform"),
    ("short", "heterogeneous"),
    ("long", "uniform"),
    ("long", "heterogeneous"),
)


@dataclass(frozen=True)
class H4ProtocolConfig:
    """Frozen H4 factorial contract (no post-hoc tuning)."""

    n_short: int = 3
    n_long: int = 12
    uniform_delay_steps: int = 4
    hetero_delay_steps: tuple[int, int] = (2, 8)
    edge_weight: float = 6.0
    syn_tau_ms: float = 3.0
    n_steps: int = 80
    h3: H3ProtocolConfig = field(
        default_factory=lambda: H3ProtocolConfig(
            delta_h=0.2,
            beta_h=0.5,
            rbd_family="f1",
            lag_steps=(2, 5, 10, 20, 35),
            train_seeds=tuple(range(100, 110)),
            test_seeds=tuple(range(200, 210)),
            n_shuffle=8,
        )
    )


def build_ring_params_edges(
    n_neurons: int,
    *,
    delay_steps: np.ndarray | int,
    weight: float = 6.0,
    tau_ms: float = 3.0,
) -> tuple[IzhikevichParams, EdgeList]:
    """Directed one-neighbor ring with matched local coupling."""
    if n_neurons < 2:
        raise ValueError("ring requires n_neurons >= 2")
    jdtype = jnp.float32
    labels = tuple("E" for _ in range(n_neurons))
    params = IzhikevichParams(
        v0=jnp.full((n_neurons,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n_neurons,), dtype=jdtype),
        a=jnp.full((n_neurons,), 0.02, dtype=jdtype),
        b=jnp.full((n_neurons,), 0.2, dtype=jdtype),
        c=jnp.full((n_neurons,), -65.0, dtype=jdtype),
        d=jnp.full((n_neurons,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n_neurons,), dtype=jdtype),
        sign=jnp.ones((n_neurons,), dtype=jdtype),
        W=jnp.zeros((n_neurons, n_neurons), dtype=jdtype),
        source_scale=jnp.ones((n_neurons,), dtype=jdtype),
        labels=labels,
        layer_labels=labels,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    pre = np.arange(n_neurons, dtype=np.int32)
    post = np.roll(pre, -1)
    if isinstance(delay_steps, int):
        ds = np.full((n_neurons,), int(delay_steps), dtype=np.int32)
    else:
        ds = np.asarray(delay_steps, dtype=np.int32)
        if ds.shape != (n_neurons,):
            raise ValueError(f"delay_steps must have shape ({n_neurons},)")
    edges = EdgeList(
        pre=jnp.asarray(pre, dtype=jnp.int32),
        post=jnp.asarray(post, dtype=jnp.int32),
        weight=jnp.full((n_neurons,), float(weight), dtype=jdtype),
        receptor_index=jnp.zeros((n_neurons,), dtype=jnp.int32),
        tau_ms=jnp.full((n_neurons,), float(tau_ms), dtype=jdtype),
        delay_steps=jnp.asarray(ds, dtype=jnp.int32),
    )
    return params, edges


def build_ring_params_edges_km(
    n_neurons: int,
    k_neighbors: int,
    *,
    arc_delay_steps: np.ndarray,
    weight: float = 6.0,
    tau_ms: float = 3.0,
) -> tuple[IzhikevichParams, EdgeList]:
    """Directed K-neighbor ring; ``arc_delay_steps[m-1]`` = delay steps for skip m.

    Skip m connects neuron i to neuron (i + m) mod n at arc length m*a
    (a = 2*pi*R/n). Edge ordering mirrors ``build_ring_params_edges``: blocks
    of skip 1..K, each block ordered by pre index.
    """
    if n_neurons < 2:
        raise ValueError("ring requires n_neurons >= 2")
    if not 1 <= k_neighbors < n_neurons:
        raise ValueError(f"k_neighbors must be in [1, {n_neurons})")
    arc_delay_steps = np.asarray(arc_delay_steps, dtype=np.int32)
    if arc_delay_steps.shape != (k_neighbors,):
        raise ValueError(f"arc_delay_steps must have shape ({k_neighbors},)")
    jdtype = jnp.float32
    labels = tuple("E" for _ in range(n_neurons))
    params = IzhikevichParams(
        v0=jnp.full((n_neurons,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n_neurons,), dtype=jdtype),
        a=jnp.full((n_neurons,), 0.02, dtype=jdtype),
        b=jnp.full((n_neurons,), 0.2, dtype=jdtype),
        c=jnp.full((n_neurons,), -65.0, dtype=jdtype),
        d=jnp.full((n_neurons,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n_neurons,), dtype=jdtype),
        sign=jnp.ones((n_neurons,), dtype=jdtype),
        W=jnp.zeros((n_neurons, n_neurons), dtype=jdtype),
        source_scale=jnp.ones((n_neurons,), dtype=jdtype),
        labels=labels,
        layer_labels=labels,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    n_edges = n_neurons * k_neighbors
    pre_blocks = [np.arange(n_neurons, dtype=np.int32) for _ in range(k_neighbors)]
    post_blocks = [
        np.roll(np.arange(n_neurons, dtype=np.int32), -m) for m in range(1, k_neighbors + 1)
    ]
    edges = EdgeList(
        pre=jnp.asarray(np.concatenate(pre_blocks), dtype=jnp.int32),
        post=jnp.asarray(np.concatenate(post_blocks), dtype=jnp.int32),
        weight=jnp.full((n_edges,), float(weight), dtype=jdtype),
        receptor_index=jnp.zeros((n_edges,), dtype=jnp.int32),
        tau_ms=jnp.full((n_edges,), float(tau_ms), dtype=jdtype),
        delay_steps=jnp.concatenate(
            [jnp.full((n_neurons,), int(d), dtype=jnp.int32) for d in arc_delay_steps]
        ),
    )
    return params, edges


def ring_delay_pattern(
  n_neurons: int,
  *,
  delay_class: DelayClass,
  cfg: H4ProtocolConfig,
) -> np.ndarray:
    if delay_class == "uniform":
        return np.full((n_neurons,), int(cfg.uniform_delay_steps), dtype=np.int32)
    lo, hi = cfg.hetero_delay_steps
    return np.asarray(
        [lo if (i % 2 == 0) else hi for i in range(n_neurons)],
        dtype=np.int32,
    )


def loop_return_time_ms(edges: EdgeList, *, dt_ms: float) -> float:
    r"""``T_loop = sum_{e in loop} tau_e`` with ``tau_e = delay_steps[e] * dt_ms``."""
    delays = np.asarray(edges.delay_steps, dtype=np.int64)
    return float(delays.sum() * float(dt_ms))


def cell_key_str(length: LengthClass, delay: DelayClass) -> str:
    return f"{length}_{delay}"


def build_cell_circuit(
    length: LengthClass,
    delay: DelayClass,
    *,
    cfg: H4ProtocolConfig,
) -> tuple[IzhikevichParams, EdgeList, dict[str, Any]]:
    n = cfg.n_short if length == "short" else cfg.n_long
    ds = ring_delay_pattern(n, delay_class=delay, cfg=cfg)
    params, edges = build_ring_params_edges(
        n,
        delay_steps=ds,
        weight=cfg.edge_weight,
        tau_ms=cfg.syn_tau_ms,
    )
    receipt = {
        "length_class": length,
        "delay_class": delay,
        "n_neurons": n,
        "delay_steps": ds.tolist(),
        "edge_weight": cfg.edge_weight,
        "syn_tau_ms": cfg.syn_tau_ms,
        "t_loop_ms": loop_return_time_ms(edges, dt_ms=cfg.h3.dt_ms),
    }
    return params, edges, receipt


def factorial_design_matrix() -> tuple[np.ndarray, list[CellKey]]:
    """Effect-coded 2×2 design: columns [1, L, D, L×D]."""
    rows = []
    keys: list[CellKey] = []
    for length in ("short", "long"):
        for delay in ("uniform", "heterogeneous"):
            L = 1.0 if length == "long" else 0.0
            D = 1.0 if delay == "heterogeneous" else 0.0
            rows.append([1.0, L, D, L * D])
            keys.append((length, delay))
    return np.asarray(rows, dtype=np.float64), keys


def fit_factorial_mx(
    cell_mx: Mapping[str, float],
) -> dict[str, float]:
    """OLS fit ``M_X = μ + α_L L + α_D D + α_{L×D} LD + ε`` on four cells."""
    X, keys = factorial_design_matrix()
    y = np.asarray([float(cell_mx[cell_key_str(*k)]) for k in keys], dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return {
        "mu": float(beta[0]),
        "alpha_length": float(beta[1]),
        "alpha_heterogeneity": float(beta[2]),
        "alpha_interaction": float(beta[3]),
        "residuals": {cell_key_str(*k): float(r) for k, r in zip(keys, resid)},
    }


def detect_curve_peaks(
    m_curve: Mapping[int, float],
    m_shuffle: Mapping[int, float],
    *,
    min_excess: float = 0.02,
) -> list[dict[str, float]]:
    """Local maxima of excess decodability ``[M-M_shuffle]_+``."""
    lags = sorted(int(l) for l in m_curve.keys())
    excess = [
        max(float(m_curve[l]) - float(m_shuffle.get(l, 0.0)), 0.0) for l in lags
    ]
    peaks: list[dict[str, float]] = []
    for i, lag in enumerate(lags):
        left = excess[i - 1] if i > 0 else -1.0
        right = excess[i + 1] if i + 1 < len(lags) else -1.0
        if excess[i] >= left and excess[i] >= right and excess[i] >= min_excess:
            peaks.append({"lag_steps": float(lag), "excess_mx": float(excess[i])})
    return peaks


def loop_peak_alignment_table(
    peaks: Sequence[Mapping[str, float]],
    *,
    t_loop_ms: float,
    dt_ms: float,
    max_harmonic: int = 3,
) -> list[dict[str, float]]:
    """Diagnostic only: compare peak lags to ``k * T_loop`` (not scored)."""
    rows: list[dict[str, float]] = []
    for peak in peaks:
        lag_ms = float(peak["lag_steps"]) * float(dt_ms)
        for k in range(1, int(max_harmonic) + 1):
            pred = k * float(t_loop_ms)
            rows.append(
                {
                    "peak_lag_steps": float(peak["lag_steps"]),
                    "peak_lag_ms": lag_ms,
                    "peak_excess_mx": float(peak["excess_mx"]),
                    "harmonic_k": float(k),
                    "predicted_loop_ms": pred,
                    "abs_error_ms": abs(lag_ms - pred),
                }
            )
    return rows


def _study_summary(study: Mapping[str, Any]) -> dict[str, Any]:
    curves = study["curves"]
    shuffle = study["shuffle_curves"]
    mx = curves["X"]
    mx_shuf = shuffle["X"]
    mh = curves["H"]
    d_h = study["diagnostics"]["D_H"]
    m_x_area = float(study["area_above_shuffle"]["X"])
    peaks = detect_curve_peaks(mx, mx_shuf)
    secondary = peaks[1:] if len(peaks) > 1 else []
    return {
        "M_X": {str(k): float(v) for k, v in mx.items()},
        "M_X_shuffle": {str(k): float(v) for k, v in mx_shuf.items()},
        "M_H": {str(k): float(v) for k, v in mh.items()},
        "D_H": {str(k): float(v) for k, v in d_h.items()},
        "M_X_area": m_x_area,
        "M_X_excess_peaks": peaks,
        "M_X_secondary_peaks": secondary,
    }


def run_h4_matrix(
    cfg: H4ProtocolConfig | None = None,
    *,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Execute the frozen 2×2 factorial and return receipt-ready outputs."""
    cfg = cfg or H4ProtocolConfig()
    cells: dict[str, Any] = {}
    cell_mx: dict[str, float] = {}
    alignment: dict[str, Any] = {}

    for length, delay in CELL_KEYS:
        key = cell_key_str(length, delay)
        params, edges, circuit = build_cell_circuit(length, delay, cfg=cfg)
        study = run_h3_decodability_study(
            params,
            edges,
            n_steps=cfg.n_steps,
            cfg=cfg.h3,
            rng_seed=rng_seed,
        )
        summary = _study_summary(study)
        cell_mx[key] = summary["M_X_area"]
        peaks_for_align = summary["M_X_secondary_peaks"] or summary["M_X_excess_peaks"]
        alignment[key] = {
            "t_loop_ms": circuit["t_loop_ms"],
            "peaks_used": peaks_for_align,
            "harmonic_table": loop_peak_alignment_table(
                peaks_for_align,
                t_loop_ms=circuit["t_loop_ms"],
                dt_ms=cfg.h3.dt_ms,
            ),
        }
        cells[key] = {
            "circuit": circuit,
            "summary": summary,
            "chance": float(study["chance"]),
            "n_classes": int(study["n_classes"]),
        }

    factorial = fit_factorial_mx(cell_mx)
    return {
        "protocol_id": "protocol_h_rbd_memory_h4",
        "scientific_question": "Does recurrent geometry alter activity-expressed RBD memory?",
        "primary_endpoint": "M_X_area = integral [M_X(Delta)-M_X_shuffle(Delta)]_+ dDelta",
        "interpretation_scope": (
            "Positive H4 supports distributed fading dynamical memory in RBD only; "
            "not long-term memory, plastic memory, predictive coding, or surprise minimization."
        ),
        "config": asdict(cfg),
        "cells": cells,
        "cell_M_X_area": cell_mx,
        "factorial": factorial,
        "loop_peak_alignment": alignment,
        "interpretation_gates": {
            "alpha_length_gt_0": "length contribution",
            "alpha_heterogeneity_gt_0": "delay-heterogeneity contribution",
            "alpha_interaction_gt_0": "synergy",
            "all_alphas_near_0": "valid falsification of topology extension under tested regime",
        },
    }
