"""Internal HDP adaptive-state layout (not public API).

Separates adaptive mathematics, parameter channel identity, and model storage.
Supports node-local and population H localities for the restoring controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

Locality = Literal["node", "population"]
ChannelKind = Literal["edge", "intrinsic"]

INTERNAL_HDP_RULE_POPULATION = "population_vector_restoring"


@dataclass(frozen=True)
class ThetaChannelSpec:
    """One adaptive coordinate channel (internal; names are diagnostic only)."""

    name: str
    kind: ChannelKind
    initial: float
    bounds_lo: float
    bounds_hi: float
    edge_mask: jnp.ndarray | None = None
    neuron_mask: jnp.ndarray | None = None


@dataclass(frozen=True)
class PopulationRestoringLayout:
    """Frozen population restoring controller layout parsed from hdp_params."""

    h_dim: int
    channels: tuple[ThetaChannelSpec, ...]
    B: jnp.ndarray
    lambda_: float
    tau_H_s: float
    tau_theta_s: float
    setpoint_E_hz: float
    setpoint_I_hz: float
    e_mask: jnp.ndarray
    i_mask: jnp.ndarray
    w_sign: jnp.ndarray
    w_baseline: jnp.ndarray


def resolve_h_state_locality(hp: Mapping[str, Any]) -> Locality:
    loc = hp.get("h_state_locality")
    if loc is None:
        if hp.get("hdp_rule") == INTERNAL_HDP_RULE_POPULATION:
            return "population"
        return "node"
    if loc not in ("node", "population"):
        raise ValueError(f"h_state_locality must be 'node' or 'population', got {loc!r}")
    return loc  # type: ignore[return-value]


def normalize_hdp_params_boundary(hp: Mapping[str, Any]) -> dict[str, Any]:
    """Map public H-state configuration to kernel-ready ``hdp_params``.

    Public population-H semantics are expressed via ``h_state_locality``
    and adaptive-parameter coefficients. The internal restoring-controller
    dispatch identifier is injected here and must not be required from callers.
    """
    out = dict(hp)
    locality = resolve_h_state_locality(out)
    out["h_state_locality"] = locality
    if locality == "population":
        out["hdp_rule"] = INTERNAL_HDP_RULE_POPULATION
    return out


def expected_h_shape(
    *,
    locality: Locality,
    n_neurons: int,
    h_state_dim: int,
) -> tuple[int, ...]:
    if h_state_dim < 1:
        raise ValueError("h_state_dim must be a positive integer")
    if locality == "node":
        return (n_neurons,) if h_state_dim == 1 else (n_neurons, h_state_dim)
    if locality == "population":
        return (h_state_dim,)
    raise ValueError(f"unknown locality {locality!r}")


def parse_population_restoring_layout(
    hp: Mapping[str, Any],
    *,
    edges_weight: jnp.ndarray,
    labels: tuple[str, ...],
    dtype: jnp.dtype,
) -> PopulationRestoringLayout:
    """Build internal layout for population restoring HDP from runtime hdp_params."""
    if hp.get("controller_B") is None:
        raise ValueError("population restoring HDP requires controller_B in hdp_params")
    h_dim = int(hp.get("h_state_dim", 2))
    if h_dim != 2:
        raise ValueError(
            f"population-restoring HDP is a two-coordinate controller "
            f"(rate error channels E/I, theta channels edge/intrinsic); "
            f"h_state_dim must be exactly 2, got {h_dim}. The node-locality "
            f"path supports other h_state_dim values."
        )
    B = jnp.asarray(hp["controller_B"], dtype=dtype)
    if B.shape != (h_dim, h_dim):
        raise ValueError(f"controller_B must be ({h_dim}, {h_dim}), got {B.shape}")

    m_ei_mask = hp.get("m_ei_edge_mask")
    e_neuron_mask = hp.get("e_neuron_mask")
    if m_ei_mask is None:
        raise ValueError("population restoring HDP requires m_ei_edge_mask")
    mei_mask = jnp.asarray(m_ei_mask, dtype=dtype)
    if e_neuron_mask is None:
        e_mask = jnp.array([str(label).startswith("E") for label in labels], dtype=dtype)
    else:
        e_mask = jnp.asarray(e_neuron_mask, dtype=dtype)
    i_mask = 1.0 - e_mask

    m_bounds = hp.get("theta_m_EI_bounds", (0.1, 5.0))
    eta_bounds = hp.get("theta_eta_a_bounds", (0.25, 4.0))
    theta_init = hp.get("controller_theta_S_init")
    if theta_init is None:
        theta0 = (1.0, 1.0)
    else:
        theta0 = tuple(float(x) for x in theta_init)
    if len(theta0) != 2:
        raise ValueError(f"controller_theta_S_init must have length 2, got {theta0}")

    w_baseline = edges_weight.astype(dtype)
    w_sign = jnp.sign(jnp.where(w_baseline == 0.0, 1.0, w_baseline))

    channels = (
        ThetaChannelSpec(
            name="edge_channel_0",
            kind="edge",
            initial=float(theta0[0]),
            bounds_lo=float(m_bounds[0]),
            bounds_hi=float(m_bounds[1]),
            edge_mask=mei_mask,
        ),
        ThetaChannelSpec(
            name="intrinsic_channel_1",
            kind="intrinsic",
            initial=float(theta0[1]),
            bounds_lo=float(eta_bounds[0]),
            bounds_hi=float(eta_bounds[1]),
            neuron_mask=e_mask,
        ),
    )

    setpoint_E = hp.get("controller_rate_setpoint_E_hz")
    setpoint_I = hp.get("controller_rate_setpoint_I_hz")
    if setpoint_E is None or setpoint_I is None:
        raise ValueError(
            "population restoring HDP requires controller_rate_setpoint_E_hz "
            "and controller_rate_setpoint_I_hz in hdp_params"
        )

    return PopulationRestoringLayout(
        h_dim=h_dim,
        channels=channels,
        B=B,
        lambda_=float(hp.get("controller_lambda", 0.45)),
        tau_H_s=float(hp.get("controller_tau_H_s", 0.2)),
        tau_theta_s=float(hp.get("controller_tau_theta_s", 2.0)),
        setpoint_E_hz=float(setpoint_E),
        setpoint_I_hz=float(setpoint_I),
        e_mask=e_mask,
        i_mask=i_mask,
        w_sign=w_sign,
        w_baseline=w_baseline,
    )


def initial_theta_vector(
    layout: PopulationRestoringLayout,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    vals = [ch.initial for ch in layout.channels]
    lo = jnp.asarray([ch.bounds_lo for ch in layout.channels], dtype=dtype)
    hi = jnp.asarray([ch.bounds_hi for ch in layout.channels], dtype=dtype)
    return jnp.clip(jnp.asarray(vals, dtype=dtype), lo, hi)


def theta_bounds(layout: PopulationRestoringLayout, *, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    lo = jnp.asarray([ch.bounds_lo for ch in layout.channels], dtype=dtype)
    hi = jnp.asarray([ch.bounds_hi for ch in layout.channels], dtype=dtype)
    return lo, hi


def bind_theta_to_plant(
    theta: jnp.ndarray,
    layout: PopulationRestoringLayout,
    *,
    a_base: jnp.ndarray,
    w_ceiling: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map adaptive coordinates to effective edge weights and intrinsic parameters."""
    w_eff = layout.w_baseline
    a_eff = a_base
    for i, ch in enumerate(layout.channels):
        if ch.kind == "edge":
            if ch.edge_mask is None:
                raise ValueError("edge channel requires edge_mask")
            w_eff = jnp.where(ch.edge_mask > 0.5, layout.w_sign * theta[i], w_eff)
        elif ch.kind == "intrinsic":
            if ch.neuron_mask is None:
                raise ValueError("intrinsic channel requires neuron_mask")
            a_eff = jnp.where(ch.neuron_mask > 0.5, a_base * theta[i], a_eff)
    w_eff = jnp.clip(w_eff, -w_ceiling, w_ceiling)
    return w_eff, a_eff


def population_rate_error(
    prev_spikes: jnp.ndarray,
    layout: PopulationRestoringLayout,
    *,
    dt_ms: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    n_e = jnp.maximum(jnp.sum(layout.e_mask), jnp.asarray(1.0, dtype=dtype))
    n_i = jnp.maximum(jnp.sum(layout.i_mask), jnp.asarray(1.0, dtype=dtype))
    r_E = jnp.sum(prev_spikes * layout.e_mask) / n_e * (jnp.asarray(1000.0, dtype=dtype) / dt_ms)
    r_I = jnp.sum(prev_spikes * layout.i_mask) / n_i * (jnp.asarray(1000.0, dtype=dtype) / dt_ms)
    return jnp.asarray(
        [r_E - layout.setpoint_E_hz, r_I - layout.setpoint_I_hz],
        dtype=dtype,
    )


def population_restoring_derivatives(
    H: jnp.ndarray,
    e_vec: jnp.ndarray,
    layout: PopulationRestoringLayout,
    *,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lam = jnp.asarray(layout.lambda_, dtype=dtype)
    tau_H = jnp.asarray(layout.tau_H_s, dtype=dtype)
    tau_theta = jnp.asarray(layout.tau_theta_s, dtype=dtype)
    dH = (-e_vec - lam * H) / tau_H
    d_theta = (layout.B @ H) / tau_theta
    return dH, d_theta


def reject_population_continuation(locality: Locality, *, context: str) -> None:
    if locality == "population":
        raise ValueError(
            f"{context}: population H-state locality is not supported for "
            "full-state continuation in this release; use Model.simulate() "
            "or reject explicitly"
        )


def population_layout_fingerprint(layout: PopulationRestoringLayout) -> tuple:
    """Hashable fingerprint for JIT cache keys (arrays via content hash)."""
    parts: list[Any] = [
        ("h_dim", layout.h_dim),
        ("lambda", layout.lambda_),
        ("tau_H_s", layout.tau_H_s),
        ("tau_theta_s", layout.tau_theta_s),
        ("setpoint_E_hz", layout.setpoint_E_hz),
        ("setpoint_I_hz", layout.setpoint_I_hz),
    ]
    for arr_name, arr in (
        ("B", np.asarray(layout.B)),
        ("e_mask", np.asarray(layout.e_mask)),
        ("i_mask", np.asarray(layout.i_mask)),
        ("w_sign", np.asarray(layout.w_sign)),
        ("w_baseline", np.asarray(layout.w_baseline)),
    ):
        parts.append((arr_name, arr.shape, str(arr.dtype), hash(arr.tobytes())))
    for i, ch in enumerate(layout.channels):
        if ch.edge_mask is not None:
            em = np.asarray(ch.edge_mask)
            parts.append((f"ch{i}_edge", em.shape, hash(em.tobytes())))
        if ch.neuron_mask is not None:
            nm = np.asarray(ch.neuron_mask)
            parts.append((f"ch{i}_neuron", nm.shape, hash(nm.tobytes())))
    return tuple(parts)
