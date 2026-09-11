"""Class-shared edge attribute storage (23-REP-01).

High-R_k edge fields (``tau_ms``, uniform ``delay_steps``) may be stored as
execution layouts derived from class tables rather than per-edge materialization.
``receptor_index`` remains one index per edge; it may be stored as uint8 when
the realized class range is losslessly narrowable. Compaction is allowed only
when the expanded view is **derivable**, not merely correlated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .emitters import (
    EdgeList,
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
    resolve_edge_weight,
    resolve_receptor_index,
    sign_only_tau_exc_ms,
    sign_only_tau_inh_ms,
)


def attribute_cardinality(array: Any) -> dict[str, Any]:
    """Return element count, distinct count, and redundancy ratio R_k."""
    a = np.asarray(array)
    n = int(a.size)
    if n == 0:
        return {"size": 0, "n_unique": 0, "R_k": None}
    n_unique = int(np.unique(a).size)
    return {
        "size": n,
        "n_unique": n_unique,
        "R_k": float(n) / float(n_unique) if n_unique else None,
    }


def build_declared_mechanism_weight_magnitude_table(
    metadata: Mapping[str, Any],
) -> np.ndarray | None:
    """Per-mechanism declared weight magnitudes in circuit declaration order.

    Returns ``None`` when mechanisms are absent, any mechanism lacks a resolvable
    rule weight, or rules disagree on magnitude for the same mechanism. Magnitudes
    are unsigned; presynaptic intrinsic sign is applied at materialization.
    """
    mechanisms = (metadata.get("circuit") or {}).get("mechanisms", [])
    connections = (metadata.get("circuit") or {}).get("connections", [])
    if not mechanisms or not connections:
        return None
    name_to_idx = {str(m.get("name")): i for i, m in enumerate(mechanisms)}
    magnitudes: dict[int, float] = {}
    for rule in connections:
        mech = rule.get("mechanism")
        if mech is None or mech not in name_to_idx:
            return None
        sign_str = rule.get("sign")
        if sign_str in ("excitatory", "inhibitory"):
            return None
        w_spec = rule.get("weight")
        if w_spec is None:
            return None
        mag = float(abs(w_spec))
        if not np.isfinite(mag):
            return None
        idx = name_to_idx[str(mech)]
        prev = magnitudes.get(idx)
        if prev is not None and prev != mag:
            return None
        magnitudes[idx] = mag
    if len(magnitudes) != len(mechanisms):
        return None
    return np.asarray([magnitudes[i] for i in range(len(mechanisms))], dtype=np.float64)


def build_declared_scalar_weight_magnitude(metadata: Mapping[str, Any]) -> float | None:
    """Single shared magnitude when every connection rule agrees on unsigned weight."""
    connections = (metadata.get("circuit") or {}).get("connections", [])
    if not connections:
        return None
    magnitudes: list[float] = []
    for rule in connections:
        sign_str = rule.get("sign")
        if sign_str in ("excitatory", "inhibitory"):
            return None
        w_spec = rule.get("weight")
        if w_spec is None:
            return None
        mag = float(abs(w_spec))
        if not np.isfinite(mag):
            return None
        magnitudes.append(mag)
    if not magnitudes or len(set(magnitudes)) != 1:
        return None
    return magnitudes[0]


def build_declared_mechanism_tau_table(metadata: Mapping[str, Any]) -> np.ndarray | None:
    """Build the declared per-mechanism tau table in circuit declaration order.

    Returns ``None`` when mechanisms are absent or any entry lacks an explicit
    finite positive ``tau_ms``. This table is authoritative metadata only --
    it is never inferred from ``standard_receptor_specs`` or weight sign.
    """
    mechanisms = (metadata.get("circuit") or {}).get("mechanisms", [])
    if not mechanisms:
        return None
    taus: list[float] = []
    for m in mechanisms:
        params = dict(m.get("params", {}))
        tau = params.get("tau_ms", m.get("tau_ms"))
        if tau is None:
            return None
        tau_f = float(tau)
        if not np.isfinite(tau_f) or tau_f <= 0.0:
            return None
        taus.append(tau_f)
    return np.asarray(taus, dtype=np.float64)


def audit_edge_list_storage(
    edges: EdgeList, *, presynaptic_sign: np.ndarray | None = None
) -> dict[str, Any]:
    """Summarize per-edge redundancy and active compaction modes."""
    ri_resolved = np.asarray(resolve_receptor_index(edges))
    tau_resolved = np.asarray(resolve_edge_tau_ms(edges, edges.weight.dtype))
    delay_resolved = np.asarray(resolve_edge_delay_steps(edges))
    if edges.weight_storage == "per_edge":
        weight_resolved = np.asarray(edges.weight)
    elif presynaptic_sign is not None:
        weight_resolved = np.asarray(
            resolve_edge_weight(
                edges,
                edges.weight.dtype,
                presynaptic_sign=np.asarray(presynaptic_sign, dtype=np.float64),
            )
        )
    else:
        weight_resolved = np.asarray(edges.weight)
    redundancy = {
        "pre": attribute_cardinality(edges.pre),
        "post": attribute_cardinality(edges.post),
        "weight": attribute_cardinality(weight_resolved),
        "receptor_index": attribute_cardinality(ri_resolved),
        "tau_ms": attribute_cardinality(tau_resolved),
        "delay_steps": attribute_cardinality(delay_resolved),
    }
    return {
        "n_edges": int(edges.n_edges),
        "tau_storage": edges.tau_storage,
        "delay_storage": edges.delay_storage,
        "uniform_delay_steps": int(edges.uniform_delay_steps),
        "receptor_index_storage": edges.receptor_index_storage,
        "weight_storage": edges.weight_storage,
        "weight_magnitude": (
            float(np.asarray(edges.weight_magnitude))
            if edges.weight_magnitude is not None and np.asarray(edges.weight_magnitude).size
            else None
        ),
        "mechanism_weight_magnitude_table": (
            [float(x) for x in np.asarray(edges.mechanism_weight_magnitude_table)]
            if edges.mechanism_weight_magnitude_table is not None
            else None
        ),
        "mechanism_tau_table": (
            [float(x) for x in np.asarray(edges.mechanism_tau_table)]
            if edges.mechanism_tau_table is not None
            else None
        ),
        "redundancy": redundancy,
        "tau_sign_table_compatible": tau_ms_matches_sign_receptor_table(edges),
        "delay_uniform_zero": delay_steps_are_uniform_zero(edges),
        "weight_sign_receptor_compatible": weight_matches_sign_receptor_magnitude(
            edges, presynaptic_sign=presynaptic_sign
        ),
    }


def _resolved_weight_numpy(
    edges: EdgeList, presynaptic_sign: np.ndarray | None = None
) -> np.ndarray:
    if edges.weight_storage == "per_edge":
        return np.asarray(edges.weight, dtype=np.float64)
    if presynaptic_sign is None:
        raise ValueError("presynaptic_sign required for compact edge weight audit")
    return np.asarray(
        resolve_edge_weight(
            edges,
            np.float64,
            presynaptic_sign=np.asarray(presynaptic_sign, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def weight_matches_sign_receptor_magnitude(
    edges: EdgeList,
    magnitude: float | None = None,
    *,
    presynaptic_sign: np.ndarray | None = None,
) -> bool:
    """True when weight equals +/- magnitude from exc/inh receptor_index."""
    if edges.n_edges == 0:
        return True
    if edges.weight_storage == "sign_from_receptor":
        return True
    if edges.weight_storage == "magnitude_times_presynaptic_sign":
        return False
    ri = np.asarray(resolve_receptor_index(edges), dtype=np.int32)
    w = _resolved_weight_numpy(edges, presynaptic_sign)
    if magnitude is None:
        mags = np.unique(np.abs(w))
        if mags.size != 1:
            return False
        magnitude = float(mags[0])
    expected = np.where(ri == 0, float(magnitude), -float(magnitude))
    return bool(np.array_equal(w, expected))


def weight_matches_presynaptic_sign_magnitude(
    edges: EdgeList,
    presynaptic_sign: np.ndarray,
    *,
    magnitude: float | None = None,
    mechanism_weight_magnitude_table: np.ndarray | None = None,
) -> bool:
    """True when weight equals magnitude * presynaptic_sign[pre] for every edge."""
    if edges.n_edges == 0:
        return True
    if edges.weight_storage == "magnitude_times_presynaptic_sign":
        return True
    pre = np.asarray(edges.pre, dtype=np.int64)
    w = _resolved_weight_numpy(edges, presynaptic_sign)
    sign_pre = np.asarray(presynaptic_sign, dtype=np.float64)[pre]
    if mechanism_weight_magnitude_table is not None:
        ri = np.asarray(resolve_receptor_index(edges), dtype=np.int64)
        table = np.asarray(mechanism_weight_magnitude_table, dtype=np.float64)
        if ri.min() < 0 or ri.max() >= table.size:
            return False
        expected = table[ri] * sign_pre
    else:
        if magnitude is None:
            ratios = w / np.where(sign_pre == 0, 1.0, sign_pre)
            mags = np.unique(np.abs(ratios))
            if mags.size != 1:
                return False
            magnitude = float(mags[0])
        expected = float(magnitude) * sign_pre
    return bool(np.allclose(w, expected))


def tau_ms_matches_sign_receptor_table(edges: EdgeList) -> bool:
    """True when per-edge tau equals the explicitly qualified sign-only 2/5 ms map."""
    if edges.n_edges == 0:
        return True
    if edges.tau_storage == "sign_from_receptor":
        return True
    ri = np.asarray(resolve_receptor_index(edges), dtype=np.int32)
    tau = np.asarray(resolve_edge_tau_ms(edges, np.float64), dtype=np.float64)
    expected = np.where(ri == 0, sign_only_tau_exc_ms(), sign_only_tau_inh_ms())
    return bool(np.array_equal(tau, expected))


def tau_ms_matches_declared_mechanism_table(
    edges: EdgeList, mechanism_tau_table: np.ndarray | None
) -> bool:
    """True when tau is exactly ``table[receptor_index]`` for every edge."""
    if edges.n_edges == 0:
        return True
    if mechanism_tau_table is None or mechanism_tau_table.size == 0:
        return False
    if edges.tau_storage == "from_mechanism_table":
        return True
    ri = np.asarray(resolve_receptor_index(edges), dtype=np.int64)
    if ri.min() < 0 or ri.max() >= mechanism_tau_table.size:
        return False
    tau = np.asarray(resolve_edge_tau_ms(edges, np.float64), dtype=np.float64)
    expected = mechanism_tau_table[ri]
    return bool(np.array_equal(tau, expected))


def delay_steps_are_uniform_zero(edges: EdgeList) -> bool:
    if edges.delay_storage == "uniform_zero":
        return True
    if edges.n_edges == 0:
        return True
    ds = np.asarray(resolve_edge_delay_steps(edges), dtype=np.int32)
    return ds.size > 0 and bool(np.all(ds == 0))


def try_compact_receptor_index_storage(edges: EdgeList) -> EdgeList:
    """Narrow receptor_index to uint8 when lossless for the realized class range."""
    if edges.n_edges == 0 or edges.receptor_index_storage == "per_edge_uint8":
        return edges
    ri = np.asarray(edges.receptor_index)
    if ri.size == 0:
        return edges
    if ri.min() < 0 or ri.max() > 255:
        return edges
    narrowed = ri.astype(np.uint8)
    if not np.array_equal(narrowed.astype(np.int32), ri.astype(np.int32)):
        return edges
    return replace(
        edges,
        receptor_index=jnp.asarray(narrowed, dtype=jnp.uint8),
        receptor_index_storage="per_edge_uint8",
    )


def materialize_edge_list_arrays(
    edges: EdgeList,
    *,
    presynaptic_sign: np.ndarray | None = None,
) -> EdgeList:
    """Expand class-compacted fields to full per-edge arrays (``per_edge`` storage)."""
    if edges.n_edges == 0:
        return edges
    jdtype = edges.weight.dtype
    tau = resolve_edge_tau_ms(edges, jdtype)
    delay = resolve_edge_delay_steps(edges)
    ri = resolve_receptor_index(edges)
    weight = resolve_edge_weight(
        edges,
        jdtype,
        presynaptic_sign=(
            jnp.asarray(presynaptic_sign, dtype=jdtype)
            if presynaptic_sign is not None
            else None
        ),
    )
    return replace(
        edges,
        pre=edges.pre,
        post=edges.post,
        weight=weight,
        receptor_index=ri.astype(jnp.int32),
        tau_ms=tau,
        delay_steps=delay,
        tau_storage="per_edge",
        delay_storage="per_edge",
        uniform_delay_steps=0,
        receptor_index_storage="per_edge_int32",
        mechanism_tau_table=None,
        weight_storage="per_edge",
        weight_magnitude=None,
        mechanism_weight_magnitude_table=None,
    )


def try_compact_edge_list_class_storage(
    edges: EdgeList,
    *,
    declared_mechanism_tau_table: np.ndarray | None = None,
    presynaptic_sign: np.ndarray | None = None,
    declared_mechanism_weight_magnitude_table: np.ndarray | None = None,
    declared_scalar_weight_magnitude: float | None = None,
) -> EdgeList:
    """Apply safe class compaction without changing resolved kernel semantics."""
    if edges.n_edges == 0:
        return edges
    needs_resolve = (
        edges.tau_storage != "per_edge"
        or edges.delay_storage != "per_edge"
        or edges.weight_storage != "per_edge"
    )
    work = (
        materialize_edge_list_arrays(edges, presynaptic_sign=presynaptic_sign)
        if needs_resolve
        else edges
    )
    jdtype = work.weight.dtype
    kwargs: dict[str, Any] = {}

    if presynaptic_sign is not None and weight_matches_presynaptic_sign_magnitude(
        work,
        presynaptic_sign,
        magnitude=declared_scalar_weight_magnitude,
        mechanism_weight_magnitude_table=declared_mechanism_weight_magnitude_table,
    ):
        kwargs["weight_storage"] = "magnitude_times_presynaptic_sign"
        kwargs["weight"] = jnp.zeros((0,), dtype=jdtype)
        if declared_mechanism_weight_magnitude_table is not None:
            kwargs["mechanism_weight_magnitude_table"] = jnp.asarray(
                declared_mechanism_weight_magnitude_table, dtype=jdtype
            )
            kwargs["weight_magnitude"] = None
        else:
            mag = declared_scalar_weight_magnitude
            if mag is None:
                w = np.asarray(work.weight, dtype=np.float64)
                pre = np.asarray(work.pre, dtype=np.int64)
                sign_pre = np.asarray(presynaptic_sign, dtype=np.float64)[pre]
                ratios = w / np.where(sign_pre == 0, 1.0, sign_pre)
                mag = float(np.unique(np.abs(ratios))[0])
            kwargs["weight_magnitude"] = jnp.asarray(mag, dtype=jdtype)
            kwargs["mechanism_weight_magnitude_table"] = None
    elif weight_matches_sign_receptor_magnitude(
        work, declared_scalar_weight_magnitude, presynaptic_sign=presynaptic_sign
    ):
        w = np.asarray(work.weight, dtype=np.float64)
        mag = declared_scalar_weight_magnitude
        if mag is None:
            mag = float(np.unique(np.abs(w))[0])
        kwargs["weight_storage"] = "sign_from_receptor"
        kwargs["weight"] = jnp.zeros((0,), dtype=jdtype)
        kwargs["weight_magnitude"] = jnp.asarray(mag, dtype=jdtype)
        kwargs["mechanism_weight_magnitude_table"] = None

    if delay_steps_are_uniform_zero(work):
        kwargs["delay_storage"] = "uniform_zero"
        kwargs["delay_steps"] = jnp.zeros((0,), dtype=jnp.int32)
        kwargs["uniform_delay_steps"] = 0

    if tau_ms_matches_sign_receptor_table(work):
        kwargs["tau_storage"] = "sign_from_receptor"
        kwargs["tau_ms"] = jnp.zeros((0,), dtype=jdtype)
        kwargs["mechanism_tau_table"] = None
    elif tau_ms_matches_declared_mechanism_table(work, declared_mechanism_tau_table):
        kwargs["tau_storage"] = "from_mechanism_table"
        kwargs["tau_ms"] = jnp.zeros((0,), dtype=jdtype)
        kwargs["mechanism_tau_table"] = jnp.asarray(declared_mechanism_tau_table, dtype=jdtype)

    compacted = replace(work, **kwargs) if kwargs else work
    return try_compact_receptor_index_storage(compacted)
