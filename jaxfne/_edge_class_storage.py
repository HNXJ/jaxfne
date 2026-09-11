"""Class-shared edge attribute storage (23-REP-01).

High-R_k edge fields (``tau_ms``, uniform ``delay_steps``) may be stored as
execution layouts derived from class tables rather than per-edge materialization.
Realized connectivity (``pre``, ``post``, ``weight``, ``receptor_index``) stays
authoritative per edge.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from .emitters import (
    EdgeList,
    resolve_edge_delay_steps,
    resolve_edge_tau_ms,
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


def audit_edge_list_storage(edges: EdgeList) -> dict[str, Any]:
    """Summarize per-edge redundancy and active compaction modes."""
    fields = {
        "pre": edges.pre,
        "post": edges.post,
        "weight": edges.weight,
        "receptor_index": edges.receptor_index,
        "tau_ms": edges.tau_ms,
        "delay_steps": edges.delay_steps,
    }
    redundancy = {name: attribute_cardinality(arr) for name, arr in fields.items()}
    return {
        "n_edges": int(edges.n_edges),
        "tau_storage": edges.tau_storage,
        "delay_storage": edges.delay_storage,
        "uniform_delay_steps": int(edges.uniform_delay_steps),
        "redundancy": redundancy,
        "tau_sign_table_compatible": tau_ms_matches_sign_receptor_table(edges),
        "delay_uniform_zero": delay_steps_are_uniform_zero(edges),
    }


def tau_ms_matches_sign_receptor_table(edges: EdgeList) -> bool:
    """True when per-edge tau equals the sign-only 2.0/5.0 receptor map."""
    if edges.n_edges == 0:
        return True
    if edges.tau_storage == "sign_from_receptor":
        return True
    ri = np.asarray(edges.receptor_index, dtype=np.int32)
    tau = np.asarray(edges.tau_ms, dtype=np.float64)
    expected = np.where(ri == 0, sign_only_tau_exc_ms(), sign_only_tau_inh_ms())
    return bool(np.array_equal(tau, expected))


def delay_steps_are_uniform_zero(edges: EdgeList) -> bool:
    if edges.delay_storage == "uniform_zero":
        return True
    if edges.n_edges == 0:
        return True
    ds = np.asarray(edges.delay_steps, dtype=np.int32)
    return ds.size > 0 and bool(np.all(ds == 0))


def materialize_edge_list_arrays(edges: EdgeList) -> EdgeList:
    """Expand class-compacted fields to full per-edge arrays (``per_edge`` storage)."""
    if edges.n_edges == 0:
        return edges
    jdtype = edges.weight.dtype
    tau = resolve_edge_tau_ms(edges, jdtype)
    delay = resolve_edge_delay_steps(edges)
    return replace(
        edges,
        tau_ms=tau,
        delay_steps=delay,
        tau_storage="per_edge",
        delay_storage="per_edge",
        uniform_delay_steps=0,
    )


def try_compact_edge_list_class_storage(edges: EdgeList) -> EdgeList:
    """Apply safe class compaction without changing resolved kernel semantics."""
    if edges.n_edges == 0:
        return edges
    jdtype = edges.tau_ms.dtype if edges.tau_ms.size else edges.weight.dtype
    kwargs: dict[str, Any] = {}
    if tau_ms_matches_sign_receptor_table(edges):
        kwargs["tau_storage"] = "sign_from_receptor"
        kwargs["tau_ms"] = jnp.zeros((0,), dtype=jdtype)
    if delay_steps_are_uniform_zero(edges):
        kwargs["delay_storage"] = "uniform_zero"
        kwargs["delay_steps"] = jnp.zeros((0,), dtype=jnp.int32)
        kwargs["uniform_delay_steps"] = 0
    if not kwargs:
        return edges
    return replace(edges, **kwargs)
