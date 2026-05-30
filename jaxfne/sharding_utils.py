"""Distributed sharding mesh stubs for jaxfne v0.3.18.

Provides trace-safe, single-axis logical mesh construction for candidate-population
parallelism across JAX devices. Falls back to no-op on single-device (CPU/single-GPU)
environments so callers need no branching logic.

Scope
-----
- ``truth_safe_unverified``: laminar proxy output only, no PDE field solver.
- ``physical_amplitude_claim_allowed: false``
- All sharding specs here are *stubs*: they set up the topology but do not yet
  drive any real multi-device dispatch in the AGSDR loop.  Full integration is
  planned for v0.3.20+.

Public API
----------
make_population_mesh          -- build a 1-D named Mesh across all visible devices
make_candidate_sharding       -- PartitionSpec for candidate-batch axis
make_replicated_sharding      -- PartitionSpec for replicated (model-param) arrays
get_sharding_context          -- convenience bundle: (mesh, cand_spec, rep_spec) or None
"""
from __future__ import annotations

from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_POPULATION_AXIS: str = "population_sweep"
"""Logical mesh axis name for the candidate-population sharding dimension."""


# ---------------------------------------------------------------------------
# Core mesh construction
# ---------------------------------------------------------------------------


def make_population_mesh() -> Optional[Mesh]:
    """Return a 1-D named :class:`jax.sharding.Mesh` across all visible JAX devices.

    On single-device environments (CPU, single GPU) returns ``None``; callers
    should treat ``None`` as "no sharding, run locally".

    Returns
    -------
    Mesh or None
        A ``Mesh`` with axis name ``"population_sweep"`` if ``len(jax.devices()) > 1``,
        otherwise ``None``.

    Examples
    --------
    >>> mesh = make_population_mesh()
    >>> if mesh is not None:
    ...     with mesh:
    ...         pass  # distribute work here
    """
    devices = jax.devices()
    if len(devices) <= 1:
        return None
    device_array = np.array(devices).reshape(-1)
    return Mesh(device_array, axis_names=(_POPULATION_AXIS,))


# ---------------------------------------------------------------------------
# PartitionSpec factories
# ---------------------------------------------------------------------------


def make_candidate_sharding(mesh: Mesh) -> NamedSharding:
    """Return a :class:`jax.sharding.NamedSharding` that slices the first
    (batch/population) axis across ``mesh``.

    Shape convention: ``(B, n_params)`` — partition on axis 0, replicate axis 1.

    Parameters
    ----------
    mesh : Mesh
        A mesh previously created by :func:`make_population_mesh`.

    Returns
    -------
    NamedSharding
        Candidate array shard: ``PartitionSpec('population_sweep', None)``.
    """
    return NamedSharding(mesh, PartitionSpec(_POPULATION_AXIS, None))


def make_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return a :class:`jax.sharding.NamedSharding` that fully replicates an array
    across all devices in ``mesh``.

    Use for model-parameter tensors (weight matrices, emitter params) that must
    not be partitioned to avoid cross-device gradient communication overhead.

    Parameters
    ----------
    mesh : Mesh
        A mesh previously created by :func:`make_population_mesh`.

    Returns
    -------
    NamedSharding
        Replicated shard: ``PartitionSpec(None)``.
    """
    return NamedSharding(mesh, PartitionSpec(None))


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------


def get_sharding_context() -> Optional[dict]:
    """Return a dict with ``mesh``, ``candidate``, and ``replicated`` sharding specs.

    Returns ``None`` on single-device environments so downstream code can
    gate sharding logic with a simple ``if ctx is not None`` check.

    Returns
    -------
    dict or None
        ::

            {
                "mesh":       Mesh,          # the logical device mesh
                "candidate":  NamedSharding, # slice batch dim across devices
                "replicated": NamedSharding, # replicate on every device
            }

        or ``None`` if only one device is available.

    Examples
    --------
    >>> ctx = get_sharding_context()
    >>> if ctx is not None:
    ...     with ctx["mesh"]:
    ...         candidate_arr = jax.device_put(
    ...             candidate_arr, ctx["candidate"]
    ...         )
    """
    mesh = make_population_mesh()
    if mesh is None:
        return None
    return {
        "mesh": mesh,
        "candidate": make_candidate_sharding(mesh),
        "replicated": make_replicated_sharding(mesh),
    }
