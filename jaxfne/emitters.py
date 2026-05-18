"""Emitter kernels for :mod:`jaxfne`.

The current implementation is a small E/I/G-like Izhikevich scaffold.  It is a
reduced emitter: its native drive is **not** a physical current in amperes unless
an explicit calibration bridge is supplied later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ReceptorSpec:
    """Metadata declaration for a synaptic receptor. Not a biological kernel."""

    name: str
    receptor_index: int
    sign: int
    tau_ms: float
    reversal_mV: float | None
    source_calibration_status: str = "metadata_only_uncalibrated"
    claim_level: str = "computational_scaffold"


@dataclass(frozen=True)
class SynapseSpec:
    """Metadata declaration for a synapse. Not a biological kernel."""

    receptors: tuple[ReceptorSpec, ...]
    source_calibration_status: str = "metadata_only_uncalibrated"
    physical_amplitude_claim_allowed: bool = False


def standard_receptor_specs() -> dict[str, ReceptorSpec]:
    """Provide standard declarative receptor metadata. No biological claim."""
    return {
        "AMPA": ReceptorSpec(
            name="AMPA", receptor_index=0, sign=1, tau_ms=2.0, reversal_mV=0.0
        ),
        "GABA_A": ReceptorSpec(
            name="GABA_A", receptor_index=1, sign=-1, tau_ms=5.0, reversal_mV=-80.0
        ),
        "NMDA": ReceptorSpec(
            name="NMDA", receptor_index=2, sign=1, tau_ms=100.0, reversal_mV=0.0
        ),
        "GABA_B": ReceptorSpec(
            name="GABA_B", receptor_index=3, sign=-1, tau_ms=150.0, reversal_mV=-95.0
        ),
    }


@dataclass(frozen=True)
class IzhikevichParams:
    """Parameter container for a reduced Izhikevich population."""

    a: jax.Array
    b: jax.Array
    c: jax.Array
    d: jax.Array
    drive: jax.Array
    sign: jax.Array
    W: jax.Array
    v0: jax.Array
    u0: jax.Array
    source_scale: jax.Array
    labels: tuple[str, ...]
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"

    @property
    def n_neurons(self) -> int:
        return int(self.v0.shape[0])


@dataclass(frozen=True)
class EIGNetwork:
    """Lightweight description of an E/PV/SST/VIP-like reduced network."""

    params: IzhikevichParams
    positions: jax.Array
    metadata: dict

    @property
    def n_neurons(self) -> int:
        return self.params.n_neurons


def _dtype_from_policy(dtype: str) -> jnp.dtype:
    if dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
        return jnp.float64
    return jnp.float32


def _cell_labels(n: int, cell_type_fractions: Mapping[str, float]) -> tuple[str, ...]:
    labels: list[str] = []
    remaining = int(n)
    items = list(cell_type_fractions.items()) or [("E", 1.0)]
    for idx, (name, frac) in enumerate(items):
        count = int(round(n * float(frac))) if idx < len(items) - 1 else remaining
        count = max(0, min(count, remaining))
        labels.extend([str(name)] * count)
        remaining -= count
    labels = labels[:n] + ["E"] * max(0, n - len(labels))
    return tuple(labels[:n])


def _default_eig_connectivity(sign: jax.Array, dtype: jnp.dtype) -> jax.Array:
    n = sign.shape[0]
    pre_sign = sign[None, :]
    weights = jnp.ones((n, n), dtype=dtype) * pre_sign
    weights = weights * (1.0 - jnp.eye(n, dtype=dtype)) / jnp.sqrt(jnp.maximum(1, n))
    return 0.5 * weights


def izhikevich_eig_params(
    n: int,
    cell_type_fractions: Mapping[str, float],
    *,
    dtype: str = "float32",
) -> IzhikevichParams:
    """Create E/PV/SST/VIP-like Izhikevich parameters.

    Labels:
    - ``E``: regular-spiking-like excitatory emitter.
    - ``PV`` or ``Inl``: fast-spiking local inhibitory emitter.
    - ``SST`` or ``Ing``: low-threshold/dendrite-related inhibitory emitter.
    - ``VIP``: placeholder inhibitory/disinhibitory class.
    """

    jdtype = _dtype_from_policy(dtype)
    labels = _cell_labels(int(n), cell_type_fractions)
    a: list[float] = []
    b: list[float] = []
    c: list[float] = []
    d: list[float] = []
    drive: list[float] = []
    sign: list[float] = []

    for name in labels:
        if name == "E":
            a.append(0.02)
            b.append(0.20)
            c.append(-65.0)
            d.append(8.0)
            drive.append(5.0)
            sign.append(1.0)
        elif name in {"PV", "Inl"}:
            a.append(0.10)
            b.append(0.20)
            c.append(-65.0)
            d.append(2.0)
            drive.append(3.0)
            sign.append(-1.0)
        elif name in {"SST", "Ing"}:
            a.append(0.02)
            b.append(0.25)
            c.append(-65.0)
            d.append(2.0)
            drive.append(3.5)
            sign.append(-1.0)
        else:
            a.append(0.05)
            b.append(0.20)
            c.append(-65.0)
            d.append(4.0)
            drive.append(3.0)
            sign.append(-1.0)

    sign_array = jnp.asarray(sign, dtype=jdtype)
    return IzhikevichParams(
        a=jnp.asarray(a, dtype=jdtype),
        b=jnp.asarray(b, dtype=jdtype),
        c=jnp.asarray(c, dtype=jdtype),
        d=jnp.asarray(d, dtype=jdtype),
        drive=jnp.asarray(drive, dtype=jdtype),
        sign=sign_array,
        W=_default_eig_connectivity(sign_array, jdtype),
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.asarray(b, dtype=jdtype) * jnp.asarray(-65.0, dtype=jdtype),
        source_scale=jnp.asarray(1.0, dtype=jdtype),
        labels=labels,
    )


def make_eig_network(
    n: int,
    cell_type_fractions: Mapping[str, float] | None = None,
    *,
    dtype: str = "float32",
) -> EIGNetwork:
    """Build a minimal EIG network with laminar depth positions."""

    if cell_type_fractions is None:
        cell_type_fractions = {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}
    params = izhikevich_eig_params(n, cell_type_fractions, dtype=dtype)
    jdtype = _dtype_from_policy(dtype)
    depth = jnp.linspace(0.0, 1.0, int(n), dtype=jdtype)
    positions = jnp.stack([jnp.zeros(n, dtype=jdtype), jnp.zeros(n, dtype=jdtype), depth], axis=1)
    return EIGNetwork(
        params=params,
        positions=positions,
        metadata={
            "emitter_family": "izhikevich",
            "source_calibration_status": params.source_calibration_status,
            "position_units": "relative_laminar_depth_proxy",
        },
    )


def simulate_eig_izhikevich(
    params: IzhikevichParams,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Simulate a reduced EIG Izhikevich scaffold using ``jax.lax.scan``.

    When ``drive_schedule`` is None the existing scan path is preserved exactly.
    When provided, it must have shape ``(n_steps, n_neurons)`` and is added to
    ``params.drive`` at each timestep as native (uncalibrated) current.
    No physical-amplitude or calibration claim is introduced.
    """

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    weights = params.W.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        key,
    )

    if drive_schedule is None:
        def step(carry, _):
            v, u, prev_spikes, rng = carry
            rng, noise_key = jax.random.split(rng)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            syn = weights @ prev_spikes
            current_native = drive + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, rng), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=None, length=int(n_steps))
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, sched_t):
            v, u, prev_spikes, rng = carry
            rng, noise_key = jax.random.split(rng)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            syn = weights @ prev_spikes
            current_native = drive + sched_t + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, rng), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=sched)

    return voltages, spikes, sources


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EdgeList:
    """Sparse recurrent connectivity as a JAX pytree.

    Edges carry signed native weights and first-order synaptic decay constants.
    This is a computational backend for recurrent reduced emitters; weights are
    native/unphysical unless a future calibration bridge declares otherwise.
    """

    pre: jax.Array
    post: jax.Array
    weight: jax.Array
    receptor_index: jax.Array
    tau_ms: jax.Array
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    def tree_flatten(self):
        children = (self.pre, self.post, self.weight, self.receptor_index, self.tau_ms)
        aux = {"source_calibration_status": self.source_calibration_status}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        pre, post, weight, receptor_index, tau_ms = children
        return cls(pre, post, weight, receptor_index, tau_ms, aux["source_calibration_status"])

    def to_dict(self) -> dict:
        from .io import json_safe
        return json_safe({
            "backend": "edge_list_recurrent_v0.0.9",
            "n_edges": self.n_edges,
            "receptors": {"0": "excitatory_native", "1": "inhibitory_native"},
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_claim_allowed": False,
            "truth_mode": "truth_safe_unverified",
        })


def make_edge_list_from_dense(
    weights: jax.Array,
    *,
    threshold: float = 1e-12,
    dtype: str = "float32",
) -> EdgeList:
    """Convert a dense recurrent weight matrix into a sparse EdgeList.

    The dense matrix uses rows as postsynaptic targets and columns as
    presynaptic sources, matching ``weights @ spikes`` in the baseline backend.
    """

    jdtype = _dtype_from_policy(dtype)
    W = jnp.asarray(weights, dtype=jdtype)
    post, pre = jnp.nonzero(jnp.abs(W) > jnp.asarray(threshold, dtype=jdtype))
    signed_weight = W[post, pre].astype(jdtype)
    receptor_index = (signed_weight < 0).astype(jnp.int32)
    tau_exc = jnp.asarray(2.0, dtype=jdtype)
    tau_inh = jnp.asarray(5.0, dtype=jdtype)
    tau_ms = jnp.where(receptor_index == 0, tau_exc, tau_inh).astype(jdtype)
    return EdgeList(
        pre=pre.astype(jnp.int32),
        post=post.astype(jnp.int32),
        weight=signed_weight,
        receptor_index=receptor_index,
        tau_ms=tau_ms,
    )


def simulate_edge_recurrent_izhikevich(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simulate reduced Izhikevich emitters with sparse recurrent synapses.

    The implementation uses ``jax.lax.scan`` over time and
    ``jax.ops.segment_sum`` over edges. It is JIT/vmap compatible and preserves
    the uncalibrated proxy-source truth status.

    When ``drive_schedule`` is None the existing scan path is preserved exactly.
    When provided, it must have shape ``(n_steps, n_neurons)`` and is added as
    native uncalibrated current at each timestep.
    """

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    n_neurons = params.v0.shape[0]

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        jnp.zeros((edges.n_edges,), dtype=jdtype),
        key,
    )

    if drive_schedule is None:
        def step(carry, _):
            v, u, prev_spikes, syn_state, rng = carry
            rng, noise_key = jax.random.split(rng)
            edge_current = weight * syn_state
            syn = jax.ops.segment_sum(edge_current, post, n_neurons)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            current_native = drive + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next, rng), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=None, length=int(n_steps))
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, sched_t):
            v, u, prev_spikes, syn_state, rng = carry
            rng, noise_key = jax.random.split(rng)
            edge_current = weight * syn_state
            syn = jax.ops.segment_sum(edge_current, post, n_neurons)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            current_native = drive + sched_t + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next, rng), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=sched)

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
    }
    return voltages, spikes, sources, final_state

def standard_receptor_tau_table(dtype: str = "float32") -> jax.Array:
    """Return the receptor_index → tau_ms lookup table used by v0.0.11.

    The table is built from :func:`standard_receptor_specs` so the kernel and
    the declarative receptor metadata cannot drift apart. It carries no
    biological-calibration claim: the entries are native time constants for a
    reduced exponential synaptic state, not patch-clamp-derived kinetics.
    """

    jdtype = _dtype_from_policy(dtype)
    specs = standard_receptor_specs()
    by_index = {spec.receptor_index: spec for spec in specs.values()}
    n = max(by_index) + 1 if by_index else 0
    return jnp.asarray(
        [float(by_index[i].tau_ms) for i in range(n)],
        dtype=jdtype,
    )


def _edge_tau_from_receptor_index(
    receptor_index: jax.Array, dtype: str = "float32"
) -> jax.Array:
    """Map ``edges.receptor_index`` to the v0.0.11 standard tau table."""

    jdtype = _dtype_from_policy(dtype)
    table = standard_receptor_tau_table(dtype=dtype)
    idx = jnp.clip(receptor_index.astype(jnp.int32), 0, table.shape[0] - 1)
    return jnp.take(table, idx).astype(jdtype)


def simulate_receptor_exponential_izhikevich(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """v0.0.11 receptor-indexed exponential recurrent kernel.

    The kernel keeps one scalar synaptic state per edge (``syn_state.shape ==
    (n_edges,)``) and selects the exponential decay per edge from
    ``edges.receptor_index`` via :func:`standard_receptor_tau_table`. Two
    different receptor channels on the same anatomical connection are
    represented as two separate edges with identical ``pre``/``post`` but
    different ``receptor_index``; the kernel does not expand state to
    ``(n_edges, n_receptors)``.

    The aggregation rule ``segment_sum(weight * syn_state, post, n_neurons)``
    guarantees each edge contributes exactly once to its postsynaptic native
    recurrent input. Receptor reversal potentials are metadata-only and are
    not used in the current computation; weights remain native/unphysical and
    no conductance equation ``g * (V - E_rev)`` is computed.

    When ``drive_schedule`` is None the existing scan path is preserved exactly.
    When provided, it must have shape ``(n_steps, n_neurons)`` and is added as
    native uncalibrated current at each timestep.
    """

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_per_edge = jnp.maximum(
        _edge_tau_from_receptor_index(edges.receptor_index, dtype=dtype),
        jnp.asarray(1e-6, dtype=jdtype),
    )
    decay = jnp.exp(-dt / tau_per_edge)
    n_neurons = params.v0.shape[0]

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        jnp.zeros((edges.n_edges,), dtype=jdtype),
        key,
    )

    if drive_schedule is None:
        def step(carry, _):
            v, u, prev_spikes, syn_state, rng = carry
            rng, noise_key = jax.random.split(rng)
            edge_drive = weight * syn_state
            syn = jax.ops.segment_sum(edge_drive, post, n_neurons)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            current_native = drive + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next, rng), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=None, length=int(n_steps))
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, sched_t):
            v, u, prev_spikes, syn_state, rng = carry
            rng, noise_key = jax.random.split(rng)
            edge_drive = weight * syn_state
            syn = jax.ops.segment_sum(edge_drive, post, n_neurons)
            noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=v.shape).astype(jdtype)
            current_native = drive + sched_t + syn + noise
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            spikes_bool = v_next >= 30.0
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next, rng), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=sched)

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "tau_per_edge": tau_per_edge,
    }
    return voltages, spikes, sources, final_state


# Backwards-compatible name from v0.0.3.
simulate_izhikevich_eig = simulate_eig_izhikevich
