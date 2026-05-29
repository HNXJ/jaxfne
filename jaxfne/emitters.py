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
    """Parameter container for a reduced Izhikevich population.

    Fields:
    - labels: tuple of cell-type labels (E, PV, SST, VIP)
    - layer_labels: optional tuple of layer names (L1, L2/3, L4, L5, L6, etc)
    """

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
    layer_labels: tuple[str, ...] | None = None
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"

    @property
    def n_neurons(self) -> int:
        return int(self.v0.shape[0])


def _izhikevich_params_flatten(params):
    children = (
        params.a,
        params.b,
        params.c,
        params.d,
        params.drive,
        params.sign,
        params.W,
        params.v0,
        params.u0,
        params.source_scale,
    )
    aux_data = {
        "labels": params.labels,
        "layer_labels": params.layer_labels,
        "source_calibration_status": params.source_calibration_status,
    }
    return children, aux_data


def _izhikevich_params_unflatten(aux_data, children):
    return IzhikevichParams(
        a=children[0],
        b=children[1],
        c=children[2],
        d=children[3],
        drive=children[4],
        sign=children[5],
        W=children[6],
        v0=children[7],
        u0=children[8],
        source_scale=children[9],
        labels=aux_data["labels"],
        layer_labels=aux_data["layer_labels"],
        source_calibration_status=aux_data["source_calibration_status"],
    )


try:
    try:
        jax.tree_util.register_pytree_node(
            IzhikevichParams,
            _izhikevich_params_flatten,
            _izhikevich_params_unflatten,
        )
    except ValueError:
        pass

except ValueError:
    pass # Already registered


def _segment_sum(data, segment_ids, num_segments):
    """Compatibility wrapper for segment_sum across JAX versions."""
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)


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
    - ``VIP``: VIP-like inhibitory/disinhibitory emitter class.
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
        else:  # VIP / IS profile
            a.append(0.02)
            b.append(-0.10)  # Corrected from +0.20 to IS/chattering profile
            c.append(-55.0)
            d.append(6.0)
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



def izhikevich_params_from_labels(
    labels: tuple[str, ...] | list[str],
    *,
    layer_labels: tuple[str, ...] | list[str] | None = None,
    dtype: str = "float32",
    drive_overrides: Mapping[str, float] | None = None,
    source_scale: float = 1.0,
) -> IzhikevichParams:
    """Create reduced Izhikevich parameters from explicit cell labels.

    This is the package-native path used by Suite No. 2 when a notebook needs
    deterministic E/PV/SST/VIP populations without local simulator code.  The
    returned native drive values are reduced-model drive units.  They are suited
    to relative proxy readouts unless a caller supplies an external calibration
    bridge.
    """

    label_tuple = tuple(str(x) for x in labels)
    if not label_tuple:
        raise ValueError("labels must contain at least one emitter label")
    if layer_labels is not None and len(layer_labels) != len(label_tuple):
        raise ValueError("layer_labels length must match labels length")

    overrides = {str(k): float(v) for k, v in (drive_overrides or {}).items()}
    jdtype = _dtype_from_policy(dtype)
    a: list[float] = []
    b: list[float] = []
    c: list[float] = []
    d: list[float] = []
    drive: list[float] = []
    sign: list[float] = []

    for name in label_tuple:
        if name == "E":
            aa, bb, cc, dd, drv, sg = 0.02, 0.20, -65.0, 8.0, 5.0, 1.0
        elif name in {"PV", "Inl"}:
            aa, bb, cc, dd, drv, sg = 0.10, 0.20, -65.0, 2.0, 3.0, -1.0
        elif name in {"SST", "Ing"}:
            aa, bb, cc, dd, drv, sg = 0.02, 0.25, -65.0, 2.0, 3.5, -1.0
        elif name == "VIP":
            aa, bb, cc, dd, drv, sg = 0.02, -0.10, -55.0, 6.0, 3.0, -1.0
        else:
            raise ValueError(f"unknown Suite No. 2 cell type label: {name!r}")
        a.append(aa)
        b.append(bb)
        c.append(cc)
        d.append(dd)
        drive.append(overrides.get(name, drv))
        sign.append(sg)

    n = len(label_tuple)
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
        source_scale=jnp.asarray(source_scale, dtype=jdtype),
        labels=label_tuple,
        layer_labels=tuple(str(x) for x in layer_labels) if layer_labels is not None else None,
        source_calibration_status="uncalibrated_izhikevich_native_current",
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
    silence_mask: "jax.Array | None" = None,
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

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype)

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
    )

    if drive_schedule is None:
        def step(carry, noise_t):
            v, u, prev_spikes = carry
            syn = weights @ prev_spikes
            current_native = drive + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes = carry
            syn = weights @ prev_spikes
            current_native = drive + sched_t + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

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
    silence_mask: "jax.Array | None" = None,
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

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype)

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        jnp.zeros((edges.n_edges,), dtype=jdtype),
    )

    if drive_schedule is None:
        def step(carry, noise_t):
            v, u, prev_spikes, syn_state = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

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
    silence_mask: "jax.Array | None" = None,
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

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype)

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        jnp.zeros((edges.n_edges,), dtype=jdtype),
    )

    if drive_schedule is None:
        def step(carry, noise_t):
            v, u, prev_spikes, syn_state = carry
            edge_drive = weight * syn_state
            syn = _segment_sum(edge_drive, post, n_neurons)
            current_native = drive + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state = carry
            edge_drive = weight * syn_state
            syn = _segment_sum(edge_drive, post, n_neurons)
            current_native = drive + sched_t + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
            du = a * (b * v - u)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
            return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "tau_per_edge": tau_per_edge,
    }
    return voltages, spikes, sources, final_state


def simulate_dynamic_ei_coupling(
    params: IzhikevichParams,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    g_ei: float = 5.0,
    g_ie: float = 3.0,
    tau_syn_e_ms: float = 5.0,
    tau_syn_i_ms: float = 10.0,
    dtype: str = "float32",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Simulate two-neuron E/I with dynamic synaptic coupling via lax.scan.

    Uses first-order exponential synaptic traces with per-neuron decay.
    syn_traces is included in the carry tuple so it evolves across timesteps.

    Args:
        params: IzhikevichParams for the network (expects n_neurons=2,
            neurons[0]=E, neurons[1]=I).
        n_steps: Number of simulation timesteps.
        dt_ms: Timestep in milliseconds.
        key: JAX PRNG key.
        g_ei: E→I coupling conductance (excitatory, model units).
        g_ie: I→E coupling conductance (inhibitory, magnitude, model units).
        tau_syn_e_ms: Excitatory synaptic time constant (ms).
        tau_syn_i_ms: Inhibitory synaptic time constant (ms).
        dtype: Float dtype policy.

    Returns:
        Tuple of (voltages, spikes, syn_currents, sources), each shape
        (n_steps, n_neurons). syn_currents is the dynamic synaptic current
        injected into each neuron at each timestep.

    Note: source_calibration_status = uncalibrated_izhikevich_native_current.
    No physical amplitude claim is made.
    """
    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)

    # Per-synapse exponential decay constants
    # syn_traces[0] = E neuron trace (used to compute E→I current)
    # syn_traces[1] = I neuron trace (used to compute I→E current)
    tau_syn = jnp.asarray([tau_syn_e_ms, tau_syn_i_ms], dtype=jdtype)
    decay = jnp.exp(-dt / jnp.maximum(tau_syn, jnp.asarray(1e-6, dtype=jdtype)))

    # Coupling gain vector: syn_traces @ gain_matrix gives per-neuron syn current
    # E→I: g_ei * syn_traces[0] injected into neuron 1 (I)
    # I→E: -g_ie * syn_traces[1] injected into neuron 0 (E)
    g_ei_scalar = jnp.asarray(g_ei, dtype=jdtype)
    g_ie_scalar = jnp.asarray(g_ie, dtype=jdtype)

    syn_traces_init = jnp.zeros(2, dtype=jdtype)

    init = (
        params.v0.astype(jdtype),
        params.u0.astype(jdtype),
        jnp.zeros_like(params.v0, dtype=jdtype),
        syn_traces_init,  # syn_traces in carry
        key,
    )

    def step(carry, _):
        v, u, prev_spikes, syn_traces, rng = carry
        rng, noise_key = jax.random.split(rng)
        noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(
            noise_key, shape=v.shape
        ).astype(jdtype)

        # Dynamic synaptic current from traces
        # E→I: positive current into neuron 1
        # I→E: negative current into neuron 0
        syn_current_ei = g_ei_scalar * syn_traces[0]   # excitatory to I
        syn_current_ie = -g_ie_scalar * syn_traces[1]  # inhibitory to E
        syn_currents = jnp.asarray([syn_current_ie, syn_current_ei], dtype=jdtype)

        current_native = drive + syn_currents + noise
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
        du = a * (b * v - u)
        v_next = v + dt * dv
        u_next = u + dt * du
        spikes_bool = v_next >= 30.0
        spikes = spikes_bool.astype(jdtype)
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)

        # Update synaptic traces (exponential decay + spike injection)
        syn_traces_next = syn_traces * decay + spikes

        source_proxy = source_scale * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
        return (v_reset, u_reset, spikes, syn_traces_next, rng), (
            v_reset, spikes, syn_currents, source_proxy
        )

    _, (voltages, spikes, syn_currents, sources) = jax.lax.scan(
        step, init, xs=None, length=int(n_steps)
    )
    return voltages, spikes, syn_currents, sources


# Backwards-compatible name from v0.0.3.
simulate_izhikevich_eig = simulate_eig_izhikevich


# =============================================================================
# Patch C: Multi-Area Emitter Runtime
# =============================================================================

def simulate_multi_area_izhikevich(
    neurons_df: "Mapping[str, any]",
    positions_m: "jax.Array",
    W: "jax.Array",
    source_tensor: "jax.Array | None" = None,
    control_params: "Mapping[str, float] | None" = None,
    cfg: "any" = None,
    n_steps: "int | None" = None,
    dt_ms: float = 0.1,
    seed: int = 0,
    dtype: str = "float32",
) -> tuple["jax.Array", "jax.Array"]:
    """Simulate multi-area Izhikevich network with laminar connectivity.

    Parameters
    ----------
    neurons_df : Mapping[str, Any]
        Neuron dataframe with keys: area, layer, cell_type, and positional data.
    positions_m : jax.Array
        Neuron positions [N, 3].
    W : jax.Array
        Connectivity matrix [N, N].
    source_tensor : jax.Array, optional
        Driving source tensor [T, N]. If provided, used as drive_schedule.
    control_params : Mapping[str, float], optional
        Control parameters including noise_scale (default 1.0).
    cfg : Any, optional
        Configuration object (used for metadata only).
    n_steps : int, optional
        Number of simulation steps. If None, inferred from source_tensor or cfg.
    dt_ms : float
        Time step in milliseconds (default 0.1).
    seed : int
        PRNG seed (default 0).
    dtype : str
        Data type (default "float32").

    Returns
    -------
    spikes : jax.Array
        Spike raster [T, N].
    voltages : jax.Array
        Membrane potentials [T, N].
    """
    import jax
    import jax.numpy as jnp

    if control_params is None:
        control_params = {"noise_scale": 1.0}

    n = len(neurons_df.get("area", [""]))
    if n == 0:
        raise ValueError("neurons_df is empty")

    # Infer number of steps
    if n_steps is None:
        if source_tensor is not None:
            n_steps = source_tensor.shape[0]
        elif cfg is not None and hasattr(cfg, "metadata") and "duration_ms" in cfg.metadata:
            duration_ms = cfg.metadata["duration_ms"]
            n_steps = int(duration_ms / dt_ms)
        else:
            n_steps = 1000  # Default fallback

    # Create Izhikevich parameters from neuron metadata
    cell_types = neurons_df.get("cell_type", ["E"] * n)
    cell_type_fractions = {}
    for ct in set(cell_types):
        count = sum(1 for c in cell_types if c == ct)
        cell_type_fractions[ct] = count / max(1, n)

    params = izhikevich_eig_params(
        n=n,
        cell_type_fractions=cell_type_fractions,
        dtype=dtype,
    )

    # Rescale connectivity matrix to be compatible with emitter gains
    W_compat = jnp.asarray(W, dtype=_dtype_from_policy(dtype)) * 0.1

    # Update params with custom connectivity
    params = IzhikevichParams(
        a=params.a,
        b=params.b,
        c=params.c,
        d=params.d,
        drive=params.drive * control_params.get("drive_scale", 1.0),
        sign=params.sign,
        W=W_compat,
        v0=params.v0,
        u0=params.u0,
        source_scale=params.source_scale,
        labels=params.labels,
        layer_labels=tuple(neurons_df.get("layer", ["unknown"] * n)),
    )

    # Create PRNG key
    key = jax.random.PRNGKey(seed)

    # Simulate with optional source drive schedule
    voltages, spikes, _ = simulate_eig_izhikevich(
        params,
        n_steps=n_steps,
        dt_ms=dt_ms,
        key=key,
        dtype=dtype,
        drive_schedule=source_tensor,
    )

    return spikes, voltages

# -----------------------------------------------------------------------------
# Generalized emitter facade classes used by tutorials and smoke tests.
# -----------------------------------------------------------------------------
from typing import NamedTuple as _NamedTuple


class EmitterState(_NamedTuple):
    v: jax.Array
    u: jax.Array
    spikes: jax.Array
    key: jax.Array
    step_count: jax.Array


class EmitterOutput(_NamedTuple):
    voltage: jax.Array
    spikes: jax.Array
    source: jax.Array
    finite: jax.Array

    @property
    def dtype(self) -> str:
        return str(self.voltage.dtype)


class Emitter:
    """Base class for package-native emitter facades."""

    def initial_state(self, seed: int = 0) -> EmitterState:
        raise NotImplementedError("TODO: implement Emitter.initial_state in a concrete emitter")

    def step(self, state: EmitterState, input_t: jax.Array, *, dt_ms: float = 0.1) -> tuple[EmitterState, EmitterOutput]:
        raise NotImplementedError("TODO: implement Emitter.step in a concrete emitter")


class IzhikevichEmitter(Emitter):
    """Reduced Izhikevich emitter facade with JAX-native step."""

    def __init__(self, n: int | None = None, *, n_neurons: int | None = None, dtype: str = "float32", cell_type_fractions: Mapping[str, float] | None = None):
        self.n = int(n if n is not None else (n_neurons if n_neurons is not None else 1))
        if self.n <= 0:
            raise ValueError("n must be positive")
        self.dtype = dtype
        self.params = izhikevich_eig_params(self.n, cell_type_fractions or {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07}, dtype=dtype)

    def initial_state(self, seed: int = 0) -> EmitterState:
        jdtype = _dtype_from_policy(self.dtype)
        return EmitterState(
            v=self.params.v0.astype(jdtype),
            u=self.params.u0.astype(jdtype),
            spikes=jnp.zeros((self.n,), dtype=jdtype),
            key=jax.random.PRNGKey(int(seed)),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(self, state: EmitterState, input_t: jax.Array, *, dt_ms: float = 0.1) -> tuple[EmitterState, EmitterOutput]:
        jdtype = _dtype_from_policy(self.dtype)
        rng, noise_key = jax.random.split(state.key)
        input_t = jnp.asarray(input_t, dtype=jdtype)
        noise = jnp.asarray(0.5, dtype=jdtype) * jax.random.normal(noise_key, shape=state.v.shape).astype(jdtype)
        syn = self.params.W.astype(jdtype) @ state.spikes.astype(jdtype)
        current_native = self.params.drive.astype(jdtype) + input_t + syn + noise
        dt = jnp.asarray(dt_ms, dtype=jdtype)
        dv = 0.04 * state.v * state.v + 5.0 * state.v + 140.0 - state.u + current_native
        du = self.params.a.astype(jdtype) * (self.params.b.astype(jdtype) * state.v - state.u)
        v_next = state.v + dt * dv
        u_next = state.u + dt * du
        spikes_bool = v_next >= 30.0
        spikes = spikes_bool.astype(jdtype)
        v_reset = jnp.where(spikes_bool, self.params.c.astype(jdtype), v_next)
        u_reset = jnp.where(spikes_bool, u_next + self.params.d.astype(jdtype), u_next)
        source = self.params.source_scale.astype(jdtype) * (current_native + jnp.asarray(20.0, dtype=jdtype) * spikes)
        next_state = EmitterState(v=v_reset, u=u_reset, spikes=spikes, key=rng, step_count=state.step_count + 1)
        output = EmitterOutput(
            voltage=v_reset,
            spikes=spikes,
            source=source,
            finite=jnp.all(jnp.isfinite(v_reset)) & jnp.all(jnp.isfinite(source)),
        )
        return next_state, output


class GLIFEmitter(Emitter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement GLIFEmitter dynamics before exposing this emitter")


class LIFEmitter(Emitter):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO: implement LIFEmitter dynamics before exposing this emitter")


class SynapseState(_NamedTuple):
    trace: jax.Array


class SynapseLayer:
    """Exponential synapse layer returning recurrent input currents."""

    def __init__(self, n: int, W: jax.Array, tau_ms: float = 5.0, dtype: str = "float32"):
        self.n = int(n)
        self.W = jnp.asarray(W, dtype=_dtype_from_policy(dtype))
        if self.W.shape != (self.n, self.n):
            raise ValueError(f"W must have shape {(self.n, self.n)}, got {self.W.shape}")
        self.tau_ms = float(tau_ms)
        self.dtype = dtype

    def initial_state(self) -> SynapseState:
        return SynapseState(trace=jnp.zeros((self.n,), dtype=_dtype_from_policy(self.dtype)))

    def step(self, state: SynapseState, pre_spikes: jax.Array, *, dt_ms: float = 0.1) -> tuple[SynapseState, jax.Array]:
        jdtype = _dtype_from_policy(self.dtype)
        decay = jnp.exp(-jnp.asarray(dt_ms, dtype=jdtype) / jnp.asarray(self.tau_ms, dtype=jdtype))
        trace_next = state.trace.astype(jdtype) * decay + jnp.asarray(pre_spikes, dtype=jdtype)
        current = self.W.astype(jdtype) @ trace_next
        return SynapseState(trace=trace_next), current
