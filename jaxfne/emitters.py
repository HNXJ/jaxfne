"""Emitter kernels for :mod:`jaxfne`.

Docs: ``docs/api/emitters.md`` (https://jaxfne.readthedocs.io/en/latest/api/emitters/) —
update that page when this module's public API changes.

The current implementation is a small E/I/G-like Izhikevich scaffold.  It is a
reduced emitter: its native drive is **not** a physical current in amperes unless
an explicit calibration bridge is supplied later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._hdp_adaptive import (
    bind_theta_to_plant,
    expected_h_shape as _expected_h_shape,
    initial_theta_vector,
    parse_population_restoring_layout,
    population_rate_error,
    population_restoring_derivatives,
    resolve_h_state_locality,
    theta_bounds,
)
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN


def _source_proxy_from_components(
    current_native: jax.Array,
    spikes: jax.Array,
    source_scale: jax.Array,
    *,
    dtype: Any,
) -> jax.Array:
    """Compose the canonical relative source from its declared components.

    ``current_native`` already contains the drive, recurrent synaptic, and
    stochastic-current terms for the active emitter path.  The spike impulse is
    added exactly once at this boundary, using the canonical gain owned by
    :mod:`jaxfne.presets`.
    """
    spike_gain = jnp.asarray(DEFAULT_SPIKE_IMPULSE_GAIN, dtype=dtype)
    return source_scale * (current_native + spike_gain * spikes)


# Canonical Izhikevich cell-type parameter defaults.
# Single source of truth for E/PV-like/SST-like/VIP-like reduced-model parameters.
# Labels "E", "PV", "SST", "VIP" denote reduced emitter classes (E-like, PV-like,
# SST-like, VIP-like) — phenomenological Izhikevich presets that do not warrant
# literal transcriptomic/morphological identity.
IZHIKEVICH_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "E":   {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0,  "drive": 5.0, "sign":  1.0},
    "PV":  {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0,  "drive": 3.0, "sign": -1.0},
    "Inl": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0,  "drive": 3.0, "sign": -1.0},
    "SST": {"a": 0.05, "b": 0.25, "c": -65.0, "d": 2.0,  "drive": 3.5, "sign": -1.0},
    "Ing": {"a": 0.05, "b": 0.25, "c": -65.0, "d": 2.0,  "drive": 3.5, "sign": -1.0},
    "VIP": {"a": 0.02, "b": -0.10, "c": -55.0, "d": 6.0, "drive": 3.0, "sign": -1.0},
    # "I" (generic Inhibitory) is a widely-used two-population E/I shorthand
    # across examples/, scripts/evidence_figures/, and tests/ -- explicit alias
    # to VIP's parameters (2026-07-13: this matches the exact values every
    # real "I" caller was already getting via a since-removed silent VIP
    # fallback in _get_cell_type_params; made explicit, not changed).
    "I":   {"a": 0.02, "b": -0.10, "c": -55.0, "d": 6.0, "drive": 3.0, "sign": -1.0},
}


def _get_cell_type_params(name: str) -> dict[str, float]:
    """Look up canonical Izhikevich parameters for a cell type.

    Raises ValueError on an unrecognized name rather than silently falling
    back to VIP parameters -- a typo'd cell-type key (e.g. "pv" instead of
    "PV") previously simulated silently as VIP with no warning.
    """
    if name in IZHIKEVICH_CELL_TYPE_DEFAULTS:
        return IZHIKEVICH_CELL_TYPE_DEFAULTS[name]
    raise ValueError(
        f"unknown Izhikevich cell type {name!r}; expected one of "
        f"{sorted(IZHIKEVICH_CELL_TYPE_DEFAULTS)}"
    )



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
    physical_amplitude_calibrated: bool = False


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
    - labels: tuple of reduced-class labels (E-like, PV-like, SST-like,
      VIP-like — stored as "E", "PV", "SST", "VIP"; no literal identity claim)
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
        """Documented public function `n_neurons`."""
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
    jax.tree_util.register_pytree_node(
        IzhikevichParams,
        _izhikevich_params_flatten,
        _izhikevich_params_unflatten,
    )
except ValueError:
    pass  # Already registered (re-import / notebook reload)


def _segment_sum(data, segment_ids, num_segments):
    """Compatibility wrapper for segment_sum across JAX versions."""
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)


def _izhikevich_dv_du(v, u, current_native, a, b):
    """Izhikevich (2003) fast-subsystem derivatives -- shared by every scan-body
    closure in this module (F-028: was duplicated verbatim 11 times)."""
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
    du = a * (b * v - u)
    return dv, du


def _izhikevich_dv_du_recovery_h_k(v, u, current_native, a, b, h_k):
    """Protocol D1 — static recovery drive ``du = a * (H_K * b * v - u)``."""
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current_native
    du = a * (h_k * b * v - u)
    return dv, du


@dataclass(frozen=True)
class EIGNetwork:
    """Lightweight description of an E/PV-like/SST-like/VIP-like reduced network
    (labels are reduced emitter classes, not literal cell identity)."""

    params: IzhikevichParams
    positions: jax.Array
    metadata: dict

    @property
    def n_neurons(self) -> int:
        """Documented public function `n_neurons`."""
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
    """Create E/PV-like/SST-like/VIP-like Izhikevich parameters (reduced
    emitter classes — does not warrant literal PV/SST/VIP identity).

    Labels (stored without suffix; described with "-like" when claiming dynamics):
    - ``E``: E-like regular-spiking excitatory emitter.
    - ``PV`` or ``Inl``: PV-like fast-spiking local inhibitory emitter.
    - ``SST`` or ``Ing``: SST-like low-threshold/dendrite-related inhibitory emitter.
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
        p = _get_cell_type_params(name)
        a.append(p["a"])
        b.append(p["b"])
        c.append(p["c"])
        d.append(p["d"])
        drive.append(p["drive"])
        sign.append(p["sign"])

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
    build_dense_connectivity: bool = True,
) -> IzhikevichParams:
    """Create reduced Izhikevich parameters from explicit cell labels.

    This is the package-native path used by Suite No. 2 when a notebook needs
    deterministic E/PV/SST/VIP populations without local simulator code.  The
    returned native drive values are reduced-model drive units.  They are suited
    to relative proxy readouts unless a caller supplies an external calibration
    bridge.

    ``build_dense_connectivity=False`` skips materializing the default dense
    ``(N,N)`` ``W`` (returns a cheap ``(0,0)`` placeholder instead) -- for a
    caller that is about to overwrite ``W`` unconditionally anyway (e.g.
    ``jaxfne._construct._neuron_population_from_config``, whose very
    next step, ``_apply_connectivity``, replaces ``W`` in every branch
    regardless of this default). Found 2026-07-17: at N=100,000 the default
    dense build alone allocates 40GB before being discarded, a real OOM on
    real GPU hardware, not a theoretical concern. Default stays ``True`` --
    zero behavior change for every other caller.
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
        p = _get_cell_type_params(name)
        if name not in IZHIKEVICH_CELL_TYPE_DEFAULTS:
            raise ValueError(f"unknown Suite No. 2 cell type label: {name!r}")
        a.append(p["a"])
        b.append(p["b"])
        c.append(p["c"])
        d.append(p["d"])
        drive.append(overrides.get(name, p["drive"]))
        sign.append(p["sign"])

    n = len(label_tuple)
    sign_array = jnp.asarray(sign, dtype=jdtype)
    W = _default_eig_connectivity(sign_array, jdtype) if build_dense_connectivity else jnp.zeros((0, 0), dtype=jdtype)
    return IzhikevichParams(
        a=jnp.asarray(a, dtype=jdtype),
        b=jnp.asarray(b, dtype=jdtype),
        c=jnp.asarray(c, dtype=jdtype),
        d=jnp.asarray(d, dtype=jdtype),
        drive=jnp.asarray(drive, dtype=jdtype),
        sign=sign_array,
        W=W,
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.asarray(b, dtype=jdtype) * jnp.asarray(-65.0, dtype=jdtype),
        source_scale=jnp.asarray(source_scale, dtype=jdtype),
        labels=label_tuple,
        layer_labels=tuple(str(x) for x in layer_labels) if layer_labels is not None else None,
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )

def make_eig_network(
    n: int = 128,
    cell_type_fractions: Mapping[str, float] | None = None,
    *,
    dtype: str = "float32",
) -> EIGNetwork:
    """Build a minimal EIG network with laminar depth positions.

    Parameters
    ----------
    n : int, default 128
        Number of neurons; depth positions are spread evenly over [0, 1].
    cell_type_fractions : Mapping[str, float], optional
        E/PV/SST/VIP fractions. Default: ``{E:0.8, PV:0.1, SST:0.07, VIP:0.03}``.
    dtype : str, keyword-only, default "float32"
        Array dtype policy.
    """

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
    noise_scale: "jax.Array | float | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Simulate a reduced EIG Izhikevich scaffold using ``jax.lax.scan``.

    Synaptic semantics: the dense backend couples ``weights @ prev_spikes``
    — an instantaneous one-step spike-jump current with no synaptic time
    constant. This is a distinct supported synaptic model, NOT a
    trajectory-equivalent representation of the edge-list backend, whose
    synapses are receptor-filtered exponentials with per-edge ``tau_ms``
    (see :func:`simulate_edge_recurrent_izhikevich`).

    When ``drive_schedule`` is None the existing scan path is preserved exactly.
    When provided, it must have shape ``(n_steps, n_neurons)`` and is added to
    ``params.drive`` at each timestep as native (uncalibrated) current.
    ``noise_scale`` sets the stochastic-current coefficient: ``None`` keeps the
    historical 0.5 scalar; a scalar or ``(n_neurons,)`` array gives per-neuron
    control of the internal noise. No physical-amplitude or calibration claim is
    introduced.
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
    noise_coef = (jnp.asarray(0.5, dtype=jdtype) if noise_scale is None
                  else jnp.asarray(noise_scale, dtype=jdtype))

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
            """Documented public function `step`."""
            v, u, prev_spikes = carry
            syn = weights @ prev_spikes
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
            return (v_reset, u_reset, spikes), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            """Documented public function `step_sched`."""
            sched_t, noise_t = xs_t
            v, u, prev_spikes = carry
            syn = weights @ prev_spikes
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
            return (v_reset, u_reset, spikes), (v_reset, spikes, source_proxy)

        _, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

    return voltages, spikes, sources


def edge_delay_steps_from_ms(
    delay_ms: float | np.ndarray | jax.Array,
    dt_ms: float,
) -> np.ndarray:
    """Convert grid-aligned edge delays in ms to integer step counts.

  Protocol D (0.4.16): ``n_ij = delay_ms / dt_ms`` must be an integer within
  tolerance. Non-grid-aligned values reject rather than round.
    """
    if dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")
    arr = np.asarray(delay_ms, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("edge delay_ms must be >= 0")
    steps = arr / float(dt_ms)
    rounded = np.rint(steps)
    if not np.allclose(steps, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(
            "edge delay_ms must be grid-aligned to dt_ms "
            f"(delay_ms/dt_ms not integer within tolerance; dt_ms={dt_ms})"
        )
    return rounded.astype(np.int32)


def edge_list_with_delay_ms(
    edges: "EdgeList",
    delay_ms: float | np.ndarray | jax.Array,
    dt_ms: float,
) -> "EdgeList":
    """Return a copy of ``edges`` with per-edge ``delay_steps`` from ms delays."""
    steps = edge_delay_steps_from_ms(delay_ms, dt_ms)
    if np.ndim(steps) == 0:
        steps_arr = jnp.full((edges.n_edges,), int(steps), dtype=jnp.int32)
    else:
        steps_arr = jnp.asarray(steps, dtype=jnp.int32)
        if int(steps_arr.shape[0]) != edges.n_edges:
            raise ValueError(
                f"delay_ms length {steps_arr.shape[0]} != n_edges {edges.n_edges}"
            )
    return dataclass_replace(edges, delay_steps=steps_arr)


def dataclass_replace(edges: "EdgeList", **kwargs: Any) -> "EdgeList":
    """Frozen-dataclass replace helper local to emitters."""
    return EdgeList(
        pre=kwargs.get("pre", edges.pre),
        post=kwargs.get("post", edges.post),
        weight=kwargs.get("weight", edges.weight),
        receptor_index=kwargs.get("receptor_index", edges.receptor_index),
        tau_ms=kwargs.get("tau_ms", edges.tau_ms),
        delay_steps=kwargs.get("delay_steps", edges.delay_steps),
        source_calibration_status=kwargs.get(
            "source_calibration_status", edges.source_calibration_status
        ),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EdgeList:
    """Sparse recurrent connectivity as a JAX pytree.

    Edges carry signed native weights and first-order synaptic decay constants.
    This is a computational backend for recurrent reduced emitters; weights are
    native/unphysical unless a future calibration bridge declares otherwise.

    Optional ``delay_steps`` (integer axonal delay per edge, in simulation steps)
    implements Protocol D finite edge delay. When all entries are zero, the
    legacy instantaneous recurrent kernel is used unchanged.
    """

    pre: jax.Array
    post: jax.Array
    weight: jax.Array
    receptor_index: jax.Array
    tau_ms: jax.Array
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    delay_steps: jax.Array | None = None

    def __post_init__(self) -> None:
        if self.delay_steps is None:
            object.__setattr__(
                self,
                "delay_steps",
                jnp.zeros(self.pre.shape[0], dtype=jnp.int32),
            )

    @property
    def n_edges(self) -> int:
        """Documented public function `n_edges`."""
        return int(self.pre.shape[0])

    def tree_flatten(self):
        """Documented public function `tree_flatten`."""
        children = (
            self.pre,
            self.post,
            self.weight,
            self.receptor_index,
            self.tau_ms,
            self.delay_steps,
        )
        aux = {"source_calibration_status": self.source_calibration_status}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Documented public function `tree_unflatten`."""
        if len(children) == 5:
            pre, post, weight, receptor_index, tau_ms = children
            delay_steps = jnp.zeros(pre.shape[0], dtype=jnp.int32)
        else:
            pre, post, weight, receptor_index, tau_ms, delay_steps = children
        return cls(
            pre,
            post,
            weight,
            receptor_index,
            tau_ms,
            aux["source_calibration_status"],
            delay_steps,
        )

    def to_dict(self) -> dict:
        """JSON-safe serialization including full edge payloads.

        The summary keys are the historical surface; ``pre``/``post``/``weight``/
        ``receptor_index``/``tau_ms``/``delay_steps`` carry the complete state so
        :meth:`from_dict` restores a bit-exact copy under the recorded dtypes.
        """
        from .io import json_safe

        return json_safe({
            "backend": "edge_list_recurrent_v0.0.9",
            "n_edges": self.n_edges,
            "receptors": {"0": "excitatory_native", "1": "inhibitory_native"},
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": False,
            "pre": self.pre,
            "post": self.post,
            "weight": self.weight,
            "receptor_index_arr": self.receptor_index,
            "tau_ms": self.tau_ms,
            "delay_steps": self.delay_steps,
            "array_dtypes": {
                "pre": str(self.pre.dtype),
                "post": str(self.post.dtype),
                "weight": str(self.weight.dtype),
                "receptor_index": str(self.receptor_index.dtype),
                "tau_ms": str(self.tau_ms.dtype),
                "delay_steps": str(self.delay_steps.dtype),
            },
        })

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeList":
        """Reconstruct an :class:`EdgeList` from :meth:`to_dict` output.

        Bit-exact for payloads produced by the same backend version: arrays are
        rebuilt with their recorded dtypes and equality-checked lengths. Older
        summary-only payloads (no array keys) raise a clear error directing the
        caller to the constructor.
        """
        if d.get("backend") != "edge_list_recurrent_v0.0.9":
            raise ValueError(f"unsupported EdgeList payload backend: {d.get('backend')!r}")
        required = ("pre", "post", "weight", "receptor_index_arr", "tau_ms", "delay_steps")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(
                "EdgeList payload lacks full array fields "
                f"(missing: {missing}); only summary payloads predate this "
                "serializer and cannot be reconstructed"
            )
        dtypes = d.get("array_dtypes", {})

        def _arr(key: str):
            dtype = dtypes.get(key, None)
            arr = jnp.asarray(d[key])
            return arr if dtype is None else arr.astype(dtype)

        pre = _arr("pre").astype(jnp.int32)
        post = _arr("post").astype(jnp.int32)
        weight = _arr("weight")
        receptor_index = _arr("receptor_index_arr")
        tau_ms = _arr("tau_ms")
        delay_steps = _arr("delay_steps")
        n = int(d["n_edges"])
        for name, arr in (("pre", pre), ("post", post), ("weight", weight),
                          ("receptor_index", receptor_index), ("tau_ms", tau_ms),
                          ("delay_steps", delay_steps)):
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"EdgeList payload length mismatch: {name} has {arr.shape[0]} "
                    f"rows, n_edges={n}"
                )
        return cls(
            pre,
            post,
            weight,
            receptor_index,
            tau_ms,
            d.get("source_calibration_status", "uncalibrated_izhikevich_native_current"),
            delay_steps.astype(jnp.int32),
        )


def make_edge_list_from_dense(
    weights: jax.Array,
    *,
    threshold: float = 1e-12,
    dtype: str = "float32",
) -> EdgeList:
    """Convert a dense recurrent weight matrix into a sparse EdgeList.

    The dense matrix uses rows as postsynaptic targets and columns as
    presynaptic sources, matching ``weights @ spikes`` in the baseline backend.

    Semantics: this is a representation conversion, NOT a dynamics
    conversion. The dense backend couples ``weights @ prev_spikes``
    (instantaneous one-step spike jump, no synaptic time constant) while the
    edge-list backend applies receptor-filtered exponential synapses. The
    ``tau_exc``/``tau_inh`` defaults assigned here (2.0 ms / 5.0 ms by weight
    sign) are declared conversion defaults for that filter; simulated
    trajectories under the two backends are therefore not
    trajectory-equivalent and parity claims are statistical only.
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


def _edge_delay_steps_host(edges: EdgeList) -> np.ndarray:
    """Host-side delay_steps array for dispatch before JIT."""
    return np.asarray(edges.delay_steps, dtype=np.int32)


def _delayed_presynaptic_spikes(
    spikes: jax.Array,
    spike_hist: jax.Array,
    t_idx: jax.Array,
    pre: jax.Array,
    delay_steps: jax.Array,
) -> jax.Array:
    """Per-edge presynaptic spikes entering the synaptic update at step ``t_idx``.

    Indexing convention (Protocol D0/D1):
      - ``delay_steps[e] == 0``: use presynaptic spikes from the *current* step
        ``spikes[pre[e]]`` (matches the legacy kernel's ``spikes[pre]`` term).
      - ``delay_steps[e] == n > 0``: use ``spikes_{t-n}[pre[e]]`` from the ring
        buffer; invalid when ``t < n`` yields zero.
    """
    bufsize = spike_hist.shape[0]
    hist_slots = (t_idx - delay_steps) % bufsize
    from_hist = spike_hist[hist_slots, pre]
    valid_hist = (delay_steps > 0) & (t_idx >= delay_steps)
    from_hist = jnp.where(valid_hist, from_hist, jnp.zeros_like(from_hist))
    from_current = spikes[pre]
    return jnp.where(delay_steps == 0, from_current, from_hist)


def _simulate_edge_recurrent_izhikevich_delayed(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    step_indices: "jax.Array | None" = None,
    record_edge_current: bool = False,
    record_current_trace: bool = False,
    record_u_trace: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Finite edge-delay recurrent kernel (Protocol D).

    Memory: ``O(N * D_max)`` spike-history ring per neuron plus ``O(E)`` synaptic
    state, where ``D_max = max(delay_steps)``.

    Segmented continuation requires full ``init_state`` including canonical
    ``delay_state`` (legacy alias ``spike_history``) and
    ``continuation_step_offset`` (global step index at segment start).
    """
    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    delay_steps = edges.delay_steps.astype(jnp.int32)
    if int(np.min(_edge_delay_steps_host(edges))) < 0:
        raise ValueError("edge delay_steps must be >= 0")
    n_neurons = params.v0.shape[0]
    max_delay = int(np.max(_edge_delay_steps_host(edges)))
    bufsize = max_delay + 1

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )
    time_step_offset = _rbd_continuation_step_offset_array(init_state)
    if init_state is not None and "v" in init_state:
        _validate_delayed_init_state(
            init_state,
            bufsize=bufsize,
            n_neurons=n_neurons,
            n_edges=edges.n_edges,
        )
        delay0 = _rbd_delay_state_from_init(
            init_state, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
        )
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            delay0,
        )
    else:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            jnp.zeros((bufsize, n_neurons), dtype=jdtype),
        )
    if step_indices is not None:
        step_indices = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
        if int(step_indices.shape[0]) != int(n_steps):
            raise ValueError(
                "step_indices must have shape (n_steps,) when provided; got "
                f"{step_indices.shape} for n_steps={n_steps}"
            )
    else:
        off = time_step_offset
        if isinstance(off, int):
            step_indices = jnp.arange(off, off + int(n_steps), dtype=jnp.int32)
        else:
            step_indices = jnp.arange(
                off,
                off + jnp.asarray(int(n_steps), dtype=jnp.int32),
                dtype=jnp.int32,
            )

    if record_edge_current or record_current_trace or record_u_trace:
        def step_delayed(carry, xs_t):
            t_idx, noise_t = xs_t
            v, u, prev_spikes, syn_state, spike_hist = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
            syn_next = syn_state * decay + presyn
            slot = jnp.mod(t_idx, bufsize)
            spike_hist_next = spike_hist.at[slot].set(spikes)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, spike_hist_next), (
                v_reset,
                spikes,
                source_proxy,
                presyn,
                edge_current,
                current_native,
                u_reset,
            )

        if drive_schedule is not None:
            sched = drive_schedule.astype(jdtype)

            def step_delayed_sched(carry, xs_t):
                t_idx, sched_t, noise_t = xs_t
                v, u, prev_spikes, syn_state, spike_hist = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + sched_t + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
                syn_next = syn_state * decay + presyn
                slot = jnp.mod(t_idx, bufsize)
                spike_hist_next = spike_hist.at[slot].set(spikes)
                source_proxy = _source_proxy_from_components(
                    current_native, spikes, source_scale, dtype=jdtype
                )
                return (v_reset, u_reset, spikes, syn_next, spike_hist_next), (
                    v_reset,
                    spikes,
                    source_proxy,
                    presyn,
                    edge_current,
                    current_native,
                    u_reset,
                )

            final, (voltages, spikes, sources, presyn_trace, edge_current_trace, current_trace, u_trace) = jax.lax.scan(
                step_delayed_sched,
                init,
                xs=(step_indices, sched, bulk_noise),
            )
        else:
            final, (voltages, spikes, sources, presyn_trace, edge_current_trace, current_trace, u_trace) = jax.lax.scan(
                step_delayed, init, xs=(step_indices, bulk_noise)
            )
    else:
        def step_delayed(carry, xs_t):
            t_idx, noise_t = xs_t
            v, u, prev_spikes, syn_state, spike_hist = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
            syn_next = syn_state * decay + presyn
            slot = jnp.mod(t_idx, bufsize)
            spike_hist_next = spike_hist.at[slot].set(spikes)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, spike_hist_next), (
                v_reset,
                spikes,
                source_proxy,
                presyn,
            )

        if drive_schedule is not None:
            sched = drive_schedule.astype(jdtype)

            def step_delayed_sched(carry, xs_t):
                t_idx, sched_t, noise_t = xs_t
                v, u, prev_spikes, syn_state, spike_hist = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + sched_t + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
                syn_next = syn_state * decay + presyn
                slot = jnp.mod(t_idx, bufsize)
                spike_hist_next = spike_hist.at[slot].set(spikes)
                source_proxy = _source_proxy_from_components(
                    current_native, spikes, source_scale, dtype=jdtype
                )
                return (v_reset, u_reset, spikes, syn_next, spike_hist_next), (
                    v_reset,
                    spikes,
                    source_proxy,
                    presyn,
                )

            final, (voltages, spikes, sources, presyn_trace) = jax.lax.scan(
                step_delayed_sched,
                init,
                xs=(step_indices, sched, bulk_noise),
            )
        else:
            final, (voltages, spikes, sources, presyn_trace) = jax.lax.scan(
                step_delayed, init, xs=(step_indices, bulk_noise)
            )

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "delay_state": final[4],
        "spike_history": final[4],
        "delay_steps_max": jnp.asarray(max_delay, dtype=jnp.int32),
        "continuation_step_offset": step_indices[-1] + jnp.asarray(1, dtype=jnp.int32),
        "presynaptic_drive_trace": presyn_trace,
    }
    if record_edge_current:
        final_state["edge_current_trace"] = edge_current_trace
    if record_current_trace:
        final_state["current_trace"] = current_trace
    if record_u_trace:
        final_state["u_trace"] = u_trace
    return voltages, spikes, sources, final_state


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
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    step_indices: "jax.Array | None" = None,
    record_edge_current: bool = False,
    record_current_trace: bool = False,
    record_u_trace: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simulate reduced Izhikevich emitters with sparse recurrent synapses.

    Synaptic semantics: each edge carries a receptor-filtered exponential
    synaptic state (``tau_ms`` per edge); this is a distinct supported
    synaptic model and is NOT trajectory-equivalent to the dense backend's
    instantaneous ``weights @ prev_spikes`` coupling.

    The implementation uses ``jax.lax.scan`` over time and
    ``jax.ops.segment_sum`` over edges. It is JIT/vmap compatible and preserves
    the uncalibrated proxy-source truth status.

    When ``drive_schedule`` is None the existing scan path is preserved exactly.
    When provided, it must have shape ``(n_steps, n_neurons)`` and is added as
    native uncalibrated current at each timestep. ``noise_scale`` sets the
    stochastic-current coefficient: ``None`` keeps the historical 0.5 scalar; a
    scalar or ``(n_neurons,)`` array gives per-neuron control of internal noise.
    ``init_state`` optionally supplies ``v``, ``u``, ``prev_spikes``, and
    ``syn_state`` for deterministic or explicitly keyed segmented continuation.
    ``prev_spikes`` is an interface-parity compatibility carry: it is part of
    the canonical state tuple shared across kernels but is not read by this
    kernel's update (dead-by-design on this path, retained for checkpoint and
    continuation-contract stability).
    Nonzero ``edges.delay_steps`` select the finite-delay kernel; segmented
    continuation with positive delays requires full ``init_state`` including
    ``delay_state`` (legacy alias ``spike_history``).
    """

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host < 0):
        raise ValueError("edge delay_steps must be >= 0")
    if np.any(delay_host > 0):
        return _simulate_edge_recurrent_izhikevich_delayed(
            params,
            edges,
            n_steps,
            dt_ms,
            key,
            dtype=dtype,
            drive_schedule=drive_schedule,
            silence_mask=silence_mask,
            noise_scale=noise_scale,
            init_state=init_state,
            step_indices=step_indices,
            record_edge_current=record_edge_current,
            record_current_trace=record_current_trace,
            record_u_trace=record_u_trace,
        )

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (jnp.asarray(0.5, dtype=jdtype) if noise_scale is None
                  else jnp.asarray(noise_scale, dtype=jdtype))
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

    if init_state is None:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
        )
    else:
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
        )

    if drive_schedule is None:
        if record_edge_current or record_current_trace or record_u_trace:
            def step(carry, noise_t):
                """Documented public function `step`."""
                v, u, prev_spikes, syn_state = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                
                # Apply silence_mask
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                syn_next = syn_state * decay + spikes[pre]
                source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
                return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy, edge_current, current_native, u_reset)

            final, (voltages, spikes, sources, edge_current_trace, current_trace, u_trace) = jax.lax.scan(step, init, xs=bulk_noise)
        else:
            def step(carry, noise_t):
                """Documented public function `step`."""
                v, u, prev_spikes, syn_state = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                
                # Apply silence_mask
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                syn_next = syn_state * decay + spikes[pre]
                source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
                return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

            final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        if record_edge_current or record_current_trace or record_u_trace:
            def step_sched(carry, xs_t):
                """Documented public function `step_sched`."""
                sched_t, noise_t = xs_t
                v, u, prev_spikes, syn_state = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + sched_t + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                
                # Apply silence_mask
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                syn_next = syn_state * decay + spikes[pre]
                source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
                return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy, edge_current, current_native, u_reset)

            final, (voltages, spikes, sources, edge_current_trace, current_trace, u_trace) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))
        else:
            def step_sched(carry, xs_t):
                """Documented public function `step_sched`."""
                sched_t, noise_t = xs_t
                v, u, prev_spikes, syn_state = carry
                edge_current = weight * syn_state
                syn = _segment_sum(edge_current, post, n_neurons)
                current_native = drive + sched_t + syn + noise_coef * noise_t
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                
                # Apply silence_mask
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                syn_next = syn_state * decay + spikes[pre]
                source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
                return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

            final, (voltages, spikes, sources) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
    }
    if record_edge_current:
        final_state["edge_current_trace"] = edge_current_trace
    if record_current_trace:
        final_state["current_trace"] = current_trace
    if record_u_trace:
        final_state["u_trace"] = u_trace
    return voltages, spikes, sources, final_state


def simulate_edge_recurrent_izhikevich_static_h_k_recovery(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    h_k: jax.Array,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Protocol D1 — static ``H_K`` on Izhikevich recovery drive (``dot W=0``).

    Typed map (frozen D1):

    ``du = a * (H_K * b * v - u)``  equivalently ``b_eff = H_K * b``.

    ``H_K`` is constant per neuron for the full run (``dH_K/dt = 0``). When
    ``H_K \\equiv 1``, this reduces to the classical
    :func:`simulate_edge_recurrent_izhikevich` zero-delay kernel (bit-exact target
    with ``noise_scale`` matched). Nonzero edge delays are rejected — use the
    classical delay kernel without RBS for delay studies.
    """
    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host > 0):
        raise ValueError(
            "static H_K recovery kernel requires zero edge delays; "
            "use simulate_edge_recurrent_izhikevich for delayed recurrence"
        )

    jdtype = _dtype_from_policy(dtype)
    h_k_arr = jnp.asarray(h_k, dtype=jdtype)
    n_neurons = params.v0.shape[0]
    if h_k_arr.shape != (n_neurons,):
        raise ValueError(f"h_k must have shape ({n_neurons},), got {h_k_arr.shape}")
    try:
        h_host = np.asarray(jax.device_get(h_k_arr))
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
        h_host = None
    if h_host is not None and np.any(h_host <= 0):
        raise ValueError("static H_K recovery requires H_K > 0")

    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )

    if init_state is None:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
        )
    else:
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
        )

    if drive_schedule is None:

        def step(carry, noise_t):
            v, u, prev_spikes, syn_state = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(
                v, u, current_native, a, b, h_k_arr
            )
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next), (
                v_reset,
                u_reset,
                spikes,
                source_proxy,
            )

        final, (voltages, u_trace, spikes, sources) = jax.lax.scan(
            step, init, xs=bulk_noise
        )
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(
                v, u, current_native, a, b, h_k_arr
            )
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next), (
                v_reset,
                u_reset,
                spikes,
                source_proxy,
            )

        final, (voltages, u_trace, spikes, sources) = jax.lax.scan(
            step_sched, init, xs=(sched, bulk_noise)
        )

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "u_trace": u_trace,
        "H_K_static": h_k_arr,
    }
    return voltages, spikes, sources, final_state


def _advance_h_k_f1_autonomous(
    H: jax.Array,
    dt: jax.Array,
    tau_k_ms: jax.Array,
) -> jax.Array:
    """Protocol D2a — ``tau_K * dH_K/dt = 1 - H_K`` with ``kappa_K = 0`` (Euler)."""
    return H + dt * (1.0 - H) / tau_k_ms


def simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    h_k0: jax.Array,
    tau_k_ms: float,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Protocol D2a — autonomous F1 ``H_K`` relaxation on D1 recovery map.

    Dynamics (frozen D2a):

    ``tau_K * dH_K/dt = 1 - H_K``, ``kappa_K = 0``.

    Coupling (frozen D1): ``du = a * (H_K * b * v - u)``.

    Discrete contract: ``H_K^{n+1} = H_K^n + dt * (1 - H_K^n) / tau_K``.

    When ``H_K \\equiv 1``, ``dH_K/dt = 0`` and the kernel matches the classical
    :func:`simulate_edge_recurrent_izhikevich` zero-delay path (bit-exact target).
    """
    if tau_k_ms <= 0:
        raise ValueError("tau_k_ms must be > 0")

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host > 0):
        raise ValueError(
            "dynamic H_K recovery kernel requires zero edge delays"
        )

    jdtype = _dtype_from_policy(dtype)
    h_k_init = jnp.asarray(h_k0, dtype=jdtype)
    n_neurons = params.v0.shape[0]
    if h_k_init.shape != (n_neurons,):
        raise ValueError(f"h_k0 must have shape ({n_neurons},), got {h_k_init.shape}")
    try:
        h_host = np.asarray(jax.device_get(h_k_init))
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
        h_host = None
    if h_host is not None and np.any(h_host <= 0):
        raise ValueError("dynamic H_K recovery requires H_K > 0")

    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    tau_k = jnp.asarray(tau_k_ms, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )

    if init_state is None:
        H0 = h_k_init
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            H0,
        )
    else:
        H0 = jnp.asarray(
            init_state.get("H_K", init_state.get("H", h_k_init)), dtype=jdtype
        )
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            H0,
        )

    if drive_schedule is None:

        def step(carry, noise_t):
            v, u, prev_spikes, syn_state, H = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(
                v, u, current_native, a, b, H
            )
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            H_next = _advance_h_k_f1_autonomous(H, dt, tau_k)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, H_next), (
                v_reset,
                u_reset,
                spikes,
                source_proxy,
                H_next,
            )

        final, (voltages, u_trace, spikes, sources, H_trace) = jax.lax.scan(
            step, init, xs=bulk_noise
        )
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, H = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(
                v, u, current_native, a, b, H
            )
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            H_next = _advance_h_k_f1_autonomous(H, dt, tau_k)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, H_next), (
                v_reset,
                u_reset,
                spikes,
                source_proxy,
                H_next,
            )

        final, (voltages, u_trace, spikes, sources, H_trace) = jax.lax.scan(
            step_sched, init, xs=(sched, bulk_noise)
        )

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "u_trace": u_trace,
        "H_K_trace": H_trace,
        "H_K_final": final[4],
        "H_K0": H0,
        "tau_k_ms": tau_k,
    }
    return voltages, spikes, sources, final_state


def simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    h_k0: jax.Array,
    owner_mask: jax.Array,
    tau_k_ms: float = 100.0,
    dynamic: bool = False,
    gamma_h_enabled: bool = True,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    step_indices: "jax.Array | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Protocol E3 — owned D1/D2a ``H_K`` on delayed edge-recurrent Izhikevich.

    Coupling (frozen D1): ``du = a * (H_K * b * v - u)`` on all neurons; non-owners
    keep ``H_K \\equiv 1`` so ``b_eff = b``.

    Dynamics on owner support only when ``dynamic=True``:

    ``tau_K * dH_K/dt = 1 - H_K`` (D2a F1 Euler).

    When ``dynamic=False``, ``H_K`` is fixed at ``h_k0`` (E3-null / static reference).

    When ``gamma_h_enabled=False``, ``H_K`` may still evolve but coupling uses
    ``b_eff = b`` (``Gamma_H = I``); used for E5 N1 state-only null arm.

    Non-owner semantics: ``H_K`` is held at the reference coordinate ``1``; the F1
    recurrence is masked off outside ``owner_mask`` (not a separate unallocated
    coordinate).

    Supports finite edge delays (E2 composition) with ``delay_state`` continuation.
    """
    if dynamic and tau_k_ms <= 0:
        raise ValueError("dynamic owned H_K requires tau_k_ms > 0")

    jdtype = _dtype_from_policy(dtype)
    h_k_init = jnp.asarray(h_k0, dtype=jdtype)
    owner = jnp.asarray(owner_mask, dtype=jdtype)
    n_neurons = params.v0.shape[0]
    if h_k_init.shape != (n_neurons,):
        raise ValueError(f"h_k0 must have shape ({n_neurons},), got {h_k_init.shape}")
    if owner.shape != (n_neurons,):
        raise ValueError(f"owner_mask must have shape ({n_neurons},), got {owner.shape}")

    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    tau_k = jnp.asarray(tau_k_ms, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    delay_steps = edges.delay_steps.astype(jnp.int32)
    delay_host = _edge_delay_steps_host(edges)
    max_delay = int(np.max(delay_host)) if delay_host.size else 0
    bufsize = max_delay + 1

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )
    time_step_offset = _rbd_continuation_step_offset_array(init_state)
    if init_state is not None and "v" in init_state:
        if max_delay > 0:
            _validate_delayed_init_state(
                init_state,
                bufsize=bufsize,
                n_neurons=n_neurons,
                n_edges=edges.n_edges,
            )
            delay0 = _rbd_delay_state_from_init(
                init_state, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
            )
        else:
            delay0 = jnp.zeros((bufsize, n_neurons), dtype=jdtype)
        H0 = jnp.asarray(
            init_state.get("H_K", init_state.get("H", h_k_init)), dtype=jdtype
        )
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            delay0,
            H0,
        )
    else:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            jnp.zeros((bufsize, n_neurons), dtype=jdtype),
            h_k_init,
        )

    if step_indices is not None:
        step_indices = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
        if int(step_indices.shape[0]) != int(n_steps):
            raise ValueError(
                "step_indices must have shape (n_steps,) when provided; got "
                f"{step_indices.shape} for n_steps={n_steps}"
            )
    else:
        off = time_step_offset
        if isinstance(off, int):
            step_indices = jnp.arange(off, off + int(n_steps), dtype=jnp.int32)
        else:
            step_indices = jnp.arange(
                off,
                off + jnp.asarray(int(n_steps), dtype=jnp.int32),
                dtype=jnp.int32,
            )

    one = jnp.asarray(1.0, dtype=jdtype)

    def _coupling_h(H: jax.Array) -> jax.Array:
        return H if gamma_h_enabled else one

    def _h_next(H: jax.Array) -> jax.Array:
        advanced = _advance_h_k_f1_autonomous(H, dt, tau_k)
        return jnp.where(owner > 0.5, advanced, one)

    def step_delayed(carry, xs_t):
        t_idx, noise_t = xs_t
        v, u, prev_spikes, syn_state, spike_hist, H = carry
        edge_current = weight * syn_state
        syn = _segment_sum(edge_current, post, n_neurons)
        current_native = drive + syn + noise_coef * noise_t
        dv, du = _izhikevich_dv_du_recovery_h_k(v, u, current_native, a, b, _coupling_h(H))
        v_next = v + dt * dv
        u_next = u + dt * du
        v_next = jnp.where(s_mask > 0.5, v_next, c)
        spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
        spikes = spikes_bool.astype(jdtype)
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)
        presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
        syn_next = syn_state * decay + presyn
        slot = jnp.mod(t_idx, bufsize)
        spike_hist_next = spike_hist.at[slot].set(spikes)
        H_next = _h_next(H) if dynamic else H
        source_proxy = _source_proxy_from_components(
            current_native, spikes, source_scale, dtype=jdtype
        )
        return (v_reset, u_reset, spikes, syn_next, spike_hist_next, H_next), (
            v_reset,
            spikes,
            source_proxy,
            H_next,
        )

    if drive_schedule is not None:
        sched = drive_schedule.astype(jdtype)

        def step_delayed_sched(carry, xs_t):
            t_idx, sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, spike_hist, H = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(v, u, current_native, a, b, _coupling_h(H))
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps)
            syn_next = syn_state * decay + presyn
            slot = jnp.mod(t_idx, bufsize)
            spike_hist_next = spike_hist.at[slot].set(spikes)
            H_next = _h_next(H) if dynamic else H
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, spike_hist_next, H_next), (
                v_reset,
                spikes,
                source_proxy,
                H_next,
            )

        final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
            step_delayed_sched,
            init,
            xs=(step_indices, sched, bulk_noise),
        )
    else:
        final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
            step_delayed, init, xs=(step_indices, bulk_noise)
        )

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "delay_state": final[4],
        "spike_history": final[4],
        "H_K_final": final[5],
        "H_K_trace": H_trace,
        "owner_mask": owner,
        "h_k_non_owner_semantics": (
            "fixed_reference_H_K_equals_1_with_F1_recurrence_masked_off"
        ),
        "gamma_h_enabled": jnp.asarray(gamma_h_enabled),
        "gamma_h_semantics": (
            "b_eff_equals_H_K_times_b"
            if gamma_h_enabled
            else "Gamma_H_identity_b_eff_equals_b"
        ),
        "delay_steps_max": jnp.asarray(max_delay, dtype=jnp.int32),
        "continuation_step_offset": step_indices[-1] + jnp.asarray(1, dtype=jnp.int32),
    }
    return voltages, spikes, sources, final_state


def _advance_h_a_trace(
    H_A: jax.Array,
    S: jax.Array,
    dt: jax.Array,
    tau_a_ms: jax.Array,
) -> jax.Array:
    """Protocol D2b — ``tau_A * dH_A/dt = -H_A + S`` with ``S in {0, 1}`` (Euler)."""
    return H_A + dt * (-H_A + S) / tau_a_ms


def _advance_h_k_activity_coupled(
    H_K: jax.Array,
    H_A_old: jax.Array,
    dt: jax.Array,
    tau_k_ms: jax.Array,
    kappa_ak: jax.Array,
) -> jax.Array:
    """Protocol D2b — uses **old** ``H_A^n`` in the ``H_K`` update (causal one-step lag)."""
    return H_K + dt * ((1.0 - H_K) + kappa_ak * H_A_old) / tau_k_ms


def simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    h_a0: jax.Array | None = None,
    h_k0: jax.Array,
    tau_a_ms: float,
    tau_k_ms: float,
    kappa_ak: float,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Protocol D2b — two-coordinate RBS ``[H_A, H_K]`` with node-local spike drive.

    Frozen discrete updates (causal ordering):

    ``S^n`` — binary per-neuron spike indicator in ``{0, 1}`` (timestep-independent
    unit event; not scaled by ``dt``).

    ``H_A^{n+1} = H_A^n + dt/tau_A * (-H_A^n + S^n)``

    ``H_K^{n+1} = H_K^n + dt/tau_K * ((1 - H_K^n) + kappa_AK * H_A^n)``

    Emitter coupling at step ``n``: ``b_eff = H_K^n * b`` (D1 map).

    ``kappa_AK = 0`` reduces to D2a ``H_K`` dynamics; ``H_A`` may still evolve from
    local spikes but does not write ``H_K``.
    """
    if tau_a_ms <= 0 or tau_k_ms <= 0:
        raise ValueError("tau_a_ms and tau_k_ms must be > 0")

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host > 0):
        raise ValueError("activity H_K RBD kernel requires zero edge delays")

    jdtype = _dtype_from_policy(dtype)
    n_neurons = params.v0.shape[0]
    h_k_init = jnp.asarray(h_k0, dtype=jdtype)
    if h_k_init.shape != (n_neurons,):
        raise ValueError(f"h_k0 must have shape ({n_neurons},), got {h_k_init.shape}")
    if h_a0 is None:
        h_a_init = jnp.zeros((n_neurons,), dtype=jdtype)
    else:
        h_a_init = jnp.asarray(h_a0, dtype=jdtype)
        if h_a_init.shape != (n_neurons,):
            raise ValueError(f"h_a0 must have shape ({n_neurons},), got {h_a_init.shape}")

    try:
        h_host = np.asarray(jax.device_get(h_k_init))
        ha_host = np.asarray(jax.device_get(h_a_init))
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
        h_host = ha_host = None
    if h_host is not None and np.any(h_host <= 0):
        raise ValueError("D2b requires H_K > 0")
    if ha_host is not None and np.any(ha_host < 0):
        raise ValueError("D2b requires H_A >= 0")

    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    tau_a = jnp.asarray(tau_a_ms, dtype=jdtype)
    tau_k = jnp.asarray(tau_k_ms, dtype=jdtype)
    kappa = jnp.asarray(kappa_ak, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    w_initial = weight

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )

    if init_state is None:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            h_a_init,
            h_k_init,
        )
    else:
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            jnp.asarray(init_state.get("H_A", h_a_init), dtype=jdtype),
            jnp.asarray(init_state.get("H_K", h_k_init), dtype=jdtype),
        )

    def _rbd_updates(H_A: jax.Array, H_K: jax.Array, S: jax.Array) -> tuple[jax.Array, jax.Array]:
        H_A_old = H_A
        H_A_next = _advance_h_a_trace(H_A_old, S, dt, tau_a)
        H_K_next = _advance_h_k_activity_coupled(H_K, H_A_old, dt, tau_k, kappa)
        return H_A_next, H_K_next

    if drive_schedule is None:

        def step(carry, noise_t):
            v, u, prev_spikes, syn_state, H_A, H_K = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(v, u, current_native, a, b, H_K)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            S_n = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + S_n[pre]
            H_A_next, H_K_next = _rbd_updates(H_A, H_K, S_n)
            source_proxy = _source_proxy_from_components(
                current_native, S_n, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, S_n, syn_next, H_A_next, H_K_next), (
                v_reset,
                u_reset,
                S_n,
                source_proxy,
                H_A_next,
                H_K_next,
                S_n,
            )

        final, outs = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, H_A, H_K = carry
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du_recovery_h_k(v, u, current_native, a, b, H_K)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            S_n = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + S_n[pre]
            H_A_next, H_K_next = _rbd_updates(H_A, H_K, S_n)
            source_proxy = _source_proxy_from_components(
                current_native, S_n, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, S_n, syn_next, H_A_next, H_K_next), (
                v_reset,
                u_reset,
                S_n,
                source_proxy,
                H_A_next,
                H_K_next,
                S_n,
            )

        final, outs = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

    voltages, u_trace, spikes, sources, H_A_trace, H_K_trace, S_trace = outs
    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "u_trace": u_trace,
        "H_A_trace": H_A_trace,
        "H_K_trace": H_K_trace,
        "H_trace": jnp.stack([H_A_trace, H_K_trace], axis=-1),
        "S_trace": S_trace,
        "H_A_final": final[4],
        "H_K_final": final[5],
        "H_A0": h_a_init,
        "H_K0": h_k_init,
        "tau_a_ms": tau_a,
        "tau_k_ms": tau_k,
        "kappa_ak": kappa,
        "w_initial": w_initial,
        "w_final": weight,
    }
    return voltages, spikes, sources, final_state


RBD_FAMILIES = ("f0", "f1", "f2")
RbdFamily = Literal["f0", "f1", "f2"]


def _validate_rbd_family(family: str) -> str:
    fam = str(family).lower()
    if fam not in RBD_FAMILIES:
        raise ValueError(f"rbd_family must be one of {RBD_FAMILIES}, got {family!r}")
    return fam


def _rbd_restoring_term(
    family: str, H: jax.Array, *, jdtype: jnp.dtype
) -> jax.Array:
    """Restoring term ``R(H)`` in ``tau_H * dH/dt = R(H) + kappa_H * I_rel``."""
    if family == "f0":
        return jnp.zeros_like(H, dtype=jdtype)
    if family == "f1":
        return jnp.ones_like(H, dtype=jdtype) - H
    return jnp.where(H > 0, (1.0 / H) - 1.0, jnp.nan)


def _rbd_advance_h(
    family: str,
    H: jax.Array,
    I_syn: jax.Array,
    dt: jax.Array,
    tau_h_ms: jax.Array,
    kappa_h: jax.Array,
    i_ref: jax.Array,
    *,
    jdtype: jnp.dtype,
) -> jax.Array:
    """Advance scalar per-neuron RBS by one Euler step (Protocol H1, ``d_H=1``)."""
    if family == "f0":
        return jnp.ones_like(H, dtype=jdtype)
    I_rel = I_syn / i_ref
    R = _rbd_restoring_term(family, H, jdtype=jdtype)
    dH = (R + kappa_h * I_rel) / tau_h_ms
    H_next = H + dt * dH
    if family == "f2":
        valid = (H > 0) & (H_next > 0)
        H_next = jnp.where(valid, H_next, jnp.nan)
    return H_next


def _rbd_recurrent_gain_affine(
    family: str,
    H: jax.Array,
    beta_h: jax.Array,
    *,
    jdtype: jnp.dtype,
) -> jax.Array:
    """Postsynaptic recurrent gain ``G_H(H;beta_H)=1+beta_H(H-1)`` (Protocol H1c-C)."""
    one = jnp.ones_like(H, dtype=jdtype)
    if family == "f0":
        return one
    g = 1.0 + beta_h * (H - 1.0)
    g = jnp.where(beta_h == 0.0, one, g)
    return jnp.where(g > 0, g, jnp.nan)


def _rbd_compose_native_current(
    I_ext: jax.Array,
    I_rec: jax.Array,
    H: jax.Array,
    family: str,
    beta_h: jax.Array,
    noise: jax.Array,
    *,
    jdtype: jnp.dtype,
) -> jax.Array:
    """``I_drive = I_ext + G_H(H) * I_rec + noise``; ``F_H`` uses pre-gain ``I_rec``."""
    g_h = _rbd_recurrent_gain_affine(family, H, beta_h, jdtype=jdtype)
    return I_ext + g_h * I_rec + noise


def _rbd_host_validate_gain_if_concrete(
    family: str, H0: jax.Array, beta_h: float
) -> None:
    if family == "f0" or beta_h == 0.0:
        return
    try:
        h_host = np.asarray(jax.device_get(H0))
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
        return
    g0 = 1.0 + beta_h * (h_host - 1.0)
    if np.any(g0 <= 0):
        raise ValueError(
            "Protocol H1c requires G_H(H;beta_H)>0; nonpositive recurrent gain "
            "invalidates the trajectory"
        )


def _rbd_host_validate_h0_if_concrete(family: str, H0: jax.Array) -> None:
    if family != "f2":
        return
    try:
        h_host = np.asarray(jax.device_get(H0))
    except (jax.errors.TracerArrayConversionError, TypeError, ValueError):
        return
    _rbd_host_validate_h0(family, h_host)


def _rbd_host_validate_h0(family: str, H0_host: np.ndarray) -> None:
    if family == "f2" and np.any(H0_host <= 0):
        raise ValueError(
            "Protocol H F2 requires H>0; nonpositive initial H invalidates the trajectory"
        )


def _rbd_delay_state_from_init(
    init_state: "dict | None",
    *,
    bufsize: int,
    n_neurons: int,
    jdtype: jnp.dtype,
) -> jax.Array:
    """Protocol H2: canonical delay-line state ``delay_state`` (alias ``spike_history``)."""
    if init_state is None:
        return jnp.zeros((bufsize, n_neurons), dtype=jdtype)
    if "delay_state" in init_state:
        return jnp.asarray(init_state["delay_state"], dtype=jdtype)
    if "spike_history" in init_state:
        return jnp.asarray(init_state["spike_history"], dtype=jdtype)
    return jnp.zeros((bufsize, n_neurons), dtype=jdtype)


def _rbd_continuation_step_offset(init_state: "dict | None") -> int:
    return int(np.asarray(_rbd_continuation_step_offset_array(init_state)))


def _rbd_continuation_step_offset_array(init_state: "dict | None") -> int | jax.Array:
    if init_state is None:
        return 0
    if "continuation_step_offset" in init_state:
        return jnp.asarray(init_state["continuation_step_offset"], dtype=jnp.int32)
    if "time_step_offset" in init_state:
        return jnp.asarray(init_state["time_step_offset"], dtype=jnp.int32)
    return jnp.asarray(0, dtype=jnp.int32)


def _validate_delayed_init_state(
    init_state: dict,
    *,
    bufsize: int,
    n_neurons: int,
    n_edges: int,
) -> None:
    required = ("v", "u", "prev_spikes", "syn_state")
    missing = [k for k in required if k not in init_state]
    if missing:
        raise ValueError(
            "delayed continuation requires init_state keys "
            f"{list(required)}; missing {missing}"
        )
    if "delay_state" not in init_state and "spike_history" not in init_state:
        raise ValueError(
            "delayed continuation requires delay_state (or legacy spike_history)"
        )


def _rbd_validate_delayed_init_state(
    init_state: dict,
    *,
    bufsize: int,
    n_neurons: int,
    n_edges: int,
) -> None:
    _validate_delayed_init_state(
        init_state,
        bufsize=bufsize,
        n_neurons=n_neurons,
        n_edges=n_edges,
    )
    ds = init_state.get("delay_state", init_state.get("spike_history"))
    ds_host = np.asarray(ds)
    if ds_host.shape != (bufsize, n_neurons):
        raise ValueError(
            f"delay_state must have shape ({bufsize}, {n_neurons}), got {ds_host.shape}"
        )
    syn = np.asarray(init_state["syn_state"])
    if syn.shape != (n_edges,):
        raise ValueError(
            f"syn_state must have shape ({n_edges},), got {syn.shape}"
        )


def simulate_edge_recurrent_izhikevich_rbd(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    rbd_family: RbdFamily = "f1",
    tau_h_ms: float = 100.0,
    kappa_h: float = 0.0,
    i_ref: float = 1.0,
    beta_h: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Protocol H1 — RBD with fixed weights (``d_H=1``, ``dot W=0``).

    Couples the standard edge-recurrent Izhikevich kernel (including Protocol D
    finite delays when ``delay_steps > 0``) to scalar per-neuron RBS ``H_i``.

    **H1c-C (postsynaptic recurrent gain):** native drive is

    ``I_i^drive = I_i^ext + G_H(H_i; beta_H) * I_i^rec + noise``,

    with ``G_H(H; beta_H) = 1 + beta_H (H - 1)``. External drive is untouched.
    ``F_H`` sees **pre-gain** recurrent aggregate ``I_i^rec`` (not
    ``G_H * I_i^rec``). ``beta_H=0`` and ``H=1`` (F0) recover H1a activity.

    ``F_H`` families:

    * **F0** — RBS disabled: ``H_i \\equiv 1``
    * **F1** — ``tau_H * dH_i/dt = (1 - H_i) + kappa_H * I_i^rel``
    * **F2** — ``tau_H * dH_i/dt = (1/H_i - 1) + kappa_H * I_i^rel`` (requires ``H>0``)

    ``I_i^rel = I_i^rec / i_ref`` uses pre-gain recurrent input. Nonpositive
    ``G_H`` propagates non-finite activity (no clip).

    Nonzero delays use the D kernel's spike ring buffer. Protocol H2 continuation
    requires full ``init_state`` including ``delay_state`` (alias
    ``spike_history``) and ``continuation_step_offset`` (global step index at
    segment start).
    """
    family = _validate_rbd_family(rbd_family)
    if tau_h_ms <= 0:
        raise ValueError("tau_h_ms must be > 0")
    if i_ref <= 0:
        raise ValueError("i_ref must be > 0")

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host < 0):
        raise ValueError("edge delay_steps must be >= 0")

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    tau_h = jnp.asarray(tau_h_ms, dtype=jdtype)
    kappa = jnp.asarray(kappa_h, dtype=jdtype)
    beta = jnp.asarray(beta_h, dtype=jdtype)
    i_ref_arr = jnp.asarray(i_ref, dtype=jdtype)
    noise_coef = (
        jnp.asarray(0.5, dtype=jdtype)
        if noise_scale is None
        else jnp.asarray(noise_scale, dtype=jdtype)
    )
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

    if init_state is None:
        H0 = jnp.ones(n_neurons, dtype=jdtype)
    else:
        H0 = jnp.asarray(
            init_state.get("H_final", init_state.get("H", jnp.ones(n_neurons))),
            dtype=jdtype,
        )
    if H0.shape != (n_neurons,):
        raise ValueError(f"H0 must have shape ({n_neurons},), got {H0.shape}")
    _rbd_host_validate_h0_if_concrete(family, H0)
    _rbd_host_validate_gain_if_concrete(family, H0, beta_h)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(
        noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
    )

    use_delays = bool(np.any(delay_host > 0))
    if use_delays:
        delay_steps = edges.delay_steps.astype(jnp.int32)
        max_delay = int(np.max(delay_host))
        bufsize = max_delay + 1
        time_step_offset = _rbd_continuation_step_offset(init_state)
        if init_state is not None and "v" in init_state:
            _rbd_validate_delayed_init_state(
                init_state,
                bufsize=bufsize,
                n_neurons=n_neurons,
                n_edges=edges.n_edges,
            )
            delay0 = _rbd_delay_state_from_init(
                init_state, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
            )
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                delay0,
                H0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros_like(params.v0, dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                jnp.zeros((bufsize, n_neurons), dtype=jdtype),
                H0,
            )
        step_indices = jnp.arange(
            int(time_step_offset),
            int(time_step_offset) + int(n_steps),
            dtype=jnp.int32,
        )

        def step_delayed_rbd(carry, xs_t):
            t_idx, noise_t = xs_t
            v, u, prev_spikes, syn_state, spike_hist, H = carry
            edge_current = weight * syn_state
            I_rec = _segment_sum(edge_current, post, n_neurons)
            H_next = _rbd_advance_h(
                family, H, I_rec, dt, tau_h, kappa, i_ref_arr, jdtype=jdtype
            )
            current_native = _rbd_compose_native_current(
                drive,
                I_rec,
                H,
                family,
                beta,
                noise_coef * noise_t,
                jdtype=jdtype,
            )
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            presyn = _delayed_presynaptic_spikes(
                spikes, spike_hist, t_idx, pre, delay_steps
            )
            syn_next = syn_state * decay + presyn
            slot = jnp.mod(t_idx, bufsize)
            spike_hist_next = spike_hist.at[slot].set(spikes)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (
                v_reset,
                u_reset,
                spikes,
                syn_next,
                spike_hist_next,
                H_next,
            ), (v_reset, spikes, source_proxy, H_next)

        if drive_schedule is not None:
            sched = drive_schedule.astype(jdtype)

            def step_delayed_rbd_sched(carry, xs_t):
                t_idx, sched_t, noise_t = xs_t
                v, u, prev_spikes, syn_state, spike_hist, H = carry
                edge_current = weight * syn_state
                I_rec = _segment_sum(edge_current, post, n_neurons)
                H_next = _rbd_advance_h(
                    family, H, I_rec, dt, tau_h, kappa, i_ref_arr, jdtype=jdtype
                )
                current_native = _rbd_compose_native_current(
                    drive + sched_t,
                    I_rec,
                    H,
                    family,
                    beta,
                    noise_coef * noise_t,
                    jdtype=jdtype,
                )
                dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
                v_next = v + dt * dv
                u_next = u + dt * du
                v_next = jnp.where(s_mask > 0.5, v_next, c)
                spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
                spikes = spikes_bool.astype(jdtype)
                v_reset = jnp.where(spikes_bool, c, v_next)
                u_reset = jnp.where(spikes_bool, u_next + d, u_next)
                presyn = _delayed_presynaptic_spikes(
                    spikes, spike_hist, t_idx, pre, delay_steps
                )
                syn_next = syn_state * decay + presyn
                slot = jnp.mod(t_idx, bufsize)
                spike_hist_next = spike_hist.at[slot].set(spikes)
                source_proxy = _source_proxy_from_components(
                    current_native, spikes, source_scale, dtype=jdtype
                )
                return (
                    v_reset,
                    u_reset,
                    spikes,
                    syn_next,
                    spike_hist_next,
                    H_next,
                ), (v_reset, spikes, source_proxy, H_next)

            final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
                step_delayed_rbd_sched,
                init,
                xs=(step_indices, sched, bulk_noise),
            )
        else:
            final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
                step_delayed_rbd,
                init,
                xs=(step_indices, bulk_noise),
            )

        final_state = {
            "v": final[0],
            "u": final[1],
            "prev_spikes": final[2],
            "syn_state": final[3],
            "delay_state": final[4],
            "spike_history": final[4],
            "delay_steps_max": jnp.asarray(max_delay, dtype=jnp.int32),
            "continuation_step_offset": jnp.asarray(
                int(time_step_offset) + int(n_steps), dtype=jnp.int32
            ),
            "H_final": final[5],
            "H_trace": H_trace,
            "tau_h_ms": tau_h,
            "kappa_h": kappa,
            "beta_h": beta,
            "i_ref": i_ref_arr,
            "w_fixed": weight,
        }
        return voltages, spikes, sources, final_state

    if init_state is None or "v" not in init_state:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            H0,
        )
    else:
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            H0,
        )

    time_step_offset = _rbd_continuation_step_offset(init_state)

    if drive_schedule is None:

        def step_rbd(carry, noise_t):
            v, u, prev_spikes, syn_state, H = carry
            edge_current = weight * syn_state
            I_rec = _segment_sum(edge_current, post, n_neurons)
            H_next = _rbd_advance_h(
                family, H, I_rec, dt, tau_h, kappa, i_ref_arr, jdtype=jdtype
            )
            current_native = _rbd_compose_native_current(
                drive,
                I_rec,
                H,
                family,
                beta,
                noise_coef * noise_t,
                jdtype=jdtype,
            )
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, H_next), (
                v_reset,
                spikes,
                source_proxy,
                H_next,
            )

        final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
            step_rbd, init, xs=bulk_noise
        )
    else:
        sched = drive_schedule.astype(jdtype)

        def step_rbd_sched(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, H = carry
            edge_current = weight * syn_state
            I_rec = _segment_sum(edge_current, post, n_neurons)
            H_next = _rbd_advance_h(
                family, H, I_rec, dt, tau_h, kappa, i_ref_arr, jdtype=jdtype
            )
            current_native = _rbd_compose_native_current(
                drive + sched_t,
                I_rec,
                H,
                family,
                beta,
                noise_coef * noise_t,
                jdtype=jdtype,
            )
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes, syn_next, H_next), (
                v_reset,
                spikes,
                source_proxy,
                H_next,
            )

        final, (voltages, spikes, sources, H_trace) = jax.lax.scan(
            step_rbd_sched, init, xs=(sched, bulk_noise)
        )

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "continuation_step_offset": jnp.asarray(
            int(time_step_offset) + int(n_steps), dtype=jnp.int32
        ),
        "H_final": final[4],
        "H_trace": H_trace,
        "tau_h_ms": tau_h,
        "kappa_h": kappa,
        "beta_h": beta,
        "i_ref": i_ref_arr,
        "w_fixed": weight,
    }
    return voltages, spikes, sources, final_state


def simulate_edge_recurrent_izhikevich_homeostatic(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    # Homeostasis control parameters (all defaulted; k_gain=0 disables)
    r_star: float = 0.05,
    tau_r_ms: float = 300.0,
    alpha: float = 1.0,
    k_gain: float = 1.0,
    g_min: float = -12.0,
    g_max: float = 8.0,
    r_max: float = 1.0,
    # Homeostatic synaptic plasticity (off when eta == 0):
    #   dw_{j->i} = eta * (r_star - r_i) * x_j   (homeostatic sign: inputs to an
    #   over-active post-neuron i downscale; to a silent one upscale). x_j is a
    #   presynaptic activity trace (tau_x_ms). Weights are clipped to [w_min, w_max].
    eta: float = 0.0,
    tau_x_ms: float = 100.0,
    w_min: float = -10.0,
    w_max: float = 10.0,
    # Hard state bounds (numerical-stability safety net; set far outside normal
    # Izhikevich dynamics so they never alter physiological behaviour, only catch
    # overflow/underflow). With these, every step stays finite in float32 for any
    # finite — or even +/-inf — input current. v upper is already caught by the
    # spike reset; these guarantee the lower/recovery/synaptic state too.
    v_floor: float = -150.0,
    v_ceiling: float = 100.0,
    u_abs_max: float = 2000.0,
    syn_abs_max: float = 1.0e4,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simulate Izhikevich emitters with sparse recurrent synapses and per-neuron homeostasis.

    Adds a slow activity trace `r_i` per neuron that generates a homeostatic excitability
    bias `g_i = clip(k_gain * (r_star - r_i), g_min, g_max)` added to input current.

    Behavior: firing → r_i↑ → g_i↓ (harder to fire); quiet → r_i leaks → g_i↑ (subsidy floor).
    Never forces spikes; g_i only biases input. Unconditionally stable via exponential leak
    and bounded clipping.

    Args:
        params, edges, n_steps, dt_ms, key: as in simulate_edge_recurrent_izhikevich
        dtype, drive_schedule, silence_mask, noise_scale: as in simulate_edge_recurrent_izhikevich
        r_star: target activity trace (default 0.05, ~expected spikes/step)
        tau_r_ms: slow leak timescale in ms (default 300, >> dt for stability)
        alpha: per-spike jump in r_i (default 1.0)
        k_gain: homeostatic restoring gain (0 = disabled, default 1.0 — a gentle
            nudge that keeps rates in-band and synchrony low). This is a one-sided
            damper, not a bidirectional rate-setpoint controller: it can suppress
            firing below baseline but cannot reliably drive it above baseline
            (verified: the activity trace r settles well above any small r_star,
            so g=clip(k_gain*(r_star-r)) stays negative in practice; raising
            r_star/g_max doesn't change this). At the default tau_r_ms=300,
            suppression stays smooth up to about k_gain~1.5-2.0; beyond
            k_gain~2.5 the population enters a bursty bang-bang relaxation
            oscillation (full-silence windows recurring on a tau_r_ms-scale
            period) rather than settling to a lower mean rate. Check a
            20-100ms-windowed rate trace, not just the mean, before treating a
            high-k_gain result as a smooth target.
        g_min, g_max: clipped excitability bias bounds (default [-12, 8])
        r_max: clipped trace bound (default 1.0)

    Returns:
        (voltages, spikes, sources, diagnostics_dict) where diagnostics_dict includes
        "g_bias" (shape [n_steps, n_neurons]) and "r_trace" (shape [n_steps, n_neurons]).
    """

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (jnp.asarray(0.5, dtype=jdtype) if noise_scale is None
                  else jnp.asarray(noise_scale, dtype=jdtype))
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    weight = edges.weight.astype(jdtype)
    tau_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_ms)
    n_neurons = params.v0.shape[0]

    # Homeostasis parameters (cast to dtype)
    r_star_arr = jnp.asarray(r_star, dtype=jdtype)
    tau_r = jnp.asarray(tau_r_ms, dtype=jdtype)
    decay_r = jnp.exp(-dt / tau_r)  # unconditionally stable (in (0,1))
    alpha_arr = jnp.asarray(alpha, dtype=jdtype)
    k_gain_arr = jnp.asarray(k_gain, dtype=jdtype)
    g_min_arr = jnp.asarray(g_min, dtype=jdtype)
    g_max_arr = jnp.asarray(g_max, dtype=jdtype)
    r_max_arr = jnp.asarray(r_max, dtype=jdtype)
    # Hard state bounds (cast once; applied to the carried state every step).
    v_floor_arr = jnp.asarray(v_floor, dtype=jdtype)
    v_ceiling_arr = jnp.asarray(v_ceiling, dtype=jdtype)
    u_abs_max_arr = jnp.asarray(u_abs_max, dtype=jdtype)
    syn_abs_max_arr = jnp.asarray(syn_abs_max, dtype=jdtype)

    def _bound_state(v_s, u_s, syn_s):
        """Clamp carried emitter state to finite hard bounds (overflow/underflow guard)."""
        return (
            jnp.clip(v_s, v_floor_arr, v_ceiling_arr),
            jnp.clip(u_s, -u_abs_max_arr, u_abs_max_arr),
            jnp.clip(syn_s, -syn_abs_max_arr, syn_abs_max_arr),
        )

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    key, noise_key = jax.random.split(key)
    bulk_noise = jax.random.normal(noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype)

    # Carry includes r_i (activity trace). When ``init_state`` is given (the
    # ``final_state`` dict returned by a previous call) the carry resumes from it,
    # enabling exact pause/resume / continuous chunked simulation through the public
    # API. Otherwise it starts from params.v0/u0 with zero synapses and r=r_star.
    if init_state is not None:
        _r0 = init_state.get("r_final")
        if _r0 is None:
            _rt = jnp.asarray(init_state["r_trace"], dtype=jdtype)
            _r0 = _rt[-1] if _rt.ndim == 2 else _rt   # accept full trajectory or final vector
        init = (
            jnp.asarray(init_state["v"], dtype=jdtype),
            jnp.asarray(init_state["u"], dtype=jdtype),
            jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
            jnp.asarray(init_state["syn_state"], dtype=jdtype),
            jnp.asarray(_r0, dtype=jdtype),
        )
    else:
        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
            jnp.zeros((edges.n_edges,), dtype=jdtype),
            jnp.full((n_neurons,), r_star, dtype=jdtype),  # r_i initialized to r_star
        )

    # Homeostatic synaptic plasticity engages only when eta != 0; otherwise the
    # existing (byte-identical) non-plastic scan runs. eta is a Python-level kwarg,
    # so this is a trace-time branch (no jit conditional).
    enable_plasticity = float(eta) != 0.0
    if enable_plasticity:
        eta_arr = jnp.asarray(eta, dtype=jdtype)
        decay_x = jnp.exp(-dt / jnp.asarray(tau_x_ms, dtype=jdtype))
        w_min_arr = jnp.asarray(w_min, dtype=jdtype)
        w_max_arr = jnp.asarray(w_max, dtype=jdtype)
        if init_state is not None and "w" in init_state:
            w0 = jnp.asarray(init_state["w"], dtype=jdtype)
        else:
            w0 = weight
        if init_state is not None and "x" in init_state:
            x0 = jnp.asarray(init_state["x"], dtype=jdtype)
        else:
            x0 = jnp.zeros((n_neurons,), dtype=jdtype)
        init_p = (*init, w0, x0)
        sched_p = (jnp.zeros((int(n_steps), n_neurons), dtype=jdtype)
                   if drive_schedule is None else drive_schedule.astype(jdtype))

        def step_plastic(carry, xs_t):
            """Homeostasis + online homeostatic synaptic plasticity."""
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, r, w, x = carry
            edge_current = w * syn_state                      # plastic weights
            syn = _segment_sum(edge_current, post, n_neurons)
            g = jnp.clip(k_gain_arr * (r_star_arr - r), g_min_arr, g_max_arr)
            current_native = drive + sched_t + syn + noise_coef * noise_t + g
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            r_next = jnp.clip(r_star_arr + (r - r_star_arr) * decay_r + alpha_arr * spikes, 0.0, r_max_arr)
            x_next = x * decay_x + spikes                     # presynaptic trace
            # Homeostatic synaptic scaling: dw = eta*(r* - r_post)*x_pre, clipped.
            dw = eta_arr * (r_star_arr - r_next[post]) * x_next[pre]
            w_next = jnp.clip(w + dt * dw, w_min_arr, w_max_arr)
            v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
            return (v_reset, u_reset, spikes, syn_next, r_next, w_next, x_next), \
                   (v_reset, spikes, source_proxy, g, r_next, w_next)

        final, (voltages, spikes, sources, g_bias, r_trace, w_trace) = jax.lax.scan(
            step_plastic, init_p, xs=(sched_p, bulk_noise))
    elif drive_schedule is None:
        def step(carry, noise_t):
            """Step with homeostasis, no drive schedule."""
            v, u, prev_spikes, syn_state, r = carry

            # Synaptic current
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)

            # Homeostatic excitability bias: tax hyperactive, subsidize silent
            g = jnp.clip(k_gain_arr * (r_star_arr - r), g_min_arr, g_max_arr)

            # Effective input current
            current_native = drive + syn + noise_coef * noise_t + g

            # Izhikevich dynamics
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du

            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)

            # Reset voltage and recovery variable on spike
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)

            # Update synaptic state
            syn_next = syn_state * decay + spikes[pre]

            # Update activity trace: slow leak toward r_star, jump on spike
            r_next = jnp.clip(
                r_star_arr + (r - r_star_arr) * decay_r + alpha_arr * spikes,
                0.0,
                r_max_arr
            )

            # Hard state bounds: overflow/underflow guard (no effect in normal regime)
            v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)

            # Proxy current for field source
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)

            return (v_reset, u_reset, spikes, syn_next, r_next), (v_reset, spikes, source_proxy, g, r_next)

        final, (voltages, spikes, sources, g_bias, r_trace) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            """Step with homeostasis and drive schedule."""
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state, r = carry

            # Synaptic current
            edge_current = weight * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)

            # Homeostatic excitability bias
            g = jnp.clip(k_gain_arr * (r_star_arr - r), g_min_arr, g_max_arr)

            # Effective input current with drive schedule
            current_native = drive + sched_t + syn + noise_coef * noise_t + g

            # Izhikevich dynamics
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du

            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)

            # Reset voltage and recovery variable on spike
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)

            # Update synaptic state
            syn_next = syn_state * decay + spikes[pre]

            # Update activity trace
            r_next = jnp.clip(
                r_star_arr + (r - r_star_arr) * decay_r + alpha_arr * spikes,
                0.0,
                r_max_arr
            )

            # Hard state bounds: overflow/underflow guard (no effect in normal regime)
            v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)

            # Proxy current for field source
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)

            return (v_reset, u_reset, spikes, syn_next, r_next), (v_reset, spikes, source_proxy, g, r_next)

        final, (voltages, spikes, sources, g_bias, r_trace) = jax.lax.scan(step_sched, init, xs=(sched, bulk_noise))

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "r_trace": final[4],
    }
    if enable_plasticity:
        final_state["w"] = final[5]   # final plastic edge weights (n_edges,)
        final_state["x"] = final[6]   # final presynaptic trace (n_neurons,)

    diagnostics_dict = {
        **final_state,
        "r_final": final[4],   # final r_i (N,), distinct from the (T,N) r_trace below
        "g_bias": g_bias,
        "r_trace": r_trace,    # full per-step trajectory (n_steps, n_neurons)
    }
    if enable_plasticity:
        diagnostics_dict["w_final"] = final[5]
        diagnostics_dict["w_trace"] = w_trace   # (n_steps, n_edges) plastic-weight trajectory

    return voltages, spikes, sources, diagnostics_dict


# Default per-cell-type relative size used by HDP's size-scaled input-current
# time constant (tau_i = tau_0_ms * size_i**3). Matches the canonical E:I size
# ratio used elsewhere in jaxfne (large E somas integrate slower than small
# interneurons); not a calibrated morphological measurement.
DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE: dict[str, float] = {
    "E": 5.0,
    "PV": 1.0,
    "Inl": 1.0,
    "SST": 1.5,
    "Ing": 1.5,
    "VIP": 1.5,
}


@lru_cache(maxsize=64)
def _hdp_size_scale_array_np(
    labels: tuple[str, ...],
    overrides: tuple[tuple[str, float], ...],
    dtype_name: str,
) -> np.ndarray:
    """Python-loop lookup, cached by (labels, overrides, dtype).

    ``labels`` is the same static tuple across every ``simulate()`` call on a
    given tensor (construct-time metadata, not traced), so repeated calls --
    the common case in tuning/sweep loops -- hit this cache instead of
    re-running an O(n_neurons) Python loop every time.
    """
    table = dict(DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE)
    table.update(dict(overrides))
    return np.asarray([float(table.get(name, 1.0)) for name in labels], dtype=dtype_name)


def _hdp_size_scale_array(
    labels: tuple[str, ...],
    size_scale_by_cell_type: "Mapping[str, float] | None",
    dtype: jnp.dtype,
) -> jax.Array:
    overrides = tuple(sorted(size_scale_by_cell_type.items())) if size_scale_by_cell_type else ()
    dtype_name = jnp.dtype(dtype).name
    return jnp.asarray(_hdp_size_scale_array_np(tuple(labels), overrides, dtype_name))


def simulate_edge_recurrent_izhikevich_hdp(
    params: IzhikevichParams,
    edges: EdgeList,
    n_steps: int,
    dt_ms: float,
    key: jax.Array,
    *,
    dtype: str = "float32",
    drive_schedule: "jax.Array | None" = None,
    noise_schedule: "jax.Array | None" = None,
    silence_mask: "jax.Array | None" = None,
    noise_scale: "jax.Array | float | None" = None,
    init_state: "dict | None" = None,
    # Hidden-state Dependent Plasticity (HDP) parameters. With the default
    # zero H-driving terms, H_i stays at its 1.0 initial value and the
    # difference/product weight term is null regardless of K_HDP. This is the
    # default H-state/weight-term null, not a general full-system equivalence.
    H_min: float = 0.1,
    H_max: float = 10.0,
    tau_0_ms: float = 100.0,
    size_scale_by_cell_type: "Mapping[str, float] | None" = None,
    size_scale_override: "jax.Array | None" = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
    C_spike: float = 0.0,
    K_HDP: float = 1.0,
    K_ctrl: float = 0.0,
    K_w_ctrl: float = 0.0,
    rho_passive: float = 0.0,
    barrier_c: float = 0.0,
    barrier_d: float = 0.0,
    barrier_eps: float = 1.0e-3,
    w_floor: float = 1.0e-3,
    w_ceiling: float = 50.0,
    v_floor: float = -150.0,
    v_ceiling: float = 100.0,
    u_abs_max: float = 2000.0,
    syn_abs_max: float = 1.0e4,
    record_dH_components: bool = False,
    record_edge_current: bool = False,
    record_weight_trace: bool = True,
    H_boost_gain: float = 0.0,
    hdp_rule: str = "signed_linear",
    h_state_dim: int = 1,
    h_state_readout: "jax.Array | None" = None,
    h_state_coupling: "jax.Array | None" = None,
    h_state_locality: "str | None" = None,
    controller_B: "jax.Array | None" = None,
    controller_lambda: float | None = None,
    controller_tau_H_s: float | None = None,
    controller_tau_theta_s: float | None = None,
    controller_rate_setpoint_E_hz: float | None = None,
    controller_rate_setpoint_I_hz: float | None = None,
    controller_theta_S_init: "Sequence[float] | None" = None,
    m_ei_edge_mask: "jax.Array | None" = None,
    e_neuron_mask: "jax.Array | None" = None,
    theta_m_EI_bounds: "tuple[float, float]" = (0.1, 5.0),
    theta_eta_a_bounds: "tuple[float, float]" = (0.25, 4.0),
    enable_boundary_stabilization: bool = False,
    tau_r_s: float = 0.3,
    tau_H_E_s: float = 4.0,
    tau_H_I_s: float = 1.0,
    K_H: float = 0.1,
    g_H: float = 0.22,
    k_L: float = 1.0,
    k_H: float = 1.0,
    beta_softplus: float = 25.0,
    r_L: float = 0.5,
    r_H: float = 20.0,
    r_bar_init: "float | None" = 8.0,
    record_boundary_components: bool = False,
    step_indices: "jax.Array | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simulate Izhikevich emitters with sparse recurrent synapses and HDP.

    **Hidden-state Dependent Plasticity (HDP)** carries per-neuron Relative
    Biophysical State (RBS) coordinates ``H_i`` (public/API field names retain
    ``H``/``h_state_*`` for compatibility). The legacy scalar state has shape
    ``(n_neurons,)``; a generalized state has shape ``(n_neurons, h_state_dim)``.
    Componentwise RBS dynamics are the default, with optional explicit linear
    coupling. The current weight rules consume a configurable readout of the
    generalized state; this is a compatibility projection, not the definition
    of RBS.

        I_syn_i = sum_j w_ji * x_j                              (incoming synaptic current; this module's existing ``syn``)
        W_i     = sum_j |w_ij|                                  (i's own outgoing synaptic burden)
        C(H_i)  = barrier_c/(H_i-H_min) + barrier_d/(H_max-H_i) (asymmetric safety barrier, H_min<H_i<H_max)
        tau_i * dH_i/dt = alpha*I_syn_i + beta - gamma*H_i*r_i - delta*W_i
                          + rho_passive/H_i**2 - dC/dH_i        (r_i = previous step's spike indicator)
        H_i    <- H_i - C_spike                                 (discrete drain on H_i when i itself spikes)

    ``hdp_rule`` determines the weight-magnitude update family (default
    "signed_linear").  Let ``m = abs(w)`` and
    ``delta_H = H_post - H_pre``:

        difference family:
          signed_linear:    dm_E/dt = +K_HDP * delta_H * m_E
                            dm_I/dt = -K_HDP * delta_H * m_I
          signed_quadratic: dm_E/dt = +K_HDP * delta_H*abs(delta_H) * m_E
                            dm_I/dt = -K_HDP * delta_H*abs(delta_H) * m_I

        product modulation:
          hebbian_product:  dm_E/dt = +K_HDP * H_pre * H_post * m_E
                            dm_I/dt = -K_HDP * H_pre * H_post * m_I

    ``signed_linear`` and ``signed_quadratic`` are the difference-family HDP
    rules.  ``hebbian_product`` is a separate product modulation: it does not
    compare pre/post RBS coordinates and is not sign-equivalent to either
    difference rule.

    ``gamma*H_i*r_i`` is an H-taxed output drain: firing costs more for neurons
    with higher H_i (resource-abundant neurons can sustain more activity without
    resource cost, but paying the full ``gamma*H_i*r_i`` when active ensures
    resource balance). The passive-income term ``rho_passive/H_i**2`` provides
    a restoring force toward ``H_i=1`` without the explicit linear controller
    ``K_ctrl*(1-H_i)`` -- low H receives stronger push than high H.
    ``C(H_i)`` is a separate, asymmetric double-barrier safety potential
    (not a controller in its own right -- ``-dC/dH_i`` is the corresponding
    restoring force): it repels H_i from both ``H_min`` and ``H_max`` but does
    *not* by itself define the equilibrium, since for ``H_min`` far below
    ``H*=1`` and ``H_max`` far above it, placing the *minimum of C* exactly at
    ``H*=1`` forces ``barrier_d/barrier_c = ((H_max-H*)/(H*-H_min))**2``
    (with the canonical defaults, ``=100``) -- a large, deliberately
    asymmetric ratio (gentle push near the floor where small deviations
    are normal, increasingly strong rescue only very close to ``H_min``,
    and comparably gentle taxation near ``H_max`` since resource surplus
    is not pathological the way near-collapse is). Both ``rho_passive`` and
    ``barrier_c``/``barrier_d`` default to 0.0 (no contribution; fully
    backward compatible with the income/spending-only kernel above).
    ``barrier_eps`` floors the ``(H_i-H_min)``/``(H_max-H_i)`` denominators
    to avoid a divide-by-zero singularity at the exact clamp boundary.
    ``K_ctrl`` is a live linear restoring term ``K_ctrl*(1-H_i)`` (REVIVED
    2026-07-01, F-017/F-019: previously dead code -- ``rho_passive/H_i**2``
    alone is >=0 everywhere and cannot pull ``H_i`` back down from above
    ``H*=1``, confirmed via a full 20s/5-seed sweep failing at all 15
    candidates; ``K_ctrl*(1-H_i)`` is genuinely two-sided and closes that
    gap). ``K_ctrl=0.0`` (default) is the null control -- no behavior change
    unless explicitly set.

    ``pre`` and ``post`` index the presynaptic and postsynaptic endpoints in
    the weight update. ``K_HDP`` is a single global gain for the
    homeostatic-difference-driven term (or the separate product-modulation
    term when ``hdp_rule="hebbian_product"``): ``K_HDP=0`` makes that
    plasticity term null, but does not disable the H equation or the
    independent ``K_w_ctrl`` weight-restoration term. ``K_HDP<0`` is an
    explicit anti-homeostatic stress-test mode, and ``|K_HDP|>1``/``<1``
    over/under-weights the selected term.

    H_i is a resource-capacity reading, not a stress accumulator: synaptic
    *input* (``alpha*I_syn_i``) raises it (income), while the neuron's own
    *output* -- its recent firing (``gamma*H_i*r_i``, scaled by own resource
    level) and the synaptic weight it must maintain (``delta*W_i``) -- drains
    it (spending), plus the discrete ``C_spike`` drain on a spike. Sign check
    for stability: an overactive neuron spends faster than it earns, so
    ``H_i`` falls below 1; this must *weaken* its excitatory weights and
    *strengthen* its inhibitory weights to correct the overactivity. With the
    ``hdp_rule="signed_linear"`` (the default), a lower-H target has
    ``H_post-H_pre < 0``, so ``dm_E/dt < 0`` (weakens excitation) and
    ``dm_I/dt > 0`` (strengthens inhibitory magnitude) -- the restoring
    direction. The ``signed_quadratic`` rule preserves this orientation while
    scaling the difference quadratically. ``hebbian_product`` is different:
    for positive H and positive ``K_HDP`` it increases excitatory magnitude and
    decreases inhibitory magnitude unless another term or a bound opposes it.

    Update order per step (as specified): (1) synaptic current, (2) update
    H_i, (3) update plastic weights from the updated H_i, (4) integrate the
    neuron (Izhikevich + spike detection), (5) spikes consume H_i. Step 2
    uses the *previous* step's spikes for ``r_i`` so the H_i update and the
    weight update (steps 2-3) do not depend on this step's own spike
    outcome, which is only known after step 4.

    jaxfne's native synapse model is current-based: each edge carries one
    signed scalar weight (``edges.weight``) added directly to input current,
    with a binary ``edges.receptor_index`` (0 = excitatory, 1 = inhibitory).
    The four declarative :func:`standard_receptor_specs` receptor classes
    (AMPA, NMDA, GABA_A, GABA_B) are metadata only and are not instantiated as
    separate synapse populations in the constructed network, and there is no
    conductance/reversal-potential term in the dynamics. HDP's excitatory
    weight class is therefore ``receptor_index == 0`` (the AMPA+NMDA role) and
    its inhibitory weight class is ``receptor_index == 1`` (the GABA_A+GABA_B
    role) -- the weight ODEs operate on the edge's existing signed native
    weight, not on a separately-modeled conductance. ``W_i`` is computed from
    a neuron's *outgoing* edges (``edges.pre == i``); ``I_syn_i`` from its
    *incoming* edges (``edges.post == i``, the existing ``syn`` term).

    Args:
        params, edges, n_steps, dt_ms, key: as in simulate_edge_recurrent_izhikevich
        dtype, drive_schedule, silence_mask, noise_scale: as in simulate_edge_recurrent_izhikevich
        H_min, H_max: clamp bounds for H_i (default [0.1, 10.0])
        tau_0_ms: base H_i integration time constant (default 100 ms);
            per-neuron tau_i = tau_0_ms * size_i**3 (larger/slower for E,
            faster for PV, matching the existing size-scaling table; e.g.
            size_i=2.0 -> tau_i = 8 * tau_0_ms)
        size_scale_by_cell_type: override the default per-cell-type relative
            size table (see DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE)
        size_scale_override: optional explicit per-neuron size array
            (n_neurons,), e.g. computed by the caller from each neuron's
            (layer, cell_type) so size can vary by layer (the cell-type-only
            table above cannot express that, since params.labels carries
            only cell-type identity). When given, takes precedence over
            size_scale_by_cell_type entirely.
        alpha: H_i income gain on incoming synaptic current (default 0.0)
        beta: constant H_i bias term (default 0.0)
        gamma: H_i spending gain on the neuron's own (previous-step) firing
            (default 0.0)
        delta: H_i spending gain on the neuron's own outgoing synaptic
            weight burden W_i (default 0.0)
        C_spike: discrete H_i drain charged when the neuron itself spikes,
            not scaled by tau_i (default 0.0)
        K_HDP: global gain on the selected weight-modulation term (default
            1.0; 0.0 nulls that term, negative is anti-homeostatic for the
            difference family)
        K_ctrl: linear restoring-force gain, dH/dt += K_ctrl*(1-H_i) (default
            0.0, null control). Two-sided: pulls H_i up when below 1, down
            when above -- unlike rho_passive (always >=0, floor-only rescue).
        K_w_ctrl: linear restoring-force gain on the synaptic weight
            magnitude, dwmag/dt += K_w_ctrl*(wmag_baseline - wmag), where
            wmag_baseline = |edges.weight| (the network's originally-declared
            wiring, captured once from the ``edges`` argument -- NOT the
            carried/chained weight from a prior trial). Default 0.0, null
            control -- no behavior change unless explicitly set. Added
            2026-07-04 to close the same structural gap K_ctrl closed for
            H_i: previously the only thing bounding wmag was the hard
            [w_floor, w_ceiling] clip, with no restoring pull back toward
            the calibrated baseline, so chaining HDP across trials (each
            trial's final weights feeding the next via
            Model.with_hdp_initial_state) compounds unbounded drift with no
            ceiling until the hard clip saturates every edge (verified:
            reproduces the repo's chained-multi-trial HDP weight-runaway,
            see scripts/v1_pfc_continuous_aaab_smoke_test.py's carry_weights
            note). Sign-agnostic: applies identically to excitatory and
            inhibitory edges since it targets the unsigned magnitude.
        rho_passive: passive-income gain on H_i (default 0.0; positive values
            add rho_passive/H_i**2 to dH_i/dt, pulling H_i toward 1 without
            an explicit linear controller)
        hdp_rule: weight-update rule family (default "signed_linear"):
            difference family:
              "signed_linear": dw_mag ~ (H_post - H_pre)
              "signed_quadratic": dw_mag ~ (H_post - H_pre)|H_post - H_pre|
            separate product modulation:
              "hebbian_product": dw_mag ~ H_pre * H_post
        barrier_c, barrier_d: asymmetric double-barrier safety-potential
            coefficients repelling H_i from H_min/H_max respectively
            (default 0.0/0.0, no contribution); for the minimum of
            C(H)=barrier_c/(H-H_min)+barrier_d/(H_max-H) to coincide with
            H*=1 requires barrier_d/barrier_c=((H_max-1)/(1-H_min))**2 (100
            at the canonical H_min=0.1/H_max=10.0) -- but barrier_c/d are
            meant only as a safety constraint, not the equilibrium
            definition; use rho_passive for that. NOTE: hdp_network.py's
            DEFAULT_HDP preset sets barrier_c=barrier_d=0.01 (ratio 1, not
            the 100 this equilibrium condition calls for) -- a real gap
            between this docstring's stated requirement and the shipped
            preset, left as-is since the tuned/verified DEFAULT_HDP dynamics
            keep H tightly pinned near H*=1 in practice (H rarely nears
            either boundary), so the asymmetry is dormant, not exercised --
            external review 2026-07-14; re-tune before relying on the
            barrier near a boundary.
        barrier_eps: floor on the barrier denominators (default 1e-3)
        w_floor, w_ceiling: clip bounds for edge weight magnitude (default
            [1e-3, 50.0]; prevents collapse-to-zero and unbounded divergence)
        v_floor, v_ceiling, u_abs_max, syn_abs_max: hard numerical-stability
            bounds, as in simulate_edge_recurrent_izhikevich_homeostatic
        record_dH_components: if True, also return per-step, per-neuron
            decomposition of dH_i/dt's five additive terms -- income
            (alpha*I_syn), H-taxed rate-spending (-gamma*H_i*r), weight-spending
            (-delta*W), passive-income restoring (rho_passive/H_i**2), and the
            barrier force -- as "dH_income_trace"/"dH_rate_trace"/
            "dH_weight_trace"/"dH_passive_trace"/"dH_barrier_trace" (each
            ``(n_steps, n_neurons)`` for scalar H or
            ``(n_steps, n_neurons, h_state_dim)`` for vector H) in
            diagnostics_dict. Default False (no extra
            compute/memory); for isolating which term drives an observed
            H/weight runaway.
        record_edge_current: if True, also return the per-step, per-edge
            synaptic current contribution ``w * syn_state`` (the summand
            that ``segment_sum`` aggregates by post-neuron into ``syn``,
            i.e. into ``dH_income``'s ``alpha*I_syn`` term) as
            "edge_current_trace" (n_steps, n_edges) in diagnostics_dict.
            Default False; combine with edges.pre/edges.post and cell-type
            labels post-hoc to decompose I_syn by connection class
            (E->E, E->PV, PV->E, ...) and find which synaptic pathway
            drives an income-term runaway.
        record_weight_trace: if False, do not stack the per-step, per-edge
            plastic-weight snapshot into "w_trace" -- diagnostics_dict["w_trace"]
            is then None (diagnostics_dict["w_final"] is always the correct
            terminal weight state regardless). Default True (existing
            behavior, matches the documented ``Model.last_hdp_diagnostics()``
            contract). The kernel's actual H/weight dynamics are identical
            either way -- this only controls what gets returned/stacked, not
            what's computed. Set False for large N and/or long durations:
            "w_trace" is (n_steps, n_edges), which dominates memory at scale
            (e.g. 10,000 steps x 2,000,000 edges x 4 bytes = 80GB -- a real
            reproduced OOM at N=20,000/5000ms; "H_trace"/"voltages"/"spikes"
            are only (n_steps, n_neurons), ~100x smaller at typical
            max_in_degree and not the source of this OOM).
        H_boost_gain: homeostatic drive compensation -- scales each
            neuron's (drive + sched_t) input by
            ``1 + H_boost_gain * max(0, 1 - H)`` using the carry's
            incoming (previous-step) H_i, so a neuron starved below its
            H=1.0 equilibrium receives a proportionally larger drive.
            Default 0.0 reproduces existing (unboosted) behavior exactly.
        h_state_dim: number of H coordinates. ``1`` preserves the legacy
            external shape ``(n_neurons,)``; values greater than one use
            ``(n_neurons, h_state_dim)``.
        h_state_readout: optional linear readout vector used by the current
            scalar drive/weight rules. Defaults to the first coordinate.
        h_state_coupling: optional ``(h_state_dim, h_state_dim)`` matrix added
            to the componentwise H derivative. Omitted means zero coupling.

    Returns:
        (voltages, spikes, sources, diagnostics_dict) where diagnostics_dict
        includes "H_trace" (n_steps, n_neurons) for scalar H or
        (n_steps, n_neurons, h_state_dim) for vector H, "w_trace"
        (n_steps, n_edges)
        or None if record_weight_trace=False, and "*_final" vectors (always
        present regardless of record_weight_trace).
    """

    jdtype = _dtype_from_policy(dtype)
    a = params.a.astype(jdtype)
    b = params.b.astype(jdtype)
    c = params.c.astype(jdtype)
    d = params.d.astype(jdtype)
    drive = params.drive.astype(jdtype)
    source_scale = params.source_scale.astype(jdtype)
    dt = jnp.asarray(dt_ms, dtype=jdtype)
    noise_coef = (jnp.asarray(0.5, dtype=jdtype) if noise_scale is None
                  else jnp.asarray(noise_scale, dtype=jdtype))
    pre = edges.pre.astype(jnp.int32)
    post = edges.post.astype(jnp.int32)
    tau_syn_ms = jnp.maximum(edges.tau_ms.astype(jdtype), jnp.asarray(1e-6, dtype=jdtype))
    decay = jnp.exp(-dt / tau_syn_ms)
    n_neurons = params.v0.shape[0]
    exc_mask = (edges.receptor_index.astype(jnp.int32) == 0)

    if isinstance(h_state_dim, bool) or not isinstance(h_state_dim, (int, np.integer)):
        raise ValueError("h_state_dim must be a positive integer")
    h_dim = int(h_state_dim)
    if h_dim < 1:
        raise ValueError("h_state_dim must be a positive integer")

    _adaptive_hp = {
        "hdp_rule": hdp_rule,
        "h_state_locality": h_state_locality,
        "h_state_dim": h_dim,
        "controller_B": controller_B,
        "controller_lambda": controller_lambda if controller_lambda is not None else 0.45,
        "controller_tau_H_s": controller_tau_H_s if controller_tau_H_s is not None else 0.2,
        "controller_tau_theta_s": controller_tau_theta_s if controller_tau_theta_s is not None else 2.0,
        "controller_rate_setpoint_E_hz": controller_rate_setpoint_E_hz,
        "controller_rate_setpoint_I_hz": controller_rate_setpoint_I_hz,
        "controller_theta_S_init": controller_theta_S_init,
        "m_ei_edge_mask": m_ei_edge_mask,
        "e_neuron_mask": e_neuron_mask,
        "theta_m_EI_bounds": theta_m_EI_bounds,
        "theta_eta_a_bounds": theta_eta_a_bounds,
    }
    locality = resolve_h_state_locality(_adaptive_hp)
    pop_layout = None
    theta_lo = theta_hi = None
    dt_s = None
    if locality == "population":
        pop_layout = parse_population_restoring_layout(
            _adaptive_hp,
            edges_weight=edges.weight,
            labels=params.labels,
            dtype=jdtype,
        )
        theta_lo, theta_hi = theta_bounds(pop_layout, dtype=jdtype)
        dt_s = dt / jnp.asarray(1000.0, dtype=jdtype)

    def _h_component_param(value: Any, name: str) -> jax.Array:
        arr = jnp.asarray(value, dtype=jdtype)
        if arr.ndim == 0:
            return arr
        if arr.shape == (h_dim,):
            return arr
        raise ValueError(
            f"{name} must be scalar or have shape ({h_dim},), got {arr.shape}"
        )

    if h_state_readout is None:
        readout = jnp.zeros((h_dim,), dtype=jdtype).at[0].set(1.0)
    else:
        readout = jnp.asarray(h_state_readout, dtype=jdtype)
        if readout.shape != (h_dim,):
            raise ValueError(
                "h_state_readout must have shape "
                f"({h_dim},), got {readout.shape}"
            )
    if h_state_coupling is None:
        coupling = jnp.zeros((h_dim, h_dim), dtype=jdtype)
    else:
        coupling = jnp.asarray(h_state_coupling, dtype=jdtype)
        if coupling.shape != (h_dim, h_dim):
            raise ValueError(
                "h_state_coupling must have shape "
                f"({h_dim}, {h_dim}), got {coupling.shape}"
            )

    if size_scale_override is not None:
        size_arr = jnp.asarray(size_scale_override, dtype=jdtype)
    else:
        size_arr = _hdp_size_scale_array(params.labels, size_scale_by_cell_type, jdtype)
    tau_i = jnp.asarray(tau_0_ms, dtype=jdtype) * size_arr * size_arr * size_arr
    tau_i = jnp.maximum(tau_i, jnp.asarray(1e-6, dtype=jdtype))

    H_min_arr = _h_component_param(H_min, "H_min")
    H_max_arr = _h_component_param(H_max, "H_max")
    alpha_arr = jnp.asarray(alpha, dtype=jdtype)
    beta_arr = jnp.asarray(beta, dtype=jdtype)
    gamma_arr = jnp.asarray(gamma, dtype=jdtype)
    delta_arr = jnp.asarray(delta, dtype=jdtype)
    C_spike_arr = _h_component_param(C_spike, "C_spike")
    K_HDP_arr = jnp.asarray(K_HDP, dtype=jdtype)
    K_ctrl_arr = jnp.asarray(K_ctrl, dtype=jdtype)  # Live linear restoring term (revived 2026-07-01)
    K_w_ctrl_arr = jnp.asarray(K_w_ctrl, dtype=jdtype)  # Weight restoring term (added 2026-07-04)
    wmag_baseline_arr = jnp.abs(edges.weight).astype(jdtype)  # Calibrated wiring, not the carried w
    rho_passive_arr = _h_component_param(rho_passive, "rho_passive")
    barrier_c_arr = _h_component_param(barrier_c, "barrier_c")
    barrier_d_arr = _h_component_param(barrier_d, "barrier_d")
    barrier_eps_arr = jnp.asarray(barrier_eps, dtype=jdtype)
    w_floor_arr = jnp.asarray(w_floor, dtype=jdtype)
    w_ceiling_arr = jnp.asarray(w_ceiling, dtype=jdtype)
    v_floor_arr = jnp.asarray(v_floor, dtype=jdtype)
    v_ceiling_arr = jnp.asarray(v_ceiling, dtype=jdtype)
    u_abs_max_arr = jnp.asarray(u_abs_max, dtype=jdtype)
    syn_abs_max_arr = jnp.asarray(syn_abs_max, dtype=jdtype)
    H_boost_gain_arr = jnp.asarray(H_boost_gain, dtype=jdtype)

    if enable_boundary_stabilization:
        c_val = 0.01 if barrier_c == 0.0 and barrier_d == 0.0 else barrier_c
        d_val = 1.00 if barrier_c == 0.0 and barrier_d == 0.0 else barrier_d
        c_arr = _h_component_param(c_val, "barrier_c")
        d_arr = _h_component_param(d_val, "barrier_d")
        K_H_arr = jnp.asarray(K_H, dtype=jdtype)
        g_H_arr = jnp.asarray(g_H, dtype=jdtype)
        k_L_arr = jnp.asarray(k_L, dtype=jdtype)
        k_H_arr = jnp.asarray(k_H, dtype=jdtype)
        beta_arr = jnp.asarray(beta_softplus, dtype=jdtype)
        r_L_arr = jnp.asarray(r_L, dtype=jdtype)
        r_H_arr = jnp.asarray(r_H, dtype=jdtype)
        tau_r_arr = jnp.asarray(tau_r_s, dtype=jdtype)
        if e_neuron_mask is not None:
            e_mask = jnp.asarray(e_neuron_mask, dtype=jdtype)
        elif params.labels is not None and len(params.labels) == n_neurons:
            e_mask = jnp.array([str(lbl).startswith("E") for lbl in params.labels], dtype=jdtype)
        else:
            e_mask = jnp.ones((n_neurons,), dtype=jdtype)
        tau_H_arr = jnp.where(e_mask > 0.5, tau_H_E_s, tau_H_I_s).astype(jdtype)

    def _bound_state(v_s, u_s, syn_s):
        """Clamp carried emitter state to finite hard bounds (overflow/underflow guard)."""
        return (
            jnp.clip(v_s, v_floor_arr, v_ceiling_arr),
            jnp.clip(u_s, -u_abs_max_arr, u_abs_max_arr),
            jnp.clip(syn_s, -syn_abs_max_arr, syn_abs_max_arr),
        )

    if silence_mask is not None:
        s_mask = silence_mask.astype(jdtype)
    else:
        s_mask = jnp.ones(params.v0.shape[0], dtype=jdtype)

    if noise_schedule is None:
        key, noise_key = jax.random.split(key)
        bulk_noise = jax.random.normal(
            noise_key, shape=(int(n_steps), params.v0.shape[0]), dtype=jdtype
        )
    else:
        bulk_noise = jnp.asarray(noise_schedule, dtype=jdtype)
        expected_noise_shape = (int(n_steps), int(n_neurons))
        if bulk_noise.shape != expected_noise_shape:
            raise ValueError(
                "noise_schedule must have shape "
                f"{expected_noise_shape}, got {bulk_noise.shape}"
            )
    sched = (jnp.zeros((int(n_steps), n_neurons), dtype=jdtype)
             if drive_schedule is None else drive_schedule.astype(jdtype))

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host < 0):
        raise ValueError("edge delay_steps must be >= 0")
    has_nonzero_delay = bool(np.any(delay_host > 0))
    if has_nonzero_delay:
        max_delay = int(np.max(delay_host))
        bufsize = max_delay + 1
        delay_steps_arr = edges.delay_steps.astype(jnp.int32)
        time_step_offset = _rbd_continuation_step_offset_array(init_state)
        if init_state is not None and ("delay_state" in init_state or "spike_history" in init_state):
            _validate_delayed_init_state(
                init_state, bufsize=bufsize, n_neurons=n_neurons, n_edges=edges.n_edges
            )
            spike_hist0 = _rbd_delay_state_from_init(
                init_state, bufsize=bufsize, n_neurons=n_neurons, jdtype=jdtype
            )
        else:
            spike_hist0 = jnp.zeros((bufsize, n_neurons), dtype=jdtype)
        if step_indices is not None:
            step_indices_arr = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
        else:
            off = time_step_offset
            if isinstance(off, int):
                step_indices_arr = jnp.arange(off, off + int(n_steps), dtype=jnp.int32)
            else:
                step_indices_arr = jnp.arange(
                    off, off + jnp.asarray(int(n_steps), dtype=jnp.int32), dtype=jnp.int32
                )

    if pop_layout is not None:
        expected_h_shape_pop = _expected_h_shape(
            locality="population", n_neurons=n_neurons, h_state_dim=h_dim
        )
        theta_default = initial_theta_vector(pop_layout, dtype=jdtype)
        if init_state is not None:
            H0 = jnp.asarray(
                init_state.get("H_final", jnp.zeros(expected_h_shape_pop, dtype=jdtype)),
                dtype=jdtype,
            )
            if H0.shape != expected_h_shape_pop:
                raise ValueError(
                    "H_final must have shape "
                    f"{expected_h_shape_pop} for population H, got {H0.shape}"
                )
            theta0 = jnp.asarray(
                init_state.get("theta_S_final", theta_default), dtype=jdtype
            )
            if theta0.shape != (len(pop_layout.channels),):
                raise ValueError(
                    f"theta_S_final must have shape ({len(pop_layout.channels)},), "
                    f"got {theta0.shape}"
                )
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                H0,
                theta0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros_like(params.v0, dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                jnp.zeros(expected_h_shape_pop, dtype=jdtype),
                jnp.clip(theta_default, theta_lo, theta_hi),
            )
    elif enable_boundary_stabilization:
        expected_h_shape = _expected_h_shape(
            locality="node", n_neurons=n_neurons, h_state_dim=h_dim
        )
        if init_state is not None and "r_bar" in init_state:
            r_bar0 = jnp.asarray(init_state["r_bar"], dtype=jdtype)
        elif init_state is not None and "r_bar_final" in init_state:
            r_bar0 = jnp.asarray(init_state["r_bar_final"], dtype=jdtype)
        else:
            init_r = 8.0 if r_bar_init is None else float(r_bar_init)
            r_bar0 = jnp.full((n_neurons,), init_r, dtype=jdtype)

        if init_state is not None:
            H0 = jnp.asarray(
                init_state.get("H_final", jnp.ones(expected_h_shape, dtype=jdtype)),
                dtype=jdtype,
            )
            if H0.shape != expected_h_shape:
                raise ValueError(
                    "H_final must have shape "
                    f"{expected_h_shape} for h_state_dim={h_dim}, got {H0.shape}"
                )
            w0 = jnp.asarray(init_state.get("w_final", edges.weight), dtype=jdtype)
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                H0, w0, r_bar0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros_like(params.v0, dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                jnp.ones(expected_h_shape, dtype=jdtype), # H_i(0) = 1.0
                edges.weight.astype(jdtype),              # w(0) = native edge weight
                r_bar0,
            )
    else:
        expected_h_shape = _expected_h_shape(
            locality="node", n_neurons=n_neurons, h_state_dim=h_dim
        )
        if init_state is not None:
            H0 = jnp.asarray(
                init_state.get("H_final", jnp.ones(expected_h_shape, dtype=jdtype)),
                dtype=jdtype,
            )
            if H0.shape != expected_h_shape:
                raise ValueError(
                    "H_final must have shape "
                    f"{expected_h_shape} for h_state_dim={h_dim}, got {H0.shape}"
                )
            w0 = jnp.asarray(init_state.get("w_final", edges.weight), dtype=jdtype)
            init = (
                jnp.asarray(init_state["v"], dtype=jdtype),
                jnp.asarray(init_state["u"], dtype=jdtype),
                jnp.asarray(init_state["prev_spikes"], dtype=jdtype),
                jnp.asarray(init_state["syn_state"], dtype=jdtype),
                H0, w0,
            )
        else:
            init = (
                params.v0.astype(jdtype),
                params.u0.astype(jdtype),
                jnp.zeros_like(params.v0, dtype=jdtype),
                jnp.zeros((edges.n_edges,), dtype=jdtype),
                jnp.ones(expected_h_shape, dtype=jdtype), # H_i(0) = 1.0
                edges.weight.astype(jdtype),              # w(0) = native edge weight
            )

    if has_nonzero_delay:
        init = init + (spike_hist0,)

    def step(carry, xs_t):
        """HDP step: population restoring or node-local income/spending plasticity."""
        if has_nonzero_delay:
            t_idx, sched_t, noise_t = xs_t
            spike_hist = carry[-1]
            carry_core = carry[:-1]
        else:
            sched_t, noise_t = xs_t
            carry_core = carry

        if pop_layout is not None:
            v, u, prev_spikes, syn_state, H_pop, theta_S = carry_core
            w_eff, a_eff = bind_theta_to_plant(
                theta_S,
                pop_layout,
                a_base=a,
                w_ceiling=w_ceiling_arr,
            )
            edge_current = w_eff * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = (drive + sched_t) * s_mask + syn + noise_coef * noise_t
            e_vec = population_rate_error(
                prev_spikes, pop_layout, dt_ms=dt, dtype=jdtype
            )
            dH, d_theta = population_restoring_derivatives(
                H_pop, e_vec, pop_layout, dtype=jdtype
            )
            H_next = H_pop + dt_s * dH
            theta_next = jnp.clip(theta_S + dt_s * d_theta, theta_lo, theta_hi)
            dv, du = _izhikevich_dv_du(v, u, current_native, a_eff, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            if has_nonzero_delay:
                presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps_arr)
                syn_next = syn_state * decay + presyn
                slot = jnp.mod(t_idx, bufsize)
                spike_hist_next = spike_hist.at[slot].set(spikes)
            else:
                syn_next = syn_state * decay + spikes[pre]
            v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            if record_weight_trace:
                outputs = (v_reset, spikes, source_proxy, H_next, theta_next, w_eff)
            else:
                outputs = (v_reset, spikes, source_proxy, H_next, theta_next)
            carry_out = (v_reset, u_reset, spikes, syn_next, H_next, theta_next)
            if has_nonzero_delay:
                carry_out = carry_out + (spike_hist_next,)
            return carry_out, outputs

        if enable_boundary_stabilization:
            v, u, prev_spikes, syn_state, H, w, r_bar = carry_core

            # (1) Synaptic current and native input
            edge_current = w * syn_state
            syn = _segment_sum(edge_current, post, n_neurons)
            current_native = (drive + sched_t) + syn + noise_coef * noise_t

            # (2) Homeostatic current from h = H - 1
            h = H - 1.0
            I_H = -g_H_arr * h
            current_total = current_native + I_H

            # (3) Integrate neuron (Izhikevich)
            dv, du = _izhikevich_dv_du(v, u, current_total, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            if has_nonzero_delay:
                presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps_arr)
                syn_next = syn_state * decay + presyn
                slot = jnp.mod(t_idx, bufsize)
                spike_hist_next = spike_hist.at[slot].set(spikes)
            else:
                syn_next = syn_state * decay + spikes[pre]

            # (4) Update rate filter: r_bar_{n+1} = r_bar_n + (dt_s / tau_r) * (spikes / dt_s - r_bar_n)
            dt_s = dt / jnp.asarray(1000.0, dtype=jdtype)
            r_inst = spikes / dt_s
            r_bar_next = r_bar + (dt_s / tau_r_arr) * (r_inst - r_bar)

            # (5) Boundary drives S_L and S_H
            S_L = k_L_arr * (jax.nn.softplus(beta_arr * (r_L_arr - r_bar_next)) / beta_arr)
            S_H = k_H_arr * (jax.nn.softplus(beta_arr * (r_bar_next - r_H_arr)) / beta_arr)

            # (6) Barrier force: -B'(h) = c/(h+0.9)^2 - d/(9-h)^2
            dist_floor = jnp.clip(h + 0.9, barrier_eps_arr, None)
            dist_ceil = jnp.clip(9.0 - h, barrier_eps_arr, None)
            minus_B_prime = (c_arr / (dist_floor * dist_floor)) - (d_arr / (dist_ceil * dist_ceil))

            # (7) dh/dt = (-K_H * h - S_L + S_H - B'(h)) / tau_H
            dh = (-K_H_arr * h - S_L + S_H + minus_B_prime) / tau_H_arr
            h_next = h + dt_s * dh
            h_next = jnp.clip(h_next, -0.9 + barrier_eps_arr, 9.0 - barrier_eps_arr)
            H_next = h_next + 1.0

            # (8) Plastic weights
            H_pre = H_next[pre]
            H_post = H_next[post]
            if hdp_rule == "signed_linear":
                rule_basis = H_post - H_pre
            elif hdp_rule == "signed_quadratic":
                diff = H_post - H_pre
                rule_basis = diff * jnp.abs(diff)
            elif hdp_rule == "hebbian_product":
                rule_basis = H_pre * H_post
            else:
                rule_basis = H_post - H_pre

            wmag = jnp.abs(w)
            dw_exc = K_HDP_arr * rule_basis * wmag
            dw_inh = -K_HDP_arr * rule_basis * wmag
            dw_w_ctrl = K_w_ctrl_arr * (wmag_baseline_arr - wmag)
            dw = jnp.where(exc_mask, dw_exc, dw_inh) + dw_w_ctrl
            wmag_next = jnp.clip(wmag + dt * dw, w_floor_arr, w_ceiling_arr)
            w_next = jnp.where(exc_mask, wmag_next, -wmag_next)

            v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)
            source_proxy = _source_proxy_from_components(current_total, spikes, source_scale, dtype=jdtype)

            if record_weight_trace:
                outputs = (v_reset, spikes, source_proxy, H_next, w_next, r_bar_next, I_H)
            else:
                outputs = (v_reset, spikes, source_proxy, H_next, r_bar_next, I_H)
            if record_boundary_components:
                outputs = outputs + (S_L, S_H, minus_B_prime, dh)
            if record_edge_current:
                outputs = outputs + (edge_current,)
            carry_out = (v_reset, u_reset, spikes, syn_next, H_next, w_next, r_bar_next)
            if has_nonzero_delay:
                carry_out = carry_out + (spike_hist_next,)
            return carry_out, outputs

        v, u, prev_spikes, syn_state, H, w = carry_core

        # (1) Synaptic current.
        edge_current = w * syn_state
        syn = _segment_sum(edge_current, post, n_neurons)
        wmag = jnp.abs(w)
        W_burden = _segment_sum(wmag, pre, n_neurons)
        # NOTE: uses the carry (previous-step) H, one step lagged behind H_next
        # computed below in (2) -- negligible at small dt but not exact. Also
        # note dH_income below is alpha*syn only, so the extra current this
        if h_dim == 1:
            h_readout = H
            syn_h = syn
            prev_spikes_h = prev_spikes
            W_burden_h = W_burden
        else:
            h_readout = H @ readout
            syn_h = syn[:, None]
            prev_spikes_h = prev_spikes[:, None]
            W_burden_h = W_burden[:, None]
        boost = 1.0 + H_boost_gain_arr * jnp.maximum(0.0, 1.0 - h_readout)
        current_native = (drive + sched_t) * boost + syn + noise_coef * noise_t

        # (2) Update H_i: income from incoming synaptic current, spending
        # from the neuron's own previous-step firing and outgoing weight
        # burden (prev_spikes avoids circularity with this step's spikes,
        # which are only known after step 4). Passive income restores H toward
        # 1 without an explicit linear controller (rho_passive/H_i**2).
        # W_burden is an abs-sum over a neuron's outgoing edges (E and I
        # pooled) -- intentional metabolic-cost framing (both excitation and
        # inhibition consume resources), not an E/I-signed drive term; it can
        # therefore stay near-constant even while HDP redistributes weight
        # between E and I edges on the same neuron.
        dist_floor = jnp.clip(H - H_min_arr, barrier_eps_arr, None)
        dist_ceil = jnp.clip(H_max_arr - H, barrier_eps_arr, None)
        barrier_force = barrier_c_arr / (dist_floor * dist_floor) - barrier_d_arr / (dist_ceil * dist_ceil)
        dH_income = alpha_arr * syn_h + beta_arr
        dH_rate = -gamma_arr * H * prev_spikes_h  # H-taxed: output spending scaled by resource level
        dH_weight = -delta_arr * W_burden_h
        dH_passive = rho_passive_arr / jnp.maximum(H * H, 1e-8)  # Passive income: stronger at low H -- NOTE this
        # term is >=0 everywhere H>0, so it can cushion H near the floor but can NEVER pull H back
        # down from above H*=1 on its own. Root-caused 2026-07-01 (F-017/F-019): with gamma=delta=0
        # (DEFAULT_HDP's own base kwargs), dH_income/dH_rate/dH_weight/dH_passive are ALL >=0, so
        # nothing opposes upward drift except barrier_force very close to H_max -- confirmed via the
        # full rho_passive sweep (scripts/hdp_v2_rho_sweep.py): H_max_obs pinned near H_max=10 across
        # nearly the entire swept range, at every rho_passive value, because H is only ever stopped by
        # the hard clip, never by a real restoring force. K_ctrl_arr*(1-H) is genuinely two-sided
        # (positive below H*=1, negative above) -- reviving it as a live term below closes this gap.
        dH_ctrl = K_ctrl_arr * (1.0 - H)  # Revived 2026-07-01 -- was dead code (computed, unused).
        dH = (
            dH_income
            + dH_rate
            + dH_weight
            + dH_passive
            + dH_ctrl
            + barrier_force
        )
        if h_dim > 1:
            dH = dH + H @ coupling.T
        tau_factor = dt / tau_i if h_dim == 1 else (dt / tau_i)[:, None]
        H_next = jnp.clip(H + tau_factor * dH, H_min_arr, H_max_arr)

        # (3) Update plastic weights from the updated H_i using the selected rule family.
        # All rules use postsynaptic-indexed weight updates (sign safety applied via exc_mask).
        if h_dim == 1:
            H_pre = H_next[pre]
            H_post = H_next[post]
        else:
            H_pre = H_next[pre] @ readout
            H_post = H_next[post] @ readout

        # Compute rule basis per edge, depending on hdp_rule.
        # signed_linear: basis ~ (H_post - H_pre), flipped to preserve postsynaptic-indexing invariant
        # signed_quadratic: basis ~ (H_post - H_pre)|H_post - H_pre|, preserving quadratic shape
        # hebbian_product: basis ~ H_pre * H_post, applied symmetrically
        if hdp_rule == "signed_linear":
            rule_basis = H_post - H_pre
        elif hdp_rule == "signed_quadratic":
            diff = H_post - H_pre
            rule_basis = diff * jnp.abs(diff)
        elif hdp_rule == "hebbian_product":
            rule_basis = H_pre * H_post
        else:
            raise ValueError(f"Unknown hdp_rule: {hdp_rule}. Must be one of: signed_linear, signed_quadratic, hebbian_product")

        # NOTE: dw is proportional to the edge's current wmag (multiplicative
        # rule) -- an edge clipped to w_floor gets a proportionally tiny
        # learning signal and stops evolving in practice. Intentional (mirrors
        # multiplicative/log-domain plasticity rules and keeps dw scale-free
        # across a wide weight range), but not swap-in-additive without
        # re-verifying every tuned preset's dynamics -- external review 2026-07-14.
        dw_exc = K_HDP_arr * rule_basis * wmag
        dw_inh = -K_HDP_arr * rule_basis * wmag
        # Weight restoring force: pulls wmag back toward its calibrated baseline
        # magnitude, closing the same gap K_ctrl closed for H_i (see K_w_ctrl's
        # docstring). Sign-agnostic (applies to the unsigned magnitude before
        # exc_mask reapplies sign), so added once to dw rather than split by
        # E/I branch like dw_exc/dw_inh above.
        dw_w_ctrl = K_w_ctrl_arr * (wmag_baseline_arr - wmag)
        dw = jnp.where(exc_mask, dw_exc, dw_inh) + dw_w_ctrl
        wmag_next = jnp.clip(wmag + dt * dw, w_floor_arr, w_ceiling_arr)
        w_next = jnp.where(exc_mask, wmag_next, -wmag_next)

        # (4) Integrate the neuron (Izhikevich) and detect spikes.
        dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
        v_next = v + dt * dv
        u_next = u + dt * du
        v_next = jnp.where(s_mask > 0.5, v_next, c)
        spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
        spikes = spikes_bool.astype(jdtype)
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)
        if has_nonzero_delay:
            presyn = _delayed_presynaptic_spikes(spikes, spike_hist, t_idx, pre, delay_steps_arr)
            syn_next = syn_state * decay + presyn
            slot = jnp.mod(t_idx, bufsize)
            spike_hist_next = spike_hist.at[slot].set(spikes)
        else:
            syn_next = syn_state * decay + spikes[pre]

        # (5) Spikes consume H_i (discrete drain on neurons that just fired).
        # NOTE: intentionally NOT divided by tau_i (see docstring) -- at any
        # nonzero C_spike this drain is the same absolute size regardless of a
        # neuron's size-scaled tau_i, so it does not follow the "larger/slower
        # neurons adapt slower" contract the continuous dH/dt terms follow.
        # C_spike=0.0 in every shipped preset today, so this is currently
        # inert everywhere; flagged, not changed, without re-verifying presets
        # that would enable it -- external review 2026-07-14.
        spike_drain = spikes if h_dim == 1 else spikes[:, None]
        H_final = jnp.clip(
            H_next - C_spike_arr * spike_drain,
            H_min_arr,
            H_max_arr,
        )

        v_reset, u_reset, syn_next = _bound_state(v_reset, u_reset, syn_next)
        source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
        if record_weight_trace:
            outputs = (v_reset, spikes, source_proxy, H_final, w_next)
        else:
            outputs = (v_reset, spikes, source_proxy, H_final)
        if record_dH_components:
            outputs = outputs + (dH_income, dH_rate, dH_weight, dH_passive, barrier_force)
        if record_edge_current:
            outputs = outputs + (edge_current,)
        carry_out = (v_reset, u_reset, spikes, syn_next, H_final, w_next)
        if has_nonzero_delay:
            carry_out = carry_out + (spike_hist_next,)
        return carry_out, outputs

    if has_nonzero_delay:
        final, scan_outputs = jax.lax.scan(step, init, xs=(step_indices_arr, sched, bulk_noise))
    else:
        final, scan_outputs = jax.lax.scan(step, init, xs=(sched, bulk_noise))
    if pop_layout is not None:
        if record_weight_trace:
            voltages, spikes, sources, H_trace, theta_trace, w_trace = scan_outputs
        else:
            voltages, spikes, sources, H_trace, theta_trace = scan_outputs
            w_trace = None
        w_final, _ = bind_theta_to_plant(
            final[5], pop_layout, a_base=a, w_ceiling=w_ceiling_arr
        )
        diagnostics_dict = {
            "v": final[0],
            "u": final[1],
            "prev_spikes": final[2],
            "syn_state": final[3],
            "H_final": final[4],
            "theta_S_final": final[5],
            "w_final": w_final,
            "H_trace": H_trace,
            "theta_S_trace": theta_trace,
            "w_trace": w_trace,
            "h_state_locality": "population",
        }
        if has_nonzero_delay:
            diagnostics_dict["delay_state"] = final[-1]
            diagnostics_dict["spike_history"] = final[-1]
            diagnostics_dict["delay_steps_max"] = jnp.asarray(max_delay, dtype=jnp.int32)
            diagnostics_dict["continuation_step_offset"] = step_indices_arr[-1] + jnp.asarray(1, dtype=jnp.int32)
        return voltages, spikes, sources, diagnostics_dict

    if enable_boundary_stabilization:
        base_arity = 5 if record_weight_trace else 4
        if record_weight_trace:
            voltages, spikes, sources, H_trace, w_trace = scan_outputs[:5]
        else:
            voltages, spikes, sources, H_trace = scan_outputs[:4]
            w_trace = None
        r_bar_trace, I_H_trace = scan_outputs[base_arity:base_arity + 2]
        final_state = {
            "v": final[0],
            "u": final[1],
            "prev_spikes": final[2],
            "syn_state": final[3],
            "H_final": final[4],
            "w_final": final[5],
            "r_bar_final": final[6],
            "I_H_final": -g_H_arr * (final[4] - 1.0),
        }
        diagnostics_dict = {
            **final_state,
            "H_trace": H_trace,
            "w_trace": w_trace,
            "r_bar_trace": r_bar_trace,
            "I_H_trace": I_H_trace,
        }
        tail = scan_outputs[base_arity + 2:]
        if record_boundary_components:
            S_L_trace, S_H_trace, minus_B_prime_trace, dh_trace = tail[:4]
            diagnostics_dict.update({
                "S_L_trace": S_L_trace,
                "S_H_trace": S_H_trace,
                "minus_B_prime_trace": minus_B_prime_trace,
                "dh_trace": dh_trace,
            })
            tail = tail[4:]
        if record_edge_current:
            diagnostics_dict["edge_current_trace"] = tail[0]
        if has_nonzero_delay:
            diagnostics_dict["delay_state"] = final[-1]
            diagnostics_dict["spike_history"] = final[-1]
            diagnostics_dict["delay_steps_max"] = jnp.asarray(max_delay, dtype=jnp.int32)
            diagnostics_dict["continuation_step_offset"] = step_indices_arr[-1] + jnp.asarray(1, dtype=jnp.int32)
        return voltages, spikes, sources, diagnostics_dict

    base_arity = 5 if record_weight_trace else 4
    if record_weight_trace:
        voltages, spikes, sources, H_trace, w_trace = scan_outputs[:5]
    else:
        voltages, spikes, sources, H_trace = scan_outputs[:4]
        w_trace = None

    final_state = {
        "v": final[0],
        "u": final[1],
        "prev_spikes": final[2],
        "syn_state": final[3],
        "H_final": final[4],
        "w_final": final[5],
    }
    diagnostics_dict = {
        **final_state,
        "H_trace": H_trace,
        "w_trace": w_trace,
    }
    tail = scan_outputs[base_arity:]
    if record_dH_components:
        dH_income_trace, dH_rate_trace, dH_weight_trace, dH_passive_trace, dH_barrier_trace = tail[:5]
        diagnostics_dict.update({
            "dH_income_trace": dH_income_trace,
            "dH_rate_trace": dH_rate_trace,
            "dH_weight_trace": dH_weight_trace,
            "dH_passive_trace": dH_passive_trace,
            "dH_barrier_trace": dH_barrier_trace,
        })
        tail = tail[5:]
    if record_edge_current:
        diagnostics_dict["edge_current_trace"] = tail[0]
    if has_nonzero_delay:
        diagnostics_dict["delay_state"] = final[-1]
        diagnostics_dict["spike_history"] = final[-1]
        diagnostics_dict["delay_steps_max"] = jnp.asarray(max_delay, dtype=jnp.int32)
        diagnostics_dict["continuation_step_offset"] = step_indices_arr[-1] + jnp.asarray(1, dtype=jnp.int32)
    return voltages, spikes, sources, diagnostics_dict


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


def synaptic_tau_from_mechanism(
    mechanism: Sequence[str], *, dtype: str = "float32"
) -> jax.Array:
    """Map declared receptor-mechanism names to per-edge tau (Synaptic Tensor, tau stage).

    Vectorized lookup over :func:`standard_receptor_specs` -- the same table
    :func:`standard_receptor_tau_table`/:func:`_edge_tau_from_receptor_index`
    use, but keyed by mechanism name (``"AMPA"``, ``"GABA_A"``, ``"NMDA"``,
    ``"GABA_B"``) instead of an already-resolved ``receptor_index``. This is
    additive: it does not change how ``core._compile_connection_rules`` infers
    tau from weight sign today.

    Raises ``ValueError`` on an unrecognized mechanism name -- unlike
    :func:`cable_filter_tau`'s cell-type fallback, silently substituting the
    wrong synaptic mechanism is worse than failing loudly.
    """

    jdtype = _dtype_from_policy(dtype)
    specs = standard_receptor_specs()
    try:
        tau_ms = [float(specs[m].tau_ms) for m in mechanism]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized receptor mechanism {exc.args[0]!r}; valid names: "
            f"{sorted(specs)}"
        ) from exc
    return jnp.asarray(tau_ms, dtype=jdtype)


def synaptic_current_tensor(
    spikes_pre: jax.Array, tau_ms: jax.Array, dt_ms: float
) -> jax.Array:
    """Standalone single-pole synaptic current tensor (Synaptic Tensor, filter stage).

    Factors out the exact per-edge synaptic state update used inline by
    :func:`simulate_edge_recurrent_izhikevich`/
    :func:`simulate_receptor_exponential_izhikevich`
    (``syn_next = syn_state * exp(-dt/tau) + spike``) as an explicit, named,
    reusable operator -- usable outside the full ``simulate()`` orchestration
    for diagnostics or parameter sweeps. A single-exponential decay model
    (no separate rise time constant), matching the kernels exactly.

    Parameters:
        spikes_pre: per-channel spike/input trace, shape ``[T, E]``.
        tau_ms: per-channel time constant in milliseconds, shape ``[E]``.
        dt_ms: simulation timestep in milliseconds.

    Returns: synaptic state trace, shape ``[T, E]``.
    """

    if spikes_pre.shape[-1] != tau_ms.shape[0]:
        raise ValueError(
            f"spikes_pre channel dim {spikes_pre.shape[-1]} != tau_ms length "
            f"{tau_ms.shape[0]}"
        )
    jdtype = tau_ms.dtype
    decay = jnp.exp(-jnp.asarray(dt_ms, dtype=jdtype) / tau_ms)
    spikes_pre = spikes_pre.astype(jdtype)

    def step(syn, spk):
        syn_next = syn * decay + spk
        return syn_next, syn_next

    syn0 = jnp.zeros_like(tau_ms)
    _, trace = jax.lax.scan(step, syn0, spikes_pre)
    return trace


def synaptic_tensor_report(
    tau_ms: jax.Array, mechanism: "Sequence[str] | None" = None
) -> dict[str, Any]:
    """JSON-safe truth-gate report for a :func:`synaptic_current_tensor` call."""

    finite = bool(jnp.all(jnp.isfinite(tau_ms)))
    return {
        "tau_ms_mean": float(jnp.mean(tau_ms)) if finite else None,
        "tau_ms_min": float(jnp.min(tau_ms)) if finite else None,
        "tau_ms_max": float(jnp.max(tau_ms)) if finite else None,
        "mechanism": list(mechanism) if mechanism is not None else None,
        "finite_tau": finite,
        "source_calibration_status": "metadata_only_uncalibrated",
        "physical_amplitude_calibrated": False,
        "claim_level": "computational_scaffold",
    }


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

    Delay contract: this kernel has no finite-delay path; ``edges.delay_steps``
    must be all zero (negative values are rejected outright). Nonzero delays
    select ``simulate_edge_recurrent_izhikevich`` instead.
    """

    delay_host = _edge_delay_steps_host(edges)
    if np.any(delay_host < 0):
        raise ValueError("edge delay_steps must be >= 0")
    if np.any(delay_host != 0):
        raise ValueError(
            "receptor_exponential synaptic_kernel has no finite-delay path; "
            "edges.delay_steps must be all zero (use the default exponential "
            "synaptic kernel for finite edge delays)"
        )

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
            """Documented public function `step`."""
            v, u, prev_spikes, syn_state = carry
            edge_drive = weight * syn_state
            syn = _segment_sum(edge_drive, post, n_neurons)
            current_native = drive + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
            return (v_reset, u_reset, spikes, syn_next), (v_reset, spikes, source_proxy)

        final, (voltages, spikes, sources) = jax.lax.scan(step, init, xs=bulk_noise)
    else:
        sched = drive_schedule.astype(jdtype)

        def step_sched(carry, xs_t):
            """Documented public function `step_sched`."""
            sched_t, noise_t = xs_t
            v, u, prev_spikes, syn_state = carry
            edge_drive = weight * syn_state
            syn = _segment_sum(edge_drive, post, n_neurons)
            current_native = drive + sched_t + syn + jnp.asarray(0.5, dtype=jdtype) * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            
            # Apply silence_mask
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            syn_next = syn_state * decay + spikes[pre]
            source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
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
        """Documented public function `step`."""
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
        dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
        v_next = v + dt * dv
        u_next = u + dt * du
        spikes_bool = v_next >= 30.0
        spikes = spikes_bool.astype(jdtype)
        v_reset = jnp.where(spikes_bool, c, v_next)
        u_reset = jnp.where(spikes_bool, u_next + d, u_next)

        # Update synaptic traces (exponential decay + spike injection)
        syn_traces_next = syn_traces * decay + spikes

        source_proxy = _source_proxy_from_components(current_native, spikes, source_scale, dtype=jdtype)
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
    # Single O(N) pass instead of O(n_cell_types * N) set + per-type recount.
    denom = max(1, n)
    cell_type_fractions = {ct: count / denom for ct, count in Counter(cell_types).items()}

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
    """Documented public class `EmitterState`."""
    v: jax.Array
    u: jax.Array
    spikes: jax.Array
    key: jax.Array
    step_count: jax.Array


class EmitterOutput(_NamedTuple):
    """Documented public class `EmitterOutput`."""
    voltage: jax.Array
    spikes: jax.Array
    source: jax.Array
    finite: jax.Array

    @property
    def dtype(self) -> str:
        """Documented public function `dtype`."""
        return str(self.voltage.dtype)


class Emitter:
    """Base class for package-level emitter facades."""

    def initial_state(self, seed: int = 0) -> EmitterState:
        raise NotImplementedError(
            "Emitter.initial_state is not implemented on the base Emitter class; "
            "use a concrete emitter such as IzhikevichEmitter"
        )

    def step(self, state: EmitterState, input_t: jax.Array, *, dt_ms: float = 0.1) -> tuple[EmitterState, EmitterOutput]:
        raise NotImplementedError(
            "Emitter.step is not implemented on the base Emitter class; "
            "use a concrete emitter such as IzhikevichEmitter"
        )


class IzhikevichEmitter(Emitter):
    """Reduced Izhikevich emitter facade with a JAX step function."""

    def __init__(self, n: int | None = None, *, n_neurons: int | None = None, dtype: str = "float32", cell_type_fractions: Mapping[str, float] | None = None):
        self.n = int(n if n is not None else (n_neurons if n_neurons is not None else 1))
        if self.n <= 0:
            raise ValueError("n must be positive")
        self.dtype = dtype
        self.params = izhikevich_eig_params(self.n, cell_type_fractions or {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07}, dtype=dtype)

    def initial_state(self, seed: int = 0) -> EmitterState:
        """Documented public function `initial_state`."""
        jdtype = _dtype_from_policy(self.dtype)
        return EmitterState(
            v=self.params.v0.astype(jdtype),
            u=self.params.u0.astype(jdtype),
            spikes=jnp.zeros((self.n,), dtype=jdtype),
            key=jax.random.PRNGKey(int(seed)),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(self, state: EmitterState, input_t: jax.Array, *, dt_ms: float = 0.1) -> tuple[EmitterState, EmitterOutput]:
        """Documented public function `step`."""
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
        source = _source_proxy_from_components(
            current_native,
            spikes,
            self.params.source_scale.astype(jdtype),
            dtype=jdtype,
        )
        next_state = EmitterState(v=v_reset, u=u_reset, spikes=spikes, key=rng, step_count=state.step_count + 1)
        output = EmitterOutput(
            voltage=v_reset,
            spikes=spikes,
            source=source,
            finite=jnp.all(jnp.isfinite(v_reset)) & jnp.all(jnp.isfinite(source)),
        )
        return next_state, output


class GLIFEmitter(Emitter):
    """Generalized Leaky Integrate-and-Fire emitter — NOT implemented.

    Intentional placeholder (exported for API-surface stability) for a future
    multi-compartment or highly parameterized GLIF implementation. It cannot be
    constructed; instantiating it raises ``NotImplementedError``.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GLIFEmitter is an intentional placeholder with no dynamics "
            "implemented; it cannot be constructed"
        )


class LIFEmitter(Emitter):
    """Leaky Integrate-and-Fire emitter — NOT implemented.

    Intentional placeholder (exported for API-surface stability) for a future
    standard single-compartment LIF implementation. It cannot be constructed;
    instantiating it raises ``NotImplementedError``.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LIFEmitter is an intentional placeholder with no dynamics "
            "implemented; it cannot be constructed"
        )


class SynapseState(_NamedTuple):
    """Per-step synaptic state carried through the recurrent scan.

    ``trace`` is the synaptic activation/conductance trace array (one entry per
    synapse or edge), advanced by the synapse kernel at each time step. A
    NamedTuple so it registers as a JAX pytree for ``lax.scan``.
    """

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
        """Documented public function `initial_state`."""
        return SynapseState(trace=jnp.zeros((self.n,), dtype=_dtype_from_policy(self.dtype)))

    def step(self, state: SynapseState, pre_spikes: jax.Array, *, dt_ms: float = 0.1) -> tuple[SynapseState, jax.Array]:
        """Documented public function `step`."""
        jdtype = _dtype_from_policy(self.dtype)
        decay = jnp.exp(-jnp.asarray(dt_ms, dtype=jdtype) / jnp.asarray(self.tau_ms, dtype=jdtype))
        trace_next = state.trace.astype(jdtype) * decay + jnp.asarray(pre_spikes, dtype=jdtype)
        current = self.W.astype(jdtype) @ trace_next
        return SynapseState(trace=trace_next), current


# Re-export the second canonical HDP sanity emitter (jaxfne/emitters_homeostatic_ei.py)
# so `jaxfne.emitters.HomeostaticEIParams`/`.simulate_homeostatic_ei` work without a
# second import path -- the implementation lives in its own sibling module (this file
# is already 2000+ lines and organized around the Izhikevich/EIG scaffold).
from .emitters_homeostatic_ei import (  # noqa: E402
    ACTIVATION_RULES,
    CONDUCTANCE_RULES,
    HOMEOSTASIS_RULES,
    HomeostaticEIParams,
    simulate_homeostatic_ei,
    make_minimal_ei_params,
    make_hebbian_pairwise_rule,
)
