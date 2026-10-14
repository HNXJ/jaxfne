"""Current-based LIF edge-list kernel with PCL refractory terms (function-level only).

Reference: N'dri, Barbier, Teuliere, Triesch, Nat Commun 16:8880 (2025),
doi:10.1038/s41467-025-64234-z, Methods; the per-event update mirrors
``Neuron.cpp`` / ``SimpleNeuron.cpp`` / ``ComplexNeuron.cpp`` as transcribed in
``artifacts/etudes/pcl/port/pcl_event_ref.py`` (read as data): on each input of
weight ``w`` at time ``t``,

    V <- V * exp(-dt/tau_m);  A <- A * exp(-dt/tau_sra),
    V <- max(v_min, V + w - eta_rp * exp(-(t - t_last)/tau_rp) - A),

a spike when ``V > v_thresh``, then ``V <- v_reset``, ``A <- A + delta_sra``.
Inhibitory inputs subtract ``w`` the same way. Complex cells carry no ``A`` term
(``delta_sra = 0``). Never-spiked cells subtract no refractory term.

Clock-driven translation (one ``lax.scan`` step per bin, as the port's stage B):
decay runs every step, and the ``k`` arrivals of a step share one subtraction of
``k * (R + A)`` at the step time. Per-event C++ order inside a step is not
modelled; that deviation is declared here, not hidden.

Cell-type defaults: PORT_SPEC.md stage-A table. The simple threshold follows the
paper's Table 2 (10 mV); the default config file carries 30 mV, and each trained
network's own config decides. The simple adaptation increment defaults to 0
(the default config's ``ETA_SRA``); per-network values ride ``overrides``.

The registered-HDP-rule hook reuses the :class:`HDPRuleContext` /
:class:`HDPRuleUpdate` contract of :mod:`jaxfne.hdp_rule` (same field names and
rate-integration ``x + dt * dx``), so rules written against the Izhikevich
registrable kernel (``jaxfne/_hdp_registrable_kernel.py``) run here unchanged:
per-edge delayed arrivals feed ``pre_sp``, per-neuron ``bias`` feeds the input
when the rule declares the ``drive_bias`` target.

Function-level only. Configuration, construct and simulate wiring is not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as _dc_field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

NEVER = -1.0e30  # no spike yet; spiked state is t_last > HAS_SPIKED
HAS_SPIKED = -1.0e29

# PORT_SPEC.md stage-A table (paper Table 2 threshold for simple; default config
# ETA_SRA = 0, so the adaptation increment defaults to 0 and is opt-in).
LIF_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "simple": {
        "tau_m_ms": 18.0, "v_thresh_mV": 10.0, "v_reset_mV": -10.0,
        "v_min_mV": -20.0, "eta_rp_mV": 10.0, "tau_rp_ms": 5.0,
        "tau_sra_ms": 100.0, "delta_sra_mV": 0.0,
    },
    "complex": {
        "tau_m_ms": 50.0, "v_thresh_mV": 3.0, "v_reset_mV": -10.0,
        "v_min_mV": -20.0, "eta_rp_mV": 10.0, "tau_rp_ms": 5.0,
        "tau_sra_ms": math.inf, "delta_sra_mV": 0.0,
    },
}

_PARAM_FIELDS = (
    "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV", "tau_rp_ms",
    "tau_sra_ms", "delta_sra_mV",
)


class LIFParams(NamedTuple):
    """Per-neuron PCL LIF parameters: float arrays of shape [N], millivolts and ms."""

    tau_m_ms: jax.Array
    v_thresh_mV: jax.Array
    v_reset_mV: jax.Array
    v_min_mV: jax.Array
    eta_rp_mV: jax.Array
    tau_rp_ms: jax.Array
    tau_sra_ms: jax.Array
    delta_sra_mV: jax.Array
    labels: tuple[str, ...]


def make_lif_params(
    labels: Sequence[str],
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> LIFParams:
    """Build :class:`LIFParams` for one cell type per neuron.

    Defaults: N'dri et al. (2025) via PORT_SPEC.md (see the module docstring).
    ``overrides`` maps a cell type to a dict of field overrides (field names as
    in :class:`LIFParams`). Unknown cell types, unknown fields, and overrides
    for a cell type absent from ``labels`` raise ``ValueError``.
    """
    labels = tuple(str(lab) for lab in labels)
    for lab in labels:
        if lab not in LIF_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"unknown LIF cell type {lab!r}; expected one of {sorted(LIF_CELL_TYPE_DEFAULTS)}"
            )
    overrides = {} if overrides is None else overrides
    for ctype, fields in overrides.items():
        if ctype not in LIF_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"overrides name unknown LIF cell type {ctype!r}; "
                f"expected one of {sorted(LIF_CELL_TYPE_DEFAULTS)}"
            )
        if ctype not in labels:
            raise ValueError(f"overrides name cell type {ctype!r}, which is not in labels {labels}")
        for field in fields:
            if field not in _PARAM_FIELDS:
                raise ValueError(
                    f"unknown LIF parameter field {field!r} for cell type {ctype!r}; "
                    f"expected one of {_PARAM_FIELDS}"
                )
    columns = {}
    for field in _PARAM_FIELDS:
        values = [
            float(overrides.get(lab, {}).get(field, LIF_CELL_TYPE_DEFAULTS[lab][field]))
            for lab in labels
        ]
        columns[field] = jnp.asarray(np.asarray(values, dtype=np.float64))
    return LIFParams(labels=labels, **columns)


def lif_step(state, I_mV, n_arr, t_ms, params: LIFParams, dt_ms: float):
    """One clock step of the PCL LIF equations for a vector of cells.

    ``state`` is ``(V, A, t_last)``: membrane voltage (mV), adaptation (mV) and
    last spike time (ms, ``NEVER`` when the cell never spiked). ``I_mV`` is the
    summed signed input of the step; ``n_arr`` the arrival count (each arrival
    subtracts one ``R + A``, the clock translation of the per-event C++ term).
    ``t_ms`` is the step time. A cell spikes on strict ``V > v_thresh``
    (``SimpleNeuron.cpp`` / ``ComplexNeuron.cpp``); on spike ``V`` resets,
    ``A`` grows by ``delta_sra`` and ``t_last`` stamps ``t_ms``.

    Returns ``((V, A, t_last), spk)`` with boolean spike vector ``spk``. Pure
    ``jax.numpy``; dtype follows the inputs.
    """
    if not dt_ms > 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}")
    V, A, t_last = state
    p = params
    Vd = V * jnp.exp(-dt_ms / p.tau_m_ms)
    Ad = A * jnp.exp(-dt_ms / p.tau_sra_ms)
    has = t_last > HAS_SPIKED
    R = jnp.where(has, p.eta_rp_mV * jnp.exp(-(t_ms - t_last) / p.tau_rp_ms), 0.0)
    V_new = jnp.maximum(p.v_min_mV, Vd + I_mV - n_arr * (R + Ad))
    spk = V_new > p.v_thresh_mV
    V_out = jnp.where(spk, p.v_reset_mV, V_new)
    A_out = jnp.where(spk, Ad + p.delta_sra_mV, Ad)
    t_out = jnp.where(spk, t_ms, t_last)
    return (V_out, A_out, t_out), spk


_EDGE_ARGS = ("edge_pre", "edge_post", "edge_weight", "edge_delay_steps")


# Labels naming PCL cell types read their row directly; every other label
# (cortical reduced classes such as "E"/"PV", TFNE-minted types) resolves to
# the simple-cell row. The fallback is declared here, not hidden in a branch:
# the tensor route builds doctrine fixtures labelled "E"/"PV", and PCL has no
# per-class LIF row for those names.
LIF_FALLBACK_CELL_TYPE = "simple"

#: Presynaptic polarity carried on the shared edge compilers, using the same
#: convention as the current-based route's sign column (+1 excitatory, -1
#: inhibitory). Labels outside this map resolve to +1, matching the
#: simple-cell fallback above.
LIF_LABEL_SIGN: dict[str, float] = {
    "simple": 1.0, "complex": 1.0, "E": 1.0,
    "PV": -1.0, "SST": -1.0, "VIP": -1.0, "Inl": -1.0, "Ing": -1.0, "I": -1.0,
}

#: Constant drive (mV added every step) for triples with no declared drive.
#: Matches the current-based route's PV default so the same doctrine fixture
#: reads the same drive channel on either family.
LIF_DEFAULT_DRIVE_MV = 3.0


def _lif_dtype(dtype: str):
    """float32 default; float64 only with x64 enabled (same refusal as the rest of the package)."""
    if dtype == "float64" and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "dtype='float64' requested but JAX x64 is not enabled -- call "
            'jax.config.update("jax_enable_x64", True) before creating any array '
            "in this process; refusing to silently build float32"
        )
    return jnp.float64 if dtype == "float64" else jnp.float32


@dataclass(frozen=True)
class LIFNetworkParams:
    """Per-neuron PCL LIF parameters for a constructed population.

    The eight ``*_ms``/``*_mV`` arrays are the :func:`make_lif_params` columns
    in population order; ``drive`` is constant current (mV added every step)
    per neuron; ``sign``/``W``/``labels``/``layer_labels`` mirror the fields
    the shared construction machinery reads (edge compilers, triple-drive
    application, audit tables). ``W`` is a ``(0, 0)`` placeholder: LIF edges
    live only in ``params["edge_list"]``, as on the sparse-direct path.
    """

    tau_m_ms: jax.Array
    v_thresh_mV: jax.Array
    v_reset_mV: jax.Array
    v_min_mV: jax.Array
    eta_rp_mV: jax.Array
    tau_rp_ms: jax.Array
    tau_sra_ms: jax.Array
    delta_sra_mV: jax.Array
    drive: jax.Array
    sign: jax.Array
    W: jax.Array
    labels: tuple[str, ...]
    layer_labels: tuple[str, ...] | None = None
    source_calibration_status: str = "uncalibrated_lif_native_mV"

    @property
    def n_neurons(self) -> int:
        """Population size (length of the drive channel)."""
        return int(self.drive.shape[0])

    def to_lif_params(self) -> LIFParams:
        """The :class:`LIFParams` NamedTuple the edge kernel consumes."""
        return LIFParams(
            tau_m_ms=self.tau_m_ms, v_thresh_mV=self.v_thresh_mV,
            v_reset_mV=self.v_reset_mV, v_min_mV=self.v_min_mV,
            eta_rp_mV=self.eta_rp_mV, tau_rp_ms=self.tau_rp_ms,
            tau_sra_ms=self.tau_sra_ms, delta_sra_mV=self.delta_sra_mV,
            labels=self.labels,
        )


def lif_cell_type_of(label: str) -> str:
    """PCL cell-type row for ``label``: the label itself when it names one, else the fallback."""
    return str(label) if str(label) in LIF_CELL_TYPE_DEFAULTS else LIF_FALLBACK_CELL_TYPE


def lif_network_params_from_labels(
    labels: Sequence[str],
    *,
    layer_labels: Sequence[str] | None = None,
    dtype: str = "float32",
    drive_overrides: Mapping[str, float] | None = None,
    lif_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> LIFNetworkParams:
    """Build :class:`LIFNetworkParams` for one label per neuron.

    Each neuron reads the PCL row of :func:`lif_cell_type_of` (its label when
    it names a PCL cell type, else the simple-cell row). ``drive_overrides``
    maps a label to constant drive (mV/step); unmentioned labels default to
    ``LIF_DEFAULT_DRIVE_MV``. ``lif_overrides`` maps a label to a dict of the
    eight :data:`_PARAM_FIELDS` fields. Unknown override labels and unknown
    fields raise ``ValueError``; an override label matching no neuron is
    refused unless it names a PCL cell type or a built label is its fallback
    target (mirrors the declared-but-rounded-away acceptance elsewhere).
    """
    label_tuple = tuple(str(x) for x in labels)
    if not label_tuple:
        raise ValueError("labels must contain at least one label")
    if layer_labels is not None and len(tuple(layer_labels)) != len(label_tuple):
        raise ValueError("layer_labels length must match labels length")
    jdtype = _lif_dtype(dtype)
    overrides = {str(k): float(v) for k, v in (drive_overrides or {}).items()}
    for lab, val in overrides.items():
        if not math.isfinite(val):
            raise ValueError(f"drive_overrides[{lab!r}] must be finite; got {val!r}")
    lif_overrides = dict(lif_overrides or {})
    for lab, fields in lif_overrides.items():
        if not isinstance(fields, Mapping):
            raise ValueError(f"lif_overrides[{lab!r}] must be a mapping of {_PARAM_FIELDS}; got {fields!r}")
        if lab not in label_tuple and lab not in LIF_CELL_TYPE_DEFAULTS:
            raise ValueError(
                f"lif_overrides names {lab!r}, which matches no built neuron "
                f"{label_tuple} and no PCL cell type {sorted(LIF_CELL_TYPE_DEFAULTS)}"
            )
        for f in fields:
            if f not in _PARAM_FIELDS:
                raise ValueError(
                    f"unknown LIF parameter field {f!r} for {lab!r}; expected one of {_PARAM_FIELDS}"
                )
    columns: dict[str, list[float]] = {f: [] for f in _PARAM_FIELDS}
    drive: list[float] = []
    sign: list[float] = []
    for lab in label_tuple:
        row = lif_cell_type_of(lab)
        over = lif_overrides.get(lab, lif_overrides.get(row, {}))
        for f in _PARAM_FIELDS:
            columns[f].append(float(over.get(f, LIF_CELL_TYPE_DEFAULTS[row][f])))
        drive.append(overrides.get(lab, LIF_DEFAULT_DRIVE_MV))
        sign.append(LIF_LABEL_SIGN.get(lab, 1.0))
    n = len(label_tuple)
    return LIFNetworkParams(
        **{f: jnp.asarray(columns[f], dtype=jdtype) for f in _PARAM_FIELDS},
        drive=jnp.asarray(drive, dtype=jdtype),
        sign=jnp.asarray(sign, dtype=jdtype),
        W=jnp.zeros((0, 0), dtype=jdtype),
        labels=label_tuple,
        layer_labels=None if layer_labels is None else tuple(str(x) for x in layer_labels),
    )


def _check_edges(N, given):
    """Validate the four edge arrays; return numpy arrays, or None when all are absent."""
    missing = [k for k in _EDGE_ARGS if given[k] is None]
    if missing and len(missing) < len(_EDGE_ARGS):
        raise ValueError(f"edge arrays must be all given or all None; missing {missing}")
    if missing:
        return None
    arrs = {}
    for k in _EDGE_ARGS:
        a = np.asarray(given[k])
        if a.ndim != 1:
            raise ValueError(f"{k} must be one-dimensional, got shape {a.shape}")
        arrs[k] = a
    lengths = {k: a.shape[0] for k, a in arrs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"edge arrays must have equal lengths; got {lengths}")

    def _ints(name):
        a = arrs[name].astype(np.float64)
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integers")
        return a.astype(np.int64)

    pre = _ints("edge_pre")
    post = _ints("edge_post")
    delay = _ints("edge_delay_steps")
    if pre.size and (pre.min() < 0 or pre.max() >= N):
        raise ValueError(f"edge_pre must index cells in [0, {N}); got min {pre.min()}, max {pre.max()}")
    if post.size and (post.min() < 0 or post.max() >= N):
        raise ValueError(f"edge_post must index cells in [0, {N}); got min {post.min()}, max {post.max()}")
    if delay.size and delay.min() < 1:
        raise ValueError(f"edge_delay_steps must be >= 1; got min {delay.min()}")
    weight = arrs["edge_weight"].astype(np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("edge_weight must be finite")
    return {"pre": pre, "post": post, "weight": weight, "delay": delay}


def _seg_sum(values, ids, n):
    return jax.ops.segment_sum(values, ids, num_segments=n)


def simulate_lif_edge(
    params: LIFParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight=None,
    edge_delay_steps=None,
    t0_ms: float = 0.0,
    I_ext_pA=None,
    V0_mV=None,
    H0=None,
    record_v: bool = False,
    hdp_rule=None,
    hdp_rule_params=None,
    plasticity_mask=None,
    key=None,
) -> dict[str, Any]:
    """Simulate current-based PCL LIF cells on a sparse signed edge list.

    Each edge ``e`` runs from ``edge_pre[e]`` to ``edge_post[e]`` with signed
    weight ``edge_weight[e]`` (mV; inhibitory edges carry negative weights and
    subtract, as ``SimpleNeuron.cpp:61``) and delay ``edge_delay_steps[e]``
    (>= 1). Per step ``t`` at time ``t0_ms + t * dt_ms``, in order:

    1. arrivals: each edge reads the spike vector of step ``t - d_e`` from a
       bool ring of length ``max(d_e)``;
    2. input ``I = segment_sum(w_e * arrival_e)`` and count ``n`` per neuron,
       plus ``I_ext`` and the rule's per-neuron ``bias`` (0 unless the rule
       declares the ``drive_bias`` target);
    3. :func:`lif_step`;
    4. the registered rule (if any) observes the step through the
       :class:`HDPRuleContext` contract and its rate update integrates as
       ``x + dt * dx`` (H clipped to the descriptor bounds, weight magnitudes
       to ``w_bounds`` with the initial sign kept, ``plasticity_mask`` gating
       which edges learn);
    5. this step's spikes are written into the ring.

    ``I_ext_pA`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``.
    ``V0_mV`` is None (rest at 0, as the C++ ``m_potential`` starts) or shape
    ``[N]``. ``H0`` is None (relative-state reference 1 on every neuron) or
    shape ``[N]``: the rule's initial ``H`` carry, consumed only when a rule
    runs (without ``hdp_rule`` it is refused, mirroring the
    ``hdp_rule_params``-without-rule refusal). ``hdp_rule_params`` without
    ``hdp_rule`` is refused; unknown rule
    names and unknown rule-param keys fail closed through
    :func:`jaxfne.hdp_rule.check_hdp_rule_params`. ``key`` seeds the per-step
    rule-noise stream (``fold_in`` of the step index); None runs the
    deterministic ``PRNGKey(0)`` stream, documented here so receipts can quote
    it. Without a rule the run is exactly deterministic.

    Returns a dict: ``spikes`` (bool ``[n_steps, N]``), ``V_mV`` (``[n_steps,
    N]``, only if ``record_v``), finals ``V``, ``A``, ``t_last``,
    ``w_final`` (signed), and with a rule ``H_final``, ``aux_final`` and
    ``b_final`` when the rule declares ``drive_bias``. Implemented with
    ``jax.lax.scan``; no dense ``N x N`` matrix. dtype follows ``params``.

    Raises ``ValueError`` naming the argument for every refused input.
    """
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 1):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    if not dt_ms > 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms!r}")
    try:
        t0 = float(t0_ms)
    except (TypeError, ValueError):
        raise ValueError(f"t0_ms must be a number, got {t0_ms!r}")
    if not math.isfinite(t0):
        raise ValueError(f"t0_ms must be finite, got {t0_ms!r}")

    N = int(jnp.shape(params.tau_m_ms)[0])
    dtype = jnp.asarray(params.tau_m_ms).dtype

    edges = _check_edges(
        N,
        {"edge_pre": edge_pre, "edge_post": edge_post,
         "edge_weight": edge_weight, "edge_delay_steps": edge_delay_steps},
    )
    if edges is None:
        edges = {
            "pre": np.zeros((0,), np.int64),
            "post": np.zeros((0,), np.int64),
            "weight": np.zeros((0,), np.float64),
            "delay": np.ones((0,), np.int64),
        }
    E = int(edges["pre"].shape[0])

    if hdp_rule is None:
        if hdp_rule_params is not None:
            raise ValueError("hdp_rule_params is set but hdp_rule is None; pass a rule name")
        descriptor = None
        rule_step = None
        rule_params = {}
    else:
        from .hdp_rule import (  # lazy: hdp_rule pulls in emitters
            check_hdp_rule_params,
            expected_aux_shape,
            get_hdp_rule,
        )

        descriptor, rule_step = get_hdp_rule(hdp_rule)
        rule_params = check_hdp_rule_params(hdp_rule, hdp_rule_params)
        aux_shape = expected_aux_shape(
            descriptor, n_neurons=N, n_edges=E,
        )
    use_b = descriptor is not None and "drive_bias" in descriptor.theta_targets

    if plasticity_mask is not None:
        m_raw = np.asarray(plasticity_mask)
        if m_raw.ndim != 1 or m_raw.shape[0] != E:
            raise ValueError(
                f"plasticity_mask must have shape ({E},), got {m_raw.shape}"
            )
        if not bool(np.all(np.isfinite(m_raw))):
            raise ValueError("plasticity_mask must be finite (got NaN/inf)")
        plastic_edge = jnp.asarray(m_raw > 0.5)
    else:
        plastic_edge = None

    pre_j = jnp.asarray(edges["pre"], jnp.int32)
    post_j = jnp.asarray(edges["post"], jnp.int32)
    w0 = jnp.asarray(edges["weight"], dtype)
    lag_j = jnp.asarray(edges["delay"], jnp.int32)
    exc_mask = w0 >= 0
    L = int(edges["delay"].max()) if E else 1

    I_const = jnp.zeros((N,), dtype)
    I_all = None
    if I_ext_pA is not None:
        I_arr = jnp.asarray(I_ext_pA, dtype)
        if I_arr.shape == (N,):
            I_const = I_arr
        elif I_arr.shape == (n_steps, N):
            I_all = I_arr
        else:
            raise ValueError(
                f"I_ext_pA must be None, shape ({N},) or ({n_steps}, {N}); got {I_arr.shape}"
            )

    if V0_mV is None:
        V0 = jnp.zeros((N,), dtype)
    else:
        V0 = jnp.asarray(V0_mV, dtype)
        if V0.shape != (N,):
            raise ValueError(f"V0_mV must be None or shape ({N},); got {V0.shape}")

    if H0 is not None and hdp_rule is None:
        raise ValueError("H0 is set but hdp_rule is None; H seeds the rule's carry only")
    H_init: jax.Array | None = None
    if H0 is not None:
        H_init = jnp.asarray(H0, dtype)
        if H_init.shape != (N,):
            raise ValueError(f"H0 must be None or shape ({N},); got {H_init.shape}")
        if not bool(jnp.all(jnp.isfinite(H_init))):
            raise ValueError("H0 must be finite (got NaN/inf)")

    if descriptor is not None:
        h_min = jnp.asarray(descriptor.h_bounds[0], dtype)
        h_max = jnp.asarray(descriptor.h_bounds[1], dtype)
        w_floor = jnp.asarray(descriptor.w_bounds[0], dtype)
        w_ceiling = jnp.asarray(descriptor.w_bounds[1], dtype)
        H0_arr = jnp.ones((N,), dtype) if H_init is None else H_init
        aux0 = jnp.zeros(aux_shape, dtype)
        b_min = jnp.asarray(descriptor.b_bounds[0], dtype)
        b_max = jnp.asarray(descriptor.b_bounds[1], dtype)
    if key is None:
        base_key = jax.random.PRNGKey(0)
    else:
        try:
            jax.random.split(jnp.asarray(key))
        except Exception:
            raise ValueError(f"key must be a JAX PRNG key; got {key!r}")
        base_key = key
    # Per-step rule-noise stream (fold_in of the step index); the kernel
    # ignores it without a rule, so rule-free runs stay exactly deterministic.
    step_keys = jax.vmap(lambda t: jax.random.fold_in(base_key, t))(
        jnp.arange(n_steps, dtype=jnp.int32)
    )

    dt = jnp.asarray(dt_ms, dtype=dtype)

    def body(carry, I_t):
        V, A, t_last, w, H, aux, bias, ring, prev_spk, t_idx = carry
        t = t0 + t_idx * float(dt_ms)
        if E:
            arrived = ring[(t_idx - lag_j) % L, pre_j]
            I_net = _seg_sum(jnp.where(arrived, w, 0.0), post_j, N)
            n_net = _seg_sum(arrived.astype(dtype), post_j, N)
        else:
            arrived = jnp.zeros((0,), bool)
            I_net = jnp.zeros((N,), dtype)
            n_net = jnp.zeros((N,), dtype)
        I_tot = I_const if I_t is None else I_t
        I_tot = I_tot + I_net
        if use_b:
            I_tot = I_tot + bias
        (V_n, A_n, t_n), spk = lif_step((V, A, t_last), I_tot, n_net, t, params, float(dt_ms))
        if descriptor is not None:
            from .hdp_rule import HDPRuleContext  # lazy: hdp_rule pulls in emitters

            ctx = HDPRuleContext(
                H=H, aux=aux, v=V_n, u=jnp.zeros((N,), dtype),
                spikes=spk.astype(dtype),
                prev_spikes=prev_spk.astype(dtype),
                syn_state=arrived.astype(dtype), w=w, pre=pre_j, post=post_j,
                exc_mask=exc_mask, dt=dt, n_neurons=N, rule_params=rule_params,
                key=step_keys[t_idx], pre_sp=arrived,
            )
            upd = rule_step(ctx)
            undeclared = [k for k in (upd.d_theta or {}) if k not in descriptor.theta_targets]
            if undeclared:
                raise ValueError(
                    f"hdp_rule {hdp_rule!r} returned undeclared theta_targets "
                    f"{undeclared!r}; declared: {list(descriptor.theta_targets)}"
                )
            H = jnp.clip(H + dt * upd.dH, h_min, h_max)
            if upd.d_aux is not None:
                aux = aux + dt * upd.d_aux
            if upd.d_theta and "edge_weight" in upd.d_theta:
                wmag = jnp.abs(w)
                wmag_next = jnp.clip(wmag + dt * upd.d_theta["edge_weight"], w_floor, w_ceiling)
                w_next = jnp.where(exc_mask, wmag_next, -wmag_next)
                w = jnp.where(plastic_edge, w_next, w) if plastic_edge is not None else w_next
            if use_b:
                db = (upd.d_theta or {}).get("drive_bias")
                if db is not None:
                    bias = jnp.clip(bias + dt * db, b_min, b_max)
        ring = ring.at[t_idx % L].set(spk)
        out = (spk, V_n) if record_v else spk
        return (V_n, A_n, t_n, w, H, aux, bias, ring, spk, t_idx + 1), out

    if descriptor is not None:
        Hc, auxc, bc = H0_arr, aux0, jnp.zeros((N,), dtype)
    else:
        Hc, auxc, bc = None, None, None
    carry0 = (
        V0, jnp.zeros((N,), dtype), jnp.full((N,), NEVER, dtype),
        w0, Hc, auxc, bc, jnp.zeros((L, N), bool),
        jnp.zeros((N,), bool), jnp.int32(0),
    )
    (V_f, A_f, t_f, w_f, H_f, aux_f, b_f, _, _, _), ys = jax.lax.scan(
        body, carry0, I_all, length=n_steps,
    )
    result: dict[str, Any] = {
        "spikes": ys[0] if record_v else ys,
        "V": V_f,
        "A": A_f,
        "t_last": t_f,
        "w_final": w_f,
    }
    if record_v:
        result["V_mV"] = ys[1]
    if descriptor is not None:
        result["H_final"] = H_f
        result["aux_final"] = aux_f
        if use_b:
            result["b_final"] = b_f
    return result
