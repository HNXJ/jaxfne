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
from dataclasses import dataclass
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


def _grouped_first_r(raw, group_of, visit, n_rounds):
    """Keep mask: the first ``n_rounds`` crossers per group in visit order.

    ``raw`` is the bool crossing vector, ``group_of`` int ids of shape
    ``(n,)``, ``visit`` the index visit order (ascending ``arange`` or its
    reverse). Inside each group the crossers are ranked in visit order and
    the first ``n_rounds`` kept. Pure ``jax.numpy``.
    """
    n = int(raw.shape[0])
    # Visit order inside each group: sort by (group, visit position).
    order = jnp.lexsort((visit, group_of))
    crossed = raw[order].astype(jnp.int32)
    same = group_of[order]
    new_seg = jnp.concatenate(
        [jnp.ones(1, dtype=bool), same[1:] != same[:-1]])
    cum = jnp.cumsum(crossed)
    seg_start = jnp.maximum.accumulate(
        jnp.where(new_seg, jnp.arange(n), 0))
    prev = jnp.where(seg_start == 0, 0, cum[seg_start - 1])
    rank = cum - prev
    keep_sorted = raw[order] & (rank <= n_rounds)
    return jnp.zeros_like(raw).at[order].set(keep_sorted)


#: Floor for cascade potentials (``checkNegativeLimits``, oracle ``NEG_LIMIT``;
#: equals ``v_min_mV`` on both LIF rows, so the whole-step floor is untouched).
_CASCADE_NEG_LIMIT_MV = -20.0


def _cascade_step(state, I_tot, Kn, t_ms, params: LIFParams, dt_ms: float,
                  n_rounds: int, members_j, mask_j, blk_j, visit_t):
    """One step of the event-order intra-step fixed point (vectorized rounds).

    ``state`` is ``(V, A, t_last)``; ``I_tot`` the step's total input current
    (edges + feed + sustained) and ``Kn`` its event counts (edge + feed
    arrivals; sustained current confers no candidacy). Potentials decay once
    per step, then each retinotopic position resolves ``n_rounds`` winner
    rounds in its declared visit order (``visit_t`` ``(G, D)`` local ids):
    a fed cell (``Kn > 0``) past the last winner whose visit value
    ``V + I - Kn * RA`` clears strict ``V > v_thresh`` spikes at once —
    reset, adaptation increment, time stamp — and subtracts its block
    inhibition plus the target's refractory/adaptation from every later
    group-mate before the next visit. Winners keep reset-minus-upper
    inhibition, losers decayed-minus-all inhibition plus their input, both
    floored; unfed cells absorb winners' inhibition only. This is the
    binned-oracle cascade (``pcl_binned_jax._cascade``) with the
    per-event visit order on A (Hamm 2026-10-14). Pure ``jax.numpy``;
    dtype follows the inputs.
    """
    V, A, t_last = state
    p = params
    G = int(members_j.shape[0])
    D = int(members_j.shape[1])
    loc = jnp.arange(D, dtype=jnp.int32)
    Vd = V * jnp.exp(-dt_ms / p.tau_m_ms)
    Ad = A * jnp.exp(-dt_ms / p.tau_sra_ms)
    has = t_last > HAS_SPIKED
    R = jnp.where(has, p.eta_rp_mV * jnp.exp(-(t_ms - t_last) / p.tau_rp_ms), 0.0)
    RA = R + Ad
    Vg0 = Vd[members_j]
    Vg = Vg0
    Ag = Ad[members_j]
    Ig = I_tot[members_j]
    Kg = Kn[members_j]
    RAg = RA[members_j]
    RApost = (p.eta_rp_mV + Ad + p.delta_sra_mV)[members_j]
    thrg = p.v_thresh_mV[members_j]
    vrg = p.v_reset_mV[members_j]
    dsrag = p.delta_sra_mV[members_j]
    tg = t_last[members_j]
    # Visit position of each local depth; pads (``-1``) land past every
    # member, and candidacy excludes them anyway.
    kk = jnp.arange(D, dtype=Vg.dtype)
    one_hot = (jnp.asarray(visit_t)[..., None] == loc[None, None, :])
    pos = jnp.einsum("gkl,k->gl", one_hot.astype(Vg.dtype), kk).astype(jnp.int32)
    pos = jnp.where(mask_j, pos, jnp.int32(D))
    found = jnp.zeros((G, D), dtype=bool)
    lo = jnp.full((G,), -1, dtype=jnp.int32)
    for _ in range(n_rounds):
        v = jnp.maximum(Vg + Ig - Kg * RAg, _CASCADE_NEG_LIMIT_MV)
        cross = mask_j & (Kg > 0) & (pos > lo[:, None]) & (v > thrg)
        w = jnp.argmin(jnp.where(cross, pos, jnp.int32(D + 1)), axis=-1)
        has_w = jnp.any(cross, axis=-1)
        one_w = (loc[None, :] == w[:, None]) & has_w[:, None] & mask_j
        found = found | one_w
        lo = jnp.where(has_w, jnp.sum(pos * one_w.astype(jnp.int32), axis=-1), lo)
        Vg = jnp.where(one_w, vrg, Vg)
        Ag = jnp.where(one_w, Ag + dsrag, Ag)
        col = blk_j[jnp.arange(G)[:, None],
                    jnp.broadcast_to(loc[None, :], (G, D)), w[:, None]]
        app = has_w[:, None] & mask_j & ~one_w
        Vg = jnp.maximum(Vg - (col + jnp.where(found, RApost, RAg)) * app,
                         _CASCADE_NEG_LIMIT_MV)
        tg = jnp.where(one_w, jnp.asarray(t_ms, tg.dtype), tg)
    V_fin = jnp.where(
        found, Vg,
        jnp.maximum(Vg + Ig - Kg * RAg, _CASCADE_NEG_LIMIT_MV))
    # Masked scatter (``.at[ids]`` with ``jnp.where`` values: boolean-mask
    # indexing is not concrete under jit). Pad slots name cell 0, so pad
    # writes are rerouted to one dummy cell past the population (sliced
    # away): a later group's pad would otherwise clobber cell 0's real
    # results, since repeated scatter indices apply in order, last wins.
    # Every real cell is covered exactly once.
    slot_ids = members_j.reshape(-1)
    keep = mask_j.reshape(-1)
    slot_v = V_fin.reshape(-1)
    slot_a = Ag.reshape(-1)
    slot_t = tg.reshape(-1)
    slot_f = found.reshape(-1)
    N = int(Vd.shape[0])
    safe_ids = jnp.where(keep, slot_ids, N)
    V_ext = jnp.concatenate([Vd, jnp.zeros((1,), Vd.dtype)])
    A_ext = jnp.concatenate([Ad, jnp.zeros((1,), Ad.dtype)])
    t_ext = jnp.concatenate([t_last, jnp.zeros((1,), t_last.dtype)])
    V_out = V_ext.at[safe_ids].set(
        jnp.where(keep, slot_v, V_ext[safe_ids]))[:N]
    A_out = A_ext.at[safe_ids].set(
        jnp.where(keep, slot_a, A_ext[safe_ids]))[:N]
    t_out = t_ext.at[safe_ids].set(
        jnp.where(keep, slot_t, t_ext[safe_ids]))[:N]
    spk = jnp.zeros((N + 1,), bool).at[safe_ids].set(
        jnp.where(keep, slot_f, False))[:N]
    return (V_out, A_out, t_out), spk


def lif_step(state, I_mV, n_arr, t_ms, params: LIFParams, dt_ms: float,
             *, cap_rounds=None, cap_order: str = "ascending",
             cap_groups=None):
    """One clock step of the PCL LIF equations for a vector of cells.

    ``state`` is ``(V, A, t_last)``: membrane voltage (mV), adaptation (mV) and
    last spike time (ms, ``NEVER`` when the cell never spiked). ``I_mV`` is the
    summed signed input of the step; ``n_arr`` the arrival count (each arrival
    subtracts one ``R + A``, the clock translation of the per-event C++ term).
    ``t_ms`` is the step time. A cell spikes on strict ``V > v_thresh``
    (``SimpleNeuron.cpp`` / ``ComplexNeuron.cpp``); on spike ``V`` resets,
    ``A`` grows by ``delta_sra`` and ``t_last`` stamps ``t_ms``.

    ``cap_rounds``/``cap_order`` (both None by default: the step is unchanged)
    apply the declared same-step local-inhibition cap inside the step: when
    more cells cross threshold than ``cap_rounds``, only the first
    ``cap_rounds`` winners in ``cap_order`` (``"ascending"``/``"descending"``
    neuron index) keep their spikes; the rest revert to their non-spike state
    exactly (pre-reset voltage, undecayed adaptation, unstamped time), as if
    they had not crossed. Downstream (ring, HDP context) sees the capped
    spikes. A non-None ``cap_rounds`` must be a positive int; ``cap_order``
    one of the two orders — anything else is refused.

    ``cap_groups`` (None, or int ids of shape ``(n,)``) scopes the cap to one
    retinotopic position per group: the first-``cap_rounds`` rule resolves
    independently inside each group in ``cap_order`` (cells sharing an id
    compete; any label order or gaps are admitted). ``None`` keeps the
    whole-population cap. Set with ``cap_rounds`` None is refused.

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
    raw = V_new > p.v_thresh_mV
    if cap_rounds is None:
        if cap_groups is not None:
            raise ValueError(
                "cap_groups is set but cap_rounds is None; groups only scope "
                "a declared local-inhibition cap"
            )
        spk = raw
    else:
        from .pcl_topology import LOCAL_INHIBITION_ORDERS, validate_rounds

        n_rounds = validate_rounds(cap_rounds, label="local_inhibition")
        if cap_order not in LOCAL_INHIBITION_ORDERS:
            raise ValueError(
                f"local_inhibition order must be one of {list(LOCAL_INHIBITION_ORDERS)}; "
                f"got {cap_order!r}"
            )
        n = int(raw.shape[0])
        visit = jnp.arange(n) if cap_order == "ascending" else n - 1 - jnp.arange(n)
        if cap_groups is None:
            idx = visit
            ordered = raw[idx]
            keep_ordered = ordered & (jnp.cumsum(ordered.astype(jnp.int32)) <= n_rounds)
            keep = jnp.zeros_like(raw).at[idx].set(keep_ordered)
        else:
            g = jnp.asarray(cap_groups)
            if tuple(g.shape) != (n,):
                raise ValueError(
                    f"cap_groups must have shape ({n},); got {tuple(g.shape)}"
                )
            keep = _grouped_first_r(raw, g, visit, n_rounds)
        spk = raw & keep
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


def _check_inhibition_block_agreement(n, groups, blocks):
    """Shared position/magnitude agreement for the inhibition blocks.

    ``groups`` (already shape-checked ``(N,)``) scopes the positions;
    ``blocks`` must be a :func:`jaxfne.pcl_topology.local_inhibition_blocks`
    dict whose members agree with ``groups`` (same positions, each cell
    covered exactly once) with finite non-negative inhibition magnitudes.
    Returns ``(members, mask, weights)`` numpy arrays. Everything missing,
    misshapen or disagreeing is refused by name.
    """
    try:
        members = np.asarray(blocks["members"])
        mask = np.asarray(blocks["member_mask"])
        weights = np.asarray(blocks["weights"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "local_inhibition_blocks must carry 'members', 'member_mask' and "
            f"'weights'; got {exc!r}"
        ) from exc
    if members.ndim != 2 or mask.shape != members.shape:
        raise ValueError(
            "local_inhibition_blocks members/mask must share one (G, D) shape; got "
            f"{members.shape} and {mask.shape}"
        )
    ng, depth = int(members.shape[0]), int(members.shape[1])
    if weights.shape != (ng, depth, depth):
        raise ValueError(
            "local_inhibition_blocks weights must have shape "
            f"({ng}, {depth}, {depth}); got {weights.shape}"
        )
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError("local_inhibition_blocks weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError(
            "local_inhibition_blocks weights must hold non-negative inhibition "
            "magnitudes (the cascade subtracts them)"
        )
    if members.size and (
        not np.all(np.isfinite(members.astype(np.float64)))
        or bool((members < 0).any()) or bool((members >= n).any())
    ):
        raise ValueError(
            f"local_inhibition_blocks member ids must lie in [0, {n})"
        )
    grp = np.asarray(groups).ravel()
    covered = np.zeros((n,), dtype=bool)
    seen: set[int] = set()
    for g in range(ng):
        mem = members[g][np.asarray(mask[g], dtype=bool)]
        if mem.shape[0] == 0:
            raise ValueError(
                f"local_inhibition_blocks position {g} is empty; every position "
                "holds at least its cell"
            )
        gid = grp[mem.astype(np.int64)]
        if not bool((gid == gid[0]).all()):
            raise ValueError(
                f"local_inhibition_blocks position {g} spans several groups; local "
                "inhibition stays inside one retinotopic position"
            )
        if int(gid[0]) in seen:
            raise ValueError(
                "local_inhibition_blocks repeats group "
                f"{int(gid[0])}; positions and groups agree one to one"
            )
        seen.add(int(gid[0]))
        if bool(covered[mem.astype(np.int64)].any()):
            raise ValueError(
                "local_inhibition_blocks covers a cell twice; every cell is "
                "visited at exactly one position"
            )
        covered[mem.astype(np.int64)] = True
    if not bool(covered.all()):
        raise ValueError(
            "local_inhibition_blocks must cover every grouped cell exactly once; "
            f"{int((~covered).sum())} of {n} are uncovered"
        )
    return members, mask, weights


def _check_event_order_inputs(
    n, n_steps, groups, blocks, visit, dtype,
):
    """Validate the event-order fixed-point inputs; return jitted arrays.

    ``groups`` (already shape-checked ``(N,)``) scopes the positions;
    ``blocks`` must be a :func:`jaxfne.pcl_topology.local_inhibition_blocks`
    dict whose members agree with ``groups`` (same positions, each cell
    covered exactly once) with finite non-negative inhibition magnitudes;
    ``visit`` must be the int ``(n_steps, G, D)`` per-step local-id visit
    order (member slots a permutation of the position's local ids, padded
    slots ``-1``). Returns ``(members_j, mask_j, blk_j, visit_j)`` as jax
    arrays. Everything missing, misshapen or disagreeing is refused by name.
    """
    if blocks is None:
        raise ValueError(
            "local_inhibition_resolution='event_order' needs "
            "local_inhibition_blocks; without inhibition magnitudes no cascade runs"
        )
    if visit is None:
        raise ValueError(
            "local_inhibition_resolution='event_order' needs "
            "local_inhibition_visit_order; without the per-step visit no cascade runs"
        )
    members, mask, weights = _check_inhibition_block_agreement(n, groups, blocks)
    ng, depth = int(members.shape[0]), int(members.shape[1])
    vis = np.asarray(visit)
    if vis.shape != (n_steps, ng, depth):
        raise ValueError(
            "local_inhibition_visit_order must have shape "
            f"({n_steps}, {ng}, {depth}); got {vis.shape}"
        )
    if vis.dtype.kind not in "iu":
        raise ValueError(
            "local_inhibition_visit_order must hold integer local ids; "
            f"got {vis.dtype}"
        )
    for g in range(ng):
        exp = np.flatnonzero(np.asarray(mask[g], dtype=bool))
        vg = vis[:, g, :]
        pad1 = np.asarray(mask[g], dtype=bool)
        if bool((vg[:, ~pad1] != -1).any()):
            raise ValueError(
                "local_inhibition_visit_order pads non-member slots with -1; "
                f"position {g} holds another id"
            )
        # Member columns hold each local id exactly once per step (no -1:
        # padding lives in non-member columns only); the sort must recover
        # the member local ids.
        row = np.sort(vg[:, pad1], axis=1)
        want = np.sort(exp)[None, :]
        if not bool((row == want).all()):
            raise ValueError(
                "local_inhibition_visit_order must visit every member local id "
                f"exactly once per step; position {g} does not"
            )
    return (
        jnp.asarray(members, dtype=jnp.int32),
        jnp.asarray(mask, dtype=bool),
        jnp.asarray(weights, dtype=dtype),
        jnp.asarray(vis, dtype=jnp.int32),
    )


def _event_scan_loop(
    V0, A0, EV, CELL, CMASK, SOX, SOY, WS,
    FWD_IDX, FWD_W, FWD_N, LAT_IDX, LAT_W, LAT_N,
    TD_IDX, TD_W, TD_N, MEMBERS, MMASK, BLK, POS_OF, LOC_OF,
    PCOL, GRID, FLAGS, BIN0,
):
    """One exact event-oracle pass over a time-sorted event stream.

    The outer ``jax.lax.scan`` walks the ``(M, 4)`` ``[t_us, x, y, p]``
    rows in order; each event excites its covered simple cells in
    ascending index with exact inter-event decay, and each spike runs
    local inhibition, then lateral inhibition (when flagged), then the
    forward pass exactly as ``pcl_event_ref.py:663-720`` (complex spikes
    run complex-local inhibition, then top-down when flagged). State
    vectors carry one dummy cell past the population so padded target
    slots scatter without touching a real cell; outputs slice it away.
    The forward fan-out loop is gated on the spike (``cond`` around the
    ``fori_loop``): an unwound per-target loop would otherwise execute
    hundreds of masked excites for every non-spiking candidate. All
    updates are indexed scatters; the caller runs this in float64
    (integer-microsecond event times against float64 taus).
    """
    f = V0.dtype
    i32 = jnp.int32
    SMAX = CELL.shape[1]
    FMAX = FWD_IDX.shape[1]
    LMAX = LAT_IDX.shape[1]
    TMAX = TD_IDX.shape[1]
    D = MEMBERS.shape[1]
    NS = FWD_IDX.shape[0]
    N1 = V0.shape[0]
    DUM = N1 - 1
    VW, _NX, NY, ND = GRID[0], GRID[1], GRID[2], GRID[3]
    T0_US, DT_US = GRID[4], GRID[5]
    zero_f = jnp.zeros((), f)

    def _excite_one(Vc, Ac, TLc, TSc, n, w, t):
        p = PCOL[n]
        dt_ = t - TLc[n]
        Vd = Vc[n] * jnp.exp(-dt_ / p[0])
        Ad = Ac[n] * jnp.exp(-dt_ / p[1])
        R = jnp.where(TSc[n] > zero_f,
                      p[5] * jnp.exp(-(t - TSc[n]) / p[6]), zero_f)
        Vn = jnp.maximum(p[4], Vd + w - R - Ad)
        return Vd, Ad, Vn, Vn > p[2]

    def _inhibit_vec(Vc, Ac, TLc, TSc, ids, wts, use, t):
        safe = jnp.where(use, ids, DUM)
        Vo = Vc[safe]
        Ao = Ac[safe]
        Tl = TLc[safe]
        Ts = TSc[safe]
        P = PCOL[safe]
        dt_ = t - Tl
        Vd = Vo * jnp.exp(-dt_ / P[:, 0])
        Ad = Ao * jnp.exp(-dt_ / P[:, 1])
        R = jnp.where(Ts > zero_f,
                      P[:, 5] * jnp.exp(-(t - Ts) / P[:, 6]),
                      jnp.zeros_like(Ts))
        Vn = jnp.maximum(P[:, 4], Vd - wts - R - Ad)
        Vc = Vc.at[safe].set(jnp.where(use, Vn, Vo))
        Ac = Ac.at[safe].set(jnp.where(use, Ad, Ao))
        TLc = TLc.at[safe].set(jnp.where(use, t, Tl))
        return Vc, Ac, TLc

    def _event_body(carry, e):
        V, A, TL, TS, CNT, FST, BIN = carry
        ti = e[0]
        x = e[1].astype(i32)
        y = e[2].astype(i32)
        pp = e[3].astype(i32)
        t = ti.astype(f)
        q = y * VW + x
        b = ((ti - T0_US) // DT_US).astype(i32)
        slots = CELL[q]
        sm = CMASK[q]

        def _slot(s, c):
            # Padded slots (~half the slots on S1: mean pixel fanout 132
            # of SMAX 256) skip the excites entirely through a ``cond``:
            # without it every pad burns two exponentials whose results
            # are discarded through ``m``.
            return jax.lax.cond(
                sm[s], lambda c: _do_slot(s, c), lambda c: c, c)

        def _do_slot(s, c):
            V, A, TL, TS, CNT, FST, BIN = c
            m = sm[s]
            n = slots[s]
            n0 = jnp.where(m, n, DUM)
            pos = n0 // ND
            k = n0 % ND
            ii = pos // NY
            jj = pos % NY
            # Padded slots still index the tables (their results are
            # discarded through ``m``), so they ride valid indices.
            ii_s = jnp.where(m, ii, 0)
            jj_s = jnp.where(m, jj, 0)
            k_s = jnp.where(m, k, 0)
            xr = x - SOX[ii_s]
            yr = y - SOY[jj_s]
            w = jnp.where(m, WS[k_s, pp, xr, yr], jnp.zeros((), f))
            Vd, Ad, Vn, cross = _excite_one(V, A, TL, TS, n0, w, t)
            spk = m & cross
            V = V.at[n0].set(
                jnp.where(m, jnp.where(spk, PCOL[n0, 3], Vn), V[n0]))
            A = A.at[n0].set(
                jnp.where(m, Ad + jnp.where(spk, PCOL[n0, 7], zero_f), A[n0]))
            TL = TL.at[n0].set(jnp.where(m, t, TL[n0]))
            TS = TS.at[n0].set(jnp.where(spk, t, TS[n0]))
            CNT = CNT.at[n0].set(CNT[n0] + spk.astype(i32))
            FST = FST.at[n0].set(
                jnp.where(spk & (FST[n0] < jnp.int64(0)), ti, FST[n0]))
            BIN = BIN.at[b, n0].set(BIN[b, n0] | spk)

            def _cascade(c2):
                V, A, TL, TS, CNT, FST, BIN = c2
                g = POS_OF[n0]
                k = LOC_OF[n0]
                mm = MMASK[g] & (jnp.arange(D) != k)
                V, A, TL = _inhibit_vec(
                    V, A, TL, TS, MEMBERS[g], BLK[g, :, k], mm, t)

                def _lat(c3):
                    V, A, TL, TS, CNT, FST, BIN = c3
                    ln = LAT_N[n0]
                    use = jnp.arange(LMAX) < ln
                    V, A, TL = _inhibit_vec(
                        V, A, TL, TS, LAT_IDX[n0], LAT_W[n0], use, t)
                    return (V, A, TL, TS, CNT, FST, BIN)

                lat_fire = (FLAGS[0] > 0) & (FLAGS[2] > 0)
                V, A, TL, TS, CNT, FST, BIN = jax.lax.cond(
                    lat_fire, _lat, lambda c3: c3,
                    (V, A, TL, TS, CNT, FST, BIN))

                def _fwd(ff, c4):
                    V, A, TL, TS, CNT, FST, BIN = c4
                    fm = ff < FWD_N[n0]
                    c0 = jnp.where(fm, FWD_IDX[n0, ff], DUM)
                    w = FWD_W[n0, ff]
                    _Vd, _Ad, Vn, cross = _excite_one(
                        V, A, TL, TS, c0, w, t)
                    spkc = fm & cross
                    V = V.at[c0].set(jnp.where(
                        fm, jnp.where(spkc, PCOL[c0, 3], Vn), V[c0]))
                    A = A.at[c0].set(jnp.where(
                        fm,
                        _Ad + jnp.where(spkc, PCOL[c0, 7], zero_f),
                        A[c0]))
                    TL = TL.at[c0].set(jnp.where(fm, t, TL[c0]))
                    TS = TS.at[c0].set(jnp.where(spkc, t, TS[c0]))
                    CNT = CNT.at[c0].set(CNT[c0] + spkc.astype(i32))
                    FST = FST.at[c0].set(jnp.where(
                        spkc & (FST[c0] < jnp.int64(0)), ti, FST[c0]))
                    BIN = BIN.at[b, c0].set(BIN[b, c0] | spkc)

                    def _ccascade(c5):
                        V, A, TL, TS, CNT, FST, BIN = c5
                        cg = POS_OF[c0]
                        ck = LOC_OF[c0]
                        cm = MMASK[cg] & (jnp.arange(D) != ck)
                        V, A, TL = _inhibit_vec(
                            V, A, TL, TS, MEMBERS[cg], BLK[cg, :, ck], cm, t)

                        def _td(c6):
                            V, A, TL, TS, CNT, FST, BIN = c6
                            cs = c0 - NS
                            use = jnp.arange(TMAX) < TD_N[cs]
                            V, A, TL = _inhibit_vec(
                                V, A, TL, TS, TD_IDX[cs], TD_W[cs], use, t)
                            return (V, A, TL, TS, CNT, FST, BIN)

                        td_fire = (FLAGS[1] > 0) & (FLAGS[2] > 0)
                        return jax.lax.cond(
                            td_fire, _td, lambda c6: c6,
                            (V, A, TL, TS, CNT, FST, BIN))

                    return jax.lax.cond(
                        spkc, _ccascade, lambda c5: c5,
                        (V, A, TL, TS, CNT, FST, BIN))

                # The fan-out loop is gated on the spike: without the
                # ``cond`` every candidate would run FMAX masked excites.
                return jax.lax.cond(
                    spk, lambda c4: jax.lax.fori_loop(
                        0, FMAX, _fwd, c4),
                    lambda c4: c4,
                    (V, A, TL, TS, CNT, FST, BIN))

            return jax.lax.cond(
                spk, _cascade, lambda c2: c2,
                (V, A, TL, TS, CNT, FST, BIN))

        return jax.lax.fori_loop(0, SMAX, _slot, carry), jnp.int32(0)

    carry0 = (
        V0, A0,
        jnp.zeros((N1,), f), jnp.zeros((N1,), f),
        jnp.zeros((N1,), jnp.int32), jnp.full((N1,), -1, dtype=jnp.int64),
        BIN0,
    )
    (V, A, TL, TS, CNT, FST, BIN), _ = jax.lax.scan(_event_body, carry0, EV)
    return V, A, TL, TS, CNT, FST, BIN


_JIT_EVENT_SCAN = jax.jit(_event_scan_loop)


def _scan_table_int(tables, key):
    """Integer geometry entry of the event-scan tables (fail-closed)."""
    try:
        value = tables[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"event_scan_tables is missing {key!r}; the scan needs the full "
            f"cover/forward/lateral/top-down tables ({exc!r})"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"event_scan_tables[{key!r}] must be an integer; got {value!r}")
    return int(value)


def _run_event_scan(
    params, N, n_steps, t0_us, dt_us, groups, blocks,
    events, tables, V0_mV,
):
    """Validate the event-scan inputs and run the exact event pass.

    Returns the kernel result dict (``spikes`` bool ``(n_steps, N)``,
    ``counts``, ``first_spike_us``, finals ``V``/``A``/``t_last`` and an
    empty ``w_final``). ``V``/``A``/``t_last`` are float64 regardless of the
    params dtype (see the x64 note at the call). Every refusal names the
    offending channel.
    """
    grp = np.asarray(groups).ravel()
    if grp.shape != (N,) or grp.dtype.kind not in "iu":
        raise ValueError(
            "local_inhibition_groups for resolution 'event_scan' must hold "
            f"integer ids of shape ({N},); got shape {grp.shape} dtype {grp.dtype}"
        )
    members, member_mask, weights = _check_inhibition_block_agreement(
        N, grp, blocks)
    if not isinstance(tables, Mapping):
        raise ValueError(
            "event_scan_tables must be the mapping event_scan_tables builds "
            f"(cover/forward/lateral/top-down); got {tables!r}")
    nx = _scan_table_int(tables, "nx")
    ny = _scan_table_int(tables, "ny")
    nd = _scan_table_int(tables, "nd")
    cx = _scan_table_int(tables, "cx")
    cy = _scan_table_int(tables, "cy")
    cd = _scan_table_int(tables, "cd")
    ns = _scan_table_int(tables, "ns")
    nc = _scan_table_int(tables, "nc")
    vw = _scan_table_int(tables, "vw")
    vh = _scan_table_int(tables, "vh")
    for _k, _v in (("nx", nx), ("ny", ny), ("nd", nd), ("cx", cx),
                   ("cy", cy), ("cd", cd), ("ns", ns), ("nc", nc),
                   ("vw", vw), ("vh", vh)):
        if _v < 1:
            raise ValueError(
                f"event_scan_tables[{_k!r}] must be >= 1; got {_v!r}")
    if ns != nx * ny * nd or nc != cx * cy * cd or ns + nc != N:
        raise ValueError(
            "event_scan_tables geometry (nx, ny, nd, cx, cy, cd, ns, nc) "
            f"disagrees with the ({N},) population")
    try:
        labs = tuple(params.labels)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "event_scan_tables describes a simple/complex two-layer net; "
            f"params must carry labels ({exc!r})") from exc
    if labs != ("simple",) * ns + ("complex",) * nc:
        raise ValueError(
            "event_scan_tables describes a simple/complex two-layer net with "
            f"{ns} simple then {nc} complex cells; params labels disagree")
    cols = {f: np.asarray(getattr(params, f)) for f in _PARAM_FIELDS}
    for f, a in cols.items():
        if a.shape != (N,):
            raise ValueError(
                f"params {f} for resolution 'event_scan' must have shape "
                f"({N},); got {a.shape}")
    if not bool(np.all(np.isfinite(cols["tau_m_ms"]))) or bool(
            (cols["tau_m_ms"] <= 0).any()):
        raise ValueError(
            "params tau_m_ms for resolution 'event_scan' must be finite and > 0")
    if not bool(np.all(np.isfinite(cols["tau_rp_ms"]))) or bool(
            (cols["tau_rp_ms"] <= 0).any()):
        raise ValueError(
            "params tau_rp_ms for resolution 'event_scan' must be finite and > 0")
    _sra = cols["tau_sra_ms"]
    if not bool(np.all(np.isfinite(_sra) | (_sra == np.inf))) or bool(
            (_sra <= 0).any()):
        raise ValueError(
            "params tau_sra_ms for resolution 'event_scan' must be finite "
            "and > 0 (or +inf)")
    for f in ("v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
              "delta_sra_mV"):
        if not bool(np.all(np.isfinite(cols[f]))):
            raise ValueError(
                f"params {f} for resolution 'event_scan' must be finite")
    if bool((cols["delta_sra_mV"][ns:] != 0.0).any()):
        raise ValueError(
            "params delta_sra_mV for resolution 'event_scan' must be 0 on "
            "complex cells; the oracle carries no adaptation term there")
    try:
        w_s = np.asarray(tables["w_s"], dtype=np.float64)
        pix_pos = np.asarray(tables["pix_pos"])
        pix_n = np.asarray(tables["pix_n"])
        fwd_idx = np.asarray(tables["fwd_idx"])
        fwd_w = np.asarray(tables["fwd_w"], dtype=np.float64)
        fwd_n = np.asarray(tables["fwd_n"])
        lat_idx = np.asarray(tables["lat_idx"])
        lat_n = np.asarray(tables["lat_n"])
        td_idx = np.asarray(tables["td_idx"])
        td_n = np.asarray(tables["td_n"])
        sox = np.asarray(tables["s_ox"])
        soy = np.asarray(tables["s_oy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "event_scan_tables must carry the cover/forward/lateral/top-down "
            f"arrays event_scan_tables builds ({exc!r})") from exc
    sx = w_s.shape[2] if w_s.ndim == 4 else -1
    sy = w_s.shape[3] if w_s.ndim == 4 else -1
    if w_s.shape != (nd, 2, sx, sy) or sx < 1 or sy < 1:
        raise ValueError(
            f"event_scan_tables['w_s'] must have shape ({nd}, 2, sx, sy); "
            f"got {w_s.shape}")
    if not bool(np.all(np.isfinite(w_s))):
        raise ValueError("event_scan_tables['w_s'] must be finite")
    lat_w = tables.get("lat_w", None)
    lat_w = None if lat_w is None else np.asarray(lat_w, dtype=np.float64)
    td_w = tables.get("td_w", None)
    td_w = None if td_w is None else np.asarray(td_w, dtype=np.float64)
    for _k in ("lat_on", "td_on", "activation"):
        if _k not in tables or not isinstance(tables[_k], (bool, np.bool_)):
            raise ValueError(
                f"event_scan_tables[{_k!r}] must be a bool; "
                f"got {tables.get(_k, None)!r}")
    lat_on, td_on, activation = (
        bool(tables["lat_on"]), bool(tables["td_on"]),
        bool(tables["activation"]))
    npos = nx * ny
    if pix_pos.ndim != 2 or pix_pos.shape[0] != vw * vh:
        raise ValueError(
            "event_scan_tables['pix_pos'] must have one row per pixel "
            f"({vw * vh}); got {pix_pos.shape}")
    if pix_pos.dtype.kind not in "iu":
        raise ValueError("event_scan_tables['pix_pos'] must hold integers")
    pmax = pix_pos.shape[1]
    if pix_n.shape != (vw * vh,) or pix_n.dtype.kind not in "iu":
        raise ValueError(
            "event_scan_tables['pix_n'] must hold one integer count per "
            f"pixel; got shape {pix_n.shape} dtype {pix_n.dtype}")
    if bool((pix_n < 0).any()) or bool((pix_n > pmax).any()):
        raise ValueError("event_scan_tables['pix_n'] must lie in [0, cover width]")
    if bool((pix_pos < 0).any()) or bool((pix_pos >= npos).any()):
        raise ValueError(
            "event_scan_tables['pix_pos'] must name simple positions in "
            f"[0, {npos})")
    _slot = np.arange(pmax)[None, :]
    _ok = np.where(_slot < pix_n[:, None], pix_pos,
                   np.iinfo(np.int64).max)
    if pix_pos.dtype.kind not in "iu" or not bool(
            (np.diff(_ok.astype(np.int64), axis=1) >= 0).all()):
        raise ValueError(
            "event_scan_tables['pix_pos'] rows must list covered positions "
            "ascending (the oracle visits candidates in ascending index)")
    if (fwd_idx.ndim != 2 or fwd_idx.shape[0] != ns
            or fwd_idx.dtype.kind not in "iu"):
        raise ValueError(
            "event_scan_tables['fwd_idx'] must hold integer targets with one "
            f"row per simple cell ({ns}); got {fwd_idx.shape}")
    fmax = fwd_idx.shape[1]
    if fwd_w.shape != (ns, fmax) or not bool(np.all(np.isfinite(fwd_w))):
        raise ValueError(
            "event_scan_tables['fwd_w'] must be finite with shape "
            f"({ns}, {fmax})")
    if fwd_n.shape != (ns,) or fwd_n.dtype.kind not in "iu":
        raise ValueError(
            "event_scan_tables['fwd_n'] must hold one integer count per "
            "simple cell")
    if bool((fwd_n < 0).any()) or bool((fwd_n > fmax).any()):
        raise ValueError(
            "event_scan_tables['fwd_n'] must lie in [0, forward width]")
    if bool((fwd_idx < 0).any()) or bool((fwd_idx >= N).any()):
        raise ValueError(
            "event_scan_tables['fwd_idx'] must name cells in [0, "
            f"{N})")
    _fslot = np.arange(fmax)[None, :] < fwd_n[:, None]
    if bool((fwd_idx[_fslot] < ns).any()):
        raise ValueError(
            "event_scan_tables['fwd_idx'] must name complex cells "
            f"(simple cells end at {ns}); the forward pass only feeds "
            "the complex layer")
    if (lat_idx.ndim != 2 or lat_idx.shape[0] != ns
            or lat_idx.dtype.kind not in "iu"):
        raise ValueError(
            "event_scan_tables['lat_idx'] must hold integer targets with one "
            f"row per simple cell ({ns}); got {lat_idx.shape}")
    lmax = lat_idx.shape[1]
    if lat_w is not None and (
            lat_w.shape != (ns, lmax)
            or not bool(np.all(np.isfinite(lat_w)))):
        raise ValueError(
            "event_scan_tables['lat_w'] must be None or finite with shape "
            f"({ns}, {lmax})")
    if lat_n.shape != (ns,) or lat_n.dtype.kind not in "iu":
        raise ValueError(
            "event_scan_tables['lat_n'] must hold one integer count per "
            "simple cell")
    if bool((lat_n < 0).any()) or bool((lat_n > lmax).any()):
        raise ValueError(
            "event_scan_tables['lat_n'] must lie in [0, lateral width]")
    if bool((lat_idx < 0).any()) or bool((lat_idx >= ns).any()):
        raise ValueError(
            "event_scan_tables['lat_idx'] must name simple cells in [0, "
            f"{ns})")
    if (td_idx.ndim != 2 or td_idx.shape[0] != nc
            or td_idx.dtype.kind not in "iu"):
        raise ValueError(
            "event_scan_tables['td_idx'] must hold integer targets with one "
            f"row per complex cell ({nc}); got {td_idx.shape}")
    tmax = td_idx.shape[1]
    if td_w is not None and (
            td_w.shape != (nc, tmax)
            or not bool(np.all(np.isfinite(td_w)))):
        raise ValueError(
            "event_scan_tables['td_w'] must be None or finite with shape "
            f"({nc}, {tmax})")
    if td_n.shape != (nc,) or td_n.dtype.kind not in "iu":
        raise ValueError(
            "event_scan_tables['td_n'] must hold one integer count per "
            "complex cell")
    if bool((td_n < 0).any()) or bool((td_n > tmax).any()):
        raise ValueError(
            "event_scan_tables['td_n'] must lie in [0, top-down width]")
    if bool((td_idx < 0).any()) or bool((td_idx >= ns).any()):
        raise ValueError(
            "event_scan_tables['td_idx'] must name simple cells in [0, "
            f"{ns})")
    if sox.shape != (nx,) or soy.shape != (ny,):
        raise ValueError(
            "event_scan_tables['s_ox'/'s_oy'] must hold the simple grid "
            f"origins (({nx},) and ({ny},))")
    ev = np.asarray(events)
    if ev.ndim != 2 or ev.shape[1] != 4:
        raise ValueError(
            "event_scan_events must have shape (M, 4) with [t_us, x, y, p] "
            f"rows; got {ev.shape}")
    if ev.dtype.kind not in "iu":
        raise ValueError(
            f"event_scan_events must hold integers; got {ev.dtype}")
    ev = ev.astype(np.int64)
    tcol, xcol, ycol, pcol = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
    if ev.shape[0] and bool((np.diff(tcol) < 0).any()):
        raise ValueError(
            "event_scan_events must be stably sorted by time (event order, "
            "then ascending index on A)")
    if ev.shape[0]:
        if bool((xcol < 0).any()) or bool((xcol >= vw).any()):
            raise ValueError(
                f"event_scan_events x must lie on the {vw}-pixel plane")
        if bool((ycol < 0).any()) or bool((ycol >= vh).any()):
            raise ValueError(
                f"event_scan_events y must lie on the {vh}-pixel plane")
        if bool((pcol < 0).any()) or bool((pcol > 1).any()):
            raise ValueError("event_scan_events polarity must lie in {0, 1}")
        bins = (tcol - t0_us) // dt_us
        if bool((bins < 0).any()) or bool((bins >= n_steps).any()):
            raise ValueError(
                "event_scan_events times must fall on the "
                f"[{t0_us}, {t0_us + n_steps * dt_us}) microsecond grid")
    if V0_mV is None:
        v_init = np.zeros((N,), dtype=np.float64)
    else:
        v_init = np.asarray(V0_mV, dtype=np.float64)
        if v_init.shape != (N,):
            raise ValueError(
                f"V0_mV must be None or shape ({N},); got {v_init.shape}")
        if not bool(np.all(np.isfinite(v_init))):
            raise ValueError("V0_mV must be finite (got NaN/inf)")
    smax = max(1, pmax * nd)
    cell = np.full((vw * vh, smax), -1, dtype=np.int64)
    cmask = np.zeros((vw * vh, smax), dtype=bool)
    for q in range(vw * vh):
        s = 0
        for slot in range(int(pix_n[q])):
            pos = int(pix_pos[q, slot])
            base = pos * nd
            for kk in range(nd):
                cell[q, s] = base + kk
                cmask[q, s] = True
                s += 1
    ng, depth = members.shape
    row_of = np.full((int(grp.max()) + 1 if grp.size else 0,), -1,
                     dtype=np.int64)
    for g in range(ng):
        mem = members[g][np.asarray(member_mask[g], dtype=bool)]
        row_of[int(grp[mem.astype(np.int64)][0])] = g
    pos_of = np.zeros((N + 1,), dtype=np.int64)
    loc_of = np.zeros((N + 1,), dtype=np.int64)
    for c in range(N):
        pos_of[c] = row_of[int(grp[c])]
        loc_of[c] = int(np.flatnonzero(
            members[int(pos_of[c])] == c)[0])
    # Column order is the kernel's (tau_m, tau_sra, v_thresh, v_reset,
    # v_min, eta_rp, tau_rp, delta_sra), not _PARAM_FIELDS; taus ride in
    # microseconds: the oracle integrates integer-microsecond event times
    # (pcl_event_ref.py:64, excite_simple :228-231).
    pcol = np.stack(
        [cols["tau_m_ms"] * 1000.0, cols["tau_sra_ms"] * 1000.0,
         cols["v_thresh_mV"], cols["v_reset_mV"], cols["v_min_mV"],
         cols["eta_rp_mV"], cols["tau_rp_ms"] * 1000.0,
         cols["delta_sra_mV"]], axis=1).astype(np.float64)
    pcol = np.vstack([pcol, pcol[:1]])
    V0 = np.concatenate([v_init, np.zeros((1,), dtype=np.float64)])
    A0 = np.zeros((N + 1,), dtype=np.float64)
    # float64 under a local x64 context (restored on exit): event times are
    # integer microseconds and the oracle is float64 — float32 decay rounding
    # would move threshold crossings.
    with jax.enable_x64():
        out = _JIT_EVENT_SCAN(
            jnp.asarray(V0, dtype=jnp.float64),
            jnp.asarray(A0, dtype=jnp.float64),
            jnp.asarray(ev),
            jnp.asarray(cell, dtype=jnp.int32),
            jnp.asarray(cmask, dtype=bool),
            jnp.asarray(sox.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(soy.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(w_s, dtype=jnp.float64),
            jnp.asarray(fwd_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(fwd_w, dtype=jnp.float64),
            jnp.asarray(fwd_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(lat_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(np.zeros((ns, lmax), dtype=np.float64)
                       if lat_w is None else lat_w, dtype=jnp.float64),
            jnp.asarray(lat_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(td_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(np.zeros((nc, tmax), dtype=np.float64)
                       if td_w is None else td_w, dtype=jnp.float64),
            jnp.asarray(td_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(members.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(member_mask, dtype=bool),
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(pos_of, dtype=jnp.int32),
            jnp.asarray(loc_of, dtype=jnp.int32),
            jnp.asarray(pcol, dtype=jnp.float64),
            jnp.asarray([vw, nx, ny, nd, t0_us, dt_us], dtype=jnp.int64),
            jnp.asarray([lat_on, td_on, activation], dtype=jnp.int32),
            jnp.zeros((n_steps, N + 1), dtype=bool),
        )
        V, A, _TL, TS, CNT, FST, BIN = out
        t_last = jnp.where(
            CNT[:N] > 0, TS[:N] / 1000.0,
            jnp.full((N,), NEVER, dtype=jnp.float64))
        return {
            "spikes": BIN[:, :N],
            "V": V[:N],
            "A": A[:N],
            "t_last": t_last,
            "w_final": jnp.zeros((0,), dtype=jnp.float64),
            "counts": CNT[:N],
            "first_spike_us": FST[:N],
        }


def _run_event_scan_branch(
    params, N, n_steps, dt_ms, t0,
    local_inhibition_rounds, local_inhibition_order,
    local_inhibition_groups, local_inhibition_blocks,
    local_inhibition_visit_order, scan_events, scan_tables,
    edge_pre, edge_post, edge_weight, edge_delay_steps, edge_group,
    I_ext_pA, V0_mV, H0, record_v,
    hdp_rule, hdp_rule_params, plasticity_mask,
    feed_current, feed_arrivals,
):
    """Refuse every non-scan channel, then run the exact event replay.

    The scan replays the timestamped stream instead of the binned drive,
    so rounds, the visit order, edges, sustained drive, feed channels,
    rules and recording would change nothing: each is refused by name.
    ``key`` stays allowed (without a rule it seeds nothing, as on the
    main path).
    """
    if local_inhibition_rounds is not None:
        raise ValueError(
            "local_inhibition_rounds is set but resolution is 'event_scan'; "
            "the replay resolves every event, no cap"
        )
    if local_inhibition_order != "ascending":
        raise ValueError(
            f"local_inhibition_order={local_inhibition_order!r} with resolution "
            "'event_scan' is refused; the scan visits candidates in ascending "
            "index (oracle pcl_event_ref.py:666)"
        )
    if local_inhibition_visit_order is not None:
        raise ValueError(
            "local_inhibition_visit_order is set but resolution is 'event_scan'; "
            "the replay walks the event stream, no visit array"
        )
    if local_inhibition_groups is None:
        raise ValueError(
            "local_inhibition_resolution='event_scan' needs "
            "local_inhibition_groups; without positions there is nothing to replay"
        )
    if local_inhibition_blocks is None:
        raise ValueError(
            "local_inhibition_resolution='event_scan' needs "
            "local_inhibition_blocks; without inhibition magnitudes no replay runs"
        )
    if scan_events is None:
        raise ValueError(
            "local_inhibition_resolution='event_scan' needs event_scan_events; "
            "without the timestamped stream there is nothing to replay"
        )
    if scan_tables is None:
        raise ValueError(
            "local_inhibition_resolution='event_scan' needs event_scan_tables; "
            "without the cover/forward/lateral/top-down tables no replay runs"
        )
    if feed_current is not None:
        raise ValueError(
            "feed_current is set but resolution is 'event_scan'; the replay "
            "feeds from event_scan_events, not the binned feed"
        )
    if feed_arrivals is not None:
        raise ValueError(
            "feed_arrivals is set but resolution is 'event_scan'; the replay "
            "feeds from event_scan_events, not the binned feed"
        )
    set_edges = [name for name, val in (
        ("edge_pre", edge_pre), ("edge_post", edge_post),
        ("edge_weight", edge_weight), ("edge_delay_steps", edge_delay_steps),
    ) if val is not None]
    if set_edges:
        raise ValueError(
            f"{'/'.join(set_edges)} is set but resolution is 'event_scan'; "
            "the replay drives simple cells from the event cover, no edges"
        )
    if edge_group is not None:
        raise ValueError(
            "edge_group is set but resolution is 'event_scan'; the replay "
            "runs no edges"
        )
    if I_ext_pA is not None:
        raise ValueError(
            "I_ext_pA is set but resolution is 'event_scan'; the replay "
            "carries no sustained drive"
        )
    if record_v:
        raise ValueError(
            "record_v is set but resolution is 'event_scan'; the replay "
            "returns final V/A only"
        )
    if hdp_rule is not None or hdp_rule_params is not None:
        raise ValueError(
            "hdp_rule/hdp_rule_params is set but resolution is 'event_scan'; "
            "the replay runs frozen weights"
        )
    if plasticity_mask is not None:
        raise ValueError(
            "plasticity_mask is set but resolution is 'event_scan'; the "
            "replay runs frozen weights"
        )
    if H0 is not None:
        raise ValueError(
            "H0 is set but resolution is 'event_scan'; the replay carries "
            "no rule state"
        )
    dt_us_f = float(dt_ms) * 1000.0
    if dt_us_f != math.floor(dt_us_f) or dt_us_f < 1:
        raise ValueError(
            "dt_ms must resolve to whole microseconds under resolution "
            f"'event_scan'; got {dt_ms!r}"
        )
    t0_us_f = t0 * 1000.0
    if t0_us_f != math.floor(t0_us_f):
        raise ValueError(
            "t0_ms must resolve to whole microseconds under resolution "
            f"'event_scan'; got {t0!r}"
        )
    return _run_event_scan(
        params, N, n_steps, int(t0_us_f), int(dt_us_f),
        local_inhibition_groups, local_inhibition_blocks,
        scan_events, scan_tables, V0_mV,
    )


def simulate_lif_edge(
    params: LIFParams,
    *,
    n_steps: int,
    dt_ms: float,
    edge_pre=None,
    edge_post=None,
    edge_weight=None,
    edge_delay_steps=None,
    edge_group=None,
    t0_ms: float = 0.0,
    I_ext_pA=None,
    V0_mV=None,
    H0=None,
    record_v: bool = False,
    hdp_rule=None,
    hdp_rule_params=None,
    plasticity_mask=None,
    local_inhibition_rounds=None,
    local_inhibition_order: str = "ascending",
    local_inhibition_groups=None,
    local_inhibition_resolution=None,
    local_inhibition_blocks=None,
    local_inhibition_visit_order=None,
    event_scan_events=None,
    event_scan_tables=None,
    feed_current=None,
    feed_arrivals=None,
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
    3. :func:`lif_step` (with the ``local_inhibition_rounds``/``order`` cap
       when declared: at most that many winners keep their spikes per step,
       in that index order, inside the step);
    4. the registered rule (if any) observes the step through the
       :class:`HDPRuleContext` contract and its rate update integrates as
       ``x + dt * dx`` (H clipped to the descriptor bounds, weight magnitudes
       to ``w_bounds`` with the initial sign kept, ``plasticity_mask`` gating
       which edges learn);
    5. this step's spikes are written into the ring.

    ``edge_group`` (None, or int ids of shape ``(n_edges,)``) names the
    weight-sharing group of each edge: members of one group execute one
    shared weight (the group mean of ``edge_weight``) and every HDP update
    is scatter-averaged back onto the shared row, so an update arriving on
    one member moves all of them. ``None`` gives every edge its own group.

    ``local_inhibition_rounds`` (None, or a positive int) with
    ``local_inhibition_order`` (``"ascending"``/``"descending"``) is the
    generic-tensor realization of the declared B3 intra-step fixed point
    (rounds and order from A). ``local_inhibition_groups`` (None, or int
    ids of shape ``(N,)``) scopes the cap to one retinotopic position per
    group — cells sharing an id compete, so co-located depth slots resolve
    their own winners; ``None`` keeps the whole-population cap. Both rounds
    and order change who spikes, so both ride the call, and so do the
    groups.

    ``local_inhibition_resolution`` (None, ``"whole_step"``,
    ``"event_order"`` or ``"event_scan"``) selects how the step resolves:
    ``None``/``"whole_step"``
    is the grouped first-R cap above (the default, so undeclared specs run
    byte-identically); ``"event_order"`` runs the declared intra-step fixed
    point (Hamm 2026-10-14) — each fed cell visited in feed order with ties
    in ascending index (oracle ``pcl_event_ref.py:663-697``), every spike
    subtracting its local inhibition from later group-mates before the next
    visit. ``"event_scan"`` replays the event stream exactly instead of the
    binned feed: one ``jax.lax.scan`` pass over the time-sorted
    ``event_scan_events`` ``(M, 4)`` ``[t_us, x, y, p]`` rows (as
    :func:`jaxfne.pcl_topology.event_scan_events` builds them), each event
    exciting its covered simple cells in ascending index with exact
    inter-event decay, each spike running local inhibition, then lateral
    inhibition (when the tables flag it), then the forward pass exactly as
    the oracle (complex spikes run complex-local inhibition, then top-down
    when flagged). The scan computes in float64 under a local x64 context
    (restored on exit) regardless of the params dtype: event times are
    integer microseconds and float32 decay rounding would move threshold
    crossings, so ``V``/``A``/``t_last`` come back float64. ``"event_scan"``
    needs ``local_inhibition_groups``, ``local_inhibition_blocks``,
    ``event_scan_events`` and ``event_scan_tables`` (a
    :func:`jaxfne.pcl_topology.event_scan_tables` mapping); rounds, order
    other than ``"ascending"``, the visit order, edges, drive, feed, rules
    and recording would change nothing and are refused by name, as are the
    scan channels under any other resolution. ``"event_order"`` needs
    ``local_inhibition_rounds``,
    ``local_inhibition_groups``, ``local_inhibition_blocks`` (a
    :func:`jaxfne.pcl_topology.local_inhibition_blocks` dict of per-position
    inhibition magnitudes) and ``local_inhibition_visit_order`` (int
    ``(n_steps, G, D)`` per-step, per-position local-id visit order, as
    :func:`jaxfne.pcl_topology.inhibition_visit_order` builds it); each is
    refused by name when missing, misshapen or disagreeing, and the blocks
    and visit order are refused outright under ``"whole_step"`` (they would
    change nothing). Under ``"event_order"`` the order must be
    ``"ascending"`` — the visit array already encodes the tie-break, so a
    descending order would not drive the cells and is refused by name.

    ``I_ext_pA`` is None, shape ``[N]`` (every step), or ``[n_steps, N]``.
    It is sustained background current: it adds to the input without
    arrival counts, so no refractory term is subtracted for it (Gap-1
    drive numerics are unchanged by the feed channels below).

    ``feed_current``/``feed_arrivals`` (each None, or shape ``[n_steps,
    N]``) are the DVS cover-table feed as part of the input drive ``x``:
    per-step summed synaptic weights and per-step fed-event counts of the
    binned event stream (:func:`jaxfne.pcl_topology.dvs_feed_arrays`
    builds them from the oracle cover tables). Each fed event subtracts
    one ``R + A`` exactly like an edge arrival (the oracle's
    ``SimpleNeuron::membraneUpdate`` subtracts the refractory and
    adaptation terms on every synaptic input,
    ``pcl_event_ref.py:226-238``); sustained ``I_ext_pA`` never does.
    Feed arrivals do not enter the HDP observation context (they are
    frozen-weight input, as in the learning-off oracle runs).
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
    Under ``"event_scan"`` the result instead holds the exact replay:
    ``spikes`` (bool ``[n_steps, N]``), finals ``V``/``A``/``t_last``
    (float64, as above), empty ``w_final``, per-cell ``counts`` and
    ``first_spike_us`` (``-1`` where a cell never spiked).

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
    if edge_group is None:
        group_np = np.arange(E, dtype=np.int64)
    else:
        group_np = np.asarray(edge_group, dtype=np.int64).ravel()
        if group_np.shape != (E,):
            raise ValueError(
                f"edge_group must have shape ({E},); got {group_np.shape}"
            )
        if E and bool((group_np < 0).any()):
            raise ValueError("edge_group ids must be >= 0")
    n_groups = int(group_np.max() + 1) if E else 0
    group_j = jnp.asarray(group_np, dtype=jnp.int32)
    if local_inhibition_rounds is not None:
        from .pcl_topology import LOCAL_INHIBITION_ORDERS, validate_rounds

        validate_rounds(local_inhibition_rounds, label="local_inhibition")
        if local_inhibition_order not in LOCAL_INHIBITION_ORDERS:
            raise ValueError(
                "local_inhibition_order must be one of "
                f"{list(LOCAL_INHIBITION_ORDERS)}; got {local_inhibition_order!r}"
            )
    group_j_inhib = None
    if local_inhibition_groups is not None:
        # jnp.asarray (not np): this array rides the jitted kernel path.
        group_j_inhib = jnp.asarray(local_inhibition_groups)
        if tuple(group_j_inhib.shape) != (N,):
            raise ValueError(
                f"local_inhibition_groups must have shape ({N},); "
                f"got {tuple(group_j_inhib.shape)}"
            )
    from .pcl_topology import LOCAL_INHIBITION_RESOLUTIONS

    _resolution = (
        "whole_step" if local_inhibition_resolution is None
        else local_inhibition_resolution
    )
    if _resolution not in LOCAL_INHIBITION_RESOLUTIONS:
        raise ValueError(
            "local_inhibition_resolution must be one of "
            f"{list(LOCAL_INHIBITION_RESOLUTIONS)}; got {local_inhibition_resolution!r}"
        )
    if _resolution == "event_scan":
        return _run_event_scan_branch(
            params, N, n_steps, dt_ms, t0,
            local_inhibition_rounds, local_inhibition_order,
            local_inhibition_groups, local_inhibition_blocks,
            local_inhibition_visit_order, event_scan_events,
            event_scan_tables, edge_pre, edge_post, edge_weight,
            edge_delay_steps, edge_group, I_ext_pA, V0_mV, H0, record_v,
            hdp_rule, hdp_rule_params, plasticity_mask,
            feed_current, feed_arrivals,
        )
    if event_scan_events is not None or event_scan_tables is not None:
        raise ValueError(
            "event_scan_events/event_scan_tables are set but resolution is "
            f"{_resolution!r}; the event replay only runs under "
            "local_inhibition_resolution='event_scan'"
        )
    if local_inhibition_resolution is not None and local_inhibition_rounds is None:
        raise ValueError(
            "local_inhibition_resolution is set but local_inhibition_rounds is None; "
            "the resolution only resolves a declared cap"
        )
    use_cascade = _resolution == "event_order"
    if use_cascade:
        if local_inhibition_order != "ascending":
            raise ValueError(
                f"local_inhibition_order={local_inhibition_order!r} with resolution "
                "'event_order' is refused; the visit order already encodes the "
                "tie-break (ascending index) — build the descending visit instead"
            )
        if group_j_inhib is None:
            raise ValueError(
                "local_inhibition_resolution='event_order' needs "
                "local_inhibition_groups; without positions there is nothing to visit"
            )
        members_j, mask_j, blk_j, visit_j = _check_event_order_inputs(
            N, n_steps, local_inhibition_groups, local_inhibition_blocks,
            local_inhibition_visit_order, dtype,
        )
    else:
        if local_inhibition_blocks is not None:
            raise ValueError(
                "local_inhibition_blocks is set but resolution is 'whole_step'; "
                "the cap resolves without per-position blocks"
            )
        if local_inhibition_visit_order is not None:
            raise ValueError(
                "local_inhibition_visit_order is set but resolution is 'whole_step'; "
                "the cap resolves in the declared order without a visit array"
            )
    if E:
        # One shared weight per group: the group mean (signed; members of a
        # sharing group carry one plastic row).
        _sums = _seg_sum(w0, group_j, n_groups)
        _counts = _seg_sum(jnp.ones_like(w0), group_j, n_groups)
        w0 = _sums[group_j] / jnp.maximum(_counts[group_j], 1.0)
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

    if feed_current is None:
        F_cur = None
    else:
        F_cur = jnp.asarray(feed_current, dtype)
        if tuple(F_cur.shape) != (n_steps, N):
            raise ValueError(
                f"feed_current must be None or shape ({n_steps}, {N}); "
                f"got {tuple(F_cur.shape)}"
            )
        if not bool(jnp.all(jnp.isfinite(F_cur))):
            raise ValueError("feed_current must be finite (got NaN/inf)")
    if feed_arrivals is None:
        F_arr = None
    else:
        F_arr = jnp.asarray(feed_arrivals, dtype)
        if tuple(F_arr.shape) != (n_steps, N):
            raise ValueError(
                f"feed_arrivals must be None or shape ({n_steps}, {N}); "
                f"got {tuple(F_arr.shape)}"
            )
        if not bool(jnp.all(jnp.isfinite(F_arr))):
            raise ValueError("feed_arrivals must be finite (got NaN/inf)")
        if bool((F_arr < 0).any()):
            raise ValueError("feed_arrivals must hold non-negative event counts")
        if not bool((F_arr == jnp.floor(F_arr)).all()):
            raise ValueError("feed_arrivals must hold whole event counts")

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
        if F_cur is not None:
            I_tot = I_tot + F_cur[t_idx]
        if F_arr is not None:
            n_net = n_net + F_arr[t_idx]
        if use_b:
            I_tot = I_tot + bias
        if use_cascade:
            (V_n, A_n, t_n), spk = _cascade_step(
                (V, A, t_last), I_tot, n_net, t, params, float(dt_ms),
                int(local_inhibition_rounds),
                members_j, mask_j, blk_j, visit_j[t_idx],
            )
        else:
            (V_n, A_n, t_n), spk = lif_step(
                (V, A, t_last), I_tot, n_net, t, params, float(dt_ms),
                cap_rounds=local_inhibition_rounds, cap_order=local_inhibition_order,
                cap_groups=group_j_inhib,
            )
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
                w_next = jnp.where(plastic_edge, w_next, w) if plastic_edge is not None else w_next
                if E:
                    # Shared accumulation: each group's update is the mean of
                    # its plastic members' updates, so STDP arriving on one
                    # member moves the whole shared row (members stay equal).
                    if plastic_edge is not None:
                        _pmask = plastic_edge.astype(dtype)
                    else:
                        _pmask = jnp.ones_like(w)
                    _upd = w_next - w
                    _num = _seg_sum(_upd * _pmask, group_j, n_groups)[group_j]
                    _den = jnp.maximum(_seg_sum(_pmask, group_j, n_groups)[group_j], 1.0)
                    w = w + _num / _den
                else:
                    w = w_next
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
