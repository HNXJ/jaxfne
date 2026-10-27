"""Current-based LIF edge-list kernel with PCL refractory terms (function-level only).

Reference: N'dri, Barbier, Teuliere, Triesch, Nat Commun 16:8880 (2025),
doi:10.1038/s41467-025-64234-z, Methods; the per-event update mirrors
``Neuron.cpp`` / ``SimpleNeuron.cpp`` / ``ComplexNeuron.cpp`` as transcribed in
``artifacts/etudes/papers/pcl/port/pcl_event_ref.py`` (read as data): on each input of
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
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .emitters import CELL_TYPE_SIGN

NEVER = -1.0e30  # no spike yet; spiked state is t_last > HAS_SPIKED
HAS_SPIKED = -1.0e29

# PORT_SPEC.md stage-A table (paper Table 2 threshold for simple; default config
# ETA_SRA = 0, so the adaptation increment defaults to 0 and is opt-in).
LIF_CELL_TYPE_DEFAULTS: dict[str, dict[str, float]] = {
    "simple": {
        "tau_m_ms": 18.0, "v_thresh_mV": 10.0, "v_reset_mV": -10.0,
        "v_min_mV": -20.0, "eta_rp_mV": 10.0, "tau_rp_ms": 5.0,
        "tau_sra_ms": 100.0, "delta_sra_mV": 0.0, "t_ref_ms": 0.0,
        "tau_syn_ms": 0.0,
    },
    "complex": {
        "tau_m_ms": 50.0, "v_thresh_mV": 3.0, "v_reset_mV": -10.0,
        "v_min_mV": -20.0, "eta_rp_mV": 10.0, "tau_rp_ms": 5.0,
        "tau_sra_ms": math.inf, "delta_sra_mV": 0.0, "t_ref_ms": 0.0,
        "tau_syn_ms": 0.0,
    },
}

_PARAM_FIELDS = (
    "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV", "tau_rp_ms",
    "tau_sra_ms", "delta_sra_mV", "t_ref_ms", "tau_syn_ms",
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
    t_ref_ms: jax.Array
    tau_syn_ms: jax.Array
    labels: tuple[str, ...]


def _check_t_ref_ms(value: float, *, label: str) -> float:
    """Hard-refractory window in ms: finite and >= 0 (0.0 = today's behaviour).

    Bools, strings and other non-reals are refused, never coerced (mirrors
    the triple-drive and in-degree validators).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{label} t_ref_ms must be a finite number >= 0 in ms; got {value!r}"
        )
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(
            f"{label} t_ref_ms must be a finite number >= 0 in ms; got {value!r}"
        )
    return v


def _check_tau_syn_ms(value: float, *, label: str) -> float:
    """Synaptic current time constant in ms: finite and > 0 when declared.

    ``0.0`` is the unset default (the delta-summing path), so a declared
    ``tau_syn_ms`` must clear zero: non-positive, non-finite and non-real
    values are refused, never coerced (mirrors the t_ref_ms validator).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{label} tau_syn_ms must be a finite number > 0 in ms; got {value!r}"
        )
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(
            f"{label} tau_syn_ms must be a finite number > 0 in ms; got {value!r}"
        )
    return v


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
        if "t_ref_ms" in fields:
            _check_t_ref_ms(
                fields["t_ref_ms"],
                label=f"lif_overrides[{ctype!r}]",
            )
        if "tau_syn_ms" in fields:
            _check_tau_syn_ms(
                fields["tau_syn_ms"],
                label=f"lif_overrides[{ctype!r}]",
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
    per-event visit order on A (Hamm 2026-10-14): an approximation of the
    event-exact cascade, not the event-exact order itself (decay once per
    step, one shared ``k * (R + A)`` subtraction, no inter-event
    interleave; ``resolution='event_scan'`` is the exact per-event
    replay under the Stage C test conditions (20-clip receipt; exactness
    is bounded by that evidence, not universal). Pure ``jax.numpy``;
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

    Hard refractory (``params.t_ref_ms``, Brunel 2000 ``taurp``): after a
    spike at ``t_last``, while ``t_ms - t_last < round(t_ref_ms / dt_ms) *
    dt_ms`` the cell's ``V`` is held at ``v_reset_mV`` and it cannot spike;
    inputs arriving in that window are discarded. ``t_ref_ms = 0.0`` keeps
    ``V`` and spikes bit for bit (the refractory mask is all-False).

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
    # Hard refractory mask: whole steps with t_ms - t_last inside the
    # round(t_ref_ms / dt_ms)-step window. The bound sits half a step below
    # the window edge, so float32 rounding of t_ms - t_last at large t cannot
    # add or drop a step. t_ref_ms = 0.0 is all-False (t_ms - t_last >= 0 >
    # -dt/2), so the path below is today's behaviour bit for bit.
    ref = (t_ms - t_last) < (jnp.round(p.t_ref_ms / dt_ms) - 0.5) * dt_ms
    if cap_rounds is None:
        if cap_groups is not None:
            raise ValueError(
                "cap_groups is set but cap_rounds is None; groups only scope "
                "a declared local-inhibition cap"
            )
        spk = raw & ~ref
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
        spk = raw & keep & ~ref
    V_out = jnp.where(spk | ref, p.v_reset_mV, V_new)
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
#: inhibitory). Derived from the single sign table
#: (:data:`jaxfne.emitters.CELL_TYPE_SIGN`), restricted to the labels the LIF
#: route signs; kept under this name so existing imports keep working.
LIF_LABEL_SIGN: dict[str, float] = {
    lab: CELL_TYPE_SIGN[lab]
    for lab in ("simple", "complex", "E", "PV", "SST", "VIP", "Inl", "Ing", "I")
}


def lif_sign_of(label: str) -> float:
    """Presynaptic sign for a LIF-route label; refuses unlisted labels.

    An unknown label previously signed +1 silently (excitatory by default);
    a typo'd label (e.g. "SOM", which names an AdEx-table type) must not
    simulate as excitatory with no warning.
    """
    try:
        return LIF_LABEL_SIGN[label]
    except KeyError:
        raise ValueError(
            f"unknown LIF sign label {label!r}; expected one of {sorted(LIF_LABEL_SIGN)}"
        ) from None

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

    The ten ``*_ms``/``*_mV`` arrays are the :func:`make_lif_params` columns
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
    t_ref_ms: jax.Array
    tau_syn_ms: jax.Array
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
            t_ref_ms=self.t_ref_ms, tau_syn_ms=self.tau_syn_ms,
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
    it names a PCL cell type, else the simple-cell row); its sign comes from
    :func:`lif_sign_of`, which refuses a label outside ``LIF_LABEL_SIGN``. ``drive_overrides``
    maps a label to constant drive (mV/step); unmentioned labels default to
    ``LIF_DEFAULT_DRIVE_MV``. ``lif_overrides`` maps a label to a dict of the
    ten :data:`_PARAM_FIELDS` fields. Unknown override labels and unknown
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
        if "t_ref_ms" in fields:
            _check_t_ref_ms(fields["t_ref_ms"], label=f"lif_overrides[{lab!r}]")
        if "tau_syn_ms" in fields:
            _check_tau_syn_ms(fields["tau_syn_ms"], label=f"lif_overrides[{lab!r}]")
    columns: dict[str, list[float]] = {f: [] for f in _PARAM_FIELDS}
    drive: list[float] = []
    sign: list[float] = []
    for lab in label_tuple:
        row = lif_cell_type_of(lab)
        over = lif_overrides.get(lab, lif_overrides.get(row, {}))
        for f in _PARAM_FIELDS:
            columns[f].append(float(over.get(f, LIF_CELL_TYPE_DEFAULTS[row][f])))
        drive.append(overrides.get(lab, LIF_DEFAULT_DRIVE_MV))
        sign.append(lif_sign_of(lab))
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


def _log_spike(LOGT, LOGC, LOGN, spk, ti, cell):
    """Append ``(ti, cell)`` at index ``LOGN`` when ``spk``; ``LOGN`` counts every spike.

    Past the capacity the write is dropped (the index is clamped and the old
    value kept), so ``LOGN`` alone reports an overflow.
    """
    K = LOGT.shape[0]
    wr = spk & (LOGN < K)
    idx = jnp.minimum(LOGN, K - 1)
    LOGT = LOGT.at[idx].set(jnp.where(wr, ti, LOGT[idx]))
    LOGC = LOGC.at[idx].set(jnp.where(wr, cell, LOGC[idx]))
    return LOGT, LOGC, LOGN + spk.astype(LOGN.dtype)


class _ScanCarry(NamedTuple):
    """The event-scan carry: one ``lax.scan`` step per event, one field per array.

    State (one dummy cell past the population): ``V`` membrane, ``A``
    adaptation, ``TL`` last update, ``TS`` last spike, ``CNT`` spike count,
    ``FST`` first spike (us, -1 before), ``BIN`` per-bin spike flags, and the
    spike log ``LOGT``/``LOGC`` with ``LOGN`` counting every spike. The
    periodic H update: ``TH`` threshold, ``RATE`` rate, ``CSNAP`` the count
    snapshot at the last update, ``LUP`` the time of that update.
    Weights (float64, read-only in this pass): ``WS`` shared simple
    ``(nd, 2, sx, sy)``, ``WC`` shared complex ``(cd, csx, csy, nd)``, ``LATW``
    lateral ``(ns, lmax)``, ``TDW`` top-down ``(nc, tmax)`` and ``BLK`` the
    inhibition blocks ``(G, D, D)``. Each stage returns the carry through
    ``_replace``, so a later stage reaches any field, CSNAP included.
    """

    V: jax.Array
    A: jax.Array
    TL: jax.Array
    TS: jax.Array
    CNT: jax.Array
    FST: jax.Array
    BIN: jax.Array
    LOGT: jax.Array
    LOGC: jax.Array
    LOGN: jax.Array
    TH: jax.Array
    RATE: jax.Array
    CSNAP: jax.Array
    LUP: jax.Array
    WS: jax.Array
    WC: jax.Array
    LATW: jax.Array
    TDW: jax.Array
    BLK: jax.Array
    # Declared buffer plasticity (``learn`` only; dummy shapes otherwise). ``TLS``
    # is the previous spike time of each simple cell (the oracle's t_last_spk).
    # ``EVCLR[n]`` is the rank in cell n's covering-event list where its ``ev``
    # buffer starts; ``LOCLR[n]`` is the position in its group's log where its
    # ``loc`` buffer starts. ``GLT``/``GLS`` (time, source depth) are one ring of
    # capacity CAP per local-inhibition position, with push count ``GCNT``.
    # ``WIS``/``WIC`` are the shared local-inhibition matrices, padded to (M, M).
    TLS: jax.Array
    EVCLR: jax.Array
    LOCLR: jax.Array
    GLT: jax.Array
    GLS: jax.Array
    GCNT: jax.Array
    WIS: jax.Array
    WIC: jax.Array
    # Complex buffer plasticity (``clearn`` only; dummy shapes otherwise). ``CEV``
    # ``(NCP, CAP, 4)`` is one ring per complex position of (time, ex, ey, k) with
    # pass-start push count ``CEVCNT``; ``CEVCLR[c]``/``CLOCLR[c]`` are the ring
    # indices where complex cell c's ``ev``/``loc`` buffer starts; ``CKT``/``CKH``
    # are the kept spike time and its flag (the oracle's one-element vspikes).
    CEV: jax.Array
    CEVCNT: jax.Array
    CEVCLR: jax.Array
    CLOCLR: jax.Array
    CKT: jax.Array
    CKH: jax.Array


class _CPend(NamedTuple):
    """Complex state of one simple spike's forward pass (``clearn`` only).

    ``BT``/``BS`` (time, source depth) are the loc pushes that the current position's
    block of candidates has made so far, ``BN`` of them, and ``BP`` is that block's
    position. ``EV*`` (one row per candidate) are the position's ev pushes, made at the
    block's first candidate; ``LP*`` (one row per candidate) are the loc pushes of the
    complex spikes. The rings are written from these after the cond.
    """

    BT: jax.Array
    BS: jax.Array
    BN: jax.Array
    BP: jax.Array
    EVON: jax.Array
    EVCP: jax.Array
    EVJ: jax.Array
    EVV: jax.Array
    LPON: jax.Array
    LPG: jax.Array
    LPJ: jax.Array
    LPV: jax.Array


def _event_scan_loop(
    V0, A0, EV, CELL, CMASK, SOX, SOY,
    FWD_IDX, FWD_KEY, FWD_N, LAT_IDX, LAT_N, TD_IDX, TD_N,
    MEMBERS, MMASK, POS_OF, LOC_OF,
    PCOL, GRID, FLAGS, BIN0, TH0, RATE0, LUP0, HTGT, HETA, HTMIN, HSC, HON,
    WS0, WC0, LATW0, TDW0, BLK0, WIS0, WIC0, DECAY, MDS, MDC, WUP, GLT0,
    EVROW, EVIDX, EVQ, LOGT0, *, learn, clearn=False,
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
    (integer-microsecond event times against float64 taus). ``LOGT0`` (int64,
    zeros, shape ``(K,)``) sets the spike-log capacity: the log keeps every
    spike's time and cell in emission order (a simple spike precedes the
    complex spikes it forwards); ``LOGN`` counts every spike, including those
    past K.

    The declared H update (``HON`` > 0) runs the oracle's periodic update
    (``pcl_event_ref.py:746-762``) after each event's whole cascade, once the
    event lies past the measurement interval ``HSC[0]`` from the last update
    ``LUP``: each cell's rate ``RATE`` (Hz) integrates the spikes since the
    last update (``CNT - CSNAP``, the snapshot taken at that update), its
    threshold ``TH`` moves by ``HETA`` toward ``HTGT``, floored at ``HTMIN``.
    ``HSC[1]`` is the rate smoothing alpha. With ``HON`` 0 the thresholds
    stay at ``TH0`` for the whole pass. The forward weight of a spike from
    simple cell ``n`` onto the complex target at ``FWD_IDX[n, f]`` is
    ``WC[cz, ex, ey, n % ND]`` for ``FWD_KEY[n, f] = (cz, ex, ey)``. The
    weights ride the carry unchanged unless ``learn`` is set, and the final
    carry is returned.

    ``learn`` (static) declares the simple excitatory buffer plasticity
    (``pcl_event_ref.py:543-574``, ``weight_update_simple``): each simple cell
    buffers every covering event (``ev``, before its excite) and every
    group-mate's local inhibition (``loc``); at a simple spike, before its local
    inhibition, the buffered events move ``WS[k]`` (LTP then LTD, clip, L1 to
    ``norm_factor``) and the buffered inhibitions move the row ``WIS[k]`` (L1 to
    ``eta_inh``), both only when ``excite_on``; then both buffers clear. ``WUP``
    carries the floats
    ``[eLTP, eLTD, iLTP, iLTD, tau_ltp, tau_ltd, norm_factor, eta_inh, excite_on]``
    (``excite_on`` > 0 for the ``"excitatory"``/``"all"`` modes). With learning
    the local inhibition reads ``WIS``/``WIC`` (depth-indexed through ``MDS``/
    ``MDC``) in place of ``BLK``. ``GLT0`` ``(G, CAP)`` sets the loc ring
    capacity. ``DECAY`` is the per-cell learning decay.

    Under ``learn`` the covering events of simple cell ``n`` are the rows of
    ``EV`` listed by ``EVIDX[EVROW[n]:EVROW[n + 1]]`` in stream order, and
    ``EVQ[e, s]`` is the rank of covering slot ``s`` of event ``e`` in its
    cell's list (-1 for empty slots). The scan walks ``(EV, EVQ)`` together.

    ``clearn`` (static, under ``learn``) declares the complex excitatory update
    (``pcl_event_ref.py:608-646``, ``weight_update_complex``). Each complex
    position keeps a ring ``CEV`` of its forwarded events and its complex loc
    entries ride the group log, both read inside the forward pass and written
    after its cond (see ``_CPend``). Its weights are ``WUP[9:17]``:
    ``[eLTP, eLTD, iLTP, iLTD, tau_ltp, tau_ltd, norm_factor, eta_inh]``.
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
    # Static shapes of the learning state, read from the array shapes (GRID is traced).
    NDs = WS0.shape[0]
    CDs = WC0.shape[0]
    CAP = GLT0.shape[1]
    # The loc log GLT/GLS is the only O(G x CAP) state. A ``lax.cond`` copies
    # every operand it returns, so the log is kept out of the conds: each cond
    # carries dummies in its place, reads the real log as a read-only operand,
    # and returns the one entry it pushes (``push``); that entry is written
    # after the cond (measured: a 1000-deep log copied per cond cost 9x).
    no_glt = jnp.zeros((1, 1), f)
    no_gls = jnp.zeros((1, 1), i32)
    no_push = (jnp.bool_(False), jnp.int32(0), jnp.int32(0), zero_f, jnp.int32(0))
    # Complex rings (``clearn``): one CAP-deep ring per complex position, the group
    # count of the loc log, and the dummy used inside the conds.
    no_cev = jnp.zeros((1, 1, 4), f)
    NG = GLT0.shape[0]
    NCP = (N1 - 1 - NS) // CDs if clearn else 1

    def _excite_one(Vc, Ac, TLc, TSc, THc, n, w, t):
        p = PCOL[n]
        dt_ = t - TLc[n]
        Vd = Vc[n] * jnp.exp(-dt_ / p[0])
        Ad = Ac[n] * jnp.exp(-dt_ / p[1])
        R = jnp.where(TSc[n] > zero_f,
                      p[5] * jnp.exp(-(t - TSc[n]) / p[6]), zero_f)
        Vn = jnp.maximum(p[4], Vd + w - R - Ad)
        return Vd, Ad, Vn, Vn > THc[n]

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

    def _stdp_simple(c, glt, gls, n0, k, t_now, q):
        """``SimpleNeuron::weightUpdate`` (oracle :543-574) at a spike of cell ``n0``.

        Runs before the cell's local inhibition (oracle :684). ``q`` is the
        spike's rank in ``n0``'s covering-event list (the spike's own event is
        appended before its excite, :679-680), so the buffered events are the
        ranks ``max(EVCLR[n0], q - CAP + 1) .. q``: the oracle's
        ``deque(maxlen=CAP)`` after the clear at ``n0``'s previous spike. Each
        rank reads its event from ``EV`` through ``EVIDX``; the cell's offsets
        turn the event's x and y into the weight index. The buffered
        inhibitions are the group-log entries of ``_loc_push``. Both learned
        rows end with their L1 normalisation. ``WS[k]`` and ``WIS[k]`` change
        only when ``excite_on`` is set (the oracle's excitatory/all gate,
        :548-574); ``EVCLR[n0]`` moves past ``q`` either way (the buffers clear
        in every mode, :603-606).
        """
        tls = c.TLS[n0]
        dec = DECAY[n0]
        eLTP, eLTD, iLTP, iLTD = WUP[0], WUP[1], WUP[2], WUP[3]
        tau_p, tau_d, nrm, eta = WUP[4], WUP[5], WUP[6], WUP[7]
        excite_on = WUP[8] > 0.0
        pos_n = n0 // ND
        sox_n = SOX[pos_n // NY]
        soy_n = SOY[pos_n % NY]
        ev_lo = jnp.maximum(c.EVCLR[n0], q - CAP + 1)
        ev_n = q - ev_lo + 1
        ev_base = EVROW[n0] + ev_lo
        # Group log window: entries LOCLR[n0] .. GCNT-1, at most CAP. Every entry
        # after LOCLR[n0] is a group-mate's push (n0's own push sets LOCLR).
        g = POS_OF[n0]
        loc_end = c.GCNT[g]
        loc_lo = jnp.maximum(c.LOCLR[n0], loc_end - CAP)
        loc_n = loc_end - loc_lo

        W0 = c.WS[k]

        def ev_body(i, W):
            e_idx = EVIDX[ev_base + i]
            ti = EV[e_idx, 0].astype(f)
            pq = EV[e_idx, 3].astype(i32)
            xq = EV[e_idx, 1].astype(i32) - sox_n
            yq = EV[e_idx, 2].astype(i32) - soy_n
            wj = W[pq, xq, yq]
            d_ltp = (3.0 - wj) * 0.33 * dec * eLTP * 1.0 * jnp.exp(-(t_now - ti) / tau_p)
            w1 = wj + d_ltp
            d_ltd = (wj * 0.33) * eLTD * 1.0 * jnp.exp(-(ti - tls) / tau_d)
            w2 = jnp.where(tls != 0.0, w1 + d_ltd, w1)
            w3 = jnp.where(w2 < 0.0, 0.0, w2)
            return W.at[pq, xq, yq].set(w3)

        W = jax.lax.fori_loop(0, ev_n, ev_body, W0)
        total = jnp.sum(jnp.abs(W))
        W = jnp.where(total != 0.0, W * (nrm / total), W)
        W = jnp.where(excite_on, W, W0)

        def loc_body(i, Wi):
            j = (loc_lo + i) % CAP
            ti = glt[g, j]
            src = gls[g, j]
            wj = Wi[src]
            d_ltp = (50.0 - wj) * 0.02 * iLTP * jnp.exp(-(t_now - ti) / tau_p)
            w1 = wj + d_ltp
            d_ltd = (wj * 0.02) * iLTD * jnp.exp(-(ti - tls) / tau_d)
            w2 = jnp.where(tls != 0.0, w1 + d_ltd, w1)
            w3 = jnp.where(w2 < 0.0, 0.0, w2)
            return Wi.at[src].set(w3)

        Wi = jax.lax.fori_loop(0, loc_n, loc_body, c.WIS[k])
        row = Wi[:NDs]
        total_i = jnp.sum(jnp.abs(row))
        row = jnp.where(total_i != 0.0, row * (eta / total_i), row)
        Wi = Wi.at[:NDs].set(row)
        # The oracle's wi_s loop and its normalisation sit inside the same
        # excitatory/all gate as w_s (pcl_event_ref.py:548-574).
        Wi = jnp.where(excite_on, Wi, c.WIS[k])
        return c._replace(
            WS=c.WS.at[k].set(W), WIS=c.WIS.at[k].set(Wi),
            EVCLR=c.EVCLR.at[n0].set(q + 1),
        )

    def _loc_push(c, g, n0, src_dep, t_now):
        """Push ``(t_now, src_dep)`` onto position ``g``'s log, after ``n0``'s update.

        Equivalence to the oracle's per-cell ``loc`` deque: a spike of ``n0``
        clears ``n0``'s deque (its ``_stdp_simple`` read) and appends this entry
        to each group-mate except ``n0`` (``pcl_event_ref.py:684-697``). A
        mate's deque therefore holds the last CAP pushes made by the other
        members since its own last spike. The log keeps one copy of each push,
        and ``LOCLR[n0]`` is set to the log count after this push, so every
        entry at or after ``LOCLR[m]`` came from a member other than ``m``, and
        the last CAP of them are the deque's contents (``deque(maxlen=CAP)``).
        The log entry itself is returned as ``push`` and written by the caller
        after the cond (see ``no_glt``); only the small counters move here.
        """
        slot = c.GCNT[g] % CAP
        gcnt = c.GCNT.at[g].add(1)
        push = (jnp.bool_(True), g, slot, t_now, src_dep)
        return c._replace(GCNT=gcnt, LOCLR=c.LOCLR.at[n0].set(gcnt[g])), push

    def _cpend0():
        return _CPend(
            BT=jnp.zeros((CDs,), f), BS=jnp.zeros((CDs,), i32),
            BN=jnp.zeros((), i32), BP=jnp.full((), -1, i32),
            EVON=jnp.zeros((FMAX,), jnp.bool_), EVCP=jnp.zeros((FMAX,), i32),
            EVJ=jnp.zeros((FMAX,), i32), EVV=jnp.zeros((FMAX, 4), f),
            LPON=jnp.zeros((FMAX,), jnp.bool_), LPG=jnp.zeros((FMAX,), i32),
            LPJ=jnp.zeros((FMAX,), i32), LPV=jnp.zeros((FMAX, 2), f))

    def _cpush_of(p):
        """The forward pass's complex pushes, returned from the cond (none without ``clearn``)."""
        if not clearn:
            return ()
        return (p.EVON, p.EVCP, p.EVJ, p.EVV, p.LPON, p.LPG, p.LPJ, p.LPV)

    no_cpush = _cpush_of(_cpend0())

    def _cstdp(c, pend, ff, cev, glt, gls, c0, cp_, cz_, g, t_now, tls, cur):
        """``ComplexNeuron::weightUpdate`` (oracle :608-646) at a complex spike of ``c0``.

        Runs in the forward pass after the spike's count increment and before the
        cell's local inhibition (oracle :723, then :725-733). ``cp_``/``cz_`` are the
        cell's complex position and depth, ``g`` its loc-log group, ``tls`` its
        previous spike time (the oracle's t_last_spk; 0 is the never-spiked sentinel)
        and ``cur`` the ``(t, ex, ey, k)`` row this forward appended to the position's
        ev buffer. ``cev``/``glt``/``gls`` are the rings as they stood before the pass.

        Why the rings reproduce the per-cell deques. ``ev``: a complex cell's deque
        holds the last CAP entries appended since its last spike. The position's ring
        gets one entry per forward (at ``CEVCNT[cp_]``, this pass's entry is the newest);
        the cell's window is ring indices ``max(CEVCLR[c0], total - CAP) .. total - 1``
        with ``total = CEVCNT[cp_] + 1``, the last of them being ``cur``. ``loc``: the
        group-mates' pushes since the cell's last spike are ring indices from
        ``CLOCLR[c0]``; the pushes this pass made to the group sit in the pending block
        ``BT``/``BS`` at ring indices ``GCNT[g] + j``. A block is one position's cells,
        so no other pending push can reach this group. Step 1 runs only at
        ``count == 1`` (oracle :614), with vspikes ``[kept spike if any, this spike]``
        (:617, :630-631); step 2 runs in every excitatory/all pass (:632-644).
        """
        ceLTP, ceLTD, ciLTP, ciLTD = WUP[9], WUP[10], WUP[11], WUP[12]
        ctp, ctd, cnrm, ceta = WUP[13], WUP[14], WUP[15], WUP[16]
        cnt1 = (c.CNT[c0] - c.CSNAP[c0]) == 1
        kh = c.CKH[c0]
        kt = c.CKT[c0]
        g_ev = c.CEVCNT[cp_]
        tot_e = g_ev + 1
        lo_e = jnp.maximum(c.CEVCLR[c0], tot_e - CAP)
        n_e = jnp.where(cnt1, tot_e - lo_e, 0)

        def _cpair(W, x, y, z, ti, tp, use):
            # Oracle :617-625: t >= t_i is LTP (the spike follows the event), else LTD.
            wj = W[x, y, z]
            ltp = (4.0 - wj) * 0.25 * ceLTP * jnp.exp(-jnp.abs(tp - ti) / ctp)
            ltd = (wj * 0.25) * ceLTD * jnp.exp(-jnp.abs(tp - ti) / ctd)
            d = jnp.where(tp >= ti, ltp, ltd)
            return W.at[x, y, z].set(jnp.where(use, wj + d, wj))

        def ev_body(i, W):
            j = lo_e + i
            row = jnp.where(j < g_ev, cev[cp_, j % CAP], cur)
            ti = row[0]
            x = row[1].astype(i32)
            y = row[2].astype(i32)
            z = row[3].astype(i32)
            W = _cpair(W, x, y, z, ti, kt, kh)
            W = _cpair(W, x, y, z, ti, t_now, jnp.bool_(True))
            v = W[x, y, z]
            # Oracle :626-627: clip after all pairs of this event entry.
            return W.at[x, y, z].set(jnp.where(v < 0.0, 0.0, v))

        W0 = c.WC[cz_]
        W = jax.lax.fori_loop(0, n_e, ev_body, W0)
        tot = jnp.sum(jnp.abs(W))
        # Oracle :628 L1-normalises the whole w_c[cz] to norm_factor.
        W = jnp.where(tot != 0.0, W * (cnrm / tot), W)
        W = jnp.where(cnt1, W, W0)
        c = c._replace(
            WC=c.WC.at[cz_].set(W),
            CSNAP=c.CSNAP.at[c0].set(jnp.where(cnt1, c.CNT[c0], c.CSNAP[c0])),
            CKT=c.CKT.at[c0].set(jnp.where(cnt1, t_now, kt)),
            CKH=c.CKH.at[c0].set(jnp.where(cnt1, jnp.bool_(True), kh)),
        )
        # Step 2 (oracle :632-644): loc entries of the group, in push order; the
        # re-read after LTP applies, and the clip only runs without a previous spike.
        bt, bs, bn = pend.BT, pend.BS, pend.BN
        g_lp = c.GCNT[g]
        tot_l = g_lp + bn
        lo_l = jnp.maximum(c.CLOCLR[c0], tot_l - CAP)
        n_l = tot_l - lo_l

        def loc_body(i, Wi):
            j = lo_l + i
            isr = j < g_lp
            jr = j % CAP
            jb = jnp.clip(j - g_lp, 0, CDs - 1)
            ti = jnp.where(isr, glt[g, jr], bt[jb])
            src = jnp.where(isr, gls[g, jr], bs[jb])
            wj = Wi[src]
            w1 = wj + (25.0 - wj) * 0.04 * ciLTP * jnp.exp(-(t_now - ti) / ctp)
            d2 = (w1 * 0.04) * ciLTD * jnp.exp(-(ti - tls) / ctd)
            w2 = jnp.where(tls != 0.0, w1 + d2, jnp.where(w1 < 0.0, 0.0, w1))
            return Wi.at[src].set(w2)

        Wi = jax.lax.fori_loop(0, n_l, loc_body, c.WIC[cz_])
        row_i = Wi[:CDs]
        tot_i = jnp.sum(jnp.abs(row_i))
        # Oracle :644 L1-normalises the whole wi_c[cz] row to eta_inh.
        row_i = jnp.where(tot_i != 0.0, row_i * (ceta / tot_i), row_i)
        Wi = Wi.at[:CDs].set(row_i)
        # Step 3 (oracle :645-646) clears both buffers; the own loc push is made
        # after the read, as the oracle appends to the mates after the update.
        j_own = g_lp + bn
        c = c._replace(
            WIC=c.WIC.at[cz_].set(Wi),
            CEVCLR=c.CEVCLR.at[c0].set(g_ev + 1),
            CLOCLR=c.CLOCLR.at[c0].set(j_own + 1),
        )
        pend = pend._replace(
            BT=bt.at[bn].set(t_now), BS=bs.at[bn].set(cz_), BN=bn + 1,
            LPON=pend.LPON.at[ff].set(True), LPG=pend.LPG.at[ff].set(g),
            LPJ=pend.LPJ.at[ff].set(j_own),
            LPV=pend.LPV.at[ff].set(jnp.stack([t_now, cz_.astype(f)])))
        return c, pend

    def _apply_cpush(c, cpush):
        """Write one slot's complex pushes after its cond (``mode="drop"``: masked rows are dropped).

        Each covering position appends one ev entry, so its rows are distinct. A loc
        push lands at its ring index unless a later push of the same pass has already
        filled the ring past it (the ring keeps the last CAP), so the kept rows are
        distinct too and the scatter is order-free.
        """
        evon, evcp, evj, evv, lpon, lpg, lpj, lpv = cpush
        rows = jnp.where(evon, evcp, NCP)
        cev_new = c.CEV.at[rows, evj % CAP].set(evv, mode="drop")
        cevcnt = c.CEVCNT.at[rows].add(evon.astype(i32), mode="drop")
        rg = jnp.where(lpon, lpg, NG)
        tot = jnp.zeros((NG,), i32).at[rg].add(1, mode="drop")
        g_safe = jnp.where(lpon, lpg, 0)
        keep = lpon & (lpj >= c.GCNT[g_safe] + tot[g_safe] - CAP)
        rk = jnp.where(keep, lpg, NG)
        glt_new = c.GLT.at[rk, lpj % CAP].set(lpv[:, 0], mode="drop")
        gls_new = c.GLS.at[rk, lpj % CAP].set(lpv[:, 1].astype(i32), mode="drop")
        gcnt_new = c.GCNT.at[rg].add(1, mode="drop")
        return c._replace(CEV=cev_new, CEVCNT=cevcnt, GLT=glt_new, GLS=gls_new,
                          GCNT=gcnt_new)

    def _event_body(c, xs):
        # TH, RATE, CSNAP and LUP change only at the periodic update after the
        # slot loop; the slot loop and the cascade pass them through unchanged.
        # Under learn ``xs`` also carries this event's covering-slot ranks.
        if learn:
            e, qrow = xs
        else:
            e = xs
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
            # are discarded through ``m``. The loc logs and the complex ring
            # stay out of the cond.
            glt, gls, cev = c.GLT, c.GLS, c.CEV
            cs, push = jax.lax.cond(
                sm[s], lambda cc: _do_slot(s, cc, glt, gls, cev),
                lambda cc: (cc, (no_push, no_cpush)),
                c._replace(GLT=no_glt, GLS=no_gls, CEV=no_cev))
            c = cs._replace(GLT=glt, GLS=gls, CEV=cev)
            spush, cpush = push
            if learn:
                on, g, slot, t_v, src = spush
                c = c._replace(
                    GLT=c.GLT.at[g, slot].set(jnp.where(on, t_v, c.GLT[g, slot])),
                    GLS=c.GLS.at[g, slot].set(jnp.where(on, src, c.GLS[g, slot])))
            if clearn:
                c = _apply_cpush(c, cpush)
            return c

        def _do_slot(s, c, glt, gls, cev):
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
            # Under learn the covering events were appended to each cell's list
            # before the excite (:679-680); EVROW/EVIDX/EVQ already hold them.
            w = jnp.where(m, c.WS[k_s, pp, xr, yr], jnp.zeros((), f))
            Vd, Ad, Vn, cross = _excite_one(c.V, c.A, c.TL, c.TS, c.TH, n0, w, t)
            spk = m & cross
            if learn:
                # TLS takes the previous spike time before TS moves to t.
                c = c._replace(TLS=c.TLS.at[n0].set(
                    jnp.where(spk, c.TS[n0], c.TLS[n0])))
            c = c._replace(
                V=c.V.at[n0].set(
                    jnp.where(m, jnp.where(spk, PCOL[n0, 3], Vn), c.V[n0])),
                A=c.A.at[n0].set(
                    jnp.where(m, Ad + jnp.where(spk, PCOL[n0, 7], zero_f), c.A[n0])),
                TL=c.TL.at[n0].set(jnp.where(m, t, c.TL[n0])),
                TS=c.TS.at[n0].set(jnp.where(spk, t, c.TS[n0])),
                CNT=c.CNT.at[n0].set(c.CNT[n0] + spk.astype(i32)),
                FST=c.FST.at[n0].set(
                    jnp.where(spk & (c.FST[n0] < jnp.int64(0)), ti, c.FST[n0])),
                BIN=c.BIN.at[b, n0].set(c.BIN[b, n0] | spk),
            )
            LOGT, LOGC, LOGN = _log_spike(c.LOGT, c.LOGC, c.LOGN, spk, ti, n0)
            c = c._replace(LOGT=LOGT, LOGC=LOGC, LOGN=LOGN)

            def _cascade(c2):
                spush = no_push
                if learn:
                    # The weight update runs before any inhibition (oracle :684).
                    c2 = _stdp_simple(c2, glt, gls, n0, n0 % NDs, t, qrow[s])
                g = POS_OF[n0]
                k = LOC_OF[n0]
                mm = MMASK[g] & (jnp.arange(D) != k)
                if learn:
                    c2, spush = _loc_push(c2, g, n0, n0 % NDs, t)
                    wts = c2.WIS[MDS[g], n0 % NDs]
                else:
                    wts = c2.BLK[g, :, k]
                V, A, TL = _inhibit_vec(
                    c2.V, c2.A, c2.TL, c2.TS, MEMBERS[g], wts, mm, t)
                c2 = c2._replace(V=V, A=A, TL=TL)

                def _lat(c3):
                    ln = LAT_N[n0]
                    use = jnp.arange(LMAX) < ln
                    V, A, TL = _inhibit_vec(
                        c3.V, c3.A, c3.TL, c3.TS, LAT_IDX[n0], c3.LATW[n0], use, t)
                    return c3._replace(V=V, A=A, TL=TL)

                lat_fire = (FLAGS[0] > 0) & (FLAGS[2] > 0)
                c2 = jax.lax.cond(lat_fire, _lat, lambda c3: c3, c2)

                def _fwd(ff, st):
                    c4, pend = st
                    fm = ff < FWD_N[n0]
                    c0 = jnp.where(fm, FWD_IDX[n0, ff], DUM)
                    w = c4.WC[FWD_KEY[n0, ff, 0], FWD_KEY[n0, ff, 1],
                              FWD_KEY[n0, ff, 2], n0 % ND]
                    _Vd, _Ad, Vn, cross = _excite_one(
                        c4.V, c4.A, c4.TL, c4.TS, c4.TH, c0, w, t)
                    spkc = fm & cross
                    if clearn:
                        # A block of candidates is one complex position (its cells are
                        # contiguous, cz ascending, one block per covering position).
                        # The block's first candidate pushes the position's ev entry
                        # before any excite (oracle :708-720); the block's pending loc
                        # state resets there. Reads use the rings as they stood before
                        # the pass plus this pending state.
                        cp_ = jnp.where(fm, (c0 - NS) // CDs, 0).astype(i32)
                        cz_ = jnp.where(fm, (c0 - NS) % CDs, 0).astype(i32)
                        cur = jnp.stack([t, FWD_KEY[n0, ff, 1].astype(f),
                                         FWD_KEY[n0, ff, 2].astype(f),
                                         (n0 % ND).astype(f)])
                        newblk = fm & (cp_ != pend.BP)
                        pend = pend._replace(
                            BP=jnp.where(fm, cp_, pend.BP),
                            BN=jnp.where(newblk, 0, pend.BN),
                            EVON=pend.EVON.at[ff].set(newblk),
                            EVCP=pend.EVCP.at[ff].set(cp_),
                            EVJ=pend.EVJ.at[ff].set(c4.CEVCNT[cp_]),
                            EVV=pend.EVV.at[ff].set(cur))
                        ts_old = c4.TS[c0]
                    c4 = c4._replace(
                        V=c4.V.at[c0].set(jnp.where(
                            fm, jnp.where(spkc, PCOL[c0, 3], Vn), c4.V[c0])),
                        A=c4.A.at[c0].set(jnp.where(
                            fm,
                            _Ad + jnp.where(spkc, PCOL[c0, 7], zero_f),
                            c4.A[c0])),
                        TL=c4.TL.at[c0].set(jnp.where(fm, t, c4.TL[c0])),
                        TS=c4.TS.at[c0].set(jnp.where(spkc, t, c4.TS[c0])),
                        CNT=c4.CNT.at[c0].set(c4.CNT[c0] + spkc.astype(i32)),
                        FST=c4.FST.at[c0].set(jnp.where(
                            spkc & (c4.FST[c0] < jnp.int64(0)), ti, c4.FST[c0])),
                        BIN=c4.BIN.at[b, c0].set(c4.BIN[b, c0] | spkc),
                    )
                    LOGT, LOGC, LOGN = _log_spike(
                        c4.LOGT, c4.LOGC, c4.LOGN, spkc, ti, c0)
                    c4 = c4._replace(LOGT=LOGT, LOGC=LOGC, LOGN=LOGN)

                    def _ccascade(s5):
                        c5, p5 = s5
                        cg = POS_OF[c0]
                        ck = LOC_OF[c0]
                        cm = MMASK[cg] & (jnp.arange(D) != ck)
                        if clearn:
                            # The weight update runs before the cell's local inhibition
                            # (oracle :723, then :725-733).
                            c5, p5 = _cstdp(c5, p5, ff, cev, glt, gls, c0, cp_, cz_,
                                            cg.astype(i32), t, ts_old, cur)
                        if learn:
                            cwts = c5.WIC[MDC[cg], (c0 - NS) % CDs]
                        else:
                            cwts = c5.BLK[cg, :, ck]
                        V, A, TL = _inhibit_vec(
                            c5.V, c5.A, c5.TL, c5.TS, MEMBERS[cg], cwts, cm, t)
                        c5 = c5._replace(V=V, A=A, TL=TL)

                        def _td(c6):
                            cs = c0 - NS
                            use = jnp.arange(TMAX) < TD_N[cs]
                            V, A, TL = _inhibit_vec(
                                c6.V, c6.A, c6.TL, c6.TS, TD_IDX[cs], c6.TDW[cs], use, t)
                            return c6._replace(V=V, A=A, TL=TL)

                        td_fire = (FLAGS[1] > 0) & (FLAGS[2] > 0)
                        return jax.lax.cond(td_fire, _td, lambda c6: c6, c5), p5

                    return jax.lax.cond(spkc, _ccascade, lambda s5: s5, (c4, pend))

                # The fan-out loop is gated on the spike: without the
                # ``cond`` every candidate would run FMAX masked excites.
                pend0 = _cpend0() if clearn else ()
                c2, pend = jax.lax.cond(
                    spk, lambda s: jax.lax.fori_loop(0, FMAX, _fwd, s),
                    lambda s: s, (c2, pend0))
                return c2, (spush, _cpush_of(pend))

            return jax.lax.cond(
                spk, _cascade, lambda c2: (c2, (no_push, no_cpush)), c)

        c = jax.lax.fori_loop(0, SMAX, _slot, c)

        # The periodic update (pcl_event_ref.py:746-762) after this event's
        # cascade: the rate integrates the spikes since the last update, the
        # threshold moves toward the target, then the update time moves.
        dtl = (ti - c.LUP).astype(f)
        due = (HON > 0) & (dtl > HSC[0])

        def _interval(c):
            alpha = HSC[1]
            sps = (c.CNT - c.CSNAP).astype(f) * (1e6 / dtl)
            RATE_ = alpha * sps + (1.0 - alpha) * c.RATE
            TH_ = jnp.where(
                RATE_ > HTGT,
                c.TH + HETA * (1 - jnp.exp(HTGT - RATE_)),
                c.TH - HETA * (1 - jnp.exp(RATE_ - HTGT)))
            TH_ = jnp.where(TH_ < HTMIN, HTMIN, TH_)
            return c._replace(TH=TH_, RATE=RATE_, CSNAP=c.CNT, LUP=ti)

        glt, gls, cev = c.GLT, c.GLS, c.CEV
        c = jax.lax.cond(
            due, _interval, lambda c: c,
            c._replace(GLT=no_glt, GLS=no_gls, CEV=no_cev))
        return c._replace(GLT=glt, GLS=gls, CEV=cev), jnp.int32(0)

    carry0 = _ScanCarry(
        V=V0, A=A0,
        TL=jnp.zeros((N1,), f), TS=jnp.zeros((N1,), f),
        CNT=jnp.zeros((N1,), jnp.int32),
        FST=jnp.full((N1,), -1, dtype=jnp.int64),
        BIN=BIN0, LOGT=LOGT0, LOGC=jnp.zeros(LOGT0.shape, i32),
        LOGN=jnp.zeros((), i32),
        TH=TH0, RATE=RATE0, CSNAP=jnp.zeros((N1,), jnp.int32), LUP=LUP0,
        WS=WS0, WC=WC0, LATW=LATW0, TDW=TDW0, BLK=BLK0,
        TLS=jnp.zeros((N1,), f),
        EVCLR=jnp.zeros((NS + 1,), i32), LOCLR=jnp.zeros((NS + 1,), i32),
        GLT=GLT0, GLS=jnp.zeros(GLT0.shape, i32),
        GCNT=jnp.zeros((GLT0.shape[0],), i32),
        WIS=WIS0, WIC=WIC0,
        CEV=jnp.zeros((NCP, CAP, 4), f) if clearn else no_cev,
        CEVCNT=jnp.zeros((NCP,), i32) if clearn else jnp.zeros((1,), i32),
        CEVCLR=jnp.zeros((N1,), i32) if clearn else jnp.zeros((1,), i32),
        CLOCLR=jnp.zeros((N1,), i32) if clearn else jnp.zeros((1,), i32),
        CKT=jnp.zeros((N1,), f) if clearn else jnp.zeros((1,), f),
        CKH=jnp.zeros((N1,), jnp.bool_) if clearn else jnp.zeros((1,), jnp.bool_),
    )
    xs = (EV, EVQ) if learn else EV
    fin, _ = jax.lax.scan(_event_body, carry0, xs)
    return fin


_JIT_EVENT_SCAN = jax.jit(_event_scan_loop, static_argnames=("learn", "clearn"))


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


def _spike_log_capacity(n_events):
    """First-pass spike-log capacity of an event-scan run; an overflow reruns at the logged size."""
    return max(1024, 2 * int(n_events))


def _check_scan_w_update_blocks(weights, members, member_mask, ns, mds, mdc, wi_s, wi_c):
    """The declared ``wi_s``/``wi_c`` must equal the blocks between distinct group-mates.

    ``weights`` is the ``(G, D, D)`` block array (target-local row, source-local
    column). Between two distinct members of one position, a simple position
    reads ``wi_s[depth(target), depth(source)]`` and a complex one ``wi_c``,
    exactly as the scan's learned rows do. Every disagreement is refused by
    name. A position that mixes simple and complex cells has no single learned
    row to read, so it is refused too.
    """
    mem = np.asarray(members, dtype=np.int64)
    valid = np.asarray(member_mask, dtype=bool)
    is_s = valid & (mem < ns)
    is_c = valid & (mem >= ns)
    mixed = is_s.any(axis=1) & is_c.any(axis=1)
    if bool(mixed.any()):
        g = int(np.flatnonzero(mixed)[0])
        raise ValueError(
            "simulate_lif_edge event_scan_w_update: local-inhibition position "
            f"{g} mixes simple and complex cells; the declared plasticity needs "
            "one cell type per position")
    d = int(weights.shape[1])
    pair = valid[:, :, None] & valid[:, None, :] & ~np.eye(d, dtype=bool)[None]
    for name, side, mat, dep in (("wi_s", is_s, wi_s, mds), ("wi_c", is_c, wi_c, mdc)):
        sel = pair & side[:, :, None]
        expect = mat[dep[:, :, None], dep[:, None, :]]
        bad = sel & (weights != expect)
        if bool(bad.any()):
            g, dl, kl = (int(v) for v in np.argwhere(bad)[0])
            raise ValueError(
                "simulate_lif_edge event_scan_w_update: local_inhibition_blocks "
                f"disagree with the declared {name} at position {g}, target member "
                f"{dl}, source member {kl}: the blocks hold {float(weights[g, dl, kl])!r}, "
                f"{name} says {float(expect[g, dl, kl])!r}")


def _event_scan_ranks(ev, cell, cmask, vw, ns):
    """CSR index of each simple cell's covering events, and each slot's rank.

    ``ev`` is the ``(M, 4)`` int64 event array the scan walks; ``cell`` and
    ``cmask`` are the ``(vw * vh, SMAX)`` slot tables. Returns ``row`` ``(ns + 1,)``
    (start offsets into ``idx``), ``idx`` (event indices, stream order within a
    cell) and ``q`` ``(M, SMAX)``: the rank of each covering slot in its cell's
    list, -1 for empty slots. The scan appends an event to a cell's list before
    the excite, so the rank is that event's place in the list.
    """
    qpix = ev[:, 2] * vw + ev[:, 1]
    cells = cell[qpix]
    mask = cmask[qpix]
    ev_i, slot_i = np.nonzero(mask)
    n = cells[ev_i, slot_i].astype(np.int64)
    if n.size and int(n.max()) >= ns:
        raise ValueError(
            "event_scan cover tables name a cell outside the ns simple cells; "
            "the declared buffer plasticity needs simple-only covers")
    order = np.argsort(n, kind="stable")
    counts = np.bincount(n, minlength=ns)
    row = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rank = np.empty(n.size, dtype=np.int64)
    rank[order] = np.arange(n.size, dtype=np.int64) - row[n[order]]
    q = np.full(mask.shape, -1, dtype=np.int64)
    q[ev_i, slot_i] = rank
    return row, ev_i[order], q


def _run_event_scan(
    params, N, n_steps, t0_us, dt_us, groups, blocks,
    events, tables, V0_mV, h_update=None, w_update=None,
):
    """Validate the event-scan inputs and run the exact event pass.

    Returns the kernel result dict (``spikes`` bool ``(n_steps, N)``,
    ``counts``, ``first_spike_us``, finals ``V``/``A``/``t_last``, an
    empty ``w_final``, ``weights_final`` (the final float64 ``w_s``, ``w_c``,
    ``lat_w``, ``td_w`` and ``blocks_w`` carried through the pass; plus ``wi_s``
    and ``wi_c`` under a declared ``w_update``), and the spike log ``spike_times_us`` and
    ``spike_cells`` (int64, every spike in emission order)), plus the
    periodic-update receipts ``threshold_mV``, ``rate_hz`` and
    ``interval_count`` (each ``(N,)``) and ``last_update_us`` (int).
    ``V``/``A``/``t_last`` are float64 regardless of the params dtype (see
    the x64 note at the call). Every refusal names the offending channel.
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
        w_c = np.asarray(tables["w_c"], dtype=np.float64)
        pix_pos = np.asarray(tables["pix_pos"])
        pix_n = np.asarray(tables["pix_n"])
        fwd_idx = np.asarray(tables["fwd_idx"])
        fwd_w = np.asarray(tables["fwd_w"], dtype=np.float64)
        fwd_key = np.asarray(tables["fwd_key"])
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
    csx = w_c.shape[1] if w_c.ndim == 4 else -1
    csy = w_c.shape[2] if w_c.ndim == 4 else -1
    if w_c.shape != (cd, csx, csy, nd) or csx < 1 or csy < 1:
        raise ValueError(
            f"event_scan_tables['w_c'] must have shape ({cd}, csx, csy, {nd}); "
            f"got {w_c.shape}")
    if not bool(np.all(np.isfinite(w_c))):
        raise ValueError("event_scan_tables['w_c'] must be finite")
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
    if fwd_key.shape != (ns, fmax, 3) or fwd_key.dtype.kind not in "iu":
        raise ValueError(
            "event_scan_tables['fwd_key'] must hold integers (cz, ex, ey) with "
            f"shape ({ns}, {fmax}, 3); got {fwd_key.shape} dtype {fwd_key.dtype}")
    kz, kx, ky = (fwd_key[..., a][_fslot] for a in range(3))
    if (bool((kz < 0).any()) or bool((kz >= cd).any())
            or bool((kx < 0).any()) or bool((kx >= csx).any())
            or bool((ky < 0).any()) or bool((ky >= csy).any())):
        raise ValueError(
            "event_scan_tables['fwd_key'] must name shared complex weights in "
            f"[0, {cd}) x [0, {csx}) x [0, {csy}) at every used forward slot")
    src_depth = np.broadcast_to((np.arange(ns) % nd)[:, None], (ns, fmax))[_fslot]
    if not bool(np.array_equal(fwd_w[_fslot], w_c[kz, kx, ky, src_depth])):
        raise ValueError(
            "event_scan_tables['fwd_w'] must equal w_c[cz, ex, ey, n % nd] at "
            "every used forward slot (fwd_key names the shared weight)")
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
    # The declared H update (EventScanDrive.h_update): the oracle's periodic
    # rate and threshold adaptation. None keeps the thresholds fixed (h_on 0).
    v_thresh = np.asarray(cols["v_thresh_mV"], dtype=np.float64)
    if h_update is None:
        h_on = 0
        h_sc = (0.0, 0.0)
        h_last = 0
        h_th0 = v_thresh
        h_rate0 = h_tgt = h_eta = h_tmin = np.zeros((N,), dtype=np.float64)
    else:
        from .pcl_topology import check_event_scan_h_update

        hu = check_event_scan_h_update(
            h_update, N, label="simulate_lif_edge event_scan_h_update")
        h_on = 1
        h_sc = (hu["measurement_interval_us"], hu["rate_alpha"])
        h_last = hu["last_update_us"]
        h_th0 = v_thresh if hu["threshold0_mV"] is None else hu["threshold0_mV"]
        h_rate0 = hu["rate0_hz"]
        h_tgt = hu["target_rate_hz"]
        h_eta = hu["threshold_eta_mV"]
        h_tmin = hu["threshold_min_mV"]
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
    # The declared buffer plasticity (EventScanDrive.w_update, the oracle's simple
    # excitatory update). Without a declaration these arrays go unread.
    m_valid = np.asarray(member_mask, dtype=bool)
    mem_i = np.asarray(members, dtype=np.int64)
    mds = np.where(m_valid & (mem_i < ns), mem_i % nd, 0).astype(np.int64)
    mdc = np.where(m_valid & (mem_i >= ns), (mem_i - ns) % cd, 0).astype(np.int64)
    pad_m = max(nd, cd, int(mem_i.shape[1]))
    wis0 = np.zeros((pad_m, pad_m), dtype=np.float64)
    wic0 = np.zeros((pad_m, pad_m), dtype=np.float64)
    decay0 = np.zeros((N + 1,), dtype=np.float64)
    wup = np.zeros((17,), dtype=np.float64)
    if w_update is None:
        learn = False
        clearn = False
        ring_cap = 1
    else:
        from .pcl_topology import check_event_scan_w_update

        wu = check_event_scan_w_update(
            w_update, N, nd=nd, cd=cd, lateral_on=lat_on, topdown_on=td_on,
            label="simulate_lif_edge event_scan_w_update")
        _check_scan_w_update_blocks(
            weights, members, member_mask, ns, mds, mdc, wu["wi_s"], wu["wi_c"])
        learn = True
        ring_cap = wu["buffer_capacity"]
        wis0[:nd, :nd] = wu["wi_s"]
        wic0[:cd, :cd] = wu["wi_c"]
        decay0[:N] = wu["decay"]
        sp = wu["simple"]
        cp_pop = wu["complex"]
        # The complex update runs for "excitatory"/"all" (oracle :613); "inhibitory"
        # has no complex effect, so it compiles without the complex path.
        clearn = cp_pop["mode"] in ("excitatory", "all")
        wup[:9] = [
            sp["eLTP"], sp["eLTD"], sp["iLTP"], sp["iLTD"], sp["tau_ltp_us"],
            sp["tau_ltd_us"], sp["norm_factor"], sp["eta_inh"],
            1.0 if sp["mode"] in ("excitatory", "all") else 0.0,
        ]
        wup[9:] = [
            cp_pop["eLTP"], cp_pop["eLTD"], cp_pop["iLTP"], cp_pop["iLTD"],
            cp_pop["tau_ltp_us"], cp_pop["tau_ltd_us"], cp_pop["norm_factor"],
            cp_pop["eta_inh"],
        ]
    glt0 = np.zeros((ng, ring_cap), dtype=np.float64)
    if learn:
        ev_row, ev_idx, ev_q = _event_scan_ranks(ev, cell, cmask, vw, ns)
    else:
        ev_row = np.zeros((ns + 1,), dtype=np.int64)
        ev_idx = np.zeros((1,), dtype=np.int64)
        ev_q = np.zeros((1, 1), dtype=np.int64)
    V0 = np.concatenate([v_init, np.zeros((1,), dtype=np.float64)])
    A0 = np.zeros((N + 1,), dtype=np.float64)
    # float64 under a local x64 context (restored on exit): event times are
    # integer microseconds and the oracle is float64 — float32 decay rounding
    # would move threshold crossings.
    with jax.enable_x64():
        args = (
            jnp.asarray(V0, dtype=jnp.float64),
            jnp.asarray(A0, dtype=jnp.float64),
            jnp.asarray(ev),
            jnp.asarray(cell, dtype=jnp.int32),
            jnp.asarray(cmask, dtype=bool),
            jnp.asarray(sox.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(soy.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(fwd_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(fwd_key.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(fwd_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(lat_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(lat_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(td_idx.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(td_n.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(members.astype(np.int64), dtype=jnp.int32),
            jnp.asarray(member_mask, dtype=bool),
            jnp.asarray(pos_of, dtype=jnp.int32),
            jnp.asarray(loc_of, dtype=jnp.int32),
            jnp.asarray(pcol, dtype=jnp.float64),
            jnp.asarray([vw, nx, ny, nd, t0_us, dt_us], dtype=jnp.int64),
            jnp.asarray([lat_on, td_on, activation], dtype=jnp.int32),
            jnp.zeros((n_steps, N + 1), dtype=bool),
            # The dummy cell (index N) rides with cell 0's values, as PCOL does.
            jnp.asarray(np.append(h_th0, h_th0[0]), dtype=jnp.float64),
            jnp.asarray(np.append(h_rate0, 0.0), dtype=jnp.float64),
            jnp.asarray(h_last, dtype=jnp.int64),
            jnp.asarray(np.append(h_tgt, 0.0), dtype=jnp.float64),
            jnp.asarray(np.append(h_eta, 0.0), dtype=jnp.float64),
            jnp.asarray(np.append(h_tmin, 0.0), dtype=jnp.float64),
            jnp.asarray(h_sc, dtype=jnp.float64),
            jnp.asarray(h_on, dtype=jnp.int64),
            jnp.asarray(w_s, dtype=jnp.float64),
            jnp.asarray(w_c, dtype=jnp.float64),
            jnp.asarray(np.zeros((ns, lmax), dtype=np.float64)
                       if lat_w is None else lat_w, dtype=jnp.float64),
            jnp.asarray(np.zeros((nc, tmax), dtype=np.float64)
                       if td_w is None else td_w, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(wis0, dtype=jnp.float64),
            jnp.asarray(wic0, dtype=jnp.float64),
            jnp.asarray(decay0, dtype=jnp.float64),
            jnp.asarray(mds, dtype=jnp.int32),
            jnp.asarray(mdc, dtype=jnp.int32),
            jnp.asarray(wup, dtype=jnp.float64),
            jnp.asarray(glt0, dtype=jnp.float64),
            jnp.asarray(ev_row, dtype=jnp.int32),
            jnp.asarray(ev_idx, dtype=jnp.int32),
            jnp.asarray(ev_q, dtype=jnp.int32),
        )
        cap = _spike_log_capacity(int(ev.shape[0]))
        out = _JIT_EVENT_SCAN(*args, jnp.zeros((cap,), dtype=jnp.int64),
                              learn=learn, clearn=clearn)
        n_logged = int(out.LOGN)  # every spike, past the capacity too
        if n_logged > cap:
            # The first pass stored only cap entries; rerun once at the logged size.
            cap = n_logged
            out = _JIT_EVENT_SCAN(*args, jnp.zeros((cap,), dtype=jnp.int64),
                                  learn=learn, clearn=clearn)
            if int(out.LOGN) != n_logged:
                raise RuntimeError(
                    "event-scan spike log changed on rerun: "
                    f"{int(out.LOGN)} spikes against {n_logged} logged")
        fin = out
        n_spk = int(fin.LOGN)
        weights_final = {
            "w_s": fin.WS, "w_c": fin.WC, "lat_w": fin.LATW,
            "td_w": fin.TDW, "blocks_w": fin.BLK,
        }
        if learn:
            weights_final["wi_s"] = fin.WIS[:nd, :nd]
            weights_final["wi_c"] = fin.WIC[:cd, :cd]
        t_last = jnp.where(
            fin.CNT[:N] > 0, fin.TS[:N] / 1000.0,
            jnp.full((N,), NEVER, dtype=jnp.float64))
        return {
            "spikes": fin.BIN[:, :N],
            "V": fin.V[:N],
            "A": fin.A[:N],
            "t_last": t_last,
            "w_final": jnp.zeros((0,), dtype=jnp.float64),
            "weights_final": weights_final,
            "counts": fin.CNT[:N],
            "first_spike_us": fin.FST[:N],
            "spike_times_us": fin.LOGT[:n_spk],
            "spike_cells": fin.LOGC[:n_spk].astype(jnp.int64),
            "threshold_mV": fin.TH[:N],
            "rate_hz": fin.RATE[:N],
            "interval_count": fin.CNT[:N] - fin.CSNAP[:N],
            "last_update_us": int(fin.LUP),
        }


def _run_event_scan_branch(
    params, N, n_steps, dt_ms, t0,
    local_inhibition_rounds, local_inhibition_order,
    local_inhibition_groups, local_inhibition_blocks,
    local_inhibition_visit_order, scan_events, scan_tables,
    edge_pre, edge_post, edge_weight, edge_delay_steps, edge_group,
    I_ext_pA, V0_mV, H0, record_v,
    hdp_rule, hdp_rule_params, plasticity_mask,
    feed_current, feed_arrivals, scan_h_update, scan_w_update,
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
    tau_scan = np.asarray(params.tau_syn_ms, dtype=np.float64)
    if bool((tau_scan > 0).any()):
        raise ValueError(
            "tau_syn_ms is set but resolution is 'event_scan'; the replay "
            "excites cells event by event with exact inter-event decay, no "
            "binned current filter — leave tau_syn_ms undeclared"
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
        scan_events, scan_tables, V0_mV, scan_h_update, scan_w_update,
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
    event_scan_h_update=None,
    event_scan_w_update=None,
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
    point (Hamm 2026-10-14) as a per-step APPROXIMATION of the event-exact
    cascade, not the event-exact order itself: potentials decay once per
    step, the step's ``k`` arrivals share one ``k * (R + A)`` subtraction,
    same-step events do not interleave decay between visits
    (``"event_scan"`` is the exact per-event replay under the Stage C test
    conditions - bounded by the 20-clip receipt, not universal). Each fed cell
    is visited in feed order with ties
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

    ``params.tau_syn_ms`` (per neuron, 0 when undeclared) is the first-order
    synaptic current filter ``dg/dt = -g/tau_syn + sum arrivals``
    (Potjans G-TAUSYN; paper ``tau_syn = 0.5 ms``): each step's edge
    arrivals and ``feed_current`` sum into ``g`` after its
    ``exp(-dt/tau_syn)`` decay, and ``g`` (not the raw deltas) enters the
    input current. Arrival counts still subtract one ``R + A`` each, as
    before. ``0.0`` everywhere runs the historical delta-summing path bit
    for bit (the update is skipped, not computed); the exact event-scan
    replay refuses a set ``tau_syn_ms`` (it has no binned current).

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
    ``first_spike_us`` (``-1`` where a cell never spiked), and the spike log
    ``spike_times_us`` and ``spike_cells`` (int64, every spike in emission
    order). ``event_scan_h_update`` (None, or the mapping
    :func:`jaxfne.pcl_topology.check_event_scan_h_update` validates) declares
    the oracle's periodic rate and threshold update (``pcl_event_ref.py``
    :746-762) under ``"event_scan"`` only; None keeps the thresholds at
    ``v_thresh_mV``. The scan then also returns ``threshold_mV``, ``rate_hz``
    and ``interval_count`` (spikes since the last update, each ``(N,)``) and
    ``last_update_us`` (int).
    ``event_scan_w_update`` (None, or the mapping
    :func:`jaxfne.pcl_topology.check_event_scan_w_update` validates) declares
    the simple excitatory buffer plasticity (``pcl_event_ref.py:543-574``)
    under ``"event_scan"`` only; None keeps every weight frozen. The scan then
    also returns ``wi_s`` and ``wi_c`` in ``weights_final``, the final shared
    local-inhibition matrices.

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
    tau_col = jnp.asarray(params.tau_syn_ms, dtype)
    if tuple(tau_col.shape) != (N,):
        raise ValueError(
            f"params tau_syn_ms must have shape ({N},); got {tuple(tau_col.shape)}"
        )
    tau_np = np.asarray(tau_col, dtype=np.float64)
    if not bool(np.all(np.isfinite(tau_np))) or bool((tau_np < 0).any()):
        raise ValueError("params tau_syn_ms must be finite and >= 0 (0 runs the delta path)")
    use_filter = bool((tau_np > 0).any())
    has_tau = tau_col > 0
    decay_syn = jnp.where(
        has_tau,
        jnp.exp(-float(dt_ms) / jnp.maximum(tau_col, 1e-12)),
        jnp.zeros((), dtype),
    )

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
            feed_current, feed_arrivals, event_scan_h_update,
            event_scan_w_update,
        )
    if event_scan_h_update is not None:
        raise ValueError(
            "event_scan_h_update is set but resolution is "
            f"{_resolution!r}; the periodic rate and threshold update only "
            "runs under local_inhibition_resolution='event_scan'"
        )
    if event_scan_w_update is not None:
        raise ValueError(
            "event_scan_w_update is set but resolution is "
            f"{_resolution!r}; the buffer plasticity only runs under "
            "local_inhibition_resolution='event_scan'"
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
        try:
            h_host = np.asarray(jax.device_get(H_init))
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError, TypeError, ValueError):
            h_host = None  # traced under jit/map: finiteness checked on the concrete call
        if h_host is not None and not bool(np.all(np.isfinite(h_host))):
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
        V, A, t_last, w, H, aux, bias, ring, prev_spk, t_idx, g = carry
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
        if use_filter:
            syn = I_net
            if F_cur is not None:
                syn = syn + F_cur[t_idx]
            g = g * decay_syn + syn
            I_tot = I_tot + g
        else:
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
        return (V_n, A_n, t_n, w, H, aux, bias, ring, spk, t_idx + 1, g), out

    if descriptor is not None:
        Hc, auxc, bc = H0_arr, aux0, jnp.zeros((N,), dtype)
    else:
        Hc, auxc, bc = None, None, None
    carry0 = (
        V0, jnp.zeros((N,), dtype), jnp.full((N,), NEVER, dtype),
        w0, Hc, auxc, bc, jnp.zeros((L, N), bool),
        jnp.zeros((N,), bool), jnp.int32(0), jnp.zeros((N,), dtype),
    )
    (V_f, A_f, t_f, w_f, H_f, aux_f, b_f, _, _, _, _), ys = jax.lax.scan(
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
