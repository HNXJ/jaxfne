"""PCL retinotopic geometry: footprint wiring and same-step local inhibition (lane 2b, Stage B).

Reference: N'dri, Barbier, Teuliere, Triesch, Nat Commun 16:8880 (2025),
doi:10.1038/s41467-025-64234-z. The oracle transcription
``artifacts/etudes/pcl/port/pcl_event_ref.py`` (read-only) is the arbiter for
every claim below; cited as ``oracle:<line>``.

Three mechanisms, each reading posed positions from the NeuronalTensor A:

- B1 footprint connection rule: a post cell at anchor ``q`` (its posed
  position plus ``offset``) connects from every pre cell whose posed position
  ``r`` falls in ``[q, q + size)``. On lattice layouts this reproduces the
  oracle cover tables bit-exactly (``cover_x``/``cover_y``,
  oracle:418-421); on continuous layouts the same rectangle test applies.
- B2 complex pooling: the same rule between the simple and complex layers
  (``c_cover``, oracle:423-428). Pooling is a sum: each simple spike adds its
  weight to every covering complex cell (``ComplexNeuron::membraneUpdate``,
  oracle:241-253); there is no max. No ``pooling`` parameter exists to
  misdeclare this.
- B3 same-step local inhibition: at one retinotopic position, depths are
  visited in ascending order; a depth above threshold at its visit spikes and
  immediately inhibits the other depths at that position
  (``SpikingNetwork::addEvent`` visit order, oracle:671-681, and
  ``lateralLocalInhibition``, oracle:686-694). At most ``rounds`` winners
  resolve per call (the declared depth order is ascending); further crossings
  are counted as ``missed`` (binned-port ``_cascade`` semantics).

Weight sharing (``sharingType: "patch"``, oracle:369): all cells of one
depth share one plastic matrix (``w_s[k]``, oracle:454; ``w_c[cz]``,
oracle:456). Edges of one receiving-depth slot form one group; the shared
weight is one row per group (gather with :func:`expand_shared`, updates with
:func:`accumulate_shared_update`). Group ids ride the compiled rule for the
registered ``pcl_stdp`` rule to consume as its ``group`` parameter
(``jaxfne/hdp_rules_pcl.py:56`` normalizes each group to ``lam``).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: Weight-sharing modes with a core consumer. ``"patch"`` is the oracle's
#: ``sharingType`` (one plastic weight per receiving depth). Anything else is
#: refused by :func:`validate_weight_sharing`.
WEIGHT_SHARING_MODES = ("patch",)

#: Keys a footprint declaration may carry. Anything else is refused.
_FOOTPRINT_KEYS = ("size", "offset")

#: Visit orders a local-inhibition declaration may name. ``"ascending"`` is
#: the oracle's depth order (depths visited ascending, oracle:671-681);
#: ``"descending"`` visits them in reverse. Anything else is refused by
#: :func:`validate_local_inhibition`.
LOCAL_INHIBITION_ORDERS = ("ascending", "descending")

#: Values the ``resolution`` key of a local-inhibition declaration may take.
#: ``"whole_step"`` is the default: one grouped first-R cap per step, in the
#: declared order. ``"event_order"`` resolves the step's feed in event order
#: (Hamm 2026-10-14: same-step interactions are a declared intra-step fixed
#: point with rounds and order on A): each fed cell is visited in feed order
#: with ties in the declared order, and every spike subtracts the local
#: inhibition from its group-mates before the next visit. ``"event_scan"``
#: resolves the raw event stream event by event in timestamp order (event
#: order, then ascending index on A): each event excites its covered cells
#: with exact inter-event decay, and each spike inhibits, lateralizes and
#: forwards immediately, exactly as the event oracle
#: (``pcl_event_ref.py:663-697``). Anything else is refused by
#: :func:`validate_local_inhibition`.
LOCAL_INHIBITION_RESOLUTIONS = ("whole_step", "event_order", "event_scan")

#: Keys a local-inhibition declaration may carry. Anything else is refused.
#: ``resolution`` is optional and, when absent, is left out of the
#: normalized spec so undeclared specs echo byte-identical static.
_LOCAL_INHIBITION_KEYS = ("rounds", "order", "resolution")


def validate_local_inhibition(value: Mapping[str, Any] | None, *, label: str) -> dict | None:
    """Validate a same-step local-inhibition declaration (fail-closed).

    ``None`` passes through (undeclared = no intra-step fixed point). A
    declared value must be a mapping with ``rounds`` (required: positive
    integer winners resolved per retinotopic position per step, the ruled
    ``R``), ``order`` (optional, default ``"ascending"``: the depth visit
    order the rounds resolve in) and ``resolution`` (optional: one of
    ``LOCAL_INHIBITION_RESOLUTIONS``, default ``"whole_step"``; the key is
    echoed only when explicitly declared, so default specs are unchanged).
    Under ``resolution="event_scan"`` the stream order is the order (the
    replay resolves every event, no cap): ``rounds`` must be absent and
    ``order`` absent or ``"ascending"``; the normalized spec carries no
    ``rounds`` key. Returns the normalized spec.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{label} local_inhibition must be a mapping with 'rounds' "
            f"(and optional 'order'); got {value!r}"
        )
    unknown = [k for k in value if k not in _LOCAL_INHIBITION_KEYS]
    if unknown:
        raise ValueError(
            f"{label} local_inhibition carries unknown keys {unknown!r}; "
            f"expected a subset of {_LOCAL_INHIBITION_KEYS}"
        )
    if value.get("resolution") == "event_scan":
        if "rounds" in value:
            raise ValueError(
                f"{label} local_inhibition resolution 'event_scan' replays "
                f"every event with no cap; 'rounds' must be absent, got "
                f"{value['rounds']!r}"
            )
        order = value.get("order", "ascending")
        if order != "ascending":
            raise ValueError(
                f"{label} local_inhibition order must be 'ascending' under "
                f"resolution 'event_scan' (the scan visits candidates in "
                f"ascending index); got {order!r}"
            )
        return {"order": "ascending", "resolution": "event_scan"}
    if "rounds" not in value:
        raise ValueError(
            f"{label} local_inhibition needs 'rounds' (winners resolved per "
            f"position per step); got {dict(value)!r}"
        )
    rounds = validate_rounds(value["rounds"], label=f"{label} local_inhibition")
    order = value.get("order", "ascending")
    if not isinstance(order, str) or order not in LOCAL_INHIBITION_ORDERS:
        raise ValueError(
            f"{label} local_inhibition order must be one of "
            f"{list(LOCAL_INHIBITION_ORDERS)}; got {order!r}"
        )
    spec = {"rounds": rounds, "order": order}
    if "resolution" in value:
        resolution = value["resolution"]
        if not isinstance(resolution, str) or resolution not in LOCAL_INHIBITION_RESOLUTIONS:
            raise ValueError(
                f"{label} local_inhibition resolution must be one of "
                f"{list(LOCAL_INHIBITION_RESOLUTIONS)}; got {resolution!r}"
            )
        spec["resolution"] = resolution
    return spec


def pcl_grid_origins(patch: float, size: float, overlap: float, n: int) -> list[int]:
    """Origins of ``n`` cells along one grid axis (oracle:408-411).

    ``origin[i] = patch + i * (size - overlap)`` (``SpikingNetwork.cpp:297-300``).
    """
    return [int(patch) + i * (int(size) - int(overlap)) for i in range(int(n))]


def _checked_pair(value: Any, *, name: str, label: str, positive: bool) -> tuple[float, float]:
    if isinstance(value, bool) or isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} footprint {name} must be a pair of numbers; got {value!r}")
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{label} footprint {name} must be a pair of numbers; got {value!r}")
    out = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise ValueError(f"{label} footprint {name} must be a pair of numbers; got {value!r}")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0.0):
            want = "finite and > 0" if positive else "finite"
            raise ValueError(
                f"{label} footprint {name} must be {want}; got {value!r}"
            )
        out.append(number)
    return (out[0], out[1])


def validate_footprint(footprint: Mapping[str, Any] | None, *, label: str) -> dict | None:
    """Validate a footprint declaration, returning the normalized spec (fail-closed).

    ``None`` passes through (undeclared = current behaviour). A declared value
    must be a mapping with ``size`` (required: covering rectangle in the same
    units as the posed positions, both entries finite and > 0) and ``offset``
    (optional, default ``(0.0, 0.0)``: added to the post position before the
    window test). Unknown keys, bools, strings, NaN/inf and non-positive
    sizes are refused by name.
    """
    if footprint is None:
        return None
    if not isinstance(footprint, Mapping):
        raise ValueError(
            f"{label} footprint must be a mapping with keys {_FOOTPRINT_KEYS}; "
            f"got {footprint!r}"
        )
    unknown = [k for k in footprint if k not in _FOOTPRINT_KEYS]
    if unknown:
        raise ValueError(
            f"{label} footprint carries unknown keys {unknown!r}; "
            f"expected a subset of {_FOOTPRINT_KEYS}"
        )
    if "size" not in footprint:
        raise ValueError(
            f"{label} footprint needs 'size' (the covering rectangle); "
            f"got {dict(footprint)!r}"
        )
    size = _checked_pair(footprint["size"], name="'size'", label=label, positive=True)
    offset = (0.0, 0.0)
    if "offset" in footprint:
        offset = _checked_pair(footprint["offset"], name="'offset'", label=label, positive=False)
    return {"size": [size[0], size[1]], "offset": [offset[0], offset[1]]}


def validate_weight_sharing(value: str | None, *, label: str) -> str | None:
    """Validate a weight-sharing declaration (fail-closed).

    ``None`` passes through (per-edge weights). A declared value must name a
    mode in :data:`WEIGHT_SHARING_MODES`.
    """
    if value is None:
        return None
    if not isinstance(value, str) or value not in WEIGHT_SHARING_MODES:
        raise ValueError(
            f"{label} weight_sharing must be one of {list(WEIGHT_SHARING_MODES)}; "
            f"got {value!r}"
        )
    return value


def validate_rounds(rounds: Any, *, label: str) -> int:
    """Validate a declared number of inhibition winners per position (fail-closed)."""
    if isinstance(rounds, bool) or not isinstance(rounds, numbers.Integral):
        raise ValueError(f"{label} rounds must be a positive integer; got {rounds!r}")
    value = int(rounds)
    if value < 1:
        raise ValueError(f"{label} rounds must be a positive integer; got {rounds!r}")
    return value


def footprint_pairs(
    pre_ids: Sequence[int],
    pre_xy: np.ndarray,
    post_ids: Sequence[int],
    post_xy: np.ndarray,
    footprint: Mapping[str, Any],
    *,
    label: str = "footprint",
) -> tuple[list[tuple[int, int]], list[int]]:
    """Deterministic (pre, post) pairs of a footprint rule on posed positions.

    ``pre_xy``/``post_xy`` are ``(n, 2)`` arrays of posed ``(x, y)`` read from
    the neuron rows; ``pre_ids``/``post_ids`` are the matching neuron ids. A
    pair exists iff ``0 <= (pre - (post + offset)) < size`` component-wise
    (the oracle's ``s_ox[i] <= x < s_ox[i] + SX`` cover test, oracle:418-421,
    with the post anchor at its posed position). Iteration is in post order,
    then pre order. The second return is the post slot per pair: co-located
    post cells (exactly equal ``(x, y)``) numbered 0, 1, ... in post order —
    the receiving-depth index, i.e. the weight-sharing group under
    ``"patch"`` sharing. Missing/ragged coordinates and a bad declaration
    are refused by name.
    """
    spec = validate_footprint(footprint, label=label)
    assert spec is not None
    sx, sy = spec["size"]
    ox, oy = spec["offset"]
    pre_ids = list(pre_ids)
    post_ids = list(post_ids)
    pre_arr = np.asarray(pre_xy, dtype=float)
    post_arr = np.asarray(post_xy, dtype=float)
    if pre_arr.shape != (len(pre_ids), 2) or not np.all(np.isfinite(pre_arr)):
        raise ValueError(
            f"{label} footprint needs finite pre_xy of shape ({len(pre_ids)}, 2); "
            f"got shape {pre_arr.shape}"
        )
    if post_arr.shape != (len(post_ids), 2) or not np.all(np.isfinite(post_arr)):
        raise ValueError(
            f"{label} footprint needs finite post_xy of shape ({len(post_ids)}, 2); "
            f"got shape {post_arr.shape}"
        )
    slots: list[int] = []
    seen: dict[tuple[float, float], int] = {}
    for x, y in post_arr.tolist():
        key = (float(x), float(y))
        slots.append(seen.get(key, 0))
        seen[key] = slots[-1] + 1
    pairs: list[tuple[int, int]] = []
    groups: list[int] = []
    for j, (post_id, (qx, qy)) in enumerate(zip(post_ids, post_arr.tolist())):
        for pre_id, (rx, ry) in zip(pre_ids, pre_arr.tolist()):
            if 0.0 <= rx - (qx + ox) < sx and 0.0 <= ry - (qy + oy) < sy:
                pairs.append((int(pre_id), int(post_id)))
                groups.append(slots[j])
    return pairs, groups


def expand_shared(shared: np.ndarray, group_of_edge: Sequence[int]) -> np.ndarray:
    """Gather one shared weight row per edge (``shared[group]`` per edge)."""
    table = np.asarray(shared, dtype=float)
    groups = np.asarray(list(group_of_edge), dtype=np.int64)
    if table.ndim != 2:
        raise ValueError(f"shared must be a 2-D (n_groups, n_syn) table; got shape {table.shape}")
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_of_edge must be a non-empty 1-D sequence")
    if bool((groups < 0).any()) or bool((groups >= table.shape[0]).any()):
        raise ValueError(
            f"group_of_edge holds ids outside [0, {table.shape[0]}); got range "
            f"({int(groups.min())}, {int(groups.max())})"
        )
    return table[groups]


def accumulate_shared_update(
    shared: np.ndarray, group_of_edge: Sequence[int], edge_delta: np.ndarray
) -> np.ndarray:
    """Scatter-mean edge updates onto the shared table (one row per group).

    The shared weight moves by the mean update of its edges — the
    patch-sharing equivalent of applying the group's STDP once.
    """
    table = np.asarray(shared, dtype=float)
    groups = np.asarray(list(group_of_edge), dtype=np.int64)
    delta = np.asarray(edge_delta, dtype=float)
    if table.ndim != 2:
        raise ValueError(f"shared must be a 2-D (n_groups, n_syn) table; got shape {table.shape}")
    if groups.ndim != 1 or delta.ndim != 2 or groups.shape[0] != delta.shape[0]:
        raise ValueError(
            f"group_of_edge {groups.shape} and edge_delta {delta.shape} disagree"
        )
    out = table.copy()
    for g in range(table.shape[0]):
        sel = groups == g
        if bool(sel.any()):
            out[g] = out[g] + delta[sel].mean(axis=0)
    return out


def resolve_local_inhibition(
    *,
    v: np.ndarray,
    adapt: np.ndarray | None,
    t_last: np.ndarray,
    t_spike: np.ndarray,
    t_now: float,
    thr: np.ndarray | float,
    inputs: np.ndarray,
    inhib: np.ndarray,
    tau_m: float,
    tau_adapt: float,
    delta_adapt: float,
    delta_rp: float,
    tau_rp: float,
    v_reset: float,
    neg_limit: float,
    adaptive: bool,
    rounds: int,
    order: str = "ascending",
    label: str = "local_inhibition",
) -> dict[str, Any]:
    """Same-step local inhibition at one retinotopic position (B3).

    ``v``/``adapt``/``t_last``/``t_spike`` are per-depth state (``t_spike`` 0
    is the oracle's never-spiked sentinel, oracle:210-212); ``inputs`` is
    ``(M, D)`` sequential input vectors at one timestamp in visit order
    (``M == 1`` for the simple feed; one row per incoming simple spike for
    the complex layer); ``inhib[d, k]`` is the weight from spiking depth
    ``k`` onto depth ``d`` with a zero diagonal. Per input, depths are
    visited in ``order`` (``"ascending"`` is the oracle's depth order,
    oracle:671-681; ``"descending"`` visits them in reverse): decay to ``t``
    (``t`` rides ``t_last``... see below), add the input minus refractory
    (and adaptation) terms, clamp at ``neg_limit``; a depth strictly above
    ``thr`` spikes (reset to ``v_reset``, adaptation increment when
    ``adaptive``) and immediately inhibits the other depths (decay,
    subtract, clamp). At most ``rounds`` winners resolve; further crossings
    keep their excitation but neither spike nor inhibit, and are counted as
    ``missed``. Both ``rounds`` and ``order`` change who spikes, so both
    are part of the declaration on A (Hamm 2026-10-14).

    Times share one unit (the oracle uses integer microseconds,
    oracle:64-65); ``t_now`` is the single timestamp of this call.
    """
    n_rounds = validate_rounds(rounds, label=label)
    if not isinstance(order, str) or order not in LOCAL_INHIBITION_ORDERS:
        raise ValueError(
            f"{label} order must be one of {list(LOCAL_INHIBITION_ORDERS)}; "
            f"got {order!r}"
        )
    vec = np.asarray(v, dtype=float).ravel().copy()
    if vec.size == 0:
        raise ValueError(f"{label} v must hold at least one depth")
    dep = vec.size
    last = np.asarray(t_last, dtype=float).ravel().copy()
    spk = np.asarray(t_spike, dtype=float).ravel().copy()
    if last.shape != (dep,) or spk.shape != (dep,):
        raise ValueError(
            f"{label} t_last {last.shape} and t_spike {spk.shape} must "
            f"both match v ({dep},)"
        )
    if adaptive:
        if adapt is None:
            raise ValueError(f"{label} adaptive=True needs adapt state")
        ada = np.asarray(adapt, dtype=float).ravel().copy()
        if ada.shape != (dep,):
            raise ValueError(f"{label} adapt {ada.shape} must match v ({dep},)")
    else:
        ada = np.zeros(dep)
    th = np.asarray(thr, dtype=float).ravel()
    if th.size == 1:
        th = np.full(dep, float(th[0]))
    if th.shape != (dep,):
        raise ValueError(f"{label} thr {th.shape} must be scalar or ({dep},)")
    seq = np.asarray(inputs, dtype=float)
    if seq.ndim == 1:
        seq = seq[None, :]
    if seq.ndim != 2 or seq.shape[1] != dep:
        raise ValueError(
            f"{label} inputs {seq.shape} must be (M, {dep}) in visit order"
        )
    mat = np.asarray(inhib, dtype=float)
    if mat.shape != (dep, dep):
        raise ValueError(f"{label} inhib {mat.shape} must be ({dep}, {dep})")
    if bool((np.diag(mat) != 0.0).any()):
        raise ValueError(f"{label} inhib must carry a zero diagonal (no self-inhibition)")
    for name, number in (
        ("tau_m", tau_m), ("tau_rp", tau_rp), ("v_reset", v_reset),
        ("neg_limit", neg_limit),
    ):
        if not isinstance(number, numbers.Real) or isinstance(number, bool):
            raise ValueError(f"{label} {name} must be a real number; got {number!r}")
    tau_m = float(tau_m)
    tau_rp = float(tau_rp)
    if not (math.isfinite(tau_m) and tau_m > 0.0):
        raise ValueError(f"{label} tau_m must be finite and > 0; got {tau_m!r}")
    if not (math.isfinite(tau_rp) and tau_rp > 0.0):
        raise ValueError(f"{label} tau_rp {tau_rp} must be finite and > 0")
    if adaptive and not (math.isfinite(float(tau_adapt)) and float(tau_adapt) > 0.0):
        raise ValueError(f"{label} tau_adapt must be finite and > 0; got {tau_adapt!r}")
    if isinstance(t_now, bool) or not isinstance(t_now, numbers.Real):
        raise ValueError(f"{label} t_now must be a real timestamp; got {t_now!r}")
    t_now = float(t_now)
    winners = np.zeros(dep, dtype=bool)
    n_winners = 0
    missed = 0
    for row in seq.tolist():
        visit = range(dep) if order == "ascending" else range(dep - 1, -1, -1)
        for d in visit:
            dt = t_now - float(last[d])
            if dt < 0:
                raise ValueError(
                    f"local_inhibition t_now={t_now} precedes t_last[{d}]={last[d]}"
                )
            vec[d] *= math.exp(-dt / tau_m)
            if adaptive:
                ada[d] *= math.exp(-dt / float(tau_adapt))
            ref = 0.0 if spk[d] == 0.0 else float(delta_rp) * math.exp(-(t_now - spk[d]) / tau_rp)
            vec[d] += float(row[d]) - ref - (ada[d] if adaptive else 0.0)
            last[d] = t_now
            if vec[d] < float(neg_limit):
                vec[d] = float(neg_limit)
            if vec[d] > float(th[d]):
                if n_winners >= n_rounds:
                    missed += 1
                    continue
                winners[d] = True
                n_winners += 1
                spk[d] = t_now
                vec[d] = float(v_reset)
                if adaptive:
                    ada[d] += float(delta_adapt)
                for e in range(dep):
                    if e == d:
                        continue
                    dte = t_now - float(last[e])
                    vec[e] *= math.exp(-dte / tau_m)
                    if adaptive:
                        ada[e] *= math.exp(-dte / float(tau_adapt))
                    refe = 0.0 if spk[e] == 0.0 else float(delta_rp) * math.exp(
                        -(t_now - spk[e]) / tau_rp)
                    vec[e] -= float(mat[e, d]) + refe + (ada[e] if adaptive else 0.0)
                    last[e] = t_now
                    if vec[e] < float(neg_limit):
                        vec[e] = float(neg_limit)
    return {
        "winners": winners,
        "v_next": vec,
        "adapt_next": ada if adaptive else None,
        "t_last_next": last,
        "t_spike_next": spk,
        "missed": int(missed),
        "n_winners": int(n_winners),
    }


def _checked_feed_inputs(
    event_t_us: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_p: np.ndarray,
    *,
    dt_ms: float,
    n_steps: int,
    origins_x: Sequence[float],
    origins_y: Sequence[float],
    size: tuple[float, float],
    weights: np.ndarray,
    nx: int,
    ny: int,
    nd: int,
    t0_us: int = 0,
    label: str = "dvs_feed",
) -> dict[str, Any]:
    """Validate the shared DVS-feed builder inputs (fail-closed, by name).

    Returns the parsed grid (bins, origins, weight table, counts) both
    :func:`dvs_feed_arrays` and :func:`dvs_feed_ranks` build from, so the
    two agree on which (step, cell) pairs the stream touches.
    """
    if not isinstance(dt_ms, numbers.Real) or isinstance(dt_ms, bool):
        raise ValueError(f"{label} dt_ms must be a real number; got {dt_ms!r}")
    dt_ms_f = float(dt_ms)
    if not math.isfinite(dt_ms_f) or dt_ms_f <= 0.0:
        raise ValueError(f"{label} dt_ms must be finite and > 0; got {dt_ms!r}")
    dt_us = dt_ms_f * 1000.0
    if dt_us != math.floor(dt_us):
        raise ValueError(
            f"{label} dt_ms must resolve to whole microseconds; got {dt_ms!r}"
        )
    dt_us_i = int(dt_us)
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral):
        raise ValueError(f"{label} n_steps must be a positive integer; got {n_steps!r}")
    n_steps_i = int(n_steps)
    if n_steps_i < 1:
        raise ValueError(f"{label} n_steps must be a positive integer; got {n_steps!r}")
    for name, value in (("nx", nx), ("ny", ny), ("nd", nd)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
            raise ValueError(
                f"{label} {name} must be a positive integer; got {value!r}"
            )
    nx_i, ny_i, nd_i = int(nx), int(ny), int(nd)
    ox = np.asarray(list(origins_x), dtype=float)
    oy = np.asarray(list(origins_y), dtype=float)
    if ox.shape != (nx_i,) or not np.all(np.isfinite(ox)):
        raise ValueError(
            f"{label} origins_x must hold {nx_i} finite values; got shape {ox.shape}"
        )
    if oy.shape != (ny_i,) or not np.all(np.isfinite(oy)):
        raise ValueError(
            f"{label} origins_y must hold {ny_i} finite values; got shape {oy.shape}"
        )
    sx, sy = _checked_pair(size, name="'size'", label=label, positive=True)
    if sx != math.floor(sx) or sy != math.floor(sy):
        raise ValueError(
            f"{label} footprint 'size' must hold whole pixels (weight indices); "
            f"got {size!r}"
        )
    sx_i, sy_i = int(sx), int(sy)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (nd_i, 2, sx_i, sy_i) or not np.all(np.isfinite(w)):
        raise ValueError(
            f"{label} weights must be a finite array of shape "
            f"({nd_i}, 2, {sx_i}, {sy_i}); got shape {w.shape}"
        )
    t = np.asarray(event_t_us).ravel()
    x = np.asarray(event_x).ravel()
    y = np.asarray(event_y).ravel()
    p = np.asarray(event_p).ravel()
    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError(
            f"{label} event arrays disagree in length: "
            f"t{t.shape} x{x.shape} y{y.shape} p{p.shape}"
        )
    if t.size and not np.all(np.isfinite(t)):
        raise ValueError(f"{label} event_t_us must be finite")
    if t.size and ((x < 0).any() or (y < 0).any()):
        bad = int(np.flatnonzero((x < 0) | (y < 0))[0])
        raise ValueError(
            f"{label} event {bad} is off the pixel plane "
            f"(x={x[bad]!r}, y={y[bad]!r})"
        )
    if t.size and not np.all((p == 0) | (p == 1)):
        bad = int(np.flatnonzero(~((p == 0) | (p == 1)))[0])
        raise ValueError(
            f"{label} event {bad} has polarity {p[bad]!r}; expected 0 or 1"
        )
    if not isinstance(t0_us, numbers.Integral) or isinstance(t0_us, bool):
        raise ValueError(f"{label} t0_us must be an integer; got {t0_us!r}")
    bins = ((t.astype(np.int64) - int(t0_us)) // dt_us_i) if t.size else t.astype(np.int64)
    if t.size and (bool((bins < 0).any()) or bool((bins >= n_steps_i).any())):
        bad = int(np.flatnonzero((bins < 0) | (bins >= n_steps_i))[0])
        raise ValueError(
            f"{label} event {bad} falls outside the {n_steps_i}-step grid "
            f"(t={t[bad]!r}, bin={bins[bad]!r})"
        )
    ns = nx_i * ny_i * nd_i
    return {
        "t": t, "x": x, "y": y, "p": p, "bins": bins,
        "ox": ox, "oy": oy, "sx_i": sx_i, "sy_i": sy_i, "w": w,
        "ns": ns, "nx_i": nx_i, "ny_i": ny_i, "nd_i": nd_i,
        "dt_us_i": dt_us_i, "n_steps_i": n_steps_i, "t0": int(t0_us),
    }


def dvs_feed_arrays(
    event_t_us: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_p: np.ndarray,
    *,
    dt_ms: float,
    n_steps: int,
    origins_x: Sequence[float],
    origins_y: Sequence[float],
    size: tuple[float, float],
    weights: np.ndarray,
    nx: int,
    ny: int,
    nd: int,
    t0_us: int = 0,
    label: str = "dvs_feed",
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a DVS event stream onto simple cells through the footprint rule.

    Event ``m`` at pixel ``(x, y)`` with polarity ``p`` reaches every post
    cell whose anchor window covers the pixel (the oracle's ``cover_x`` /
    ``cover_y`` test, oracle:418-421): post ``(i, j, k)`` gains
    ``weights[k, p, x - origins_x[i], y - origins_y[j]]`` current and one
    arrival in bin ``(t - t0_us) // dt_us``. Returns ``(current, arrivals)``
    float32 arrays of shape ``(n_steps, nx * ny * nd)`` in the oracle's flat
    order (``(i * ny + j) * nd + k``), ready for
    :func:`jaxfne.emitters_lif.simulate_lif_edge` ``feed_current`` /
    ``feed_arrivals``. Accumulation is exact (float64, then cast).

    ``size`` is the covering rectangle in pixels (both entries positive
    integers: they index ``weights`` of shape ``(nd, 2, sx, sy)``).
    ``dt_ms`` must resolve to a whole number of microseconds. An event
    outside the time grid, off the non-negative pixel plane, or with a
    polarity outside ``{0, 1}`` is refused by name; a pixel no window
    covers contributes nothing (the oracle loops over empty covers).
    """
    g = _checked_feed_inputs(
        event_t_us, event_x, event_y, event_p, dt_ms=dt_ms, n_steps=n_steps,
        origins_x=origins_x, origins_y=origins_y, size=size, weights=weights,
        nx=nx, ny=ny, nd=nd, t0_us=t0_us, label=label,
    )
    t, x, y, p, bins = g["t"], g["x"], g["y"], g["p"], g["bins"]
    ox, oy, sx_i, sy_i, w = g["ox"], g["oy"], g["sx_i"], g["sy_i"], g["w"]
    ns, nx_i, ny_i, nd_i = g["ns"], g["nx_i"], g["ny_i"], g["nd_i"]
    n_steps_i = g["n_steps_i"]
    current = np.zeros((n_steps_i, ns), dtype=np.float64)
    arrivals = np.zeros((n_steps_i, ns), dtype=np.float64)
    if t.size == 0:
        return current.astype(np.float32), arrivals.astype(np.float32)
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    key = xi * (int(yi.max()) + 1) + yi
    order = np.argsort(key, kind="stable")
    skey = key[order]
    bounds = np.flatnonzero(np.diff(skey, prepend=skey[0] - 1))
    starts = np.concatenate([bounds, [skey.shape[0]]])
    for lo, hi in zip(starts[:-1], starts[1:]):
        m0 = int(order[lo])
        px, py = int(xi[m0]), int(yi[m0])
        cover_i = [i for i in range(nx_i) if ox[i] <= px < ox[i] + sx_i]
        cover_j = [j for j in range(ny_i) if oy[j] <= py < oy[j] + sy_i]
        if not cover_i or not cover_j:
            continue
        cells = np.array(
            [(i * ny_i + j) * nd_i + k for i in cover_i for j in cover_j
             for k in range(nd_i)],
            dtype=np.int64,
        )
        tab = np.array(
            [[w[k, q, px - int(ox[i]), py - int(oy[j])]
              for i in cover_i for j in cover_j for k in range(nd_i)]
             for q in (0, 1)],
            dtype=np.float64,
        )
        sel = order[lo:hi]
        bsel = bins[sel]
        psel = p[sel].astype(np.int64)
        # np.add.at: repeated (bin, cell) pairs accumulate (fancy += is
        # last-wins on duplicates, so it would drop binned events).
        np.add.at(current, (bsel[:, None], cells[None, :]), tab[psel])
        np.add.at(arrivals, (bsel[:, None], cells[None, :]), 1.0)
    return current.astype(np.float32), arrivals.astype(np.float32)


def dvs_feed_ranks(
    event_t_us: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_p: np.ndarray,
    *,
    dt_ms: float,
    n_steps: int,
    origins_x: Sequence[float],
    origins_y: Sequence[float],
    size: tuple[float, float],
    weights: np.ndarray,
    nx: int,
    ny: int,
    nd: int,
    t0_us: int = 0,
    label: str = "dvs_feed_ranks",
) -> np.ndarray:
    """First-event time of each fed (step, cell) pair, for event-order resolution.

    Same inputs and cover rule as :func:`dvs_feed_arrays` (validated by the
    same checker, so both agree on which pairs the stream touches).
    ``ranks[b, n]`` is the offset in microseconds of cell ``n``'s first
    event inside step ``b`` (``t - (t0_us + b * dt_us)``), or ``0.0`` when
    the cell is fed no event in that step (such cells are never visited:
    :func:`feed_visit_order` only visits pairs with arrivals). Returns a
    float32 array of shape ``(n_steps, nx * ny * nd)``; exact (integer
    microseconds, ``np.minimum.at`` accumulation).
    """
    g = _checked_feed_inputs(
        event_t_us, event_x, event_y, event_p, dt_ms=dt_ms, n_steps=n_steps,
        origins_x=origins_x, origins_y=origins_y, size=size, weights=weights,
        nx=nx, ny=ny, nd=nd, t0_us=t0_us, label=label,
    )
    t, x, y = g["t"], g["x"], g["y"]
    ox, oy, sx_i, sy_i = g["ox"], g["oy"], g["sx_i"], g["sy_i"]
    ns, nx_i, ny_i, nd_i = g["ns"], g["nx_i"], g["ny_i"], g["nd_i"]
    n_steps_i, dt_us_i, t0 = g["n_steps_i"], g["dt_us_i"], g["t0"]
    ranks = np.full((n_steps_i, ns), np.inf, dtype=np.float64)
    if t.size == 0:
        return np.zeros((n_steps_i, ns), dtype=np.float32)
    bins = g["bins"]
    step_start = t0 + bins.astype(np.int64) * dt_us_i
    off = (t.astype(np.int64) - step_start).astype(np.float64)
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    key = xi * (int(yi.max()) + 1) + yi
    order = np.argsort(key, kind="stable")
    skey = key[order]
    bounds = np.flatnonzero(np.diff(skey, prepend=skey[0] - 1))
    starts = np.concatenate([bounds, [skey.shape[0]]])
    for lo, hi in zip(starts[:-1], starts[1:]):
        m0 = int(order[lo])
        px, py = int(xi[m0]), int(yi[m0])
        cover_i = [i for i in range(nx_i) if ox[i] <= px < ox[i] + sx_i]
        cover_j = [j for j in range(ny_i) if oy[j] <= py < oy[j] + sy_i]
        if not cover_i or not cover_j:
            continue
        cells = np.array(
            [(i * ny_i + j) * nd_i + k for i in cover_i for j in cover_j
             for k in range(nd_i)],
            dtype=np.int64,
        )
        sel = order[lo:hi]
        np.minimum.at(ranks, (bins[sel][:, None], cells[None, :]), off[sel][:, None])
    ranks[~np.isfinite(ranks)] = 0.0
    return ranks.astype(np.float32)


def feed_visit_order(
    ranks: np.ndarray,
    arrivals: np.ndarray,
    *,
    label: str = "feed_visit",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step fed-cell visit order for the event-order fixed point.

    Step ``b`` visits the cells with ``arrivals[b] > 0`` sorted by
    (``ranks[b]``, cell index): event order first, the declared index order
    on ties (the oracle visits same-time candidates in ascending index,
    oracle:666). Returns ``(visit, visit_len)``: int32 ``visit`` of shape
    ``(n_steps, K)`` (``K`` the busiest step's fed-cell count, rows padded)
    and int32 ``visit_len`` of shape ``(n_steps,)``. Both inputs must be
    finite 2-D arrays of one shape; arrivals must hold non-negative whole
    counts. Anything else is refused by name.
    """
    r = np.asarray(ranks)
    a = np.asarray(arrivals)
    if r.ndim != 2 or a.ndim != 2 or r.shape != a.shape:
        raise ValueError(
            f"{label} ranks and arrivals must be finite 2-D arrays of one "
            f"shape; got {r.shape} and {a.shape}"
        )
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{label} ranks must be finite")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{label} arrivals must be finite")
    if bool((a < 0).any()):
        raise ValueError(f"{label} arrivals must hold non-negative event counts")
    if not bool((a == np.floor(a)).all()):
        raise ValueError(f"{label} arrivals must hold whole event counts")
    n_steps = int(r.shape[0])
    fed = [np.flatnonzero(a[b] > 0) for b in range(n_steps)]
    width = max([int(v.shape[0]) for v in fed] + [0])
    visit = np.zeros((n_steps, width), dtype=np.int32)
    lens = np.zeros((n_steps,), dtype=np.int32)
    for b in range(n_steps):
        v = fed[b]
        if v.shape[0] == 0:
            continue
        seq = v[np.lexsort((v, r[b, v]))]
        visit[b, :seq.shape[0]] = seq.astype(np.int32)
        lens[b] = int(seq.shape[0])
    return visit, lens


def inhibition_visit_order(
    ranks: np.ndarray,
    blocks: dict[str, np.ndarray],
    *,
    label: str = "inhibition_visit",
) -> np.ndarray:
    """Per-step, per-position depth visit order for the event-order fixed point.

    ``ranks`` is ``(n_steps, N)`` finite first-event offsets (``0.0`` where a
    cell is fed no event in the step, as :func:`dvs_feed_ranks` leaves them;
    cells a second feed never touches, e.g. edge-driven complex cells, read
    ``0.0`` everywhere and keep index order). ``blocks`` is a
    :func:`local_inhibition_blocks` dict (``members`` ``(G, D)`` and
    ``member_mask`` scope the positions). Returns int32 ``(n_steps, G, D)``:
    step ``b`` visits each position's member depths in ``(rank, local
    index)`` order — event order first, ascending index on ties (the
    oracle visits same-time candidates in ascending index, oracle:666).
    Padded slots hold ``-1``. Anything misshapen, non-finite or
    N-disagreeing is refused by name.
    """
    r = np.asarray(ranks)
    try:
        members = np.asarray(blocks["members"])
        mask = np.asarray(blocks["member_mask"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{label} blocks must carry 'members' and 'member_mask'; got {exc!r}"
        ) from exc
    if members.ndim != 2 or mask.shape != members.shape:
        raise ValueError(
            f"{label} blocks members/mask must share one (G, D) shape; got "
            f"{members.shape} and {mask.shape}"
        )
    n_cells = int(np.asarray(blocks.get("cell_group", members.reshape(-1)[:0])).shape[0])
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError(f"{label} ranks must be a finite 2-D array; got shape {r.shape}")
    n = int(r.shape[1])
    if n_cells and n != n_cells:
        raise ValueError(
            f"{label} ranks cover {n} cells but the blocks span {n_cells}"
        )
    n_steps, ng, depth = int(r.shape[0]), int(members.shape[0]), int(members.shape[1])
    order = np.full((n_steps, ng, depth), -1, dtype=np.int32)
    r64 = np.asarray(r, dtype=np.float64)
    for g in range(ng):
        mem = members[g][mask[g]]
        if mem.shape[0] == 0:
            continue
        if bool((mem < 0).any()) or bool((mem >= n).any()):
            raise ValueError(
                f"{label} blocks member ids must lie in [0, {n}); "
                f"group {g} holds {mem!r}"
            )
        loc = np.flatnonzero(mask[g])
        # One C argsort per position (not one lexsort per step): the key
        # ``rank * depth + local`` sorts by (rank, local index) exactly —
        # whole-microsecond ranks times depth stay integral in float64.
        seq = loc[np.argsort(
            r64[:, mem] * float(depth) + loc[None, :].astype(np.float64),
            axis=1, kind="stable")]
        order[:, g, :seq.shape[1]] = seq.astype(np.int32)
    return order


def local_inhibition_blocks(
    groups: np.ndarray,
    li_pre: np.ndarray,
    li_post: np.ndarray,
    li_weight: np.ndarray,
    *,
    label: str = "local_inhibition",
) -> dict[str, np.ndarray]:
    """Dense per-position inhibition blocks for the event-order fixed point.

    ``groups`` (int, shape ``(N,)``) scopes the positions; the sparse
    ``(li_pre, li_post, li_weight)`` triple carries the inhibition magnitude
    each spike subtracts from its group-mates (non-negative mV: pass the
    negated edge weight, since executed inhibitory edges carry negative
    weights and the cascade subtracts). Group ids are compacted to ``0..G - 1``. Returns
    ``{"members", "member_mask", "weights", "cell_group", "cell_local"}``:
    int ``members`` ``(G, D)`` (padded with 0 where masked), bool
    ``member_mask``, float64 ``weights`` ``(G, D, D)`` (target-local row,
    source-local column), int ``cell_group`` ``(N,)`` and int
    ``cell_local`` ``(N,)``. Every edge must stay inside one group (local
    inhibition is within-position); anything misshapen, non-finite,
    out of range or cross-group is refused by name.
    """
    grp = np.asarray(groups)
    if grp.ndim != 1:
        raise ValueError(
            f"{label} groups must have shape (N,); got {grp.shape}"
        )
    n = int(grp.shape[0])
    pre = np.asarray(li_pre).ravel()
    post = np.asarray(li_post).ravel()
    wgt = np.asarray(li_weight, dtype=np.float64).ravel()
    if not (pre.shape == post.shape == wgt.shape):
        raise ValueError(
            f"{label} inhibition pre/post/weight disagree in length: "
            f"{pre.shape} {post.shape} {wgt.shape}"
        )
    if pre.shape[0] and (
        not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post))
        or bool((pre < 0).any()) or bool((post < 0).any())
        or bool((pre >= n).any()) or bool((post >= n).any())
    ):
        raise ValueError(
            f"{label} inhibition endpoints must be whole ids in [0, {n})"
        )
    if not bool(np.all(np.isfinite(wgt))):
        raise ValueError(f"{label} inhibition weights must be finite")
    if bool((wgt < 0).any()):
        raise ValueError(
            f"{label} inhibition weights must hold non-negative magnitudes "
            "(the cascade subtracts them; negate executed edge weights)"
        )
    pre_i = pre.astype(np.int64)
    post_i = post.astype(np.int64)
    if pre_i.shape[0] and bool((grp[pre_i] != grp[post_i]).any()):
        bad = int(np.flatnonzero(grp[pre_i] != grp[post_i])[0])
        raise ValueError(
            f"{label} inhibition edge {bad} crosses groups "
            f"({int(pre_i[bad])}->{int(post_i[bad])}); local inhibition "
            "stays inside one retinotopic position"
        )
    uniq = np.unique(grp)
    remap = {int(v): k for k, v in enumerate(uniq)}
    cell_group = np.array([remap[int(v)] for v in grp], dtype=np.int64)
    ng = int(uniq.shape[0])
    sizes = np.bincount(cell_group, minlength=ng)
    depth = int(sizes.max()) if ng else 0
    members = np.zeros((ng, depth), dtype=np.int64)
    member_mask = np.zeros((ng, depth), dtype=bool)
    cell_local = np.zeros((n,), dtype=np.int64)
    weights = np.zeros((ng, depth, depth), dtype=np.float64)
    slots = np.zeros((ng,), dtype=np.int64)
    for c in range(n):
        gg = int(cell_group[c])
        lc = int(slots[gg])
        slots[gg] = lc + 1
        members[gg, lc] = c
        member_mask[gg, lc] = True
        cell_local[c] = lc
    loc_pre = cell_local[pre_i]
    loc_post = cell_local[post_i]
    grp_e = cell_group[pre_i]
    for e in range(pre_i.shape[0]):
        weights[int(grp_e[e]), int(loc_post[e]), int(loc_pre[e])] += float(wgt[e])
    return {
        "members": members,
        "member_mask": member_mask,
        "weights": weights,
        "cell_group": cell_group.astype(np.int64),
        "cell_local": cell_local,
    }


def _scan_whole(value: Any, *, name: str, label: str) -> int:
    """A whole-number grid count (bools refused)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} {name} must be a whole number; got {value!r}")
    return int(value)


def _scan_origin(value: Any, *, name: str, label: str) -> int:
    """A whole-number grid origin (``2.0`` admits; NaN/fractions refused)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} {name} must be a whole number; got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number != math.floor(number):
        raise ValueError(f"{label} {name} must be a whole number; got {value!r}")
    return int(number)


def event_scan_tables(
    nx, ny, nd, cx, cy, cd,
    origins_x, origins_y, size, weights,
    c_cover, c_ox, c_oy, c_size, c_weights,
    vw, vh, inhib_range=(0, 0),
    lateral_on=False, lateral_weights=None,
    topdown_on=False, topdown_weights=None,
    activation=True,
    label="event_scan",
) -> dict[str, Any]:
    """Static tables for the event-scan intra-step fixed point.

    The tables transcribe the oracle's wiring (``pcl_event_ref.py:393-477``)
    into padded per-source target lists the scan kernel walks event by
    event: pixel cover positions (``cover_x``/``cover_y``, oracle:418-421),
    simple-to-complex forwarding (``c_cover``, oracle:423-428), the lateral
    window (``neuronInhibitionRange``, oracle:443-446) and the top-down
    receptive field (``td_mask``, oracle:431-439). Local-inhibition
    magnitudes are NOT carried here: they ride
    :func:`local_inhibition_blocks`, built from the same groups the kernel
    takes, so the declared magnitudes stay owned by the declaration.

    ``weights`` is ``(nd, 2, sx, sy)`` (shared simple matrix per depth,
    oracle:454); ``c_weights`` is ``(cd, csx, csy, nd)`` (shared complex
    matrix per depth, oracle:456); ``c_cover`` maps every simple ``(i, j)``
    to its covering complex ``(cx, cy)`` cells (recomputed from the geometry
    and required exact — a missing position or an off-grid cell is refused).
    ``lateral_weights`` is None (an all-zero map, as the S1 local-only run
    leaves it) or ``(NS, NS)`` with rows read as targets (``LAT[m, n]`` is
    the weight from source ``n`` onto target ``m``, oracle:702); anything
    else is refused. ``topdown_weights`` is None or ``(NC, NS)`` with rows
    read as sources (row ``c`` holds source ``c``'s weights onto its simple
    targets); anything else is refused. All-zero maps store no weight
    matrix (``lat_w``/``td_w`` None) but the calls still execute — a
    zero-weight inhibition still subtracts ``R + A`` and stamps the time
    (oracle:256-269). ``inhib_range`` is the lateral ``(Rx, Ry)`` window;
    nonzero while lateral is off changes nothing and is refused by name.
    ``activation`` is the oracle's ``m_activation`` (gates lateral and
    top-down, oracle:695, :734).

    Returns a dict of numpy arrays (``pix_pos``/``pix_n`` pixel cover
    positions, ``fwd_idx``/``fwd_w``/``fwd_n`` forward targets,
    ``lat_idx``/``lat_w``/``lat_n`` + ``lat_on``,
    ``td_idx``/``td_w``/``td_n`` + ``td_on``, ``activation``) plus the
    geometry and the shared ``w_s`` the kernel feeds from.
    """
    name = str(label)
    nx_i, ny_i, nd_i = _scan_whole(nx, name="nx", label=name), _scan_whole(ny, name="ny", label=name), _scan_whole(nd, name="nd", label=name)
    cx_i, cy_i, cd_i = _scan_whole(cx, name="cx", label=name), _scan_whole(cy, name="cy", label=name), _scan_whole(cd, name="cd", label=name)
    vw_i, vh_i = _scan_whole(vw, name="vw", label=name), _scan_whole(vh, name="vh", label=name)
    for tag, val in (("nx", nx_i), ("ny", ny_i), ("nd", nd_i), ("cx", cx_i),
                     ("cy", cy_i), ("cd", cd_i), ("vw", vw_i), ("vh", vh_i)):
        if val < 1:
            raise ValueError(f"{name} {tag} must be >= 1; got {val!r}")
    if not isinstance(origins_x, Sequence) or len(list(origins_x)) != nx_i:
        raise ValueError(f"{name} origins_x must hold nx={nx_i} origins; got {origins_x!r}")
    if not isinstance(origins_y, Sequence) or len(list(origins_y)) != ny_i:
        raise ValueError(f"{name} origins_y must hold ny={ny_i} origins; got {origins_y!r}")
    ox = [_scan_origin(v, name=f"origins_x[{k}]", label=name) for k, v in enumerate(list(origins_x))]
    oy = [_scan_origin(v, name=f"origins_y[{k}]", label=name) for k, v in enumerate(list(origins_y))]
    if not isinstance(size, Sequence) or len(list(size)) != 2:
        raise ValueError(f"{name} size must be a pair of whole pixels; got {size!r}")
    sx, sy = (_scan_whole(v, name=f"size[{k}]", label=name) for k, v in enumerate(list(size)))
    sx, sy = int(sx), int(sy)
    if sx < 1 or sy < 1:
        raise ValueError(f"{name} size must hold positive pixels; got {list(size)!r}")
    w_s = np.asarray(weights, dtype=np.float64)
    if w_s.shape != (nd_i, 2, sx, sy):
        raise ValueError(
            f"{name} weights must have shape ({nd_i}, 2, {sx}, {sy}); got {w_s.shape}")
    if not bool(np.all(np.isfinite(w_s))):
        raise ValueError(f"{name} weights must be finite")
    if not isinstance(c_ox, Sequence) or len(list(c_ox)) != cx_i:
        raise ValueError(f"{name} c_ox must hold cx={cx_i} origins; got {c_ox!r}")
    if not isinstance(c_oy, Sequence) or len(list(c_oy)) != cy_i:
        raise ValueError(f"{name} c_oy must hold cy={cy_i} origins; got {c_oy!r}")
    cox = [_scan_origin(v, name=f"c_ox[{k}]", label=name) for k, v in enumerate(list(c_ox))]
    coy = [_scan_origin(v, name=f"c_oy[{k}]", label=name) for k, v in enumerate(list(c_oy))]
    if not isinstance(c_size, Sequence) or len(list(c_size)) != 2:
        raise ValueError(f"{name} c_size must be a pair of whole pixels; got {c_size!r}")
    csx, csy = (_scan_whole(v, name=f"c_size[{k}]", label=name) for k, v in enumerate(list(c_size)))
    csx, csy = int(csx), int(csy)
    if csx < 1 or csy < 1:
        raise ValueError(f"{name} c_size must hold positive pixels; got {list(c_size)!r}")
    w_c = np.asarray(c_weights, dtype=np.float64)
    if w_c.shape != (cd_i, csx, csy, nd_i):
        raise ValueError(
            f"{name} c_weights must have shape ({cd_i}, {csx}, {csy}, {nd_i}); got {w_c.shape}")
    if not bool(np.all(np.isfinite(w_c))):
        raise ValueError(f"{name} c_weights must be finite")
    # Complex cover: recomputed from the geometry (oracle:423-428) and
    # required exact, so a missing position or an off-grid cell is refused.
    want_cover = {
        (i, j): [(ax, ay) for ax in range(cx_i) for ay in range(cy_i)
                  if cox[ax] <= i < cox[ax] + csx and coy[ay] <= j < coy[ay] + csy]
        for i in range(nx_i) for j in range(ny_i)
    }
    if not isinstance(c_cover, Mapping):
        raise ValueError(f"{name} c_cover must map every simple (i, j) to complex cells")
    got_cover: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key, val in c_cover.items():
        try:
            pair = (int(key[0]), int(key[1]))
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"{name} c_cover carries a bad position {key!r}") from exc
        if isinstance(val, (str, bytes)) or not isinstance(val, Sequence):
            raise ValueError(f"{name} c_cover[{pair}] must list complex (cx, cy) cells")
        cells = []
        for item in list(val):
            try:
                cell = (int(item[0]), int(item[1]))
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"{name} c_cover[{pair}] carries a bad complex cell {item!r}") from exc
            cells.append(cell)
        got_cover[pair] = cells
    if set(got_cover) != set(want_cover) or any(
            sorted(got_cover[k]) != sorted(v) for k, v in want_cover.items()):
        raise ValueError(f"{name} c_cover must name every simple position's covering complex cells")
    ns, nc = nx_i * ny_i * nd_i, cx_i * cy_i * cd_i
    # Pixel cover positions (oracle:418-421), ascending.
    cover_x = [[i for i in range(nx_i) if ox[i] <= x < ox[i] + sx] for x in range(vw_i)]
    cover_y = [[j for j in range(ny_i) if oy[j] <= y < oy[j] + sy] for y in range(vh_i)]
    pix_lists = [
        [i * ny_i + j for i in cover_x[x] for j in cover_y[y]]
        for y in range(vh_i) for x in range(vw_i)
    ]
    pmax = max(1, max([len(v) for v in pix_lists] + [0]))
    pix_pos = np.zeros((vw_i * vh_i, pmax), dtype=np.int64)
    pix_n = np.zeros((vw_i * vh_i,), dtype=np.int64)
    for q, lst in enumerate(pix_lists):
        pix_n[q] = len(lst)
        if lst:
            pix_pos[q, :len(lst)] = np.asarray(lst, dtype=np.int64)
    # Forward targets per simple cell (ascending complex index, oracle:708-720).
    fwd_idx_lists: list[list[int]] = [[] for _ in range(ns)]
    fwd_w_lists: list[list[float]] = [[] for _ in range(ns)]
    for i in range(nx_i):
        for j in range(ny_i):
            for k in range(nd_i):
                n = (i * ny_i + j) * nd_i + k
                for (ax, ay) in want_cover[(i, j)]:
                    ex, ey = i - cox[ax], j - coy[ay]
                    base = ns + (ax * cy_i + ay) * cd_i
                    for cz in range(cd_i):
                        fwd_idx_lists[n].append(base + cz)
                        fwd_w_lists[n].append(float(w_c[cz, ex, ey, k]))
    fmax = max(1, max([len(v) for v in fwd_idx_lists] + [0]))
    fwd_idx = np.zeros((ns, fmax), dtype=np.int64)
    fwd_w = np.zeros((ns, fmax), dtype=np.float64)
    fwd_n = np.zeros((ns,), dtype=np.int64)
    for n in range(ns):
        fwd_n[n] = len(fwd_idx_lists[n])
        if fwd_idx_lists[n]:
            fwd_idx[n, :len(fwd_idx_lists[n])] = np.asarray(fwd_idx_lists[n], dtype=np.int64)
            fwd_w[n, :len(fwd_w_lists[n])] = np.asarray(fwd_w_lists[n], dtype=np.float64)
    # Lateral window (oracle:443-446): neighbor positions, own excluded.
    if not isinstance(inhib_range, Sequence) or len(list(inhib_range)) != 2:
        raise ValueError(f"{name} inhib_range must be a pair of whole pixels; got {inhib_range!r}")
    rx, ry = (_scan_whole(v, name=f"inhib_range[{k}]", label=name)
              for k, v in enumerate(list(inhib_range)))
    rx, ry = int(rx), int(ry)
    if rx < 0 or ry < 0:
        raise ValueError(f"{name} inhib_range must hold non-negative pixels; got {list(inhib_range)!r}")
    if not isinstance(lateral_on, bool):
        raise ValueError(f"{name} lateral_on must be a bool; got {lateral_on!r}")
    if not isinstance(topdown_on, bool):
        raise ValueError(f"{name} topdown_on must be a bool; got {topdown_on!r}")
    if not isinstance(activation, bool):
        raise ValueError(f"{name} activation must be a bool; got {activation!r}")
    if not lateral_on and (rx or ry):
        raise ValueError(
            f"{name} inhib_range {(rx, ry)} is set but lateral is off; "
            "the window only windows lateral neighborhoods")
    if lateral_weights is not None and not lateral_on:
        raise ValueError(
            f"{name} lateral_weights is set but lateral is off; without the "
            "lateral calls no weight executes")
    if topdown_weights is not None and not topdown_on:
        raise ValueError(
            f"{name} topdown_weights is set but topdown is off; without the "
            "top-down calls no weight executes")
    lat_mat = None
    if lateral_weights is not None:
        lat_mat = np.asarray(lateral_weights, dtype=np.float64)
        if lat_mat.shape != (ns, ns):
            raise ValueError(
                f"{name} lateral_weights must have shape ({ns}, {ns}); got {lat_mat.shape}")
        if not bool(np.all(np.isfinite(lat_mat))):
            raise ValueError(f"{name} lateral_weights must be finite")
        if bool((lat_mat < 0).any()):
            raise ValueError(f"{name} lateral_weights must hold non-negative magnitudes")
    lat_idx_lists: list[list[int]] = [[] for _ in range(ns)]
    lat_w_lists: list[list[float]] = [[] for _ in range(ns)]
    if lateral_on:
        si = np.repeat(np.arange(nx_i), ny_i)
        sj = np.tile(np.arange(ny_i), nx_i)
        for n in range(ns):
            q = n // nd_i
            keep = (np.abs(si - si[q]) <= rx) & (np.abs(sj - sj[q]) <= ry)
            keep[q] = False
            for qq in np.flatnonzero(keep):
                base = int(qq) * nd_i
                for d in range(nd_i):
                    m = base + d
                    lat_idx_lists[n].append(m)
                    lat_w_lists[n].append(float(lat_mat[m, n]) if lat_mat is not None else 0.0)
    lmax = max(1, max([len(v) for v in lat_idx_lists] + [0]))
    lat_idx = np.zeros((ns, lmax), dtype=np.int64)
    lat_n = np.zeros((ns,), dtype=np.int64)
    lat_w: np.ndarray | None = None
    if any(len(v) for v in lat_w_lists) and any(
            v != 0.0 for lst in lat_w_lists for v in lst):
        lat_w = np.zeros((ns, lmax), dtype=np.float64)
    for n in range(ns):
        lat_n[n] = len(lat_idx_lists[n])
        if lat_idx_lists[n]:
            lat_idx[n, :len(lat_idx_lists[n])] = np.asarray(lat_idx_lists[n], dtype=np.int64)
            if lat_w is not None:
                lat_w[n, :len(lat_w_lists[n])] = np.asarray(lat_w_lists[n], dtype=np.float64)
    # Top-down targets per complex source (oracle:431-439, :737-744), ascending.
    td_mat = None
    if topdown_weights is not None:
        td_mat = np.asarray(topdown_weights, dtype=np.float64)
        if td_mat.shape != (nc, ns):
            raise ValueError(
                f"{name} topdown_weights must have shape ({nc}, {ns}) "
                f"(rows are complex sources); got {td_mat.shape}")
        if not bool(np.all(np.isfinite(td_mat))):
            raise ValueError(f"{name} topdown_weights must be finite")
        if bool((td_mat < 0).any()):
            raise ValueError(f"{name} topdown_weights must hold non-negative magnitudes")
    td_idx_lists: list[list[int]] = [[] for _ in range(nc)]
    td_w_lists: list[list[float]] = [[] for _ in range(nc)]
    if topdown_on:
        for c in range(nc):
            q = c // cd_i
            axi, ayi = q // cy_i, q % cy_i
            for i in range(nx_i):
                if not cox[axi] <= i < cox[axi] + csx:
                    continue
                for j in range(ny_i):
                    if not coy[ayi] <= j < coy[ayi] + csy:
                        continue
                    base = (i * ny_i + j) * nd_i
                    for k in range(nd_i):
                        s = base + k
                        td_idx_lists[c].append(s)
                        td_w_lists[c].append(float(td_mat[c, s]) if td_mat is not None else 0.0)
    tmax = max(1, max([len(v) for v in td_idx_lists] + [0]))
    td_idx = np.zeros((nc, tmax), dtype=np.int64)
    td_n = np.zeros((nc,), dtype=np.int64)
    td_w: np.ndarray | None = None
    if any(len(v) for v in td_w_lists) and any(
            v != 0.0 for lst in td_w_lists for v in lst):
        td_w = np.zeros((nc, tmax), dtype=np.float64)
    for c in range(nc):
        td_n[c] = len(td_idx_lists[c])
        if td_idx_lists[c]:
            td_idx[c, :len(td_idx_lists[c])] = np.asarray(td_idx_lists[c], dtype=np.int64)
            if td_w is not None:
                td_w[c, :len(td_w_lists[c])] = np.asarray(td_w_lists[c], dtype=np.float64)
    return {
        "nx": nx_i, "ny": ny_i, "nd": nd_i, "cx": cx_i, "cy": cy_i, "cd": cd_i,
        "ns": ns, "nc": nc, "vw": vw_i, "vh": vh_i,
        "s_ox": np.asarray(ox, dtype=np.int64), "s_oy": np.asarray(oy, dtype=np.int64),
        "w_s": w_s,
        "pix_pos": pix_pos, "pix_n": pix_n,
        "fwd_idx": fwd_idx, "fwd_w": fwd_w, "fwd_n": fwd_n,
        "lat_idx": lat_idx, "lat_w": lat_w, "lat_n": lat_n,
        "lat_on": bool(lateral_on),
        "td_idx": td_idx, "td_w": td_w, "td_n": td_n,
        "td_on": bool(topdown_on),
        "activation": bool(activation),
    }


def event_scan_events(
    event_t_us, event_x, event_y, event_p,
    *,
    dt_ms, n_steps, vw, vh, t0_us=0,
    label="event_scan_events",
) -> np.ndarray:
    """The event-scan visit list: the raw stream in timestamp order.

    Returns int64 ``(M, 4)`` rows ``[t_us, x, y, p]`` stably sorted by time
    (file order within equal timestamps — the oracle visits same-time
    candidates in ascending index, oracle:666, and the scan kernel visits
    each event's cells ascending). ``dt_ms`` must resolve to a whole number
    of microseconds; an event outside the ``[t0_us, t0_us + n_steps *
    dt_us)`` grid, off the ``vw``-by-``vh`` pixel plane, or with a polarity
    outside ``{0, 1}`` is refused by name.
    """
    name = str(label)
    vecs = []
    for tag, arr in (("event_t_us", event_t_us), ("event_x", event_x),
                     ("event_y", event_y), ("event_p", event_p)):
        a = np.asarray(arr)
        if a.ndim != 1:
            raise ValueError(f"{name} {tag} must be one-dimensional; got shape {a.shape}")
        if a.dtype.kind not in "iu":
            raise ValueError(f"{name} {tag} must hold integers; got {a.dtype}")
        vecs.append(a.astype(np.int64))
    t, x, y, p = vecs
    if not (x.shape == y.shape == p.shape == t.shape):
        raise ValueError(
            f"{name} event arrays must share one length; got {t.shape}, {x.shape}, "
            f"{y.shape}, {p.shape}")
    if not isinstance(dt_ms, numbers.Real) or isinstance(dt_ms, bool):
        raise ValueError(f"{name} dt_ms must be a real number; got {dt_ms!r}")
    dt_f = float(dt_ms)
    if not math.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError(f"{name} dt_ms must be > 0; got {dt_ms!r}")
    dt_us = dt_f * 1000.0
    if dt_us != math.floor(dt_us) or dt_us < 1:
        raise ValueError(f"{name} dt_ms must resolve to whole microseconds; got {dt_ms!r}")
    dt_i = int(dt_us)
    n_i = _scan_whole(n_steps, name="n_steps", label=name)
    vw_i = _scan_whole(vw, name="vw", label=name)
    vh_i = _scan_whole(vh, name="vh", label=name)
    if n_i < 1 or vw_i < 1 or vh_i < 1:
        raise ValueError(f"{name} n_steps/vw/vh must be >= 1")
    t0_i = _scan_whole(t0_us, name="t0_us", label=name)
    if t0_i < 0:
        raise ValueError(f"{name} t0_us must be >= 0; got {t0_us!r}")
    if t.shape[0]:
        if bool((x < 0).any()) or bool((x >= vw_i).any()):
            raise ValueError(f"{name} event x must lie on the {vw_i}-pixel plane")
        if bool((y < 0).any()) or bool((y >= vh_i).any()):
            raise ValueError(f"{name} event y must lie on the {vh_i}-pixel plane")
        if bool((p < 0).any()) or bool((p > 1).any()):
            raise ValueError(f"{name} event polarity must lie in {{0, 1}}")
        bins = (t - t0_i) // dt_i
        if bool((bins < 0).any()) or bool((bins >= n_i).any()):
            raise ValueError(
                f"{name} event times must fall on the [{t0_i}, {t0_i + n_i * dt_i}) "
                "microsecond grid")
    order = np.argsort(t, kind="stable")
    return np.stack([t[order], x[order], y[order], p[order]], axis=1)


def scan_inhibition_triplets(
    nx, ny, nd, cx, cy, cd, ns, wi_s, wi_c,
    label="scan_inhibition",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local-inhibition triplets for the event-scan replay.

    Transcribes the per-position depth pairs the oracle's
    ``SpikingNetwork`` cascade walks: every ordered depth pair inside one
    retinotopic position carries the local-inhibition magnitude, and
    nothing crosses positions. Simple positions run row-major
    (``(i, j)`` depth block ``((i * ny + j) * nd + k)``); complex cells
    follow at the ``ns`` offset in ``(cx, cy)`` row-major blocks. Pair
    ``(pre, post)`` carries ``wi_s[post_depth, pre_depth]`` (simple) or
    ``wi_c[post_depth, pre_depth]`` (complex): the magnitude a ``pre``
    spike subtracts from ``post``. ``wi_s`` is ``(nd, nd)`` and ``wi_c``
    is ``(cd, cd)`` (shared per-depth matrices, oracle:454); ``ns`` must
    equal ``nx * ny * nd``. Returns int64 ``(pre, post)`` and float64
    ``mag``. Anything misshapen, non-finite or count-disagreeing is
    refused by name (the magnitudes ride :func:`local_inhibition_blocks`
    from here, so the declared magnitudes stay owned by the declaration).
    """
    name = str(label)
    nx_i = _scan_whole(nx, name="nx", label=name)
    ny_i = _scan_whole(ny, name="ny", label=name)
    nd_i = _scan_whole(nd, name="nd", label=name)
    cx_i = _scan_whole(cx, name="cx", label=name)
    cy_i = _scan_whole(cy, name="cy", label=name)
    cd_i = _scan_whole(cd, name="cd", label=name)
    ns_i = _scan_whole(ns, name="ns", label=name)
    for tag, val in (("nx", nx_i), ("ny", ny_i), ("nd", nd_i), ("cx", cx_i),
                     ("cy", cy_i), ("cd", cd_i), ("ns", ns_i)):
        if val < 1:
            raise ValueError(f"{name} {tag} must be >= 1; got {val!r}")
    if ns_i != nx_i * ny_i * nd_i:
        raise ValueError(
            f"{name} ns={ns_i} disagrees with the simple grid "
            f"nx * ny * nd={nx_i * ny_i * nd_i}; the scan indexes simples "
            "by grid position"
        )
    mat_s = np.asarray(wi_s, dtype=np.float64)
    if mat_s.shape != (nd_i, nd_i):
        raise ValueError(
            f"{name} wi_s must have shape ({nd_i}, {nd_i}); got {mat_s.shape}")
    if not bool(np.all(np.isfinite(mat_s))):
        raise ValueError(f"{name} wi_s must be finite")
    mat_c = np.asarray(wi_c, dtype=np.float64)
    if mat_c.shape != (cd_i, cd_i):
        raise ValueError(
            f"{name} wi_c must have shape ({cd_i}, {cd_i}); got {mat_c.shape}")
    if not bool(np.all(np.isfinite(mat_c))):
        raise ValueError(f"{name} wi_c must be finite")
    pre: list[int] = []
    post: list[int] = []
    mag: list[float] = []
    for i in range(nx_i):
        for j in range(ny_i):
            base = (i * ny_i + j) * nd_i
            for k in range(nd_i):
                for d in range(nd_i):
                    if d == k:
                        continue
                    pre.append(base + k)
                    post.append(base + d)
                    mag.append(float(mat_s[d, k]))
    for ax in range(cx_i):
        for ay in range(cy_i):
            base = ns_i + (ax * cy_i + ay) * cd_i
            for k in range(cd_i):
                for d in range(cd_i):
                    if d == k:
                        continue
                    pre.append(base + k)
                    post.append(base + d)
                    mag.append(float(mat_c[d, k]))
    return (
        np.asarray(pre, dtype=np.int64),
        np.asarray(post, dtype=np.int64),
        np.asarray(mag, dtype=np.float64),
    )


class EventScanDrive:
    """The event-scan replay as a declared x-source paradigm object.

    Carries the timestamped stream (``events``: int ``(M, 4)`` rows
    ``[t_us, x, y, p]`` as :func:`event_scan_events` builds them), the
    wiring transcription (``tables`` as :func:`event_scan_tables` builds
    it), the inhibition magnitudes (``blocks`` as
    :func:`local_inhibition_blocks` builds them, from
    :func:`scan_inhibition_triplets` triplets) and the retinotopic
    position of every cell (``groups``: int ``(N,)``) for a run of
    ``n_steps`` steps of ``dt_ms``. ``simulate(paradigm=drive)`` on a
    tensor declaring ``local_inhibition`` ``resolution="event_scan"`
    replays exactly this stream through ``simulate_lif_edge``; every
    other simulate input that would change nothing (schedules, poisson,
    HDP, compiled edges) is refused by name there. Anything misshapen
    here is refused naming ``EventScanDrive``.
    """

    def __init__(
        self,
        events: np.ndarray,
        tables: Mapping[str, Any],
        blocks: Mapping[str, np.ndarray],
        groups: np.ndarray,
        *,
        dt_ms: float,
        n_steps: int,
    ) -> None:
        name = type(self).__name__
        ev = np.asarray(events)
        if ev.ndim != 2 or ev.shape[1] != 4:
            raise ValueError(
                f"{name} events must have shape (M, 4) with [t_us, x, y, p] "
                f"rows; got shape {ev.shape}"
            )
        if ev.dtype.kind not in "iu":
            raise ValueError(
                f"{name} events must hold integers; got {ev.dtype}")
        if not isinstance(tables, Mapping):
            raise ValueError(
                f"{name} tables must be the event_scan_tables mapping; "
                f"got {tables!r}")
        try:
            ns = int(tables["ns"])
            nc = int(tables["nc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} tables must carry whole 'ns'/'nc' geometry; got "
                f"{exc!r}") from exc
        grp = np.asarray(groups)
        if grp.ndim != 1 or grp.dtype.kind not in "iu":
            raise ValueError(
                f"{name} groups must hold one integer position per cell; "
                f"got shape {grp.shape} dtype {grp.dtype}")
        if int(grp.shape[0]) != ns + nc:
            raise ValueError(
                f"{name} groups cover {int(grp.shape[0])} cells but the "
                f"tables span ns + nc = {ns + nc}")
        if not isinstance(blocks, Mapping):
            raise ValueError(
                f"{name} blocks must be the local_inhibition_blocks "
                f"mapping; got {blocks!r}")
        for key in ("members", "member_mask", "weights", "cell_group",
                    "cell_local"):
            if key not in blocks:
                raise ValueError(
                    f"{name} blocks must carry {key!r}; got "
                    f"{sorted(blocks, key=repr)}")
        if not isinstance(dt_ms, numbers.Real) or isinstance(dt_ms, bool):
            raise ValueError(f"{name} dt_ms must be a real number; got {dt_ms!r}")
        if not math.isfinite(float(dt_ms)) or float(dt_ms) <= 0.0:
            raise ValueError(f"{name} dt_ms must be > 0; got {dt_ms!r}")
        n_i = _scan_whole(n_steps, name="n_steps", label=name)
        if n_i < 1:
            raise ValueError(f"{name} n_steps must be >= 1")
        self.events = ev.astype(np.int64)
        self.tables = tables
        self.blocks = blocks
        self.groups = grp.astype(np.int64)
        self.dt_ms = float(dt_ms)
        self.n_steps = n_i

    def to_dict(self) -> dict[str, Any]:
        """Small receipt (shapes and grid only; the arrays ride the run)."""
        tables = self.tables
        return {
            "kind": type(self).__name__,
            "n_events": int(self.events.shape[0]),
            "n_steps": int(self.n_steps),
            "dt_ms": float(self.dt_ms),
            "ns": int(tables["ns"]),
            "nc": int(tables["nc"]),
        }


def build_s1_scan_tensor(ns: int, nc: int, *, name: str = "s1-scan") -> Any:
    """A scan-ready S1 tensor: simples-first population, scan declaration.

    One area with a simple layer (``ns`` ``"simple"`` cells) followed by a
    complex layer (``nc`` ``"complex"`` cells), so the constructed
    population order is ``["simple"] * ns + ["complex"] * nc`` — the
    order the event-scan kernel indexes. Positions stay the default
    ``uniform_random`` geometry (the scan replays the declared
    :class:`EventScanDrive` groups, not construct geometry). A single
    declaration connection carries ``local_inhibition``
    ``{"resolution": "event_scan"}`` (rounds-free: the replay resolves
    every event, no cap); construct strips it from the bridge (it
    compiles no edges — the scan has no edge expression) and records
    the spec, so ``simulate(paradigm=EventScanDrive(...))`` replays the
    stream exactly. ``ns``/``nc`` must be positive whole numbers.
    """
    from .neuronal_tensor import (
        Area,
        InterConnection,
        Layer,
        NeuronalTensor,
        NeuronType,
        StaticParams,
    )

    label = "build_s1_scan_tensor"
    ns_i = _scan_whole(ns, name="ns", label=label)
    nc_i = _scan_whole(nc, name="nc", label=label)
    if ns_i < 1 or nc_i < 1:
        raise ValueError(
            f"{label} ns/nc must be >= 1; got {ns!r}/{nc!r}")
    simple_layer = f"{name}-simple"
    complex_layer = f"{name}-complex"
    return NeuronalTensor(
        name=str(name),
        areas=[Area(
            name="S1",
            layers=[
                Layer(name=simple_layer, n_neurons=ns_i,
                      neuron_types=[NeuronType.make("simple")]),
                Layer(name=complex_layer, n_neurons=nc_i,
                      neuron_types=[NeuronType.make("complex")]),
            ],
            inter_connections=[InterConnection(
                source_layer=simple_layer,
                source_neuron_type="simple",
                target_layer=complex_layer,
                target_neuron_type="complex",
                mechanism="AMPA",
                static=StaticParams(dT_ms=5.0),
                local_inhibition={"resolution": "event_scan"},
            )],
        )],
    )
