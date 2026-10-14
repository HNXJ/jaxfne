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
    label: str = "local_inhibition",
) -> dict[str, Any]:
    """Same-step local inhibition at one retinotopic position (B3).

    ``v``/``adapt``/``t_last``/``t_spike`` are per-depth state (``t_spike`` 0
    is the oracle's never-spiked sentinel, oracle:210-212); ``inputs`` is
    ``(M, D)`` sequential input vectors at one timestamp in visit order
    (``M == 1`` for the simple feed; one row per incoming simple spike for
    the complex layer); ``inhib[d, k]`` is the weight from spiking depth
    ``k`` onto depth ``d`` with a zero diagonal. Per input, depths are
    visited ascending: decay to ``t`` (``t`` rides ``t_last``... see below),
    add the input minus refractory (and adaptation) terms, clamp at
    ``neg_limit``; a depth strictly above ``thr`` spikes (reset to
    ``v_reset``, adaptation increment when ``adaptive``) and immediately
    inhibits the other depths (decay, subtract, clamp). At most ``rounds``
    winners resolve; further crossings keep their excitation but neither
    spike nor inhibit, and are counted as ``missed``.

    Times share one unit (the oracle uses integer microseconds,
    oracle:64-65); ``t_now`` is the single timestamp of this call.
    """
    n_rounds = validate_rounds(rounds, label=label)
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
        for d in range(dep):
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
