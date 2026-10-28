"""v0.3.30 connectivity compiler.

Compiles declarative connection rules (v0.3.28 ``metadata["circuit"]``) into
sparse, finite edge arrays compatible with :class:`jaxfne.EdgeList`.

Design (locked):
- **Sparse-first probabilistic sampling**: a ``probability`` rule draws
  ``round(p * n_pre * n_post)`` candidate edges directly with an explicit PRNG
  key; it never materializes a dense ``n_pre x n_post`` mask. Dense arrays are
  only used in the explicit ``matrix`` weight mode.
- **Free-function core** (:func:`compile_connection_rules`) plus a thin
  ``Model.compile_connections()`` wrapper (in ``core.py``).
- Host-side only: this runs outside JIT (Python spec dicts, deterministic
  sampling). The output edge arrays are JAX arrays ready for the simulation
  kernels.

Truth gates: this is a computational scaffold. Plasticity/dynamic-weight
metadata is carried as ``declared_not_simulated``; no biological learning,
calibrated amplitude, or solved-field claim is made.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import warnings
from functools import partial
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .pcl_topology import footprint_pairs, validate_footprint, validate_local_inhibition, validate_rounds, validate_weight_sharing

__all__ = [
    "compile_connection_rules",
    "ConnectionCompileResult",
    "compile_connection_rules_jax",
    "delay_steps_from_ms",
]


def delay_steps_from_ms(delay_ms: float, dt_ms: float) -> int:
    """Realize a configured delay (ms) as integer simulation steps (0.5.2 decision 0b).

    ``delay_steps = round(delay_ms / dt_ms)`` (round half up) at the timestep
    the model is constructed for. A positive delay that rounds to 0 steps is
    refused rather than silently dropped; negative or non-finite delays and
    non-positive timesteps are refused alongside. Deliberately distinct from
    :func:`jaxfne.emitters.edge_delay_steps_from_ms` (Protocol D), which
    rejects non-grid-aligned values instead of rounding: the TFNE/compiler
    chain follows the authorized 0b rule, the hand-set EdgeList surface keeps
    its grid-alignment contract.
    """
    try:
        ms = float(delay_ms)
    except (TypeError, ValueError):
        raise ValueError(f"delay_ms must be a number in ms; got {delay_ms!r}")
    try:
        dt = float(dt_ms)
    except (TypeError, ValueError):
        raise ValueError(f"dt_ms must be a positive number; got {dt_ms!r}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}")
    if not math.isfinite(ms) or ms < 0.0:
        raise ValueError(f"delay_ms must be finite and >= 0; got {delay_ms!r}")
    steps = int(math.floor(ms / dt + 0.5))
    if ms > 0.0 and steps <= 0:
        raise ValueError(
            f"delay_ms={ms} at dt_ms={dt} rounds to 0 steps; a positive delay "
            "that fits inside one timestep is refused rather than dropped"
        )
    return steps


def _validate_rule_delay_ms(rule_name: str, delay: Any) -> "float | None":
    """Validate one connection rule's configured ``delay_ms`` (or pass None through)."""
    if delay is None:
        return None
    try:
        ms = float(delay)
    except (TypeError, ValueError):
        raise ValueError(f"connection {rule_name!r} delay_ms must be a number in ms; got {delay!r}")
    if not math.isfinite(ms) or ms < 0.0:
        raise ValueError(
            f"connection {rule_name!r} delay_ms must be finite and >= 0; got {delay!r}"
        )
    return ms


#: ``jax.random.fold_in`` tags for the per-rule spread streams. Pair selection
#: consumes ``rkey`` and scalar/random weight modes consume ``fold_in(rkey, 1)``,
#: so weight spread, delay spread and the per-edge weight range fold 2, 3 and 4:
#: declaring one perturbs no pre-existing stream and omitting one leaves every
#: stream byte-identical.
_WEIGHT_SPREAD_FOLD = 2
_DELAY_SPREAD_FOLD = 3
_WEIGHT_RANGE_FOLD = 4


def _validate_spread_std(rule_name: str, value: Any, key: str) -> "float | None":
    """Validate one connection rule's per-edge spread (``weight_std``/``delay_std_ms``).

    None = undeclared = one constant per rule (current behaviour); a declared
    value must be a finite real number >= 0 (bools and strings refused, never
    coerced). Negative or non-finite values are refused, never clamped.
    """
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(
            f"connection {rule_name!r} {key} must be a finite number >= 0; "
            f"got {value!r}"
        )
    try:
        std = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"connection {rule_name!r} {key} must be a finite number >= 0; "
            f"got {value!r}"
        )
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(
            f"connection {rule_name!r} {key} must be finite and >= 0; got {value!r}"
        )
    return std


def _validate_weight_range(
    rule_name: str, value: Any, *, weight: Any, weight_std: Any
) -> "tuple[float, float] | None":
    """Validate one connection rule's per-edge weight range (``weight_range``).

    None = undeclared = the rule's existing weights. A declared value must be a
    2-sequence ``(lo, hi)`` of finite reals with ``0 <= lo < hi``; each edge's
    magnitude is then drawn U[lo, hi). The range replaces a scalar ``weight``
    and excludes ``weight_std``, so either conflict is refused, never resolved.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"connection {rule_name!r} weight_range must be a 2-sequence (lo, hi) "
            f"of finite reals; got {value!r}"
        )
    bounds: list[float] = []
    for end in value:
        if (
            isinstance(end, bool)
            or not isinstance(end, numbers.Real)
            or not math.isfinite(float(end))
        ):
            raise ValueError(
                f"connection {rule_name!r} weight_range must be a 2-sequence (lo, hi) "
                f"of finite reals; got {value!r}"
            )
        bounds.append(float(end))
    lo, hi = bounds
    if weight is not None:
        raise ValueError(
            f"connection {rule_name!r} declares weight_range={value!r} with "
            f"weight={weight!r}; the range draws each edge's magnitude -- declare "
            "weight_range or a scalar weight, not both"
        )
    if weight_std is not None:
        raise ValueError(
            f"connection {rule_name!r} declares weight_range={value!r} with "
            f"weight_std={weight_std!r}; the range and the Gaussian spread both set "
            "each edge's weight -- declare one"
        )
    if lo < 0.0:
        raise ValueError(f"connection {rule_name!r} weight_range lo must be >= 0; got {value!r}")
    if lo >= hi:
        raise ValueError(f"connection {rule_name!r} weight_range must satisfy lo < hi; got {value!r}")
    return (lo, hi)


def _validate_in_degree(rule_name: str, in_degree: Any) -> "int | None":
    """Validate one connection rule's exact fixed ``in_degree`` (or pass None through).

    None = undeclared = the density path; a declared value must be a
    positive integer (bools, non-integral numbers and values < 1 are
    refused, never coerced). When set, the rule compiles through the
    exact fixed-indegree sampler instead of the ``probability`` density
    path, so the two are mutually exclusive.
    """
    if in_degree is None:
        return None
    if isinstance(in_degree, bool) or not isinstance(in_degree, numbers.Integral):
        raise ValueError(
            f"connection {rule_name!r} in_degree must be a positive integer; "
            f"got {in_degree!r}"
        )
    degree = int(in_degree)
    if degree < 1:
        raise ValueError(
            f"connection {rule_name!r} in_degree must be a positive integer; "
            f"got {in_degree!r}"
        )
    return degree


def _validate_multinomial_in_degree(label: str, in_degree: Any) -> "float | None":
    """Validate a ``sampling="multinomial"`` mean in-degree ``k`` (or pass None through).

    The declared ``k`` is the mean in-degree: any finite real number > 0,
    non-integer allowed, returned as a float. bools, non-numbers, non-finite
    values and values <= 0 are refused, never coerced. ``label`` names the
    declaring surface in the message.
    """
    if in_degree is None:
        return None
    if isinstance(in_degree, bool) or not isinstance(in_degree, numbers.Real):
        raise ValueError(
            f"{label} in_degree with sampling='multinomial' must be a finite "
            f"positive number (the mean in-degree); got {in_degree!r}"
        )
    value = float(in_degree)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{label} in_degree with sampling='multinomial' must be a finite "
            f"positive number (the mean in-degree); got {in_degree!r}"
        )
    return value


@dataclass(frozen=True)
class ConnectionCompileResult:
    """Compiled sparse connectivity.

    Edge arrays are 1-D and index-aligned. ``edge_mechanism`` is the receptor
    index into ``mechanism_table``. All numeric arrays are finite.
    """

    edge_pre: jax.Array
    edge_post: jax.Array
    edge_weight: jax.Array
    edge_mechanism: jax.Array
    edge_rule_id: jax.Array
    connection_table: list[dict[str, Any]]
    mechanism_table: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Configured per-edge delay in ms (0.0 where the rule declared none).
    # Realized steps are derived at construction, where dt is known, via
    # :func:`delay_steps_from_ms`; never here.
    edge_delay_ms: jax.Array | None = None

    @property
    def n_edges(self) -> int:
        """Documented public function `n_edges`."""
        return int(self.edge_pre.shape[0])

    def to_edge_list(
        self, *, default_tau_ms: float = 5.0, dtype: str = "float32", dt_ms: "float | None" = None
    ):
        """Build a :class:`jaxfne.EdgeList` from the compiled edges.

        ``tau_ms`` per edge comes from the mechanism's declared ``tau_ms`` (or
        ``default_tau_ms``); ``receptor_index`` is ``edge_mechanism``.
        ``dt_ms`` realizes configured per-rule delays (0.5.2 decision 0b):
        when given, each edge's ``delay_steps`` is
        :func:`delay_steps_from_ms` of its configured ms at that timestep
        (a positive delay rounding to 0 steps raises); when omitted, delays
        stay zero; exactly the pre-0.5.2 EdgeList.
        """
        from .emitters import EdgeList

        if dtype == "float64" and not bool(jax.config.read("jax_enable_x64")):
            raise ValueError(
                "dtype='float64' requested but JAX x64 is not enabled -- call "
                'jax.config.update("jax_enable_x64", True) before creating any array '
                "in this process; refusing to silently build float32"
            )
        jdtype = (
            jnp.float64
            if (dtype == "float64" and bool(jax.config.read("jax_enable_x64")))
            else jnp.float32
        )
        tau_by_mech = [
            float(m.get("tau_ms", default_tau_ms))
            if m.get("tau_ms") is not None
            else float(default_tau_ms)
            for m in self.mechanism_table
        ]
        if self.n_edges and tau_by_mech:
            tau_table = jnp.asarray(tau_by_mech, dtype=jdtype)
            mech_idx = jnp.asarray(self.edge_mechanism, dtype=jnp.int32)
            tau = tau_table[mech_idx]
        else:
            tau = jnp.zeros((self.n_edges,), dtype=jdtype)
        if dt_ms is None:
            delay_steps = None
        else:
            import numpy as _np

            ms = (
                _np.asarray(self.edge_delay_ms, dtype=_np.float64)
                if self.edge_delay_ms is not None
                else _np.zeros((self.n_edges,), dtype=_np.float64)
            )
            spread_rules = {
                i
                for i, row in enumerate(self.connection_table)
                if isinstance(row, Mapping) and row.get("delay_std_ms") is not None
            }
            if not spread_rules:
                # Realize each distinct delay once; every value is still checked.
                uniq, inverse = _np.unique(ms, return_inverse=True)
                steps = _np.asarray(
                    [delay_steps_from_ms(float(v), float(dt_ms)) for v in uniq], dtype=_np.int32
                ).reshape(-1)
                delay_steps = jnp.asarray(
                    steps[inverse.reshape(-1)] if ms.size else _np.zeros((0,), _np.int32),
                    dtype=jnp.int32,
                )
            else:
                # Spread rules drew per-edge delays (folded >= 0 at compile):
                # each draw snaps half-up like the constant path, except a
                # positive draw inside one timestep floors to 1 step instead
                # of raising (the constant path keeps the refusal). Exact-zero
                # draws stay 0 steps. Non-spread edges keep the exact path.
                import math as _math

                rule_of = _np.asarray(self.edge_rule_id,
                                      dtype=_np.int64).reshape(-1)
                steps = _np.zeros(ms.shape, dtype=_np.int32)
                plain = _np.array([int(r) not in spread_rules for r in rule_of])
                if bool(_np.any(plain)):
                    uniq, inverse = _np.unique(ms[plain], return_inverse=True)
                    psteps = _np.asarray(
                        [delay_steps_from_ms(float(v), float(dt_ms)) for v in uniq],
                        dtype=_np.int32,
                    ).reshape(-1)
                    steps[plain] = psteps[inverse.reshape(-1)]
                dt = float(dt_ms)
                for idx in _np.where(~plain)[0]:
                    v = float(ms[idx])
                    if v == 0.0:
                        steps[idx] = 0
                    else:
                        steps[idx] = max(1, int(_math.floor(v / dt + 0.5)))
                delay_steps = jnp.asarray(steps, dtype=jnp.int32)
        return EdgeList(
            pre=jnp.asarray(self.edge_pre, dtype=jnp.int32),
            post=jnp.asarray(self.edge_post, dtype=jnp.int32),
            weight=jnp.asarray(self.edge_weight, dtype=jdtype),
            receptor_index=jnp.asarray(self.edge_mechanism, dtype=jnp.int32),
            tau_ms=tau,
            source_calibration_status="uncalibrated_proxy_compiled",
            **({} if delay_steps is None else {"delay_steps": delay_steps}),
        )


class _SelectionIndex:
    """Per-call inverted index over a neuron table (0.5.1 opt051_2).

    Built once per :func:`compile_connection_rules` call over the union of
    selector fields used by that call's rules. :meth:`match` reproduces the
    strict-AND ``str()``-coerced semantics of :func:`_select_indices`, keyed
    by encounter position; the caller converts matched positions to neuron
    ids (reading ``neuron_id`` only for matches, exactly like the scan), so
    tables with ``neuron_id`` absent on non-matching rows behave identically.
    """

    def __init__(self, neurons: Sequence[Mapping[str, Any]], fields: set[str]) -> None:
        self._neurons = neurons
        self._n = len(neurons)
        self._postings: dict[str, dict[str, list[int]]] = {}
        for f in fields:
            post: dict[str, list[int]] = {}
            for pos in range(self._n):
                post.setdefault(str(neurons[pos].get(f)), []).append(pos)
            self._postings[f] = post
        self._sets: dict[tuple[str, str], set[int]] = {}

    def match(self, keys: dict[str, Any]) -> tuple[bool, list[int]]:
        """Return (covered, positions). covered=False -> fall back to scan."""
        if not keys:
            return True, list(range(self._n))
        ranked: list[tuple[str, str, list[int]]] = []
        for f, v in keys.items():
            post = self._postings.get(f)
            if post is None:
                return False, []
            lst = post.get(str(v))
            if not lst:
                return True, []
            ranked.append((f, str(v), lst))
        ranked.sort(key=lambda t: len(t[2]))
        rest = ranked[1:]
        member: list[set[int]] = []
        for f, v, _ in rest:
            key = (f, v)
            st = self._sets.get(key)
            if st is None:
                st = set(self._postings[f][v])
                self._sets[key] = st
            member.append(st)
        return True, [q for q in ranked[0][2] if all(q in st for st in member)]


_PLURAL_SELECTOR_KEYS = {"layers": "layer", "cell_types": "cell_type"}


def _select_indices(
    neurons: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    *,
    allow_empty: bool,
    label: str,
    _index: "_SelectionIndex | None" = None,
) -> list[int]:
    """Resolve a quartet/id selector to neuron_id indices (strict AND).

    The plural keys ``layers``/``cell_types`` take a name or a list and match
    any listed value of the row field ``layer``/``cell_type``; declaring a key
    in both forms is refused.
    """
    if not isinstance(selector, Mapping):
        raise ValueError(f"{label} selector must be a mapping; got {type(selector).__name__}")
    keys = {k: v for k, v in selector.items() if k != "ids" and v is not None}
    member: dict[str, set[str]] = {}
    for plural, single in _PLURAL_SELECTOR_KEYS.items():
        if plural not in keys:
            continue
        if single in keys:
            raise ValueError(f"{label} selector declares both {single!r} and {plural!r}; declare one")
        value = keys.pop(plural)
        member[single] = {str(v) for v in ([value] if isinstance(value, str) else value)}
    ids_filter = selector.get("ids")
    ids_set = set(int(i) for i in ids_filter) if ids_filter is not None else None
    if _index is not None:
        covered, positions = _index.match(keys)
        if covered:
            out = []
            for pos in positions:
                if not all(str(neurons[pos].get(f)) in s for f, s in member.items()):
                    continue
                nid = int(neurons[pos]["neuron_id"])
                if ids_set is not None and nid not in ids_set:
                    continue
                out.append(nid)
            if not out and not allow_empty:
                raise ValueError(f"selector {dict(selector)!r} ({label}) matched zero neurons")
            return out
    out: list[int] = []
    for row in neurons:
        if not all(str(row.get(k)) == str(v) for k, v in keys.items()):
            continue
        if not all(str(row.get(f)) in s for f, s in member.items()):
            continue
        nid = int(row["neuron_id"])
        if ids_set is not None and nid not in ids_set:
            continue
        out.append(nid)
    if not out and not allow_empty:
        raise ValueError(f"selector {dict(selector)!r} ({label}) matched zero neurons")
    return out


def _resolve_mechanism(name: Optional[str], mech_index: Mapping[str, int], rule_name: str) -> int:
    if name is None:
        raise ValueError(f"connection {rule_name!r} has no mechanism; declare one and reference it")
    if name not in mech_index:
        raise ValueError(
            f"connection {rule_name!r} references unknown mechanism {name!r}; "
            f"declared mechanisms: {sorted(mech_index)}"
        )
    return mech_index[name]


def _candidate_pairs(
    pre_ids: list[int],
    post_ids: list[int],
    *,
    probability: Optional[float],
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Return (pre, post) edge pairs.

    probability=None -> full bipartite (sparse list, no dense mask).
    probability=p    -> target edge *density*: the realized unique edge count is
                        round(p*n_pre*n_post), except when allow_self=False and
                        populations overlap (same neurons in pre and post), in which
                        case self-connections are excluded and the count may be capped
                        below target. ``probability`` is the target density fraction,
                        NOT an independent per-pair Bernoulli rate; p>=1 yields full
                        all-to-all (minus self unless allowed). Sampling is deterministic
                        given ``key`` and never materializes a dense n_pre x n_post mask
                        in the sparse regime. A warning is issued when self-connection
                        exclusion prevents reaching the target.
    """
    n_pre, n_post = len(pre_ids), len(post_ids)
    if n_pre == 0 or n_post == 0:
        return []

    def _all_pairs():
        return [(a, b) for a in pre_ids for b in post_ids if allow_self or a != b]

    if probability is None:
        return _all_pairs()
    total = n_pre * n_post
    n_target = min(int(round(float(probability) * total)), total)
    if n_target <= 0:
        return []
    if n_target >= total:
        # p>=1: full connectivity (minus self unless allowed).
        return _all_pairs()

    if n_target * 2 <= total:
        # Sparse regime: top-up rejection sampling to exactly n_target distinct
        # edges (fixes birthday-paradox under-sampling) at O(n_target) memory.
        seen: set[tuple[int, int]] = set()
        k = key
        attempts = 0
        max_attempts = 40 * n_target + 100
        while len(seen) < n_target and attempts < max_attempts:
            k, sub = jax.random.split(k)
            draw = max((n_target - len(seen)) * 2, 16)
            pre_pick = np.asarray(jax.random.randint(sub, (draw,), 0, n_pre))
            post_pick = np.asarray(
                jax.random.randint(jax.random.fold_in(sub, 1), (draw,), 0, n_post)
            )
            for a, b in zip(pre_pick, post_pick):
                pr, po = pre_ids[int(a)], post_ids[int(b)]
                if not allow_self and pr == po:
                    continue
                seen.add((pr, po))
                if len(seen) >= n_target:
                    break
            attempts += draw
        return sorted(seen)

    # Dense-ish regime (n_target > total/2): enumerate then subsample without
    # replacement. The full pair list is the same O(total) we'd build for the
    # p>=1 branch, so this stays consistent and exact.
    all_pairs = _all_pairs()
    take = min(n_target, len(all_pairs))

    # Warn if self-connection exclusion prevents reaching the probability target
    if take < n_target:
        max_possible = len(all_pairs)
        warnings.warn(
            f"Dense-regime edge target {n_target} exceeds non-self pairs {max_possible} "
            f"(probability={probability:.3f} on {n_pre}→{n_post}); "
            f"realized {take} edges instead. Use sparse populations or lower probability.",
            UserWarning,
            stacklevel=4,
        )

    idx = np.asarray(jax.random.choice(key, len(all_pairs), shape=(take,), replace=False))
    return sorted(all_pairs[int(i)] for i in idx)


#: Tile budget (candidate pairs) for chunked Bernoulli sampling: a rule whose
#: n_pre x n_post product exceeds this never materializes the dense pair grid;
#: uniform draws are taken per tile instead. Matches the sign-only compiler's
#: ``_CONNECTIONS_EXACT_PRODUCT_CAP`` so both paths switch regime together.
_BERNOULLI_TILE_ELEMENTS = 4_000_000


def _candidate_pairs_bernoulli(
    pre_ids: list[int],
    post_ids: list[int],
    *,
    probability: Optional[float],
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Return (pre, post) edge pairs via independent per-pair Bernoulli draws.

    Each (pre, post) candidate is included independently with probability ``p``;
    with ``allow_self=False``, ``i == i`` pairs are excluded after the draw.
    ``probability=None`` -> full bipartite (sparse list, no dense mask);
    ``p >= 1`` -> full all-to-all (minus self unless allowed); ``p <= 0`` ->
    no edges. Sampling is deterministic given ``key``. Rules larger than
    :data:`_BERNOULLI_TILE_ELEMENTS` candidate pairs are drawn per tile of
    post cells (tiled over pre cells too if one column still exceeds the
    budget), so no dense n_pre x n_post array is ever materialized.
    """
    n_pre, n_post = len(pre_ids), len(post_ids)
    if n_pre == 0 or n_post == 0:
        return []

    def _all_pairs():
        return [(a, b) for a in pre_ids for b in post_ids if allow_self or a != b]

    if probability is None:
        return _all_pairs()
    p = float(probability)
    if p <= 0.0:
        return []
    if p >= 1.0:
        return _all_pairs()

    pre_arr = np.asarray(pre_ids)
    post_arr = np.asarray(post_ids)
    c_post = max(1, min(n_post, _BERNOULLI_TILE_ELEMENTS // max(n_pre, 1)))
    c_pre = max(1, min(n_pre, _BERNOULLI_TILE_ELEMENTS // max(c_post, 1)))
    pairs: list[tuple[int, int]] = []
    tile = 0
    for js in range(0, n_post, c_post):
        chunk_post = post_arr[js : js + c_post]
        for i_s in range(0, n_pre, c_pre):
            chunk_pre = pre_arr[i_s : i_s + c_pre]
            u = np.asarray(
                jax.random.uniform(jax.random.fold_in(key, tile), (chunk_pre.size, chunk_post.size))
            )
            tile += 1
            hit = np.argwhere(u < p)
            if hit.size == 0:
                continue
            sel_pre = chunk_pre[hit[:, 0]].tolist()
            sel_post = chunk_post[hit[:, 1]].tolist()
            if allow_self:
                pairs.extend(zip(sel_pre, sel_post))
            else:
                pairs.extend(
                    (int(a), int(b)) for a, b in zip(sel_pre, sel_post) if a != b
                )
    return sorted(pairs)


def _candidate_pairs_capped_bernoulli(
    pre_ids: list[int],
    post_ids: list[int],
    *,
    probability: Optional[float],
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Bernoulli draws truncated to a per-post-cell cap.

    The draws are exactly :func:`_candidate_pairs_bernoulli`'s (same tiles,
    same folded keys, same ``key``), then each post cell keeps its first
    ``C = int(n_pre * p)`` hits scanning pre cells in order (self pairs
    excluded first when ``allow_self`` is False, so the cap counts kept
    edges). ``probability=None`` -> full bipartite; ``p <= 0`` -> no edges;
    ``p >= 1`` -> full all-to-all (minus self unless allowed), as for
    Bernoulli. When the cap binds no post cell (``C`` covers every
    candidate), the result equals the Bernoulli draw bit-for-bit.
    Deterministic given ``key``.
    """
    n_pre = len(pre_ids)
    if probability is None:
        return [(a, b) for a in pre_ids for b in post_ids if allow_self or a != b]
    p = float(probability)
    if p <= 0.0:
        return []
    if p >= 1.0:
        return [(a, b) for a in pre_ids for b in post_ids if allow_self or a != b]
    cap = int(n_pre * p)
    pairs = _candidate_pairs_bernoulli(
        pre_ids, post_ids, probability=probability, allow_self=allow_self, key=key
    )
    kept: list[tuple[int, int]] = []
    seen_post: dict[int, int] = {}
    for a, b in pairs:  # sorted by (pre, post): the pre-order scan per post cell
        n_kept = seen_post.get(b, 0)
        if n_kept >= cap:
            continue
        seen_post[b] = n_kept + 1
        kept.append((a, b))
    return kept


def _candidate_pairs_fixed_indegree(
    pre_ids: list[int],
    post_ids: list[int],
    *,
    in_degree: int,
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Return (pre, post) edge pairs with an exact per-post-cell in-degree.

    Every post cell draws exactly ``in_degree`` distinct presynaptic
    partners uniformly without replacement from the pre pool (the self
    pair excluded first when ``allow_self`` is False, so the count holds
    kept edges). Each post cell keeps the ``in_degree`` smallest of
    independent uniforms over its pool, a uniform draw of a subset of that
    size; the uniforms come from one NumPy generator seeded by ``key``, so
    draws are deterministic given ``key`` and independent across post cells,
    and the whole projection costs one host-side pass (no per-cell device
    call). A post cell with fewer than ``in_degree`` available partners
    raises ``ValueError`` rather than filling short: the exact count is the
    contract (Brunel 2000: fixed CE/CI per cell).
    """
    if not pre_ids or not post_ids or in_degree <= 0:
        return []
    pre_arr = np.asarray(pre_ids, dtype=np.int64)
    post_arr = np.asarray(post_ids, dtype=np.int64)
    n_pre = pre_arr.size
    pre_pos = {int(a): i for i, a in enumerate(pre_ids)}
    # Pool position of each post cell's own pair, -1 when it stays a candidate.
    self_pos = np.full(post_arr.size, -1, dtype=np.int64)
    if not allow_self:
        for j, post_id in enumerate(post_ids):
            self_pos[j] = pre_pos.get(int(post_id), -1)
    available = n_pre - (self_pos >= 0)
    short = np.flatnonzero(available < in_degree)
    if short.size:
        j = int(short[0])
        raise ValueError(
            f"fixed_indegree in_degree={in_degree} exceeds the "
            f"{int(available[j])} available presynaptic neurons for post "
            f"neuron {post_ids[j]!r} (allow_self={allow_self}); lower "
            "in_degree or widen the source selector"
        )
    rng = np.random.default_rng(
        np.asarray(jax.random.bits(key, (4,), dtype=jnp.uint32))
    )
    rows_per_chunk = max(1, (1 << 22) // n_pre)
    out_pre, out_post = [], []
    for lo in range(0, post_arr.size, rows_per_chunk):
        hi = min(post_arr.size, lo + rows_per_chunk)
        u = rng.random((hi - lo, n_pre))
        own = np.flatnonzero(self_pos[lo:hi] >= 0)
        u[own, self_pos[lo:hi][own]] = np.inf
        if in_degree < n_pre:
            idx = np.argpartition(u, in_degree - 1, axis=1)[:, :in_degree]
        else:
            idx = np.broadcast_to(np.arange(n_pre), (hi - lo, n_pre))
        out_pre.append(pre_arr[idx].ravel())
        out_post.append(np.repeat(post_arr[lo:hi], in_degree))
    edge_pre = np.concatenate(out_pre)
    edge_post = np.concatenate(out_post)
    order = np.lexsort((edge_post, edge_pre))
    return list(zip(edge_pre[order].tolist(), edge_post[order].tolist()))


def _candidate_pairs_multinomial(
    pre_ids: list[int],
    post_ids: list[int],
    *,
    probability: Optional[float],
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Return exactly ``K = round_half_up(p * n_post)`` (pre, post) edge pairs
    drawn with replacement (multapses allowed).

    Each of the K synapses draws its presynaptic and postsynaptic neuron
    independently and uniformly, giving binomial in/out-degree; a (pre, post)
    pair may repeat. ``probability=None`` -> ``p = 1.0``; ``p <= 0`` -> no
    edges; the count rounds half-up (the delay grid's convention). A rule
    declared with ``in_degree=k`` passes its mean in-degree ``k`` as
    ``probability``, so ``K = round_half_up(k * n_post)`` for any finite
    ``k > 0``. With ``allow_self=False`` and overlapping pools, self pairs are
    excluded by drawing over the reduced valid pair space arithmetically, so exactly K
    kept pairs always come back; a rule whose valid pair space is empty while
    ``K > 0`` is refused. Sampling is deterministic given ``key`` and never
    materializes a dense n_pre x n_post pair grid.
    """
    n_pre, n_post = len(pre_ids), len(post_ids)
    if n_pre == 0 or n_post == 0:
        return []
    p = 1.0 if probability is None else float(probability)
    if p <= 0.0:
        return []
    k = int(math.floor(p * n_post + 0.5))
    if k <= 0:
        return []
    pre_arr = np.asarray(pre_ids)
    post_arr = np.asarray(post_ids)
    shared = set(map(int, pre_ids)) & set(map(int, post_ids))
    if allow_self or not shared:
        # Every flat pair index is valid: draw K flat indices with replacement
        # and map them arithmetically (no dense pair grid materialized).
        flat = np.asarray(
            jax.random.choice(key, n_pre * n_post, shape=(k,), replace=True)
        )
        pre_pick, post_pick = np.divmod(flat, n_post)
        return sorted(zip(pre_arr[pre_pick].tolist(), post_arr[post_pick].tolist()))
    # Overlapping pools without self: the valid pair space drops the
    # pre==post pairs. Draw K indices over the reduced space and map them
    # per post cell, so exactly K kept pairs always come back (no rejection
    # top-up that could come back short).
    valid_per_post = np.asarray(
        [n_pre - (1 if int(b) in shared else 0) for b in post_arr.tolist()],
        dtype=np.int64,
    )
    total_valid = int(valid_per_post.sum())
    if total_valid <= 0:
        raise ValueError(
            "multinomial sampling draws K>0 synapses but every (pre, post) pair "
            "of the rule is a refused self pair; allow self pairs or enlarge the pools"
        )
    offsets = np.concatenate(([0], np.cumsum(valid_per_post)))
    flat = np.asarray(jax.random.choice(key, total_valid, shape=(k,), replace=True))
    post_pick = np.searchsorted(offsets, flat, side="right") - 1
    within = flat - offsets[post_pick]
    pre_idx = np.asarray(within, dtype=np.int64).copy()
    for j in np.flatnonzero(valid_per_post != n_pre):
        skip = int(np.flatnonzero(pre_arr == post_arr[j])[0])
        sel = post_pick == j
        pre_idx[sel] += (within[sel] >= skip).astype(np.int64)
    return sorted(zip(pre_arr[pre_idx].tolist(), post_arr[post_pick].tolist()))


def _spatial_bucket_key(pos: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    return tuple(int(math.floor(c / cell_size)) for c in pos)


_LOCALIZED_CANDIDATE_POOL_CAP = 500


def _candidate_pairs_localized(
    pre_ids: list[int],
    post_ids: list[int],
    positions: Mapping[int, np.ndarray],
    *,
    max_in_degree: int,
    spatial_sigma: float,
    allow_self: bool,
    key: jax.Array,
) -> list[tuple[int, int]]:
    """Distance-limited edge sampling with a constant per-post in-degree cap.

    Unlike ``_candidate_pairs``'s ``probability`` mode (a target edge *density*
    that scales as O(n_pre*n_post) and is memory-catastrophic at large N), this
    targets a CONSTANT in-degree per post-neuron, independent of N, matching
    real cortical anatomy (fan-in bounded by dendritic arbor reach, not by
    total tissue volume).

    Candidate pre-neurons are restricted to a spatial neighborhood via grid
    bucketing (cell size = 3*spatial_sigma, covering ~3 std of the Gaussian
    falloff; the grid is anchored at the pre population's minimum corner) rather
    than computing a full n_pre x n_post distance matrix -- that full computation
    would reintroduce the same O(N^2) cost one step earlier even though the
    resulting edge count is sparse.

    PERFORMANCE NOTE (found 2026-07-01): the grid-bucket neighborhood's raw
    candidate count scales with local NEURON DENSITY, not with N or
    max_in_degree -- at fixed column volume, higher N means higher density,
    so the same spatial_sigma that gives ~tens of candidates at N=500 gives
    ~40,000 at N=100k in the same volume (confirmed: density * (3*sigma)^3 * 27
    neighbor cells). Without a cap, that's an O(density) cost PER POST-NEURON,
    times N post-neurons -- the dominant cost at scale, not the sampling call
    itself. Candidates are pre-subsampled uniformly down to
    ``_LOCALIZED_CANDIDATE_POOL_CAP`` (500) BEFORE the more expensive Gaussian-
    weight computation and without-replacement draw, bounding per-neuron cost
    to O(cap) regardless of density. This is a uniform pre-filter within an
    already-local neighborhood, not a bias away from "nearest" -- the Gaussian
    weighting still governs which of the (now-capped) candidates are chosen.

    Uses ``numpy``'s RNG throughout, not ``jax.random`` -- this function is
    host-side/non-traced Python (per this module's own design doctrine), and
    numpy avoids per-call JAX dispatch overhead accumulating across N
    iterations. ``key`` is still accepted and folded into a numpy seed so
    sampling stays deterministic given the same JAX key, matching the rest of
    this module's determinism contract.

    Within each post-neuron's (capped) neighborhood, pre-neurons are sampled
    without replacement, weighted by a Gaussian distance kernel, up to
    ``max_in_degree`` (or all available candidates if fewer).
    """
    if not pre_ids or not post_ids or max_in_degree <= 0:
        return []
    cell_size = 3.0 * max(float(spatial_sigma), 1e-9)
    # Anchor the grid at the pre population's minimum corner: a rigid translation of the
    # population then leaves every bucket key, and so every candidate set, unchanged.
    origin = np.min(np.stack([positions[nid] for nid in pre_ids]), axis=0)

    pre_buckets: dict[tuple[int, int, int], list[int]] = {}
    for nid in pre_ids:
        bucket = _spatial_bucket_key(positions[nid] - origin, cell_size)
        pre_buckets.setdefault(bucket, []).append(nid)

    neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    pairs: list[tuple[int, int]] = []
    rng = np.random.default_rng(_stable_fold(int(jax.random.randint(key, (), 0, 2**31 - 1))))
    for post_id in post_ids:
        post_pos = positions[post_id]
        bx, by, bz = _spatial_bucket_key(post_pos - origin, cell_size)
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(pre_buckets.get((bx + dx, by + dy, bz + dz), ()))
        if not allow_self and post_id in candidates:
            candidates = [c for c in candidates if c != post_id]
        if not candidates:
            continue
        # The pool cap and the draw both index this list: sorting by neuron id makes the
        # selection a function of ids and distances only.
        candidates.sort()
        if len(candidates) > _LOCALIZED_CANDIDATE_POOL_CAP:
            keep = rng.choice(len(candidates), size=_LOCALIZED_CANDIDATE_POOL_CAP, replace=False)
            candidates = [candidates[i] for i in keep]

        cand_pos = np.stack([positions[c] for c in candidates])
        dist_sq = np.sum((cand_pos - post_pos) ** 2, axis=1)
        raw_w = np.exp(-dist_sq / (2.0 * spatial_sigma * spatial_sigma))
        mass = raw_w.sum()
        if mass <= 0.0:
            continue
        probs = raw_w / mass

        n_take = min(max_in_degree, len(candidates))
        idx = rng.choice(len(candidates), size=n_take, replace=False, p=probs)
        for i in idx:
            pairs.append((int(candidates[int(i)]), post_id))
    return pairs


def _stable_fold(seed: Any) -> int:
    """Deterministic 31-bit fold value from a weight-spec ``seed``.

    Uses the integer directly when possible, otherwise a content hash
    (``hashlib.sha256``); never the builtin ``hash()``, which is randomized
    per-process for strings (PEP 456 / ``PYTHONHASHSEED``) and would break the
    determinism contract for string seeds.
    """
    if isinstance(seed, bool):
        seed = int(seed)
    if isinstance(seed, int):
        return seed & 0x7FFFFFFF
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _edge_weights(
    weight: Any,
    pairs: list[tuple[int, int]],
    pre_pos: dict[int, int],
    post_pos: dict[int, int],
    *,
    key: jax.Array,
    rule_name: str,
    artifacts: Mapping[str, Any],
) -> np.ndarray:
    """Compute a finite weight per edge.

    ``weight`` may be a scalar (v0.3.28 form -> scalar mode) or a dict
    WeightInitSpec with ``mode`` in {scalar, random_uniform, random_normal,
    matrix, artifact_ref}.
    """
    n = len(pairs)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # Scalar (v0.3.28 connections() stores weight as a float) or None default.
    if weight is None:
        return np.ones((n,), dtype=np.float64)
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        w = float(weight)
        if not math.isfinite(w):
            raise ValueError(f"connection {rule_name!r} scalar weight must be finite")
        return np.full((n,), w, dtype=np.float64)
    if not isinstance(weight, Mapping):
        raise ValueError(
            f"connection {rule_name!r} weight must be a number or WeightInitSpec mapping"
        )

    mode = weight.get("mode", "scalar")
    if mode == "scalar":
        w = float(weight.get("value", 1.0))
        if not math.isfinite(w):
            raise ValueError(f"connection {rule_name!r} scalar value must be finite")
        return np.full((n,), w, dtype=np.float64)
    if mode in ("random_uniform", "random_normal"):
        scale = float(weight.get("scale", 1.0))
        wkey = jax.random.fold_in(key, _stable_fold(weight.get("seed", 0)))
        if mode == "random_uniform":
            low = float(weight.get("low", 0.0))
            high = float(weight.get("high", 1.0))
            vals = np.asarray(jax.random.uniform(wkey, (n,), minval=low, maxval=high)) * scale
        else:
            loc = float(weight.get("value", 0.0))
            vals = loc + np.asarray(jax.random.normal(wkey, (n,))) * scale
        vals = vals.astype(np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"connection {rule_name!r} produced non-finite random weights")
        return vals
    if mode == "matrix":
        mat = np.asarray(weight.get("array"))
        n_pre, n_post = len(pre_pos), len(post_pos)
        if mat.shape != (n_pre, n_post):
            raise ValueError(
                f"connection {rule_name!r} matrix shape {mat.shape} != (n_pre, n_post)=({n_pre}, {n_post})"
            )
        vals = np.array([mat[pre_pos[a], post_pos[b]] for a, b in pairs], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"connection {rule_name!r} matrix contains non-finite weights")
        return vals
    if mode == "artifact_ref":
        for req in ("path", "array_name", "sha256"):
            if not weight.get(req):
                raise ValueError(f"connection {rule_name!r} artifact_ref requires {req!r}")
        array_name = weight["array_name"]
        if array_name not in artifacts:
            raise ValueError(
                f"connection {rule_name!r} artifact_ref array {array_name!r} not supplied; "
                "pass it via artifacts={name: array} (compiler does not read files)"
            )
        arr = np.ascontiguousarray(np.asarray(artifacts[array_name]))
        # Verify the declared content hash against the supplied array so the
        # sha256 field is a real integrity gate, not just a present-key check.
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest != str(weight["sha256"]):
            raise ValueError(
                f"connection {rule_name!r} artifact_ref sha256 mismatch for {array_name!r}: "
                f"declared {weight['sha256']!r} != supplied-array digest {digest!r} "
                f"(hash of np.ascontiguousarray(array).tobytes())"
            )
        return _edge_weights(
            {"mode": "matrix", "array": arr},
            pairs,
            pre_pos,
            post_pos,
            key=key,
            rule_name=rule_name,
            artifacts=artifacts,
        )
    raise ValueError(f"connection {rule_name!r} unknown weight mode {mode!r}")


_DEFAULT_SPATIAL_SIGMA = 0.1


def compile_connection_rules(
    neurons: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    allow_empty: bool = False,
    allow_self_connections: bool = False,
    artifacts: Optional[Mapping[str, Any]] = None,
    dtype: str = "float32",
    allow_intra_step_rounds: bool = False,
) -> ConnectionCompileResult:
    """Compile declared connection rules into sparse finite edge arrays.

    Parameters
    ----------
    neurons:
        Neuron-table rows (e.g. ``Model.neuron_table()``) with ``neuron_id`` and
        selector fields (area/layer/cell_type/...).
    connections:
        Connection-rule dicts (``metadata["circuit"]["connections"]``).
    mechanisms:
        Mechanism declaration dicts (``metadata["circuit"]["mechanisms"]``).
    seed:
        Base PRNG seed; each rule folds in its index for deterministic sampling.
    allow_empty:
        If False (default), a rule whose pre/post selector matches zero neurons
        raises; if True, the rule is skipped and recorded in diagnostics.
    allow_self_connections:
        If False (default), self-edges (pre == post) are dropped.
    artifacts:
        Optional mapping ``array_name -> ndarray`` for ``artifact_ref`` weights.

    A connection rule may set ``max_in_degree`` (int) instead of
    ``probability`` to sample via :func:`_candidate_pairs_localized`: a
    constant per-post-neuron in-degree cap with spatially-local, Gaussian
    distance-weighted sampling (``spatial_sigma``, default 0.1, same units as
    neuron ``x``/``y``/``z``) rather than a flat O(n_pre*n_post) density
    target.
    Requires every selected neuron row to carry ``x``/``y``/``z``.

    A rule may set ``sampling`` to ``"density"`` (default: the realized unique
    edge count is exactly ``round(p*n_pre*n_post)``), ``"bernoulli"`` (each
    (pre, post) pair is included independently with probability ``p`` via
    :func:`_candidate_pairs_bernoulli`, tiled past 4M candidate pairs so no
    dense mask is materialized), ``"capped_bernoulli"`` (the Bernoulli
    draws truncated to ``int(n_pre*p)`` kept edges per post cell, scanning
    pre cells in order, via :func:`_candidate_pairs_capped_bernoulli`) or
    ``"fixed_indegree"`` (each post cell draws exactly ``in_degree`` distinct
    presynaptic partners uniformly without replacement via
    :func:`_candidate_pairs_fixed_indegree`; it needs ``in_degree`` and
    refuses ``probability``, ``max_in_degree``, ``footprint`` and the
    Bernoulli samplings) or ``"multinomial"`` (exactly ``round(p*n_post)``
    synapses per rule drawn with replacement via
    :func:`_candidate_pairs_multinomial`; with ``in_degree=k`` in place of
    ``probability`` it draws ``round(k*n_post)`` instead, ``k`` being a finite
    mean in-degree > 0, non-integer allowed; it refuses ``in_degree`` together
    with ``probability``).
    Any other ``sampling`` value raises ``ValueError``.

    A rule may set ``delay_ms`` (configured axonal delay in ms, None =
    undeclared): validated here (finite, >= 0), fanned out per edge onto
    ``edge_delay_ms``, and recorded on the connection-table row. Realization
    to integer steps happens at construction via :func:`delay_steps_from_ms`
    (0.5.2 decision 0b), where ``dt`` is known; never here.

    A rule may set ``intra_step_rounds`` (None = undeclared): the declared
    intra-step fixed point for zero-delay edges, realized on the adex route
    only. Any other caller refuses it here, naming the adex route; pass
    ``allow_intra_step_rounds=True`` on the adex route, where it is validated
    (a positive int) and carried on the rule for the kernel.
    """
    artifacts = dict(artifacts or {})
    if dtype == "float64" and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "dtype='float64' requested but JAX x64 is not enabled -- call "
            'jax.config.update("jax_enable_x64", True) before creating any array '
            "in this process; refusing to silently build float32"
        )
    jdtype = (
        jnp.float64
        if (dtype == "float64" and bool(jax.config.read("jax_enable_x64")))
        else jnp.float32
    )

    from .emitters import INSTANTANEOUS_MECHANISM

    mechanism_table: list[dict[str, Any]] = []
    mech_index: dict[str, int] = {}
    for i, m in enumerate(mechanisms):
        name = m.get("name")
        if name is None:
            raise ValueError("mechanism declaration missing 'name'")
        params = dict(m.get("params", {}))
        tau = params.get("tau_ms", m.get("tau_ms"))
        if name == INSTANTANEOUS_MECHANISM:
            # P-051: the tau -> 0 limit is the one zero tau, and it must be declared exactly.
            if tau is None or float(tau) != 0.0:
                raise ValueError(
                    f"mechanism {name!r} is the tau -> 0 limit: declare tau_ms=0.0; got {tau!r}"
                )
        elif tau is not None and (not math.isfinite(float(tau)) or float(tau) <= 0.0):
            raise ValueError(f"mechanism {name!r} tau_ms must be finite and positive; got {tau!r}")
        mech_index[name] = i
        mechanism_table.append(
            {
                "receptor_index": i,
                "name": name,
                "kind": m.get("kind"),
                "sign": params.get("sign", m.get("sign")),
                "tau_ms": float(tau) if tau is not None else None,
                "status": "declared_not_simulated",
            }
        )

    base_key = jax.random.PRNGKey(int(seed))
    all_pre: list[int] = []
    all_post: list[int] = []
    all_weight: list[float] = []
    all_mech: list[int] = []
    all_rule: list[int] = []
    all_delay_ms: list[float] = []
    connection_table: list[dict[str, Any]] = []
    skipped: list[str] = []

    _sel_fields: set[str] = set()
    for _rule in connections:
        for _side in (_rule.get("source", {}), _rule.get("target", {})):
            if isinstance(_side, Mapping):
                _sel_fields.update(k for k, v in _side.items() if k != "ids" and v is not None)
    _sel_index = _SelectionIndex(neurons, _sel_fields)

    for rule_id, rule in enumerate(connections):
        rname = rule.get("name", f"rule_{rule_id}")
        rule_self = bool(rule.get("allow_self_connections", allow_self_connections))
        # Configured delay (ms) is a per-rule constant: validated once here,
        # fanned out per edge below, realized to steps at construction.
        rule_delay_ms = _validate_rule_delay_ms(rname, rule.get("delay_ms"))
        # Validated before any skip, so an empty or localized rule cannot carry a bad value.
        declared_rounds = rule.get("intra_step_rounds")
        if declared_rounds is not None:
            if not allow_intra_step_rounds:
                raise ValueError(
                    f"connection {rname!r} declares intra_step_rounds={declared_rounds!r}: "
                    "the intra-step fixed point is realized on the adex route only "
                    "(emitter family 'adex'); omit it on this route"
                )
            validate_rounds(declared_rounds, label=f"connection {rname!r} intra_step_rounds")
        sampling = rule.get("sampling")
        sampling = "density" if sampling is None else sampling
        if not isinstance(sampling, str) or sampling not in (
            "density", "bernoulli", "capped_bernoulli", "fixed_indegree",
            "multinomial",
        ):
            raise ValueError(
                f"connection {rname!r} sampling must be 'density', 'bernoulli', "
                f"'capped_bernoulli', 'fixed_indegree' or 'multinomial'; got {sampling!r}"
            )
        if sampling == "multinomial":
            in_degree = _validate_multinomial_in_degree(
                f"connection {rname!r}", rule.get("in_degree"))
        else:
            in_degree = _validate_in_degree(rname, rule.get("in_degree"))
        if sampling == "fixed_indegree" and in_degree is None:
            raise ValueError(
                f"connection {rname!r} declares sampling='fixed_indegree' "
                "without in_degree; the fixed-indegree sampler draws exactly "
                "in_degree presynaptic partners per post cell -- declare both"
            )
        if sampling in ("bernoulli", "capped_bernoulli") and in_degree is not None:
            raise ValueError(
                f"connection {rname!r} sampling={sampling!r} is not implemented "
                f"with in_degree={in_degree!r} (exact fixed in-degree); "
                "omit one of them"
            )
        if in_degree is not None and rule.get("probability") is not None:
            raise ValueError(
                f"connection {rname!r} declares in_degree={in_degree!r} with "
                f"probability={rule.get('probability')!r}; the fixed-indegree "
                "sampler draws an exact per-post-cell count, not a density "
                "fraction -- declare one or neither"
            )
        if in_degree is not None and rule.get("max_in_degree") is not None:
            raise ValueError(
                f"connection {rname!r} declares in_degree={in_degree!r} with "
                f"max_in_degree={rule.get('max_in_degree')!r}; exact fixed "
                "in-degree and localized (distance-weighted) sampling are "
                "mutually exclusive sampling modes -- declare one or neither"
            )
        if sampling in ("bernoulli", "capped_bernoulli", "multinomial") and rule.get("max_in_degree") is not None:
            raise ValueError(
                f"connection {rname!r} sampling={sampling!r} is not implemented with "
                "max_in_degree (localized sampling); omit one of them"
            )
        # A declared spatial_sigma has no consumer on the density path: it
        # only parametrizes localized (max_in_degree) sampling, so carrying
        # it without the cap is refused rather than silently ignored.
        if rule.get("max_in_degree") is None and rule.get("spatial_sigma") is not None:
            raise ValueError(
                f"connection {rname!r} declares spatial_sigma={rule.get('spatial_sigma')!r} "
                "without max_in_degree; spatial_sigma needs max_in_degree "
                "(localized sampling) -- declare both or neither"
            )
        # A declared footprint selects pairs deterministically from posed
        # positions (jaxfne.pcl_topology.footprint_pairs), so the stochastic
        # pair selectors must not also apply: probability, max_in_degree and
        # a non-default sampling alongside it are refused rather than
        # silently ignored. Weight sharing without a footprint is refused:
        # sharing groups are the footprint's co-located post slots.
        footprint_spec = validate_footprint(
            rule.get("footprint"), label=f"connection {rname!r}")
        sharing_mode = validate_weight_sharing(
            rule.get("weight_sharing"), label=f"connection {rname!r}")
        if footprint_spec is None:
            if sharing_mode is not None:
                raise ValueError(
                    f"connection {rname!r} declares weight_sharing={sharing_mode!r} "
                    "without footprint; sharing groups are the footprint rule's "
                    "co-located post slots -- declare both or neither"
                )
        else:
            if rule.get("probability") is not None:
                raise ValueError(
                    f"connection {rname!r} declares footprint with "
                    f"probability={rule.get('probability')!r}; the footprint "
                    "rule selects pairs deterministically -- declare one or neither"
                )
            if rule.get("max_in_degree") is not None:
                raise ValueError(
                    f"connection {rname!r} declares footprint with "
                    f"max_in_degree={rule.get('max_in_degree')!r}; the footprint "
                    "rule selects pairs deterministically -- declare one or neither"
                )
            if in_degree is not None:
                raise ValueError(
                    f"connection {rname!r} declares footprint with "
                    f"in_degree={in_degree!r}; the footprint "
                    "rule selects pairs deterministically -- declare one or neither"
                )
            if sampling != "density":
                raise ValueError(
                    f"connection {rname!r} declares footprint with "
                    f"sampling={sampling!r}; the footprint rule selects pairs "
                    "deterministically -- leave sampling at its default"
                )
        # _select_indices honors allow_empty internally (returns [] vs raises on
        # zero-match) and ALWAYS raises on a structurally invalid (non-mapping)
        # selector. No try/except here: structural errors must surface loudly,
        # never be silently recorded as an empty-selector skip.
        pre_ids = _select_indices(
            neurons,
            rule.get("source", {}),
            allow_empty=allow_empty,
            label=f"{rname}.source",
            _index=_sel_index,
        )
        post_ids = _select_indices(
            neurons,
            rule.get("target", {}),
            allow_empty=allow_empty,
            label=f"{rname}.target",
            _index=_sel_index,
        )
        if not pre_ids or not post_ids:
            skipped.append(rname)
            connection_table.append(
                {"name": rname, "n_edges": 0, "status": "skipped_empty_selector"}
            )
            continue

        mech_idx = _resolve_mechanism(rule.get("mechanism"), mech_index, rname)
        rkey = jax.random.fold_in(base_key, rule_id)
        max_in_degree = rule.get("max_in_degree")
        sharing_groups: list[int] | None = None
        if footprint_spec is not None:
            try:
                xy_by_id = {
                    int(row["neuron_id"]): (float(row["x"]), float(row["y"]))
                    for row in neurons
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"connection {rname!r} declares footprint but a neuron row "
                    f"lacks x/y posed positions ({exc})"
                )
            if any(
                not (math.isfinite(x) and math.isfinite(y))
                for x, y in xy_by_id.values()
            ):
                raise ValueError(
                    f"connection {rname!r} declares footprint but a neuron row "
                    "carries a non-finite x/y posed position"
                )
            pairs_all, groups_all = footprint_pairs(
                [int(i) for i in pre_ids],
                np.array([xy_by_id[int(i)] for i in pre_ids], dtype=float),
                [int(i) for i in post_ids],
                np.array([xy_by_id[int(i)] for i in post_ids], dtype=float),
                {"size": footprint_spec["size"], "offset": footprint_spec["offset"]},
                label=f"connection {rname!r}",
            )
            if not rule_self:
                kept = [(a, b, g) for (a, b), g in zip(pairs_all, groups_all) if a != b]
            else:
                kept = [(a, b, g) for (a, b), g in zip(pairs_all, groups_all)]
            pairs = [(a, b) for a, b, _ in kept]
            if sharing_mode is not None:
                sharing_groups = [g for _, _, g in kept]
        elif max_in_degree is not None:
            neuron_xyz = {
                int(row["neuron_id"]): np.asarray([row["x"], row["y"], row["z"]], dtype=float)
                for row in neurons
            }
            pairs = _candidate_pairs_localized(
                pre_ids,
                post_ids,
                neuron_xyz,
                max_in_degree=int(max_in_degree),
                # `.connections()` always writes the key, using None for "unset", so
                # `.get(key, default)` never fires. Coalesce explicitly -- not with `or`,
                # which would also swallow a deliberate 0.0.
                spatial_sigma=float(
                    _DEFAULT_SPATIAL_SIGMA
                    if rule.get("spatial_sigma") is None
                    else rule["spatial_sigma"]
                ),
                allow_self=rule_self,
                key=rkey,
            )
        elif in_degree is not None and sampling == "multinomial":
            # The declared mean in-degree k drives the probability form's
            # with-replacement draw: K = round_half_up(k * n_post).
            pairs = _candidate_pairs_multinomial(
                pre_ids,
                post_ids,
                probability=in_degree,
                allow_self=rule_self,
                key=rkey,
            )
        elif in_degree is not None:
            pairs = _candidate_pairs_fixed_indegree(
                pre_ids,
                post_ids,
                in_degree=in_degree,
                allow_self=rule_self,
                key=rkey,
            )
        else:
            if sampling in ("density", "bernoulli", "capped_bernoulli", "multinomial"):
                # The builder validates too; the compiler owns the rule as well,
                # and a direct compile_connection_rules call bypasses the builder.
                p_rule = rule.get("probability")
                if p_rule is not None and (
                    isinstance(p_rule, bool)
                    or not isinstance(p_rule, numbers.Real)
                    or not math.isfinite(float(p_rule))
                    or not (0.0 <= float(p_rule) <= 1.0)
                ):
                    raise ValueError(
                        f"connection {rname!r} probability={p_rule!r} "
                        "must be a finite number in [0, 1]"
                    )
            if sampling == "bernoulli":
                pairs = _candidate_pairs_bernoulli(
                    pre_ids,
                    post_ids,
                    probability=rule.get("probability"),
                    allow_self=rule_self,
                    key=rkey,
                )
            elif sampling == "capped_bernoulli":
                pairs = _candidate_pairs_capped_bernoulli(
                    pre_ids,
                    post_ids,
                    probability=rule.get("probability"),
                    allow_self=rule_self,
                    key=rkey,
                )
            elif sampling == "multinomial":
                pairs = _candidate_pairs_multinomial(
                    pre_ids,
                    post_ids,
                    probability=rule.get("probability"),
                    allow_self=rule_self,
                    key=rkey,
                )
            else:
                pairs = _candidate_pairs(
                    pre_ids,
                    post_ids,
                    probability=rule.get("probability"),
                    allow_self=rule_self,
                    key=rkey,
                )
        pre_pos = {nid: i for i, nid in enumerate(pre_ids)}
        post_pos = {nid: i for i, nid in enumerate(post_ids)}
        weights = _edge_weights(
            rule.get("weight"),
            pairs,
            pre_pos,
            post_pos,
            key=jax.random.fold_in(rkey, 1),
            rule_name=rname,
            artifacts=artifacts,
        )
        weight_range = _validate_weight_range(
            rname,
            rule.get("weight_range"),
            weight=rule.get("weight"),
            weight_std=rule.get("weight_std"),
        )
        if weight_range is not None:
            # Per-edge magnitudes U[lo, hi) from the rule's own stream (fold 4).
            lo, hi = weight_range
            u = np.asarray(
                jax.random.uniform(jax.random.fold_in(rkey, _WEIGHT_RANGE_FOLD), (len(pairs),)),
                dtype=np.float64,
            )
            weights = lo + (hi - lo) * u
        # Per-rule Gaussian spreads (G-WDIST/G-DDIST): one Normal draw per
        # edge from the rule's own stream, absolute-value folded at zero so
        # the paper's fixed sign (SOURCE_P1.md section 2: Gaussian w +- delta_w
        # sign-preserving; Gaussian d +- delta_d positive) holds per edge.
        # Undeclared spreads keep the constant path above byte-identical.
        weight_std = _validate_spread_std(rname, rule.get("weight_std"), "weight_std")
        if weight_std is not None:
            w_spec = rule.get("weight")
            if w_spec is not None and (
                isinstance(w_spec, bool) or not isinstance(w_spec, (int, float))
            ):
                raise ValueError(
                    f"connection {rname!r} declares weight_std={weight_std!r} with "
                    f"a non-scalar weight={w_spec!r}; the spread draws around one "
                    "scalar mean -- declare a scalar weight or neither"
                )
            mean = 0.0 if w_spec is None else float(w_spec)
            noise = np.asarray(
                jax.random.normal(jax.random.fold_in(rkey, _WEIGHT_SPREAD_FOLD),
                                  (len(pairs),)), dtype=np.float64)
            weights = np.abs(mean + float(weight_std) * noise)
        delay_std = _validate_spread_std(
            rname, rule.get("delay_std_ms"), "delay_std_ms")
        if delay_std is not None and rule_delay_ms is None:
            raise ValueError(
                f"connection {rname!r} declares delay_std_ms={delay_std!r} without "
                "delay_ms; the spread draws around one scalar mean -- declare both"
            )
        if delay_std is None:
            for (a, b), w in zip(pairs, weights):
                all_pre.append(int(a))
                all_post.append(int(b))
                all_weight.append(float(w))
                all_mech.append(int(mech_idx))
                all_rule.append(int(rule_id))
                all_delay_ms.append(0.0 if rule_delay_ms is None else float(rule_delay_ms))
        else:
            dnoise = np.asarray(
                jax.random.normal(jax.random.fold_in(rkey, _DELAY_SPREAD_FOLD),
                                  (len(pairs),)), dtype=np.float64)
            draws = np.abs(float(rule_delay_ms) + float(delay_std) * dnoise)
            for (a, b), w, dd in zip(pairs, weights, draws):
                all_pre.append(int(a))
                all_post.append(int(b))
                all_weight.append(float(w))
                all_mech.append(int(mech_idx))
                all_rule.append(int(rule_id))
                all_delay_ms.append(float(dd))
        row = {
            "name": rname,
            "n_pre": len(pre_ids),
            "n_post": len(post_ids),
            "n_edges": len(pairs),
            "mechanism": rule.get("mechanism"),
            "probability": rule.get("probability"),
            "max_in_degree": rule.get("max_in_degree"),
            "in_degree": rule.get("in_degree"),
            "sign": rule.get("sign"),
            "control_key": rule.get("control_key"),
            "delay_ms": rule_delay_ms,
            "status": "compiled",
        }
        if weight_range is not None:
            # Echoed only when declared so undeclared rows stay byte-identical.
            row["weight_range"] = [weight_range[0], weight_range[1]]
        if weight_std is not None:
            # Echoed only when declared so undeclared rows stay byte-identical.
            row["weight_std"] = weight_std
        if delay_std is not None:
            row["delay_std_ms"] = delay_std
        if footprint_spec is not None:
            # Echoed so the realized geometry stays inspectable; the
            # per-edge groups align with this rule's edges (global indices
            # estart:estart + n_edges) for the HDP rule to consume.
            row["footprint"] = {
                "size": list(footprint_spec["size"]),
                "offset": list(footprint_spec["offset"]),
            }
            row["weight_sharing"] = sharing_mode
            if sharing_groups is not None:
                row["sharing_group_of_edge"] = [int(g) for g in sharing_groups]
                row["n_sharing_groups"] = (
                    max(sharing_groups) + 1 if sharing_groups else 0
                )
        inhibition_spec = validate_local_inhibition(
            rule.get("local_inhibition"), label=f"connection {rname!r}")
        if inhibition_spec is not None:
            # Echoed so the declared intra-step fixed point (R and the round
            # order on A) stays inspectable through the manifest. The
            # resolution echoes only when explicitly declared, so default
            # specs echo byte-identically.
            row["local_inhibition"] = {
                "rounds": int(inhibition_spec["rounds"]),
                "order": str(inhibition_spec["order"]),
            }
            if "resolution" in inhibition_spec:
                row["local_inhibition"]["resolution"] = str(
                    inhibition_spec["resolution"])
        if sampling in ("bernoulli", "capped_bernoulli", "fixed_indegree", "multinomial"):
            # Recorded only when non-default, as in connections(), so density rows are unchanged.
            row["sampling"] = sampling
        if sampling == "multinomial" and in_degree is not None:
            # The declared mean in-degree is row["in_degree"]; the realized
            # with-replacement count K is recorded beside it.
            row["realized_k"] = len(pairs)
        connection_table.append(row)

    # Lists go through NumPy first: jnp.asarray walks a Python list element by element.
    edge_pre = jnp.asarray(np.asarray(all_pre, dtype=np.int32), dtype=jnp.int32)
    edge_post = jnp.asarray(np.asarray(all_post, dtype=np.int32), dtype=jnp.int32)
    edge_weight = jnp.asarray(np.asarray(all_weight, dtype=np.float64), dtype=jdtype)
    edge_mechanism = jnp.asarray(np.asarray(all_mech, dtype=np.int32), dtype=jnp.int32)
    edge_rule_id = jnp.asarray(np.asarray(all_rule, dtype=np.int32), dtype=jnp.int32)
    edge_delay_ms = jnp.asarray(np.asarray(all_delay_ms, dtype=np.float64), dtype=jdtype)
    if edge_weight.shape[0] and not bool(jnp.all(jnp.isfinite(edge_weight))):
        raise ValueError("compiled edge weights contain non-finite values")

    diagnostics = {
        "n_rules": len(connections),
        "n_mechanisms": len(mechanism_table),
        "n_edges_total": int(edge_pre.shape[0]),
        "skipped_rules": skipped,
        "seed": int(seed),
        "allow_self_connections": bool(allow_self_connections),
        "field_solver_status": "linear_solver",
        "physical_amplitude_calibrated": False,
        "connectivity_compiled": True,
    }
    return ConnectionCompileResult(
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_weight=edge_weight,
        edge_mechanism=edge_mechanism,
        edge_rule_id=edge_rule_id,
        connection_table=connection_table,
        mechanism_table=mechanism_table,
        diagnostics=diagnostics,
        edge_delay_ms=edge_delay_ms,
    )


def with_rule_allow_self_connections(cfg: Any, rule_name: str, value: bool = True) -> Any:
    """Return a copy of ``cfg`` with ``allow_self_connections`` set on one rule.

    :meth:`Configuration.connections` has no ``allow_self_connections`` kwarg,
    so a caller that wires one (the NeuronalTensor bridge's per-connection
    ``allow_self`` opt-in) records it through here instead of metadata
    surgery: the named rule entry is copied with
    ``allow_self_connections=bool(value)`` and everything else byte-identical.
    The rule must exist exactly once; a non-bool ``value`` is refused
    (fail-closed: only an explicit ``True``/``False`` opts a rule in or out).
    :func:`compile_connection_rules` reads the per-rule key on every sampling
    path; an absent key keeps the global default (self-edges dropped).
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"allow_self_connections for rule {rule_name!r} must be a bool; got {value!r}"
        )
    metadata = dict(cfg.metadata)
    circuit = {k: list(v) for k, v in dict(metadata.get("circuit", {})).items()}
    rules = list(circuit.get("connections", []))
    hits = [i for i, rule in enumerate(rules) if rule.get("name") == rule_name]
    if len(hits) != 1:
        raise ValueError(
            f"rule {rule_name!r} must exist exactly once to opt into "
            f"allow_self_connections; found {len(hits)}"
        )
    patched = dict(rules[hits[0]], allow_self_connections=value)
    rules[hits[0]] = patched
    circuit["connections"] = rules
    metadata["circuit"] = circuit
    metadata["circuit_compiled"] = False
    return _dc_replace(cfg, metadata=metadata)


def with_rule_local_inhibition(cfg: Any, rule_name: str, spec: Any) -> Any:
    """Return a copy of ``cfg`` with ``local_inhibition`` set on one rule.

    Same idiom as :func:`with_rule_allow_self_connections` (:meth:
    `Configuration.connections` has no ``local_inhibition`` kwarg): the
    named rule entry is copied with ``local_inhibition={"rounds", "order"}``
    (plus ``"resolution"`` when explicitly declared, so default specs stay
    byte-identical) and everything else byte-identical. ``spec`` is revalidated here
    (fail-closed); :func:`compile_connection_rules` echoes it onto the
    connection-table row so the declared intra-step fixed point rides the
    manifest. The rule must exist exactly once.
    """
    normalized = validate_local_inhibition(spec, label=f"rule {rule_name!r}")
    assert normalized is not None
    metadata = dict(cfg.metadata)
    circuit = {k: list(v) for k, v in dict(metadata.get("circuit", {})).items()}
    rules = list(circuit.get("connections", []))
    hits = [i for i, rule in enumerate(rules) if rule.get("name") == rule_name]
    if len(hits) != 1:
        raise ValueError(
            f"rule {rule_name!r} must exist exactly once to declare "
            f"local_inhibition; found {len(hits)}"
        )
    patched = dict(
        rules[hits[0]],
        local_inhibition={
            "rounds": normalized["rounds"], "order": normalized["order"],
            **({"resolution": normalized["resolution"]}
               if "resolution" in normalized else {}),
        },
    )
    rules[hits[0]] = patched
    circuit["connections"] = rules
    metadata["circuit"] = circuit
    metadata["circuit_compiled"] = False
    return _dc_replace(cfg, metadata=metadata)


@partial(jax.jit, static_argnums=(4,))
def compile_connection_rules_jax(
    pre_indices: jax.Array,
    post_indices: jax.Array,
    probability: float,
    key: jax.Array,
    max_edges: int,
    weight_val: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Tensorized JAX connectivity compiler producing static-shape edge outputs.

    Samples connection pairs using a probability mask and pads/truncates to max_edges.
    Negative one (-1) pre/post values denote padded/inactive connections.

    Note on sparsity: this still touches an O(n_pre*n_post) array of per-pair
    random scores (unavoidable under a single static-shape jit call while
    preserving the independent-Bernoulli-then-cap-at-max_edges semantics), but
    it no longer materializes the two full (n_pre, n_post) pre/post INDEX grids
    the earlier implementation built via jnp.tile -- those are derived only for
    the selected max_edges candidates via flat-index arithmetic instead.
    """
    n_pre = pre_indices.shape[0]
    n_post = post_indices.shape[0]
    total = n_pre * n_post

    rand_vals = jax.random.uniform(key, shape=(total,))
    mask = rand_vals < probability

    # Push unselected edges to score 2.0, selected edges keep random score < 1.0
    score = jnp.where(mask, rand_vals, 2.0)
    sorted_idx = jnp.argsort(score)
    selected_idx = sorted_idx[:max_edges]

    valid_mask = score[selected_idx] < 2.0

    # Flat-index arithmetic on only the selected candidates -- never a dense
    # (n_pre, n_post) pre/post index grid.
    sel_pre_pos = selected_idx // n_post
    sel_post_pos = selected_idx % n_post

    edge_pre = jnp.where(valid_mask, pre_indices[sel_pre_pos], -1)
    edge_post = jnp.where(valid_mask, post_indices[sel_post_pos], -1)
    edge_weight = jnp.where(valid_mask, jnp.full((max_edges,), weight_val), 0.0)

    return edge_pre, edge_post, edge_weight
