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
import warnings
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["compile_connection_rules", "ConnectionCompileResult", "compile_connection_rules_jax"]


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

    @property
    def n_edges(self) -> int:
        """Documented public function `n_edges`."""
        return int(self.edge_pre.shape[0])

    def to_edge_list(self, *, default_tau_ms: float = 5.0, dtype: str = "float32"):
        """Build a :class:`jaxfne.EdgeList` from the compiled edges.

        ``tau_ms`` per edge comes from the mechanism's declared ``tau_ms`` (or
        ``default_tau_ms``); ``receptor_index`` is ``edge_mechanism``.
        """
        from .emitters import EdgeList

        jdtype = jnp.float64 if (dtype == "float64" and bool(jax.config.read("jax_enable_x64"))) else jnp.float32
        tau_by_mech = [
            float(m.get("tau_ms", default_tau_ms)) if m.get("tau_ms") is not None else float(default_tau_ms)
            for m in self.mechanism_table
        ]
        if self.n_edges and tau_by_mech:
            tau_table = jnp.asarray(tau_by_mech, dtype=jdtype)
            mech_idx = jnp.asarray(self.edge_mechanism, dtype=jnp.int32)
            tau = tau_table[mech_idx]
        else:
            tau = jnp.zeros((self.n_edges,), dtype=jdtype)
        return EdgeList(
            pre=jnp.asarray(self.edge_pre, dtype=jnp.int32),
            post=jnp.asarray(self.edge_post, dtype=jnp.int32),
            weight=jnp.asarray(self.edge_weight, dtype=jdtype),
            receptor_index=jnp.asarray(self.edge_mechanism, dtype=jnp.int32),
            tau_ms=tau,
            source_calibration_status="uncalibrated_proxy_compiled",
        )


def _select_indices(
    neurons: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    *,
    allow_empty: bool,
    label: str,
) -> list[int]:
    """Resolve a quartet/id selector to neuron_id indices (strict AND)."""
    if not isinstance(selector, Mapping):
        raise ValueError(f"{label} selector must be a mapping; got {type(selector).__name__}")
    keys = {k: v for k, v in selector.items() if k != "ids" and v is not None}
    ids_filter = selector.get("ids")
    ids_set = set(int(i) for i in ids_filter) if ids_filter is not None else None
    out: list[int] = []
    for row in neurons:
        if not all(str(row.get(k)) == str(v) for k, v in keys.items()):
            continue
        nid = int(row["neuron_id"])
        if ids_set is not None and nid not in ids_set:
            continue
        out.append(nid)
    if not out and not allow_empty:
        raise ValueError(f"selector {dict(selector)!r} ({label}) matched zero neurons")
    return out


def _resolve_mechanism(
    name: Optional[str], mech_index: Mapping[str, int], rule_name: str
) -> int:
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
            post_pick = np.asarray(jax.random.randint(jax.random.fold_in(sub, 1), (draw,), 0, n_post))
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
    falloff) rather than computing a full n_pre x n_post distance matrix --
    that full computation would reintroduce the same O(N^2) cost one step
    earlier even though the resulting edge count is sparse.

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

    pre_buckets: dict[tuple[int, int, int], list[int]] = {}
    for nid in pre_ids:
        pre_buckets.setdefault(_spatial_bucket_key(positions[nid], cell_size), []).append(nid)

    neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    pairs: list[tuple[int, int]] = []
    rng = np.random.default_rng(_stable_fold(int(jax.random.randint(key, (), 0, 2**31 - 1))))
    for post_id in post_ids:
        post_pos = positions[post_id]
        bx, by, bz = _spatial_bucket_key(post_pos, cell_size)
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(pre_buckets.get((bx + dx, by + dy, bz + dz), ()))
        if not allow_self and post_id in candidates:
            candidates = [c for c in candidates if c != post_id]
        if not candidates:
            continue
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
    (``hashlib.sha256``) — never the builtin ``hash()``, which is randomized
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
        raise ValueError(f"connection {rule_name!r} weight must be a number or WeightInitSpec mapping")

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
            pairs, pre_pos, post_pos, key=key, rule_name=rule_name, artifacts=artifacts,
        )
    raise ValueError(f"connection {rule_name!r} unknown weight mode {mode!r}")


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
    """
    artifacts = dict(artifacts or {})
    jdtype = jnp.float64 if (dtype == "float64" and bool(jax.config.read("jax_enable_x64"))) else jnp.float32

    mechanism_table: list[dict[str, Any]] = []
    mech_index: dict[str, int] = {}
    for i, m in enumerate(mechanisms):
        name = m.get("name")
        if name is None:
            raise ValueError("mechanism declaration missing 'name'")
        params = dict(m.get("params", {}))
        tau = params.get("tau_ms", m.get("tau_ms"))
        if tau is not None and (not math.isfinite(float(tau)) or float(tau) <= 0.0):
            raise ValueError(f"mechanism {name!r} tau_ms must be finite and positive; got {tau!r}")
        mech_index[name] = i
        mechanism_table.append({
            "receptor_index": i,
            "name": name,
            "kind": m.get("kind"),
            "sign": params.get("sign", m.get("sign")),
            "tau_ms": float(tau) if tau is not None else None,
            "status": "declared_not_simulated",
        })

    base_key = jax.random.PRNGKey(int(seed))
    all_pre: list[int] = []
    all_post: list[int] = []
    all_weight: list[float] = []
    all_mech: list[int] = []
    all_rule: list[int] = []
    connection_table: list[dict[str, Any]] = []
    skipped: list[str] = []

    for rule_id, rule in enumerate(connections):
        rname = rule.get("name", f"rule_{rule_id}")
        rule_self = bool(rule.get("allow_self_connections", allow_self_connections))
        # _select_indices honors allow_empty internally (returns [] vs raises on
        # zero-match) and ALWAYS raises on a structurally invalid (non-mapping)
        # selector. No try/except here: structural errors must surface loudly,
        # never be silently recorded as an empty-selector skip.
        pre_ids = _select_indices(neurons, rule.get("source", {}), allow_empty=allow_empty, label=f"{rname}.source")
        post_ids = _select_indices(neurons, rule.get("target", {}), allow_empty=allow_empty, label=f"{rname}.target")
        if not pre_ids or not post_ids:
            skipped.append(rname)
            connection_table.append({"name": rname, "n_edges": 0, "status": "skipped_empty_selector"})
            continue

        mech_idx = _resolve_mechanism(rule.get("mechanism"), mech_index, rname)
        rkey = jax.random.fold_in(base_key, rule_id)
        max_in_degree = rule.get("max_in_degree")
        if max_in_degree is not None:
            neuron_xyz = {
                int(row["neuron_id"]): np.asarray([row["x"], row["y"], row["z"]], dtype=float)
                for row in neurons
            }
            pairs = _candidate_pairs_localized(
                pre_ids, post_ids, neuron_xyz,
                max_in_degree=int(max_in_degree),
                spatial_sigma=float(rule.get("spatial_sigma", 0.1)),
                allow_self=rule_self, key=rkey,
            )
        else:
            pairs = _candidate_pairs(
                pre_ids, post_ids, probability=rule.get("probability"), allow_self=rule_self, key=rkey
            )
        pre_pos = {nid: i for i, nid in enumerate(pre_ids)}
        post_pos = {nid: i for i, nid in enumerate(post_ids)}
        weights = _edge_weights(
            rule.get("weight"), pairs, pre_pos, post_pos,
            key=jax.random.fold_in(rkey, 1), rule_name=rname, artifacts=artifacts,
        )
        for (a, b), w in zip(pairs, weights):
            all_pre.append(int(a))
            all_post.append(int(b))
            all_weight.append(float(w))
            all_mech.append(int(mech_idx))
            all_rule.append(int(rule_id))
        connection_table.append({
            "name": rname,
            "n_pre": len(pre_ids),
            "n_post": len(post_ids),
            "n_edges": len(pairs),
            "mechanism": rule.get("mechanism"),
            "probability": rule.get("probability"),
            "max_in_degree": rule.get("max_in_degree"),
            "sign": rule.get("sign"),
            "control_key": rule.get("control_key"),
            "status": "compiled",
        })

    edge_pre = jnp.asarray(all_pre, dtype=jnp.int32)
    edge_post = jnp.asarray(all_post, dtype=jnp.int32)
    edge_weight = jnp.asarray(all_weight, dtype=jdtype)
    edge_mechanism = jnp.asarray(all_mech, dtype=jnp.int32)
    edge_rule_id = jnp.asarray(all_rule, dtype=jnp.int32)
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
    )


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

