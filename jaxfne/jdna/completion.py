"""JDNA developmental completion: defaults, K_D sampling, value origins.

TFNE states constraints and may leave quantities open that realization
requires. This module is the single owner of completing them, per
``docs/doctrine/tfne_jdna_boundary.md`` and project source 7 S29.1:

    A (TFNE constraints) + D (rules/defaults) + K_D (development RNG) -> M.

Every completed value carries its origin in
{TFNE-declared, JDNA-derived, JDNA-default, JDNA-sampled}, where "declared"
means stated in the input specification (TFNE constraints, or equivalently
PseudoGenome rules on the genome path).

K_D discipline: callers pass an integer development seed from the K_D domain
(distinct from construction/simulation K_S and optimizer K_A). Keys are
derived by splitting once per completed scope in sorted order; a key is never
reused across scopes. Same seed reproduces the same completion; different
seeds realize different phenotypes within the same constraint bands.

Positions are float32 arrays with documented axes: columns are (x, y, z) in
relative units. Compute in float32 (matching the JAX default); no accumulation
is performed here, so no precision is at stake beyond representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp

ORIGIN_DECLARED = "TFNE-declared"
ORIGIN_DERIVED = "JDNA-derived"
ORIGIN_DEFAULT = "JDNA-default"
ORIGIN_SAMPLED = "JDNA-sampled"

ORIGINS = frozenset({ORIGIN_DECLARED, ORIGIN_DERIVED, ORIGIN_DEFAULT, ORIGIN_SAMPLED})

#: Default developmental domain per axis: the unit cube in relative units.
DEFAULT_DOMAIN_3D: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)

#: Default position distribution when none is declared.
DEFAULT_GEOMETRY_DISTRIBUTION = "uniform"

#: Working precision of realized positions (JAX default; representation only).
POSITIONS_DTYPE = "float32"


@dataclass(frozen=True)
class CompletionRule:
    """One completable quantity: how JDNA resolves it when undeclared."""

    name: str
    policy: str  # "default" | "derive" | "sample" | "required"
    default: Any = None


def _require_policy(rule: CompletionRule) -> None:
    if rule.policy not in ("default", "derive", "sample", "required"):
        raise ValueError(f"unknown completion policy {rule.policy!r}")


#: The canonical defaults table. "required" means the quantity is
#: consequential without a canonical default: JDNA refuses rather than
#: invents. Mechanism kinetics stay "required" by design even though the
#: TFNE mechanism vocabulary has landed (TFNE2-07, resolve_mechanism):
#: JDNA completes structure, not kinetics; tau resolution happens
#: downstream at the tensor bridge into StaticParams.
COMPLETION_RULES: Mapping[str, CompletionRule] = {
    "geometry_distribution": CompletionRule(
        name="geometry_distribution", policy="default", default=DEFAULT_GEOMETRY_DISTRIBUTION
    ),
    "geometry_domain": CompletionRule(
        name="geometry_domain", policy="default", default=DEFAULT_DOMAIN_3D
    ),
    # Cell allocation is never defaulted or sampled: exact integer counts
    # must be derived from declared N/P (largest-remainder), or refused.
    "cell_allocation": CompletionRule(name="cell_allocation", policy="derive"),
    "mechanism_tau_ms": CompletionRule(name="mechanism_tau_ms", policy="required"),
}


def resolve(
    name: str,
    declared: Any = None,
    *,
    derive: Optional[Callable[[], Any]] = None,
    sample: Optional[Callable[[Any], Any]] = None,
    key: Any = None,
) -> tuple[Any, str]:
    """Resolve one completable quantity to (value, origin).

    A declared value always wins (TFNE-declared). Otherwise the quantity's
    rule applies: canonical default, deterministic derivation, K_D sampling,
    or refusal. Unknown quantity names are refused even when declared —
    accepting them would let a misspelled quantity pass as completed while
    downstream ignores it.
    """
    rule = COMPLETION_RULES.get(name)
    if rule is None:
        raise ValueError(
            f"unknown completable quantity {name!r}; JDNA completes only "
            f"declared quantities: {sorted(COMPLETION_RULES)}"
        )
    if declared is not None:
        return declared, ORIGIN_DECLARED
    _require_policy(rule)
    if rule.policy == "default":
        return rule.default, ORIGIN_DEFAULT
    if rule.policy == "derive":
        if derive is None:
            raise ValueError(
                f"{name!r} must be derived from another constraint, "
                "but no derive function was given"
            )
        return derive(), ORIGIN_DERIVED
    if rule.policy == "sample":
        if sample is None or key is None:
            raise ValueError(
                f"{name!r} is stochastic: it requires a sample function and an explicit K_D key"
            )
        return sample(key), ORIGIN_SAMPLED
    raise ValueError(
        f"{name!r} is required but has no canonical default; refusing "
        "rather than inventing a consequential value"
    )


def _axis_range(declared: Mapping[str, Any], axis: str) -> tuple[float, float]:
    """Resolve one axis domain: declared pair, default pair, or refusal."""
    lo_key, hi_key = f"{axis}0", f"{axis}1"
    range_key = f"{axis}_range"
    if range_key in declared:
        lo, hi = declared[range_key]
        return float(lo), float(hi)
    lo = declared.get(lo_key)
    hi = declared.get(hi_key)
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    if lo is None and hi is None:
        return DEFAULT_DOMAIN_3D["xyz".index(axis)]
    raise ValueError(
        f"partial {axis} domain declares only one bound "
        f"({lo_key}={lo!r}, {hi_key}={hi!r}); a half-domain cannot be "
        "honoured exactly, so it is refused rather than half-defaulted"
    )


def realize_geometry(
    declared: Optional[Mapping[str, Any]], n: int, seed: int
) -> tuple[Any, dict[str, str]]:
    """Realize positions for `n` neurons under K_D.

    `declared` is the TFNE `G` body (possibly empty): per-axis bounds as
    `{ax}0`/`{ax}1` (or `{ax}_range` pairs) plus an optional `distribution`.
    Missing axes fall back to the unit interval (JDNA-default); a declared
    axis is obeyed (TFNE-declared); a half-declared axis is refused. Only
    `uniform` sampling is supported; any other declared distribution is
    refused rather than silently uniformed.

    Returns `(positions, origins)` with positions a float32 `(n, 3)` array
    of `(x, y, z)` columns and origins per axis/distribution/positions.
    `n == 0` yields an empty `(0, 3)` array deterministically.
    """
    body: Mapping[str, Any] = declared or {}
    if n < 0:
        raise ValueError(f"realize_geometry needs n >= 0; got {n}")
    distribution, dist_origin = resolve("geometry_distribution", body.get("distribution"))
    if distribution != "uniform":
        raise ValueError(
            f"unsupported geometry distribution {distribution!r}; only "
            "'uniform' can be honoured exactly"
        )
    domain: list[tuple[float, float]] = []
    origins: dict[str, str] = {"distribution": dist_origin}
    for axis in "xyz":
        lo, hi = _axis_range(body, axis)
        domain.append((lo, hi))
        if (axis + "0" in body and axis + "1" in body) or (axis + "_range" in body):
            origins[axis] = ORIGIN_DECLARED
        else:
            origins[axis] = ORIGIN_DEFAULT
        if not hi > lo:
            raise ValueError(f"degenerate {axis} domain [{lo}, {hi}]; bounds must order")
    key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(key, 3)
    cols = [
        jax.random.uniform(
            keys[i], shape=(n,), minval=domain[i][0], maxval=domain[i][1], dtype=jnp.float32
        )
        for i in range(3)
    ]
    positions = jnp.stack(cols, axis=1).reshape((n, 3))
    origins["positions"] = ORIGIN_SAMPLED
    return positions, origins


def complete_tfne(realization: Any, seed: int) -> dict[str, Any]:
    """Complete a TFNE realization's per-leaf geometry under K_D.

    Reads each realized leaf's neuron count from `I["neuron_paths"]` and its
    declared `G` body from `s["geometry"]`; realizes positions per leaf with
    one split of the development key each, in sorted leaf order. The input
    realization is never mutated — the TFNE bridge is untouched; this is the
    JDNA-side path that owns coordinates.

    Returns `{"positions", "origins", "seed", "domain"}` with positions and
    origins keyed by leaf path.
    """
    paths: list[str] = list(realization.I["neuron_paths"])
    counts: dict[str, int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1
    leaves = sorted(counts)
    master = jax.random.PRNGKey(int(seed))
    subseeds = [
        int(v) for v in jax.random.randint(master, shape=(len(leaves),), minval=0, maxval=2**31 - 1)
    ]
    positions: dict[str, Any] = {}
    origins: dict[str, Any] = {}
    geometry: Mapping[str, Any] = realization.s.get("geometry", {})
    for leaf, subseed in zip(leaves, subseeds):
        pos, org = realize_geometry(geometry.get(leaf), counts[leaf], subseed)
        positions[leaf] = pos
        origins[leaf] = org
    return {"positions": positions, "origins": origins, "seed": int(seed), "domain": "K_D"}
