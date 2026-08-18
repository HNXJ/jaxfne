"""PseudoGenome specification and the ``develop`` operator.

The module implements the JDNA grammar

    PseudoGenome --develop--> NeuronalTensor

A PseudoGenome is a generative specification: it declares structural rules
(area/layer organization, cell-type fractions with tolerance bands, geometry,
connection schemes) that determine the development of a neuronal phenotype.
It never stores the terminal phenotype (no positions, no edges).

Development is deterministic in the development PRNG domain K_D:
``develop(G, seed) == develop(G, seed)`` for every seed. Different K_D values
realize different phenotypes within the genome-declared constraint bands.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import jax

from ..io import save_json, load_json
from ..neuronal_tensor import (
    Area,
    AreaConnection,
    Geometry3D,
    InterConnection,
    Layer,
    NeuronalTensor,
    NeuronType,
    Pose3D,
    RuntimeConfiguration,
)

PSEUDOGENOME_SCHEMA_VERSION = "pseudogenome_v1"


@dataclass(frozen=True)
class LayerGenome:
    """Generative rule for one laminar layer.

    ``cell_type_fractions`` declares the base population composition;
    ``fraction_tolerance`` (optional, per cell type, absolute band in [0, 1])
    declares the developmental constraint within which the realized
    composition may vary. Counts are allocated to exactly ``n_neurons``.
    """

    name: str
    n_neurons: int
    depth_band: tuple[float, float]
    cell_type_fractions: Mapping[str, float]
    fraction_tolerance: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(
        default_factory=lambda: {
            "distribution": "uniform_random",
            "x_range": (0.0, 1.0),
            "y_range": (0.0, 1.0),
            "value_tag": "relative",
        }
    )
    relative_sizes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionRuleGenome:
    """Declared within-area connection rule (a typed synapse scheme)."""

    source_layer: str
    source_neuron_type: str
    target_layer: str
    target_neuron_type: str
    mechanism: str


@dataclass(frozen=True)
class AreaGenome:
    """Generative rule for one area: layers plus within-area connection scheme."""

    name: str
    layers: Sequence[LayerGenome] = field(default_factory=tuple)
    inter_connections: Sequence[ConnectionRuleGenome] = field(default_factory=tuple)
    pose: Mapping[str, Any] = field(
        default_factory=lambda: {
            "plane": "xy",
            "rotation_deg": 0.0,
            "translation": (0.0, 0.0, 0.0),
            "value_tag": "relative",
        }
    )


@dataclass(frozen=True)
class PseudoGenome:
    """A finite, model-defined generative specification of a neuronal phenotype.

    The genome declares structural rules only. Development (``develop``) maps
    ``(G, K_D)`` to a :class:`NeuronalTensor` whose composition satisfies the
    genome-declared constraints; the tensor carries the development provenance
    (genome identity hash, schema version, development seed, development
    parameters, phenotype hash).
    """

    name: str
    schema_version: str = PSEUDOGENOME_SCHEMA_VERSION
    description: str = ""
    areas: Sequence[AreaGenome] = field(default_factory=tuple)
    area_connections: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    development_parameters: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _canonical(obj: Any) -> Any:
    """Canonicalize a JSON-safe object (sorted keys, tuples -> lists) for hashing."""
    if isinstance(obj, Mapping):
        return {k: _canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")


def genome_rules_hash(genome: PseudoGenome) -> str:
    """Deterministic sha256 of the generative rules (excludes ``description``)."""
    payload = {
        "schema_version": genome.schema_version,
        "name": genome.name,
        "development_parameters": dict(genome.development_parameters),
        "areas": [_area_to_dict(a) for a in genome.areas],
        "area_connections": [dict(c) for c in genome.area_connections],
    }
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def phenotype_sha256(tensor: NeuronalTensor) -> str:
    """Deterministic sha256 of the developed phenotype (provenance excluded)."""
    payload = dict(tensor.to_dict())
    payload.pop("provenance", None)
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _area_to_dict(area: AreaGenome) -> dict[str, Any]:
    return {
        "name": area.name,
        "pose": dict(area.pose),
        "layers": [
            {
                "name": layer.name,
                "n_neurons": layer.n_neurons,
                "depth_band": list(layer.depth_band),
                "cell_type_fractions": dict(layer.cell_type_fractions),
                "fraction_tolerance": {
                    k: list(v) for k, v in layer.fraction_tolerance.items()
                },
                "geometry": dict(layer.geometry),
                "relative_sizes": dict(layer.relative_sizes),
            }
            for layer in area.layers
        ],
        "inter_connections": [
            {
                "source_layer": c.source_layer,
                "source_neuron_type": c.source_neuron_type,
                "target_layer": c.target_layer,
                "target_neuron_type": c.target_neuron_type,
                "mechanism": c.mechanism,
            }
            for c in area.inter_connections
        ],
    }


def _parse_layer(raw: Mapping[str, Any]) -> LayerGenome:
    return LayerGenome(
        name=str(raw["name"]),
        n_neurons=int(raw["n_neurons"]),
        depth_band=tuple(float(x) for x in raw["depth_band"]),
        cell_type_fractions=dict(raw["cell_type_fractions"]),
        fraction_tolerance={
            k: tuple(float(x) for x in v) for k, v in raw.get("fraction_tolerance", {}).items()
        },
        geometry=_canonical_geometry(dict(raw.get("geometry", {}))),
        relative_sizes=dict(raw.get("relative_sizes", {})),
    )


def _canonical_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize geometry range values (``x_range``/``y_range``/``z_range``)
    to tuples so a load(save(G)) roundtrip is semantically equal to G."""
    out = dict(geometry)
    for key in ("x_range", "y_range", "z_range"):
        if key in out and not isinstance(out[key], tuple):
            out[key] = tuple(float(x) for x in out[key])
    return out


def _canonical_pose(pose: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pose ``translation`` to a tuple (semantic roundtrip)."""
    out = dict(pose)
    if "translation" in out and not isinstance(out["translation"], tuple):
        out["translation"] = tuple(float(x) for x in out["translation"])
    return out


def _parse_rule(raw: Mapping[str, Any]) -> ConnectionRuleGenome:
    return ConnectionRuleGenome(
        source_layer=str(raw["source_layer"]),
        source_neuron_type=str(raw["source_neuron_type"]),
        target_layer=str(raw["target_layer"]),
        target_neuron_type=str(raw["target_neuron_type"]),
        mechanism=str(raw["mechanism"]),
    )


def pseudogenome_from_dict(raw: Mapping[str, Any]) -> PseudoGenome:
    """Build a :class:`PseudoGenome` from a JSON-safe dict (``pseudogenome_v1``).

    Unknown schema versions are rejected explicitly (no silent
    interpretation as v1); a registered migration path is the only way to
    consume a future schema.
    """
    declared = str(raw.get("schema_version", PSEUDOGENOME_SCHEMA_VERSION))
    if declared != PSEUDOGENOME_SCHEMA_VERSION:
        raise ValueError(
            f"PseudoGenome declares schema_version={declared!r}; "
            f"this package only supports {PSEUDOGENOME_SCHEMA_VERSION!r}. "
            f"Register a migration path to load future schemas."
        )
    areas = []
    for area_raw in raw.get("areas", []):
        areas.append(
            AreaGenome(
                name=str(area_raw["name"]),
                layers=tuple(_parse_layer(l) for l in area_raw.get("layers", [])),
                inter_connections=tuple(_parse_rule(c) for c in area_raw.get("inter_connections", [])),
                pose=_canonical_pose(dict(area_raw.get("pose", {}))),
            )
        )
    return PseudoGenome(
        name=str(raw["name"]),
        schema_version=declared,
        description=str(raw.get("description", "")),
        areas=tuple(areas),
        area_connections=tuple(dict(c) for c in raw.get("area_connections", [])),
        development_parameters=dict(raw.get("development_parameters", {})),
    )


def genomes_dir() -> Path:
    """Path to the shipped canonical PseudoGenome JSON library."""
    return Path(__file__).resolve().parent / "genomes"


def list_canonical_pseudogenomes() -> list[str]:
    """Names (without ``.json``) of every canonical PseudoGenome shipped."""
    return sorted(p.stem for p in genomes_dir().glob("*.json"))


def load_pseudogenome(path: str | Path | Mapping[str, Any]) -> PseudoGenome:
    """Load a PseudoGenome from a JSON file (or dict)."""
    if isinstance(path, Mapping):
        return pseudogenome_from_dict(path)
    raw = load_json(Path(path))
    return pseudogenome_from_dict(raw)


def load_canonical_pseudogenome(name: str) -> PseudoGenome:
    """Load a canonical PseudoGenome by name from :func:`genomes_dir`."""
    stem = name[:-5] if name.endswith(".json") else name
    path = genomes_dir() / f"{stem}.json"
    if not path.exists():
        available = ", ".join(list_canonical_pseudogenomes())
        raise FileNotFoundError(
            f"No canonical PseudoGenome named {stem!r} in {genomes_dir()}. "
            f"Available: {available}"
        )
    return load_pseudogenome(path)


def save_pseudogenome(genome: PseudoGenome, path: str | Path) -> str:
    """Serialize a PseudoGenome as JSON (data, never code).

    ``schema_version`` is preserved verbatim from the in-memory genome;
    serialization never silently upgrades or downgrades schema versions.
    """
    payload = dict(_area_to_dict_outer(genome))
    payload["schema_version"] = genome.schema_version
    save_json(payload, Path(path))
    return str(path)


def _area_to_dict_outer(genome: PseudoGenome) -> dict[str, Any]:
    return {
        "name": genome.name,
        "description": genome.description,
        "development_parameters": dict(genome.development_parameters),
        "areas": [_area_to_dict(a) for a in genome.areas],
        "area_connections": [dict(c) for c in genome.area_connections],
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_genome(genome: PseudoGenome) -> None:
    """Validate genome structure; raise ``ValueError`` on any violation."""
    if not genome.name:
        raise ValueError("PseudoGenome.name must be non-empty")
    if not genome.areas:
        raise ValueError("PseudoGenome must declare at least one area")
    area_names: set[str] = set()
    for area in genome.areas:
        if area.name in area_names:
            raise ValueError(f"duplicate area name {area.name!r}")
        area_names.add(area.name)
        if not area.layers:
            raise ValueError(f"area {area.name!r} must declare at least one layer")
        layer_names: set[str] = set()
        for layer in area.layers:
            if layer.name in layer_names:
                raise ValueError(f"duplicate layer name {layer.name!r} in area {area.name!r}")
            layer_names.add(layer.name)
            if layer.n_neurons <= 0:
                raise ValueError(
                    f"layer {layer.name!r}: n_neurons must be positive, got {layer.n_neurons}"
                )
            lo, hi = layer.depth_band
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"layer {layer.name!r}: depth_band must satisfy 0 <= lo < hi <= 1, "
                    f"got {layer.depth_band}"
                )
            total = sum(float(f) for f in layer.cell_type_fractions.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"layer {layer.name!r}: cell_type_fractions must sum to 1, got {total}"
                )
            for ct, frac in layer.cell_type_fractions.items():
                if not (0.0 <= float(frac) <= 1.0):
                    raise ValueError(
                        f"layer {layer.name!r}: cell type {ct!r} fraction must be in "
                        f"[0, 1], got {frac}"
                    )
            for ct, (tlo, thi) in layer.fraction_tolerance.items():
                if ct not in layer.cell_type_fractions:
                    raise ValueError(
                        f"layer {layer.name!r}: tolerance for undeclared cell type {ct!r}"
                    )
                if not (0.0 <= tlo <= thi <= 1.0):
                    raise ValueError(
                        f"layer {layer.name!r}: tolerance band {ct!r} must satisfy "
                        f"0 <= lo <= hi <= 1, got {(tlo, thi)}"
                    )
                base = float(layer.cell_type_fractions[ct])
                if not (tlo <= base <= thi):
                    raise ValueError(
                        f"layer {layer.name!r}: base fraction {ct!r}={base} lies outside "
                        f"its declared tolerance band {(tlo, thi)}"
                    )
            frac_lower_sum = sum(
                float(layer.fraction_tolerance.get(ct, (f, f))[0])
                for ct, f in layer.cell_type_fractions.items()
            )
            frac_upper_sum = sum(
                float(layer.fraction_tolerance.get(ct, (f, f))[1])
                for ct, f in layer.cell_type_fractions.items()
            )
            if not (frac_lower_sum <= 1.0 + 1e-9 and frac_upper_sum >= 1.0 - 1e-9):
                raise ValueError(
                    f"layer {layer.name!r}: tolerance bands jointly infeasible "
                    f"(sum lo={frac_lower_sum:.4f}, sum hi={frac_upper_sum:.4f}); "
                    f"a feasible probability vector requires sum(lo) <= 1 <= sum(hi)"
                )
        layer_map = {l.name: l for l in area.layers}
        for rule in area.inter_connections:
            for role in ("source_layer", "target_layer"):
                if getattr(rule, role) not in layer_map:
                    raise ValueError(
                        f"area {area.name!r}: connection rule references unknown "
                        f"{role} {getattr(rule, role)!r}"
                    )
            for role in ("source_neuron_type", "target_neuron_type"):
                lname = (
                    rule.source_layer
                    if role == "source_neuron_type"
                    else rule.target_layer
                )
                if getattr(rule, role) not in layer_map[lname].cell_type_fractions:
                    raise ValueError(
                        f"area {area.name!r}: connection rule references unknown "
                        f"{role} {getattr(rule, role)!r} in layer {lname!r}"
                    )

    for raw in genome.area_connections:
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"area_connections entries must be mappings, got {type(raw).__name__}"
            )
        for area_role in ("source_area", "target_area"):
            aname = str(raw.get(area_role, ""))
            if aname not in area_names:
                raise ValueError(
                    f"area_connections entry references unknown {area_role} {aname!r}"
                )
        for area_role, layer_role in (
            ("source_area", "source_layer"),
            ("target_area", "target_layer"),
        ):
            aname = str(raw.get(area_role, ""))
            lname = str(raw.get(layer_role, ""))
            layer_map = {l.name: l for l in next(
                a for a in genome.areas if a.name == aname
            ).layers}
            if lname not in layer_map:
                raise ValueError(
                    f"area_connections entry references unknown {layer_role} {lname!r} "
                    f"in area {aname!r}"
                )
        for area_role, layer_role, nt_role in (
            ("source_area", "source_layer", "source_neuron_type"),
            ("target_area", "target_layer", "target_neuron_type"),
        ):
            aname = str(raw.get(area_role, ""))
            lname = str(raw.get(layer_role, ""))
            nt = str(raw.get(nt_role, ""))
            layer = next(
                l for a in genome.areas if a.name == aname
                for l in a.layers if l.name == lname
            )
            if nt not in layer.cell_type_fractions:
                raise ValueError(
                    f"area_connections entry references unknown {nt_role} {nt!r} "
                    f"in layer {lname!r}"
                )


def declared_constraints(genome: PseudoGenome) -> dict[str, Any]:
    """Machine-readable genome-level constraints (used by tests and audits).

    Returns a dict with per-area, per-layer exact neuron counts and integer
    cell-type count bands derived from the declared fraction tolerance.
    """
    out: dict[str, Any] = {"schema_version": genome.schema_version, "areas": {}}
    for area in genome.areas:
        area_entry: dict[str, Any] = {"layers": {}}
        for layer in area.layers:
            bands: dict[str, list[int]] = {}
            for ct, frac in layer.cell_type_fractions.items():
                tol = layer.fraction_tolerance.get(ct, (frac, frac))
                bands[ct] = [
                    int(math.floor(layer.n_neurons * tol[0])),
                    int(math.ceil(layer.n_neurons * tol[1])),
                ]
            area_entry["layers"][layer.name] = {
                "n_neurons": layer.n_neurons,
                "cell_type_count_bands": bands,
            }
        out["areas"][area.name] = area_entry
    return out


# --------------------------------------------------------------------------- #
# Development
# --------------------------------------------------------------------------- #


def _project_box_simplex(
    z: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
) -> list[float]:
    """Project ``z`` onto the box-constrained simplex.

    ``C = {p : sum_i p_i = 1, lo_i <= p_i <= hi_i}``.

    Deterministic KKT bisection on the dual variable ``mu``:

        p_i = clip(z_i - mu, lo_i, hi_i),   sum(p) = 1.

    ``p`` is non-increasing in ``mu``, so a simple bisection on ``f(mu) =
    sum(clip(z - mu, lo, hi)) - 1`` converges deterministically. Requires
    ``sum(lo) <= 1 <= sum(hi)`` (joint feasibility, enforced by
    ``validate_genome``); raises ``ValueError`` otherwise.
    """
    n = len(z)
    lo_f = [float(x) for x in lo]
    hi_f = [float(x) for x in hi]
    if sum(lo_f) > 1.0 + 1e-9 or sum(hi_f) < 1.0 - 1e-9:
        raise ValueError(
            f"box-constrained simplex infeasible: sum(lo)={sum(lo_f):.4f}, "
            f"sum(hi)={sum(hi_f):.4f}"
        )

    def f(mu: float) -> float:
        return sum(min(max(z[i] - mu, lo_f[i]), hi_f[i]) for i in range(n)) - 1.0

    mu_lo = min(z) - max(hi_f)
    mu_hi = max(z) - min(lo_f)
    for _ in range(200):
        mid = 0.5 * (mu_lo + mu_hi)
        if f(mid) > 0.0:
            mu_lo = mid
        else:
            mu_hi = mid
    mu = 0.5 * (mu_lo + mu_hi)
    p = [min(max(z[i] - mu, lo_f[i]), hi_f[i]) for i in range(n)]
    s = sum(p)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"box-simplex projection did not converge (sum={s:.12f})")
    return p


def _allocate_counts(
    layer: LayerGenome,
    sigma: float,
    key: jax.Array,
) -> dict[str, int]:
    """Allocate exact integer cell-type counts for ``n_neurons`` neurons.

    With declared tolerance bands and ``sigma > 0``, base fractions are
    jittered with Gaussian noise (deterministic in the PRNG key) and then
    projected onto the box-constrained simplex (bands + sum-to-one) by
    ``_project_box_simplex``; counts are then allocated by the
    largest-remainder method so they sum exactly to ``n_neurons``. Without
    declared tolerance, base fractions are used exactly.
    """
    n = layer.n_neurons
    types = list(layer.cell_type_fractions.keys())
    base = {ct: float(layer.cell_type_fractions[ct]) for ct in types}
    bands = layer.fraction_tolerance

    if sigma > 0.0 and bands:
        z = jax.random.normal(key, shape=(len(types),))
        lo = [
            bands[ct][0] if ct in bands else float(base[ct]) for ct in types
        ]
        hi = [
            bands[ct][1] if ct in bands else float(base[ct]) for ct in types
        ]
        jittered = [base[ct] + sigma * float(z[i]) for i, ct in enumerate(types)]
        p = _project_box_simplex(jittered, lo, hi)
        weighted = {ct: n * p[i] for i, ct in enumerate(types)}
    else:
        weighted = {ct: n * base[ct] for ct in types}

    floors = {ct: int(w) for ct, w in weighted.items()}
    remainder = n - sum(floors.values())
    if remainder < 0:
        raise ValueError(
            f"layer {layer.name!r}: fractional allocation exceeds n_neurons={n}"
        )
    if remainder > 0:
        order = sorted(types, key=lambda ct: weighted[ct] - floors[ct], reverse=True)
        for i, ct in enumerate(order):
            if i >= remainder:
                break
            floors[ct] += 1
    return floors


def _check_realized_constraints(genome: PseudoGenome, tensor: NeuronalTensor) -> None:
    """Assert the developed tensor satisfies the genome-declared constraints."""
    for area in genome.areas:
        for layer in area.layers:
            realized = next(
                (l for a in tensor.areas if a.name == area.name for l in a.layers if l.name == layer.name),
                None,
            )
            if realized is None:
                raise ValueError(f"developed tensor missing layer {layer.name!r}")
            if realized.n_neurons != layer.n_neurons:
                raise ValueError(
                    f"layer {layer.name!r}: developed {realized.n_neurons} neurons, "
                    f"genome declares {layer.n_neurons}"
                )
            counts = {nt.name: 0 for nt in realized.neuron_types}
            for nt in realized.neuron_types:
                counts[nt.name] = int(round(realized.n_neurons * (nt.fraction or 0.0)))
            if sum(counts.values()) != realized.n_neurons:
                raise ValueError(f"layer {layer.name!r}: counts do not sum to n_neurons")
            for ct, frac in layer.cell_type_fractions.items():
                tol = layer.fraction_tolerance.get(ct, (frac, frac))
                lo = int(math.floor(layer.n_neurons * tol[0]))
                hi = int(math.ceil(layer.n_neurons * tol[1]))
                if not (lo <= counts.get(ct, 0) <= hi):
                    raise ValueError(
                        f"layer {layer.name!r} cell type {ct!r}: count {counts.get(ct, 0)} "
                        f"outside declared band [{lo}, {hi}]"
                    )


def develop(
    genome: PseudoGenome,
    seed: int = 0,
    *,
    development_parameters: Optional[Mapping[str, Any]] = None,
) -> NeuronalTensor:
    """Develop a PseudoGenome into a NeuronalTensor.

    Parameters
    ----------
    genome:
        The generative specification.
    seed:
        Development PRNG key (domain ``K_D``). The same genome and seed
        reproduce the exact same phenotype; different seeds realize different
        phenotypes within the genome-declared constraint bands.
    development_parameters:
        Optional overrides merged over the genome's declared development
        parameters (e.g. ``{"fraction_jitter_sigma": 0.0}`` to disable
        population-composition jitter).

    Returns
    -------
    NeuronalTensor
        The terminal structural phenotype, carrying development provenance
        (genome identity hash, schema version, development seed, development
        parameters, phenotype hash) in ``tensor.provenance``.

    Notes
    -----
    The returned tensor is a build-time specification: geometry positions and
    edge realization are resolved by the ordinary ``construct``/``simulate``
    pipeline under the runtime PRNG domain ``K_S`` (``RuntimeConfiguration``).
    Development determines structure; construction realizes it.
    """
    validate_genome(genome)
    params = dict(genome.development_parameters)
    if development_parameters is not None:
        params.update(development_parameters)
    sigma = float(params.get("fraction_jitter_sigma", 0.0))

    key = jax.random.PRNGKey(int(seed))
    areas: list[Area] = []
    for area in genome.areas:
        key, area_key = jax.random.split(key)
        layers: list[Layer] = []
        for layer in area.layers:
            area_key, layer_key = jax.random.split(area_key)
            counts = _allocate_counts(layer, sigma, layer_key)
            total = sum(counts.values())
            neuron_types = [
                NeuronType.make(
                    ct,
                    relative_size=layer.relative_sizes.get(ct),
                    fraction=count / total,
                )
                for ct, count in counts.items()
            ]
            geom_raw = dict(layer.geometry)
            x_range = tuple(float(v) for v in geom_raw.get("x_range", (0.0, 1.0)))
            y_range = tuple(float(v) for v in geom_raw.get("y_range", (0.0, 1.0)))
            layers.append(
                Layer(
                    name=layer.name,
                    neuron_types=neuron_types,
                    geometry=Geometry3D(
                        distribution=str(geom_raw.get("distribution", "uniform_random")),
                        x_range=x_range,
                        y_range=y_range,
                        z_range=tuple(float(v) for v in layer.depth_band),
                        value_tag="relative",
                    ),
                    n_neurons=layer.n_neurons,
                )
            )
        connections = [
            InterConnection(
                source_layer=c.source_layer,
                source_neuron_type=c.source_neuron_type,
                target_layer=c.target_layer,
                target_neuron_type=c.target_neuron_type,
                mechanism=c.mechanism,
            )
            for c in area.inter_connections
        ]
        pose_raw = dict(area.pose)
        pose = Pose3D(
            plane=str(pose_raw.get("plane", "xy")),
            rotation_deg=float(pose_raw.get("rotation_deg", 0.0)),
            translation=tuple(float(v) for v in pose_raw.get("translation", (0.0, 0.0, 0.0))),
            value_tag="relative",
        )
        areas.append(
            Area(
                name=area.name,
                layers=layers,
                inter_connections=connections,
                pose=pose,
            )
        )

    area_connections: list[AreaConnection] = []
    for raw in genome.area_connections:
        area_connections.append(
            AreaConnection(
                source_area=str(raw["source_area"]),
                source_layer=str(raw["source_layer"]),
                source_neuron_type=str(raw["source_neuron_type"]),
                target_area=str(raw["target_area"]),
                target_layer=str(raw["target_layer"]),
                target_neuron_type=str(raw["target_neuron_type"]),
                mechanism=str(raw.get("mechanism", "monotonic_cable_synapse")),
            )
        )

    tensor = NeuronalTensor(
        areas=areas,
        area_connections=area_connections,
        name=genome.name,
        provenance={
            "genome": genome.name,
            "genome_sha256": genome_rules_hash(genome),
            "schema_version": genome.schema_version,
            "development_seed": int(seed),
            "development_parameters": dict(params),
            "phenotype_sha256": None,  # set below
        },
    )
    tensor.provenance["phenotype_sha256"] = phenotype_sha256(tensor)
    _check_realized_constraints(genome, tensor)
    return tensor