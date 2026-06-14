"""Tensor-first HPC/JAX architecture contracts for jaxfne.

This module is intentionally experimental. It records the target API boundaries
for the 0.3.28+ refactor without changing the current stable behavior.

Every not-yet-implemented method raises ``NotImplementedError(">TBI-not-ready")``
so unfinished APIs cannot silently succeed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

_TBI = ">TBI-not-ready"


def _tbi() -> None:
    raise NotImplementedError(_TBI)


@dataclass(frozen=True)
class RuntimeStatic:
    """Hashable static runtime spec for JIT compilation.

    This object must contain only compile-time values: shape-relevant integers,
    dtype policy, backend flags, synaptic kernel names, and feature switches.
    Dynamic arrays such as drives, parameters, or PRNG keys must be passed as
    separate JAX arrays to compiled functions.
    """

    n_steps: int
    dt_ms: float
    dtype: str = "float32"
    recurrent_backend: str = "edge_list"
    synaptic_kernel: str = "exponential"
    record_sources: bool = True
    record_fields: bool = True

    def validate(self) -> dict[str, Any]:
        """Return JSON-safe validation diagnostics for static runtime fields."""
        if self.n_steps <= 0:
            raise ValueError("runtime_static.n_steps must be positive")
        if self.dt_ms <= 0:
            raise ValueError("runtime_static.dt_ms must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("runtime_static.dtype must be 'float32' or 'float64'")
        return {"valid": True, "dtype": self.dtype, "n_steps": int(self.n_steps)}


@dataclass(frozen=True)
class CircuitSpec:
    """Declarative circuit sub-spec for Config.

    Contains cell parameters, mechanisms, connection rules, lesions, selectors,
    and geometry/cell placement metadata. This is config data, not compiled arrays.
    Raw ndarrays are not allowed unless explicitly encoded as runtime-only values
    or referenced through artifact_ref records with SHA256.
    """

    cell_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    mechanisms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    connections: tuple[Mapping[str, Any], ...] = ()
    lesions: tuple[Mapping[str, Any], ...] = ()

    def validate(self) -> dict[str, Any]:
        """Validate JSON-safety and finite scalar cell/mechanism parameters."""
        _tbi()


@dataclass(frozen=True)
class ProbeSpec:
    """Probe/readout declaration.

    Describes readout kind, contact count, geometry, units/status, and proxy/solver
    metadata. It must not imply calibrated physical amplitude unless explicitly
    backed by future solver/calibration evidence.
    """

    name: str
    kind: str
    n_contacts: int = 16
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        """Validate contact count, readout kind, and proxy-status metadata."""
        _tbi()


@dataclass(frozen=True)
class Config:
    """Canonical future config object.

    Config is the bio-circuit PCB sketch. It owns only declarative specs and
    JSON-safe metadata. It does not own compiled JAX arrays, optimizer state, or
    plotted artifacts.
    """

    schema_version: str = "0.3.28-hpc-contract"
    runtime: Mapping[str, Any] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    circuit: CircuitSpec = field(default_factory=CircuitSpec)
    probes: tuple[ProbeSpec, ...] = ()
    paradigm: Mapping[str, Any] = field(default_factory=dict)
    objective: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_runtime(self, **kwargs: Any) -> "Config":
        """Return a copy with merged runtime metadata."""
        _tbi()

    def with_circuit(self, circuit: CircuitSpec) -> "Config":
        """Return a copy with a new CircuitSpec."""
        _tbi()

    def with_probes(self, *probes: ProbeSpec) -> "Config":
        """Return a copy with probe declarations."""
        _tbi()

    def validate(self) -> dict[str, Any]:
        """Return strict JSON-safe schema validation report."""
        _tbi()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to strict JSON-safe dict with schema_version."""
        _tbi()


@dataclass(frozen=True)
class NodeIdentity:
    """Stable node identity for selector-addressable circuits."""

    global_id: int
    area: str
    area_id: str
    local_id: int
    layer: str
    cell_type: str

    @property
    def quartet(self) -> str:
        """Return canonical ``area_id:local_id:layer:cell_type`` label."""
        return f"{self.area_id}:{self.local_id:06d}:{self.layer}:{self.cell_type}"


@dataclass(frozen=True)
class SelectorSpec:
    """Selector over area/layer/cell-type/id fields."""

    area: Optional[str] = None
    area_id: Optional[str] = None
    layer: Optional[str] = None
    cell_type: Optional[str] = None
    ids: Optional[tuple[int, ...]] = None

    def resolve(self, neuron_table: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> jax.Array:
        """Resolve selector to integer node row indices (strict AND semantics).

        Each row in ``neuron_table`` must match ALL specified fields
        (``area``/``area_id``/``layer``/``cell_type`` by equality or membership,
        ``ids`` by ``neuron_id`` membership). Input row order is preserved.

        A requested field that is absent from a row raises ``KeyError`` rather
        than matching/skipping silently — selectors never invent metadata.

        Returns an int32 JAX array of row positions. Empty results raise
        ``ValueError`` unless ``allow_empty=True`` (then an empty array).
        """

        def _match(row: Mapping[str, Any], field_name: str, wanted: Any) -> bool:
            if field_name not in row:
                raise KeyError(
                    f"SelectorSpec.resolve requires neuron_table field {field_name!r}; "
                    f"row {row.get('neuron_id', '?')} is missing it"
                )
            value = row[field_name]
            if isinstance(wanted, (list, tuple, set, frozenset)):
                return value in wanted
            return value == wanted

        matches: list[int] = []
        for i, row in enumerate(neuron_table):
            if self.area is not None and not _match(row, "area", self.area):
                continue
            if self.area_id is not None and not _match(row, "area_id", self.area_id):
                continue
            if self.layer is not None and not _match(row, "layer", self.layer):
                continue
            if self.cell_type is not None and not _match(row, "cell_type", self.cell_type):
                continue
            if self.ids is not None and not _match(row, "neuron_id", self.ids):
                continue
            matches.append(i)

        if not matches and not allow_empty:
            raise ValueError(f"SelectorSpec matched no neurons: {self!r}")

        return jnp.asarray(matches, dtype=jnp.int32)


@dataclass(frozen=True)
class MechanismSpec:
    """Synaptic/receptor mechanism metadata for connection rules."""

    name: str
    kind: str
    sign: float
    g: float = 1.0
    tau_ms: float = 2.0
    plasticity: float = 0.0

    def validate(self) -> dict[str, Any]:
        """Validate finite mechanism fields and sign/kind consistency."""
        _tbi()


@dataclass(frozen=True)
class WeightInitSpec:
    """Weight initialization descriptor.

    Supported future modes: scalar, random_uniform, random_normal, matrix,
    artifact_ref. Matrix mode is runtime-only. Artifact refs validate path,
    array name, and SHA256 before construction.
    """

    mode: str
    value: Any = None
    low: float = 0.0
    high: float = 1.0
    scale: float = 1.0
    seed: int = 0
    path: Optional[str] = None
    array_name: Optional[str] = None
    sha256: Optional[str] = None

    def materialize(self, shape: tuple[int, int], *, dtype: str = "float32", artifact_root: Optional[str] = None) -> jax.Array:
        """Materialize a weight matrix outside JIT."""
        _tbi()


@dataclass(frozen=True)
class ConnectionRuleSpec:
    """Config connection rule: pre selector -> post selector + mechanism + weight."""

    name: str
    pre: SelectorSpec
    post: SelectorSpec
    mechanism: str
    weight: WeightInitSpec
    pattern: str = "all_to_all"
    probability: float = 1.0
    allow_empty: bool = False
    plasticity: Mapping[str, Any] = field(default_factory=dict)
    control_key: Optional[str] = None

    def validate(self, mechanisms: Mapping[str, MechanismSpec]) -> dict[str, Any]:
        """Validate selectors, mechanism reference, pattern, probability, and weight."""
        _tbi()


@dataclass(frozen=True)
class ConnectionCompileResult:
    """Compiled sparse edge representation from connection rules."""

    edge_pre: jax.Array
    edge_post: jax.Array
    edge_weight: jax.Array
    edge_mechanism: jax.Array
    connection_table: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]


def compile_connection_rules(
    neuron_table: Sequence[Mapping[str, Any]],
    connection_rules: Sequence[ConnectionRuleSpec],
    mechanisms: Mapping[str, MechanismSpec],
    *,
    seed: int = 0,
    dtype: str = "float32",
    artifact_root: Optional[str] = None,
) -> ConnectionCompileResult:
    """Compile selector-based connection rules to sparse edge arrays.

    Must avoid dense O(N^2) intermediate matrices except for explicit matrix mode.
    All randomness must use JAX PRNG keys derived from ``seed``.
    """
    _tbi()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FlatNet:
    """JAX-native flat network representation for JIT/vmap/pmap.

    All fields except ``static`` are JAX arrays. Metadata maps live outside traced
    numerical kernels.
    """

    neuron_params: jax.Array
    positions: jax.Array
    area_id: jax.Array
    layer_id: jax.Array
    cell_type_id: jax.Array
    edge_pre: jax.Array
    edge_post: jax.Array
    edge_weight: jax.Array
    edge_mechanism: jax.Array
    probe_positions: jax.Array
    static: Mapping[str, Any] = field(default_factory=dict)

    def tree_flatten(self):
        """Documented public function `tree_flatten`."""
        children = (
            self.neuron_params,
            self.positions,
            self.area_id,
            self.layer_id,
            self.cell_type_id,
            self.edge_pre,
            self.edge_post,
            self.edge_weight,
            self.edge_mechanism,
            self.probe_positions,
        )
        return children, dict(self.static)

    @classmethod
    def tree_unflatten(cls, aux_data: Mapping[str, Any], children: tuple[Any, ...]) -> "FlatNet":
        """Documented public function `tree_unflatten`."""
        return cls(*children, static=aux_data)

    @property
    def n_nodes(self) -> int:
        """Return node count from flat arrays."""
        return int(self.positions.shape[0])

    @property
    def n_edges(self) -> int:
        """Return edge count from flat arrays."""
        return int(self.edge_pre.shape[0])


@dataclass(frozen=True)
class TrackingMaps:
    """Static maps linking FlatNet rows back to semantic identities."""

    row_to_quartet: tuple[str, ...]
    area_vocab: Mapping[str, int]
    layer_vocab: Mapping[str, int]
    cell_type_vocab: Mapping[str, int]
    mechanism_vocab: Mapping[str, int]
    edge_rule_origin: tuple[str, ...] = ()

    def validate(self, flat: FlatNet) -> dict[str, Any]:
        """Validate one-to-one row tracking and edge provenance lengths."""
        _tbi()


@dataclass(frozen=True)
class Net:
    """Compiled user-facing biophysical circuit.

    Net owns static tables, semantic metadata, and wrappers. Its numerical path
    should delegate to FlatNet pure functions.
    """

    config: Config
    flat: FlatNet
    maps: TrackingMaps
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def initialize_state(self, *, key: Optional[jax.Array] = None) -> Mapping[str, jax.Array]:
        """Return initial dynamic state arrays for simulation."""
        _tbi()

    def simulate(self, paradigm: "Paradigm", *, runtime_static: RuntimeStatic, key: jax.Array) -> "Signals":
        """Simulate this Net under a Paradigm and return queryable Signals."""
        _tbi()

    def select(self, **selector: Any) -> jax.Array:
        """Resolve semantic selectors to node rows."""
        _tbi()

    def neuron_table(self) -> tuple[Mapping[str, Any], ...]:
        """Return JSON-safe neuron table with quartet identities."""
        _tbi()

    def connection_table(self) -> tuple[Mapping[str, Any], ...]:
        """Return JSON-safe connection table with rule provenance."""
        _tbi()

    def weights(self, *, format: str = "edge") -> Any:
        """Return edge weights or opt-in dense matrix."""
        _tbi()

    def to_config(self) -> Config:
        """Reconstruct config from this Net without dynamic state."""
        _tbi()

    def to_flat(self) -> tuple[FlatNet, TrackingMaps]:
        """Return FlatNet and tracking maps."""
        return self.flat, self.maps


def construct_net(config: Config) -> Net:
    """Compile Config into Net."""
    _tbi()


def flatten_net(net: Net) -> tuple[FlatNet, TrackingMaps]:
    """Flatten Net for JAX transforms."""
    return net.to_flat()


def unflatten_net(flat: FlatNet, maps: TrackingMaps, *, config: Optional[Config] = None) -> Net:
    """Reconstruct a Net wrapper around FlatNet and maps."""
    _tbi()


def simulate_flat_izhikevich(
    flat: FlatNet,
    runtime_static: RuntimeStatic,
    drive: jax.Array,
    key: jax.Array,
) -> "SignalTensor":
    """JIT-ready flat Izhikevich simulation.

    No Python object mutation, plotting, JSON, file I/O, NumPy, or SciPy calls are
    allowed in this function. ``runtime_static`` must be marked static by the
    caller when jitted.
    """
    _tbi()


@dataclass(frozen=True)
class SignalTensor:
    """JAX-native signal payload with explicit axis layout."""

    data: jax.Array
    layout: str
    name: str
    units_or_status: str = "proxy_units"

    def convert_layout(self, to_layout: str) -> "SignalTensor":
        """Return a layout-converted tensor using named-axis transposition."""
        _tbi()


@dataclass(frozen=True)
class Signals:
    """Queryable simulation output container."""

    tensors: Mapping[str, SignalTensor]
    time_ms: jax.Array
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get(
        self,
        key: str,
        *,
        area: Optional[str] = None,
        layer: Optional[str] = None,
        cell_type: Optional[str] = None,
        trial: Optional[int] = None,
        layout: Optional[str] = None,
        as_numpy: bool = False,
    ) -> Any:
        """Return selected signal tensor in requested layout."""
        _tbi()


def get_signal(obj: Any, key: str, **kwargs: Any) -> Any:
    """Public free-function signal query for Signals, Net outputs, or tutorial trial dicts."""
    _tbi()


@dataclass(frozen=True)
class Paradigm:
    """Task/trial/stimulus program for a Net."""

    name: str
    conditions: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def constant_dc(cls, *, target: Mapping[str, Any], amplitude: float, duration_ms: float) -> "Paradigm":
        """Build minimal constant-drive paradigm needed for construct->simulate gates."""
        _tbi()

    def compile_drive(self, net: Net, runtime_static: RuntimeStatic) -> jax.Array:
        """Compile events/stimuli into a drive array outside the JIT kernel."""
        _tbi()


@dataclass(frozen=True)
class ObjectiveOutputSpec:
    """One objective output metric or gate."""

    name: str
    source: str
    metric: str
    target: Optional[float] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    weight: float = 1.0
    selector: Mapping[str, Any] = field(default_factory=dict)
    gate: Optional[str] = None

    def evaluate(self, signals: Signals, net: Optional[Net] = None) -> Mapping[str, Any]:
        """Evaluate metric/gate against signals."""
        _tbi()


@dataclass(frozen=True)
class ObjectiveResult:
    """Aggregated objective result."""

    score: float
    metrics: Mapping[str, Any]
    gates: Mapping[str, bool]
    status: str = "computed_proxy_objective"


@dataclass(frozen=True)
class TrainingResult:
    """JSON-safe training result bundle."""

    best_config: Config
    best_parameters: Mapping[str, float]
    best_score: float
    history: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    validation: Mapping[str, Any]
    truth_gates: Mapping[str, Any]

    def save(self, path: str) -> None:
        """Save strict JSON training result."""
        _tbi()


class Trainer:
    """Base trainer contract."""

    @classmethod
    def from_config(cls, config: Config) -> "Trainer":
        """Create trainer from Config optimizer/trainable/objective specs."""
        _tbi()

    def apply_candidate(self, config: Config, candidate: Mapping[str, float]) -> Config:
        """Return candidate-updated Config."""
        _tbi()

    def evaluate(self, net: Net, objective: Sequence[ObjectiveOutputSpec], paradigm: Paradigm, *, key: jax.Array) -> ObjectiveResult:
        """Run simulation and objective evaluation for one candidate."""
        _tbi()

    def fit(self, net_or_config: Any, *, key: Optional[jax.Array] = None) -> TrainingResult:
        """Run optimizer loop and return best config and metrics."""
        _tbi()


def weld_config(
    *configs: Config,
    duplicate_policy: str = "suffix",
    preserve_connections: bool = True,
    strict_runtime: bool = True,
) -> Config:
    """Merge Config objects with deterministic area renaming.

    Welding preserves each component's internal connections. It does not create
    cross-connections between components; users add explicit rules after welding.
    """
    _tbi()
