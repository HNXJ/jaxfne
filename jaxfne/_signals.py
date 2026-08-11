"""Signals/Objective schema, trial specs, and paradigm-adjacent readout types.

Split out of ``jaxfne/core.py`` (second slice of the core.py monolith split,
see ``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports every
symbol here for backward compatibility -- import from ``jaxfne.core``, not
this module, unless you are working on core.py itself.

One-directional dependency: this module is consumed by ``Model``/``construct``
(still in core.py) via the metric/objective-evaluation helpers below; nothing
here calls back into ``Model`` or ``Configuration``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .fields import FieldOutput
from .io import config_hash, json_safe
from .experimental_hpc.contracts import SelectorSpec
from ._runtime_config import RuntimeConfig
from .paradigm import (
    ParadigmEvent,
    ParadigmCondition,
    Paradigm,
    paradigm,
    evoked_l4_drive_paradigm,
    omission_oddball_paradigm,
    coop_omission_oddball_paradigm,
    coop_omission_oddball_for_model,
    coop_omission_oddball_for_neuronal_tensor,
    paradigm_target_indices_from_model,
    general_sequential_oddball_paradigm,
    general_delayed_match_to_sample_paradigm,
)


@dataclass(frozen=True)
class Simulation:
    """Immutable specification of one simulation run.

    Fields: ``duration_ms`` and ``dt_ms`` (milliseconds; both must be positive
    and finite), ``seed`` (PRNG seed), ``plasticity`` (plasticity gain),
    ``record_sources``/``record_fields`` (recording toggles), ``poisson_drive``
    (optional drive spec), ``runtime`` (:class:`RuntimeConfig` override), and
    ``ablation`` (optional ablation label). ``n_steps`` is derived as
    ``round(duration_ms / dt_ms)`` and must be > 0.
    """

    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True
    poisson_drive: Optional[dict] = None
    runtime: RuntimeConfig | None = None
    ablation: Optional[str] = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.duration_ms) and self.duration_ms > 0):
            raise ValueError(
                f"Simulation.duration_ms must be positive and finite; got {self.duration_ms!r}"
            )
        if not (math.isfinite(self.dt_ms) and self.dt_ms > 0):
            raise ValueError(
                f"Simulation.dt_ms must be positive and finite; got {self.dt_ms!r}"
            )
        n = int(round(self.duration_ms / self.dt_ms))
        if n <= 0:
            raise ValueError(
                f"Simulation produces n_steps={n} <= 0 for "
                f"duration_ms={self.duration_ms}, dt_ms={self.dt_ms}"
            )

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_ms / self.dt_ms))

    @property
    def resolved_runtime(self) -> RuntimeConfig:
        base = self.runtime or RuntimeConfig(seed=self.seed, n_steps=self.n_steps)
        return replace(base, seed=self.seed, n_steps=self.n_steps)

    def with_plasticity(self, gain: float) -> "Simulation":
        return replace(self, plasticity=float(gain))


@dataclass(frozen=True)
class Probe:
    """Immutable declaration of a readout probe.

    ``name`` identifies the probe; ``modes`` is the sequence of requested
    readout keys (e.g. ``"spikes"``, ``"V_m"``, ``"lfp_proxy"``,
    ``"csd_proxy"``); ``metadata`` carries optional JSON-safe annotations. All
    field-derived modes are proxy readouts, not physical measurements.
    """

    name: str
    modes: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# v0.3.29 signal-access key aliases. Maps user-facing keys to the real
# underlying attribute. Only real, present readouts are listed: the core proxy
# field carries lfp/csd/phi_e/source proxies — there is NO eeg/meg/emm readout.
_SIGNALS_GET_KEY_ALIASES: dict[str, str] = {
    "vm": "V_m", "v_m": "V_m", "voltage": "V_m", "V_m": "V_m",
    "spk": "spikes", "spike": "spikes", "spikes": "spikes", "raster": "spikes",
    "src": "sources", "source": "sources", "sources": "sources",
    "lfp": "lfp_proxy", "lfp_proxy": "lfp_proxy",
    "csd": "csd_proxy", "csd_proxy": "csd_proxy",
    "phi_e": "phi_e_proxy", "phi": "phi_e_proxy", "phi_e_proxy": "phi_e_proxy",
    "field_source": "source_proxy", "source_proxy": "source_proxy",
    # Probe-only readouts (not on FieldOutput); aliases normalize config vocabulary.
    # `*_like` names are fully retired (no aliases): use `*_proxy` only.
    "eeg": "eeg_proxy", "eeg_proxy": "eeg_proxy",
    "meg": "meg_proxy", "meg_proxy": "meg_proxy",
    "emm": "emm_proxy", "emm_proxy": "emm_proxy",
}
# Signals whose neuron axis is the trailing axis (length == n_units).
_SIGNALS_GET_NEURON_AXIS_KEYS = frozenset({"V_m", "spikes", "sources"})
# Field readouts are laminar/contact-indexed, not neuron-indexed.
_SIGNALS_GET_FIELD_KEYS = frozenset({"lfp_proxy", "csd_proxy", "phi_e_proxy", "source_proxy"})


@dataclass(frozen=True)
class Signals:
    """Simulation output container holding multiple arrays."""

    time_ms: jax.Array
    V_m: jax.Array
    spikes: jax.Array
    sources: Optional[jax.Array]
    field: Optional[FieldOutput]
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe signal diagnostics for notebooks."""
        from .io import json_safe
        dt_ms = float(self.time_ms[1] - self.time_ms[0]) if self.time_ms.shape[0] > 1 else None
        return json_safe({
            "n_steps": int(self.time_ms.shape[0]),
            "n_units": int(self.V_m.shape[1]) if self.V_m.ndim == 2 else None,
            "dt_ms": dt_ms,
            "spike_count_total": float(jnp.sum(self.spikes)),
            "spike_rate_hz_mean": (
                float(jnp.mean(self.spikes) * (1000.0 / dt_ms)) if dt_ms else None
            ),
            "V_m_mean": float(jnp.mean(self.V_m)),
            "field_status": "present" if self.field is not None else "absent",
            "field_claim_level": self.metadata.get("field_claim_level", "proxy_readout"),
        })

    def get(
        self,
        key: str,
        *,
        selector: "Optional[SelectorSpec]" = None,
        area: Optional[Any] = None,
        layer: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        ids: Optional[Sequence[int]] = None,
        trial: Optional[int] = None,
        as_numpy: bool = False,
    ) -> Any:
        """Return a named signal array, optionally filtered to selected neurons.

        Key aliases (case-sensitive): ``vm``/``V_m``/``voltage`` -> V_m;
        ``spk``/``spikes``/``raster`` -> spikes; ``src``/``sources`` -> sources;
        ``lfp``/``csd``/``phi_e`` -> the corresponding laminar proxy readout;
        ``field_source`` -> field source proxy. Unknown keys raise ``KeyError``
        listing the available ones.

        A selector (either a ``SelectorSpec`` or the ``area``/``layer``/
        ``cell_type``/``ids`` fields, not both) filters neuron-indexed signals
        (V_m, spikes, sources) along their trailing axis. Selectors on laminar
        field readouts raise ``ValueError`` (no declared neuron axis). Selection
        requires ``metadata['neuron_metadata']``; if absent it raises
        ``ValueError`` rather than guessing.

        ``trial`` is not supported: core ``Signals`` has no declared trial axis,
        so any ``trial`` argument raises ``NotImplementedError``.
        """
        if trial is not None:
            raise NotImplementedError(
                "Signals.get(..., trial=...) is not supported: core Signals has no "
                "declared trial axis (V_m is (n_steps, n_units)). Use "
                "jtfne.run_trials(...)/Model.run_trials(...) or "
                "tutorial_utils.simulate_laminar_trials(...) for multi-trial data, "
                "or index the array directly once trial semantics are explicit."
            )

        field_args = (area, layer, cell_type, ids)
        if selector is not None and any(x is not None for x in field_args):
            raise ValueError(
                "Pass either selector=SelectorSpec(...) or the area/layer/cell_type/ids "
                "fields, not both."
            )
        if selector is None and any(x is not None for x in field_args):
            selector = SelectorSpec(
                area=area,
                layer=layer,
                cell_type=cell_type,
                ids=tuple(ids) if ids is not None else None,
            )

        if key not in _SIGNALS_GET_KEY_ALIASES:
            available = ", ".join(sorted(_SIGNALS_GET_KEY_ALIASES))
            raise KeyError(f"Unknown signal key {key!r}. Available: {available}")
        attr = _SIGNALS_GET_KEY_ALIASES[key]

        if attr == "V_m":
            arr = self.V_m
        elif attr == "spikes":
            arr = self.spikes
        elif attr == "sources":
            if self.sources is None:
                raise ValueError("sources not recorded (run with record_sources=True)")
            arr = self.sources
        elif attr in _SIGNALS_GET_FIELD_KEYS:
            if self.field is None:
                raise ValueError(f"field not recorded; {attr!r} unavailable (record_fields=True)")
            arr = getattr(self.field, attr, None)
            if arr is None:
                raise ValueError(f"field output has no attribute {attr!r}")
        else:  # pragma: no cover - alias map and branches kept in sync
            raise KeyError(f"Signal {attr!r} not available")

        if selector is not None:
            if attr not in _SIGNALS_GET_NEURON_AXIS_KEYS:
                raise ValueError(
                    f"Selector cannot be applied to signal {key!r}: {attr!r} is a laminar "
                    f"field readout with no declared neuron axis. Slice it directly instead."
                )
            if getattr(arr, "ndim", 0) < 2:
                raise ValueError(
                    f"Cannot apply selector to {attr!r}: expected a neuron axis "
                    f"(ndim >= 2), got shape {getattr(arr, 'shape', None)}"
                )
            n_units = int(self.V_m.shape[-1]) if self.V_m.ndim >= 2 else None
            if n_units is not None and int(arr.shape[-1]) != n_units:
                raise ValueError(
                    f"Cannot apply selector to {attr!r}: trailing axis {arr.shape[-1]} "
                    f"does not match neuron count {n_units}; no declared neuron axis."
                )
            table = self.metadata.get("neuron_metadata")
            if table is None:
                raise ValueError(
                    "Selector requested but this run has no neuron identity table "
                    "(metadata['neuron_metadata'] is None). Build the model with geometry "
                    "that provides neuron rows, or index by integer position directly."
                )
            indices = selector.resolve(table)
            arr = arr[..., indices]

        if as_numpy:
            import numpy as _np
            arr = _np.asarray(arr)
        return arr


# Backwards-compatible alias.
Signal = Signals


@dataclass(frozen=True)
class Objective:
    """Declarative objective specification: losses, regularizers, and diagnostic gates.

    All specs are stored as plain dicts (no callables) so the objective is always
    JSON-serializable.  Gate pass/fail is a computational diagnostic only — it does
    not imply empirical validation or biological calibration.
    """

    name: str = "anonymous"
    kind: str = "generic"  # "generic", "group_rate_targets", or custom
    losses: list[dict[str, Any]] = field(default_factory=list)
    regularizers: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)

    def loss(
        self,
        name: str,
        target: Optional[float] = None,
        weight: float = 1.0,
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `loss`."""
        spec: dict[str, Any] = {"name": name, "weight": float(weight)}
        if target is not None:
            spec["target"] = float(target)
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, losses=[*self.losses, spec])

    def regularizer(
        self,
        name: str,
        target: float = 0.0,
        weight: float = 1.0,
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `regularizer`."""
        spec: dict[str, Any] = {"name": name, "target": float(target), "weight": float(weight)}
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, regularizers=[*self.regularizers, spec])

    def gate(
        self,
        name: str,
        threshold: Any,
        criterion: str = "below",
        metric: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Objective":
        """Documented public function `gate`."""
        spec: dict[str, Any] = {"name": name, "threshold": threshold, "criterion": str(criterion)}
        if metric is not None:
            spec["metric"] = str(metric)
        if metadata:
            spec["metadata"] = dict(metadata)
        return replace(self, gates=[*self.gates, spec])

    def compose(self, *others: "Objective") -> "Objective":
        """Merge other Objective specs into this one, concatenating all specs."""
        all_losses = list(self.losses)
        all_regularizers = list(self.regularizers)
        all_gates = list(self.gates)
        for other in others:
            all_losses.extend(other.losses)
            all_regularizers.extend(other.regularizers)
            all_gates.extend(other.gates)
        return replace(self, losses=all_losses, regularizers=all_regularizers, gates=all_gates)


@dataclass(frozen=True)
class DatasetSpec:
    """Manifest-safe dataset/comparison declaration for observed data.

    DatasetSpec is a schema object, not a loader.  It records how an external
    dataset is aligned and interpreted so objectives can reference data without
    hard-coding paths or claiming empirical validation.
    """

    name: str = "unnamed_dataset"
    modality: str = "unspecified"
    source_format: str = "unspecified"
    comparison_label: str = "p1"
    comparison_code: int = 101
    sampling_rate_hz: Optional[float] = None
    units: str = "unspecified"
    trial_filter: dict[str, Any] = field(default_factory=dict)
    condition_map: dict[str, list[int]] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_condition_map(self, condition_map: Mapping[str, Sequence[int]]) -> "DatasetSpec":
        """Documented public function `with_condition_map`."""
        mapped = {str(k): [int(x) for x in v] for k, v in condition_map.items()}
        return replace(self, condition_map=mapped)

    def with_quality_gate(self, name: str, value: Any) -> "DatasetSpec":
        """Documented public function `with_quality_gate`."""
        gates = dict(self.quality_gates)
        gates[str(name)] = value
        return replace(self, quality_gates=gates)


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.name:
            issues.append("dataset_name_missing")
        if self.comparison_code is None:
            issues.append("comparison_code_missing")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            issues.append("sampling_rate_hz_must_be_positive")
        return {
            "valid": not issues,
            "issues": issues,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        }

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "modality": self.modality,
            "source_format": self.source_format,
            "comparison_label": self.comparison_label,
            "comparison_code": self.comparison_code,
            "sampling_rate_hz": self.sampling_rate_hz,
            "units": self.units,
            "trial_filter": self.trial_filter,
            "condition_map": self.condition_map,
            "quality_gates": self.quality_gates,
            "metadata": self.metadata,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        })


@dataclass(frozen=True)
class TrialSpec:
    """Specification for a single simulation trial."""

    trial_id: str
    condition: Optional[ParadigmCondition] = None
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "trial_id": self.trial_id,
            "condition": self.condition.to_dict() if self.condition else None,
            "seed": self.seed,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialBatch:
    """A collection of trial specifications to be run."""

    trials: tuple[TrialSpec, ...]
    batch_id: str = "anonymous_batch"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "batch_id": self.batch_id,
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialResult:
    """Result of a single simulation trial."""

    trial_id: str
    condition_label: Optional[str] = None
    signals: Optional[Signals] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation, excluding large JAX arrays."""
        from .io import json_safe
        return json_safe({
            "trial_id": self.trial_id,
            "condition_label": self.condition_label,
            "success": self.success,
            "error_message": self.error_message,
            "signals": self.signals.summary() if self.signals else None,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class TrialBatchResult:
    """Results from a batch of trials."""

    batch_id: str
    results: tuple[TrialResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "batch_id": self.batch_id,
            "n_results": len(self.results),
            "n_success": sum(1 for r in self.results if r.success),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        })




@dataclass(frozen=True)
class ReadoutSpec:
    """Declarative specification for extracting a scalar feature from Signals.

    Defines a single named metric to compute from a simulation's Signals
    object.  All fields are JSON-safe.  No physical-amplitude or calibration
    claim is made; all values are proxy or native-current units unless
    explicitly stated otherwise.

    Supported metrics (``_KNOWN_READOUT_METRICS``):
        spike_rate_hz, spike_count, mean_V_m,
        csd_abs_mean, lfp_abs_mean, source_abs_mean.

    Optional filters:
        time_window_ms: (start_ms, end_ms) tuple for temporal slice.
        n_contacts_slice: (start, end) tuple for contact-depth slice on field modes.
    """

    name: str
    metric: str
    time_window_ms: Optional[tuple[float, float]] = None
    n_contacts_slice: Optional[tuple[int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "metric": self.metric,
            "time_window_ms": list(self.time_window_ms) if self.time_window_ms else None,
            "n_contacts_slice": list(self.n_contacts_slice) if self.n_contacts_slice else None,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ReadoutResult:
    """Result of applying a ReadoutSpec to Signals.

    All scalar values are floats or None (when computation is not applicable).
    JSON-safe via to_dict().

    Status values:
        "computed"   — value was computed successfully.
        "no_field"   — metric requires field output but signals has no field.
        "unknown_metric" — metric not in _KNOWN_READOUT_METRICS.
    """

    spec_name: str
    metric: str
    value: Optional[float]
    status: str = "computed"
    claim_level: str = "computational_scaffold"
    physical_amplitude_calibrated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "spec_name": self.spec_name,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "claim_level": self.claim_level,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "metadata": self.metadata,
        })

    @property
    def name(self) -> str:
        """Compatibility alias for spec_name used by public examples.

        Allows usage like:
            for result in results:
                print(result.name, result.metric, result.value, result.status)
        """
        return self.spec_name



@dataclass(frozen=True)
class ObjectiveReport:
    """Structured, immutable result of evaluating an Objective against Signals.

    Wraps the dict returned by :meth:`Model.evaluate` into a frozen dataclass
    that is always JSON-safe and carries explicit truth gates.

    Gate pass/fail is a computational diagnostic only.  It does not imply
    empirical validation, biological calibration, or mechanism proof.

    ``readout_results`` is populated when ReadoutSpecs are passed to
    :meth:`Model.evaluate_report`.  It is an empty tuple otherwise.
    """

    objective_name: str
    evaluation_status: str
    total_loss: Optional[float]
    all_gates_pass: bool
    losses: tuple[dict[str, Any], ...]
    regularizers: tuple[dict[str, Any], ...]
    gates: tuple[dict[str, Any], ...]
    readout_results: tuple["ReadoutResult", ...] = field(default_factory=tuple)
    truth: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "objective_name": self.objective_name,
            "evaluation_status": self.evaluation_status,
            "total_loss": self.total_loss,
            "all_gates_pass": self.all_gates_pass,
            "losses": list(self.losses),
            "regularizers": list(self.regularizers),
            "gates": list(self.gates),
            "readout_results": [r.to_dict() for r in self.readout_results],
            "truth": self.truth,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class RunReceipt:
    """Complete, JSON-safe record of a single simulation run.

    Captures config fingerprint, simulation parameters, signal summary,
    and truth gates in one immutable object.  ``receipt_id`` is deterministic:
    same configuration + seed + version always yields the same ID.

    Truth status: all gates are frozen at conservative defaults and cannot be
    escalated.  No physical-amplitude, empirical-validation, or mechanism
    claim is introduced by this receipt.
    """

    receipt_id: str
    jaxfne_version: str
    config_hash: str
    simulation: dict[str, Any]
    signals_summary: dict[str, Any]
    truth: dict[str, Any]
    claim_labels: dict[str, Any]
    backend: dict[str, Any]
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "receipt_id": self.receipt_id,
            "jaxfne_version": self.jaxfne_version,
            "config_hash": self.config_hash,
            "simulation": self.simulation,
            "signals_summary": self.signals_summary,
            "truth": self.truth,
            "claim_labels": self.claim_labels,
            "backend": self.backend,
            "tags": self.tags,
        })

import numpy as _np  # used only in StimulusSchedule.to_array; no JAX tracing


def _make_poisson_drive(
    n_steps: int,
    n_neurons: int,
    rate_hz: float,
    amplitude: float,
    dt_ms: float,
    seed: int,
    target: str = "all",
) -> jax.Array:
    """Generate a Poisson stochastic drive array.
    
    Returns (n_steps, n_neurons) float32 array. Each timestep, each neuron
    has an independent Poisson event with probability rate_hz * dt_ms / 1000.
    Events inject `amplitude` native current units. Output is finite and bounded.
    """
    prob = float(rate_hz) * float(dt_ms) / 1000.0
    prob = min(max(prob, 0.0), 1.0)
    key = jax.random.PRNGKey(int(seed))
    noise = jax.random.bernoulli(key, p=prob, shape=(int(n_steps), int(n_neurons)))
    return (jnp.asarray(noise, dtype=jnp.float32) * float(amplitude))


@dataclass(frozen=True)
class StimulusSchedule:
    """Explicit drive schedule for event-aligned stimulus injection.

    Contains an ordered sequence of drive events as JSON-safe dicts with keys:
    ``onset_ms``, ``duration_ms``, ``amplitude``, ``label``, ``is_drive_event``.
    When ``is_drive_event`` is False or ``amplitude`` is 0, the event injects
    zero drive. No physical-amplitude or calibration claim is made; amplitude
    values are native Izhikevich current units.

    An event may optionally carry ``frequency_hz``: when present, the flat
    ``amplitude`` plateau is replaced by ``amplitude * sin(2*pi*frequency_hz*t)``
    with ``t`` measured in seconds since the event's own onset (so every event
    restarts the sinusoid at phase 0). Absent ``frequency_hz`` reproduces the
    original flat-amplitude behavior exactly (backward compatible).
    """

    events: tuple[dict[str, Any], ...]
    n_neurons: int
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"

    def to_array(self, n_steps: int, dt_ms: float, dtype: str = "float32") -> "jax.Array":
        """Materialize a ``(n_steps, n_neurons)`` drive schedule array.

        Supports optional target_indices in event to apply amplitude only to selected neurons.
        """
        schedule = _np.zeros((int(n_steps), int(self.n_neurons)), dtype=_np.float32)
        for ev in self.events:
            if not ev.get("is_drive_event", True):
                continue
            amp = float(ev.get("amplitude", 0.0))
            if amp == 0.0:
                continue
            onset_ms = float(ev.get("onset_ms", 0.0))
            dur_ms = float(ev.get("duration_ms", 50.0))
            start = int(round(onset_ms / dt_ms))
            end = int(round((onset_ms + dur_ms) / dt_ms))
            start = max(0, min(start, int(n_steps)))
            end = max(0, min(end, int(n_steps)))
            if start < end:
                frequency_hz = ev.get("frequency_hz", None)
                if frequency_hz:
                    t_rel_s = (_np.arange(start, end, dtype=_np.float64) - start) * (dt_ms / 1000.0)
                    values = amp * _np.sin(2.0 * _np.pi * float(frequency_hz) * t_rel_s)
                else:
                    values = None
                # Check if this event has target_indices
                target_indices = ev.get("target_indices", None)
                if target_indices is not None:
                    # Apply amplitude only to selected indices
                    idx_array = _np.asarray(target_indices, dtype=int)
                    # Validate bounds
                    if len(idx_array) > 0:
                        if idx_array.min() < 0 or idx_array.max() >= self.n_neurons:
                            raise ValueError(
                                f"target_indices out of bounds: "
                                f"min={idx_array.min()}, max={idx_array.max()}, "
                                f"n_neurons={self.n_neurons}"
                            )
                        if values is not None:
                            schedule[start:end, idx_array] += values[:, None]
                        else:
                            schedule[start:end, idx_array] += amp
                else:
                    if values is not None:
                        schedule[start:end, :] += values[:, None]
                    else:
                        # Apply amplitude to all neurons
                        schedule[start:end, :] += amp
        np_dtype = _np.float64 if dtype == "float64" else _np.float32
        return jnp.asarray(schedule.astype(np_dtype))

    def to_array_jax(self, n_steps: int, dt_ms: float, dtype: str = "float32") -> jax.Array:
        """Materialize a ``(n_steps, n_neurons)`` drive schedule array using JAX.

        Supports target_indices in event to apply amplitude only to selected neurons.
        Uses time masking and outer products for JIT/vmap compatibility.

        Note: unlike :meth:`to_array`, this path does not yet support the
        optional ``frequency_hz`` sinusoidal-drive key (flat amplitude only).
        ``Model.simulate`` calls :meth:`to_array`, not this method.
        """
        jdtype = jnp.float64 if (dtype == "float64" and bool(jax.config.read("jax_enable_x64"))) else jnp.float32
        time_ms = jnp.arange(n_steps, dtype=jdtype) * dt_ms
        
        schedule = jnp.zeros((n_steps, self.n_neurons), dtype=jdtype)
        
        for ev in self.events:
            if not ev.get("is_drive_event", True):
                continue
            amp = float(ev.get("amplitude", 0.0))
            if amp == 0.0:
                continue
            onset_ms = float(ev.get("onset_ms", 0.0))
            dur_ms = float(ev.get("duration_ms", 50.0))
            
            event_mask = (time_ms >= onset_ms) & (time_ms < onset_ms + dur_ms)
            
            target_indices = ev.get("target_indices", None)
            if target_indices is not None:
                idx_array = jnp.asarray(target_indices, dtype=jnp.int32)
                neuron_mask = jnp.zeros((self.n_neurons,), dtype=jdtype)
                neuron_mask = neuron_mask.at[idx_array].set(1.0)
                schedule = schedule + (event_mask[:, None] * neuron_mask[None, :]) * amp
            else:
                schedule = schedule + event_mask[:, None] * amp
                
        return schedule

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        # Count targeted vs non-targeted events
        n_targeted = sum(1 for ev in self.events if ev.get("target_indices") is not None)
        return json_safe({
            "stimulus_injection_status": "native_drive_schedule_v0.0.12",
            "n_drive_events": len(self.events),
            "n_targeted_events": n_targeted,
            "n_neurons": self.n_neurons,
            "events": list(self.events),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
        })


_KNOWN_LAYERS = frozenset({"L1", "L2/3", "L4", "L5", "L6", "unspecified"})


@dataclass(frozen=True)
class LaminarPopulation:
    """Metadata descriptor for one named laminar cell population.

    Depth values are normalized proxy coordinates in [0, 1] — not physical
    microns.  Overlapping depth ranges are allowed; co-located cell types
    (e.g. E and PV in the same layer) are anatomically expected.
    No physical-amplitude or calibration claim is made.
    """

    name: str
    cell_type: str
    layer: str
    depth_min: float
    depth_max: float
    n_units: int
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.name:
            issues.append("name_empty")
        if not self.cell_type:
            issues.append("cell_type_empty")
        if not self.layer:
            issues.append("layer_empty")
        if not (0.0 <= self.depth_min < self.depth_max <= 1.0):
            issues.append("depth_range_invalid")
        if self.n_units <= 0:
            issues.append("n_units_must_be_positive")
        if self.physical_amplitude_calibrated is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        if self.claim_level != "computational_scaffold":
            issues.append("claim_level_must_be_computational_scaffold")
        warnings: list[str] = []
        if self.layer not in _KNOWN_LAYERS:
            warnings.append(f"unrecognized_layer:{self.layer}")
        return {"valid": not issues, "issues": issues, "warnings": warnings}

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "cell_type": self.cell_type,
            "layer": self.layer,
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "n_units": int(self.n_units),
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
        })


@dataclass(frozen=True)
class LaminarSourceGeometry:
    """Metadata descriptor for the full laminar source geometry.

    Groups named :class:`LaminarPopulation` descriptors and can materialize
    a deterministic ``(n_units_total, 3)`` positions array for use in field
    projection.  Depths are proxy-normalized coordinates, not physical
    microns.  No physical-amplitude, PDE, or calibration claim is made.
    """

    populations: tuple[LaminarPopulation, ...]
    n_units_total: int
    position_units: str = "relative_laminar_depth_proxy"
    source_calibration_status: str = "uncalibrated_izhikevich_native_current"
    physical_amplitude_calibrated: bool = False
    claim_level: str = "computational_scaffold"


    def validate(self) -> dict[str, Any]:
        """Documented public function `validate`."""
        issues: list[str] = []
        if not self.populations:
            issues.append("populations_empty")
        pop_sum = sum(p.n_units for p in self.populations)
        if pop_sum != self.n_units_total:
            issues.append(f"n_units_total_mismatch:sum={pop_sum},declared={self.n_units_total}")
        if self.physical_amplitude_calibrated is not False:
            issues.append("physical_amplitude_claim_must_be_false")
        pop_issues: list[str] = []
        for p in self.populations:
            v = p.validate()
            pop_issues.extend([f"{p.name}:{i}" for i in v["issues"]])
        issues.extend(pop_issues)
        return {"valid": not issues, "issues": issues, "n_populations": len(self.populations)}

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "type": "laminar_source_geometry",
            "n_units_total": self.n_units_total,
            "n_populations": len(self.populations),
            "position_units": self.position_units,
            "source_calibration_status": self.source_calibration_status,
            "physical_amplitude_calibrated": self.physical_amplitude_calibrated,
            "claim_level": self.claim_level,
            "populations": [p.to_dict() for p in self.populations],
        })

    def population_slices(self) -> dict[str, slice]:
        """Map population names to neuron index slices.

        Returns:
            dict mapping population name → slice object spanning neuron indices.

        Example:
            >>> geom = LaminarSourceGeometry(...)
            >>> slices = geom.population_slices()
            >>> V_m_L4 = signals.V_m[slices["L4_E"], :]
        """
        result = {}
        start = 0
        for pop in self.populations:
            end = start + pop.n_units
            result[pop.name] = slice(start, end)
            start = end
        return result

    def positions_array(self, dtype: str = "float32") -> "jax.Array":
        """Return a deterministic ``(n_units_total, 3)`` positions array.

        x = 0, y = 0, z linearly spaced within each population's depth range.
        Population order is preserved.  No random sampling.
        """
        np_dtype = _np.float64 if dtype == "float64" else _np.float32
        rows: list[_np.ndarray] = []
        for pop in self.populations:
            n = int(pop.n_units)
            z = _np.linspace(float(pop.depth_min), float(pop.depth_max), n, dtype=np_dtype)
            xyz = _np.stack([_np.zeros(n, dtype=np_dtype), _np.zeros(n, dtype=np_dtype), z], axis=1)
            rows.append(xyz)
        arr = _np.concatenate(rows, axis=0) if rows else _np.zeros((0, 3), dtype=np_dtype)
        return jnp.asarray(arr)


_KNOWN_METRICS = frozenset({
    "spike_rate_hz_mean",
    "spike_count_total",
    "mean_V_m",
    "source_proxy_abs_mean",
    "csd_proxy_abs_mean",
    "lfp_proxy_abs_mean",
    "kappa_synchrony",
})

#: Config-metadata gate metrics — string-valued flags compared against
#: ``cfg.metadata`` rather than computed readout metrics.  These let an
#: Objective declare truth/scope gates (e.g. claim_level) that
#: are evaluated against the configuration, not the signals.
_KNOWN_CONFIG_GATE_METRICS = frozenset({
    "claim_level",
    "field_solver_status",
    "field_claim_level",
    "source_calibration_status",
    "source_projection_mode",
    "empirical_validation_status",
    "mechanism_claim_status",
})


def _finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _compute_kappa_synchrony_metric(spikes: Any) -> Optional[float]:
    """Vectorized mean pairwise spike-train correlation (kappa synchrony).

    Matches :func:`jaxfne.tutorial_utils.kappa_synchrony` semantics — mean
    Pearson correlation over neuron pairs with non-zero temporal variance —
    but computed as a single correlation matrix multiply for speed in the
    per-candidate tuning loop.  Proxy diagnostic; not a biological invariant.
    Expects ``spikes`` shaped ``[n_timesteps, n_neurons]``.
    """
    import numpy as _np

    x = _np.asarray(spikes, dtype=float)
    if x.ndim != 2:
        return 0.0
    n_timesteps, n_neurons = x.shape
    if n_neurons < 2 or n_timesteps < 1:
        return 0.0
    std = x.std(axis=0)
    valid = std > 0
    n_valid = int(valid.sum())
    if n_valid < 2:
        return 0.0
    xv = x[:, valid]
    z = (xv - xv.mean(axis=0)) / xv.std(axis=0)
    corr = (z.T @ z) / n_timesteps
    off_diagonal_sum = float(corr.sum() - _np.trace(corr))
    mean_pairwise = off_diagonal_sum / (n_valid * (n_valid - 1))
    return _finite_or_none(float(mean_pairwise))


def _compute_all_metrics(signals: "Signals", readout: Optional[dict[str, Any]] = None) -> dict[str, Optional[float]]:
    """Compute all known scalar metrics from signals."""
    dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
    m: dict[str, Optional[float]] = {}
    m["spike_rate_hz_mean"] = _finite_or_none(float(jnp.mean(signals.spikes) * (1000.0 / dt_ms)))
    m["spike_count_total"] = _finite_or_none(float(jnp.sum(signals.spikes)))
    m["mean_V_m"] = _finite_or_none(float(jnp.mean(signals.V_m)))
    m["kappa_synchrony"] = _compute_kappa_synchrony_metric(signals.spikes)
    if signals.field is not None:
        m["source_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.source_proxy))))
        m["csd_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.csd_proxy))))
        m["lfp_proxy_abs_mean"] = _finite_or_none(float(jnp.mean(jnp.abs(signals.field.lfp_proxy))))
    else:
        m["source_proxy_abs_mean"] = None
        m["csd_proxy_abs_mean"] = None
        m["lfp_proxy_abs_mean"] = None
    return m


def _check_gate_criterion(value: float, threshold: Any, criterion: str) -> bool:
    """Return True if the gate passes for the given criterion."""
    if criterion == "below":
        return float(value) < float(threshold)
    if criterion == "above":
        return float(value) > float(threshold)
    if criterion == "equal":
        return abs(float(value) - float(threshold)) < 1e-6
    if criterion == "in_range":
        lo, hi = float(threshold[0]), float(threshold[1])
        return lo <= float(value) <= hi
    return False


def _evaluate_loss_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    """Evaluate one loss spec against computed metrics."""
    result: dict[str, Any] = {"name": spec["name"], "weight": spec.get("weight", 1.0)}
    metric = spec.get("metric")
    target = spec.get("target")
    if metric is None:
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = "no_metric_specified"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        if strict:
            result["status"] = msg
            result["value"] = None
            result["weighted_value"] = None
            warnings.append(msg)
            if "metadata" in spec:
                result["metadata"] = spec["metadata"]
            return result
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if target is not None:
        raw = (value - float(target)) ** 2
    else:
        raw = value
    weighted = float(spec.get("weight", 1.0)) * raw
    result["target"] = target
    result["raw_loss"] = _finite_or_none(raw)
    result["weighted_value"] = _finite_or_none(weighted)
    result["status"] = "ok"
    if "metadata" in spec:
        result["metadata"] = spec["metadata"]
    return result


def _evaluate_regularizer_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    """Evaluate one regularizer spec."""
    result: dict[str, Any] = {
        "name": spec["name"],
        "target": spec.get("target", 0.0),
        "weight": spec.get("weight", 1.0),
    }
    metric = spec.get("metric")
    if metric is None:
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = "no_metric_specified"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
        if "metadata" in spec:
            result["metadata"] = spec["metadata"]
        return result
    target = float(spec.get("target", 0.0))
    raw = (value - target) ** 2
    weighted = float(spec.get("weight", 1.0)) * raw
    result["raw_regularizer"] = _finite_or_none(raw)
    result["weighted_value"] = _finite_or_none(weighted)
    result["status"] = "ok"
    if "metadata" in spec:
        result["metadata"] = spec["metadata"]
    return result


_CONSERVATIVE_TRUTH_DEFAULTS = {
    "physical_amplitude_calibrated": False,
    "claim_level": "computational_scaffold",
    "field_claim_level": "proxy_readout",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "linear_solver",
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}


def _evaluate_gate_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
    cfg_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate one gate spec; returns pass/fail.

    Numeric gates are checked against computed readout ``metrics``.  Gates
    whose metric is a configuration flag (see ``_KNOWN_CONFIG_GATE_METRICS``)
    are checked against ``cfg_meta`` via exact string comparison — this lets
    an Objective assert truth/scope gates (claim_level, …) that
    describe the configuration rather than the signals.
    """
    result: dict[str, Any] = {
        "name": spec["name"],
        "threshold": spec.get("threshold"),
        "criterion": spec.get("criterion", "below"),
    }
    metric = spec.get("metric")
    if metric is None:
        result["value"] = None
        result["pass"] = False
        result["status"] = "no_metric_specified"
        return result
    # Config-metadata gates: exact-match a configuration flag.
    if metric in _KNOWN_CONFIG_GATE_METRICS:
        cfg_meta = cfg_meta or {}
        value = cfg_meta.get(metric, _CONSERVATIVE_TRUTH_DEFAULTS.get(metric))
        result["metric"] = metric
        result["value"] = value
        threshold = spec.get("threshold")
        criterion = spec.get("criterion", "exact")
        if criterion == "exact":
            passes = value == threshold
        else:
            passes = _check_gate_criterion(value, threshold, criterion) if value is not None else False
        result["pass"] = bool(passes)
        result["status"] = "pass" if passes else "fail"
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        warnings.append(msg)
        result["metric"] = metric
        result["value"] = None
        result["pass"] = False
        result["status"] = msg
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["pass"] = False
        result["status"] = "metric_unavailable"
        return result
    passes = _check_gate_criterion(value, spec.get("threshold"), spec.get("criterion", "below"))
    result["pass"] = passes
    result["status"] = "pass" if passes else "fail"
    return result


# ──────────────────────────────────────────────────────────────
# v0.2.26 computation-basis contracts
# ──────────────────────────────────────────────────────────────

#: Allowed values for AxisSpec.status
_AXIS_STATUS_VALUES: frozenset[str] = frozenset({"active", "collapsed", "indexed"})

#: Allowed values for BasisSpec.space_basis
_SPACE_BASIS_VALUES: frozenset[str] = frozenset(
    {"collapsed", "xy", "xyz", "laminar_depth", "graph"}
)

#: Allowed values for BasisSpec.time_basis
_TIME_BASIS_VALUES: frozenset[str] = frozenset(
    {"continuous_ms", "discrete_steps", "slow_proxy"}
)

#: Allowed values for BasisSpec.field_regime
_FIELD_REGIME_VALUES: frozenset[str] = frozenset(
    {
        "laminar_proxy",
        "quasi_static_resistive",
        "solved_poisson",
        "future_admittive",
        "future_maxwell",
    }
)

#: Regimes that are declared future — never claim implemented=True for these
_FUTURE_FIELD_REGIMES: frozenset[str] = frozenset({"future_admittive", "future_maxwell"})

#: Allowed values for BasisSpec.source_mode
_SOURCE_MODE_BASIS_VALUES: frozenset[str] = frozenset(
    {"total_membrane_current", "decomposed_cap_ion_syn", "proxy_no_field_solve"}
)

_SOURCE_MODE_BASIS_STATUS: dict[str, dict[str, Any]] = {
    "proxy_no_field_solve": {
        "status": "active",
        "executable": True,
        "operator_type": "linear_projection",
    },
    "total_membrane_current": {
        "status": "reserved",
        "executable": False,
        "operator_type": "direct_readout",
    },
    "decomposed_cap_ion_syn": {
        "status": "reserved",
        "executable": False,
        "operator_type": "direct_readout",
    },
}

#: Allowed values for BasisSpec.probe_basis
_PROBE_BASIS_VALUES: frozenset[str] = frozenset(
    {
        "none",
        "spike_only",
        "field_proxy",
        "multimodal_proxy",
        "physical_forward_model",
    }
)


@dataclass(frozen=True)
class AxisSpec:
    """Typed descriptor for one tensor axis in the TFNE scaffold.

    Describes whether a spatial/feature dimension is actively computed
    (``active``), collapsed to a scalar or removed (``collapsed``), or
    indexed by an explicit label set (``indexed``).

    These are documentation/contract objects — they do not affect JAX
    execution. They appear in manifest output to make axis semantics
    explicit and auditable.

    Attributes
    ----------
    name : str
        Canonical dimension name (e.g. ``"x"``, ``"y"``, ``"z"``, ``"t"``).
    status : str
        One of ``"active"``, ``"collapsed"``, ``"indexed"``.
    size : int or None
        Known static size, if any. ``None`` if dynamic or not applicable.
    units_or_status : str
        Physical units (``"mm"``, ``"ms"``) or proxy status
        (``"declared"``, ``"proxy"``). Default ``"declared"``.
    """

    name: str
    status: str = "active"
    size: Optional[int] = None
    units_or_status: str = "declared"


    def validate(self) -> dict[str, Any]:
        """Return a JSON-safe validation dict."""
        issues: list[str] = []
        if not self.name:
            issues.append("name_empty")
        if self.status not in _AXIS_STATUS_VALUES:
            issues.append(f"invalid_status:{self.status!r}")
        if self.size is not None and self.size <= 0:
            issues.append(f"size_must_be_positive_got:{self.size}")
        return {
            "valid": not issues,
            "issues": issues,
            "name": self.name,
            "status": self.status,
            "size": self.size,
            "units_or_status": self.units_or_status,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dict representation."""
        return {
            "name": self.name,
            "status": self.status,
            "size": self.size,
            "units_or_status": self.units_or_status,
        }


@dataclass(frozen=True)
class BasisSpec:
    """Typed descriptor for the computation basis of a TFNE run.

    Declares the spatial, temporal, and field regime without claiming
    physical amplitude or biological validity. Future electrodynamic
    regimes (``future_maxwell``, ``future_admittive``) are recorded as
    declared-future modules: ``implemented=False``,
    ``claim_allowed=False``.

    The default matches the current v0.2.27 laminar-proxy scaffold.

    Attributes
    ----------
    space_basis : str
        Spatial computation domain. Default ``"laminar_depth"``.
    time_basis : str
        Temporal basis. Default ``"continuous_ms"``.
    field_regime : str
        Field computation regime. Default ``"laminar_proxy"``.
    source_mode : str
        Source model. Default ``"proxy_no_field_solve"``.
    probe_basis : str
        Probe operator class. Default ``"multimodal_proxy"``.
    axes : tuple of AxisSpec
        Explicit axis descriptors. Default: x/y collapsed, z active.
    """

    space_basis: str = "laminar_depth"
    time_basis: str = "continuous_ms"
    field_regime: str = "laminar_proxy"
    source_mode: str = "proxy_no_field_solve"
    probe_basis: str = "multimodal_proxy"
    axes: tuple[Any, ...] = field(
        default_factory=lambda: (
            AxisSpec(name="x", status="collapsed"),
            AxisSpec(name="y", status="collapsed"),
            AxisSpec(name="z", status="active", units_or_status="proxy"),
        )
    )

    @property
    def implemented(self) -> bool:
        """True if this regime has a runtime implementation in the current package."""
        # Future regimes are always unimplemented by doctrine
        if self.field_regime in _FUTURE_FIELD_REGIMES:
            return False
        # solved_poisson is specified but not solved in v0.2.x
        if self.field_regime == "solved_poisson":
            return False
        return True

    @property
    def claim_allowed(self) -> bool:
        """Physical amplitude claims are always False in proxy/scaffold regimes."""
        # Claims require solved field with calibrated conductivity — not in v0.2.x
        return False

    @property
    def source_mode_status(self) -> dict[str, Any]:
        """Return the execution status of the declared source mode."""
        return dict(
            _SOURCE_MODE_BASIS_STATUS.get(
                self.source_mode,
                {
                    "status": "invalid",
                    "executable": False,
                    "operator_type": "unresolved",
                },
            )
        )


    def validate(self) -> dict[str, Any]:
        """Return a JSON-safe validation dict. Raises ValueError on invalid enum."""
        issues: list[str] = []
        if self.space_basis not in _SPACE_BASIS_VALUES:
            issues.append(f"invalid_space_basis:{self.space_basis!r}")
        if self.time_basis not in _TIME_BASIS_VALUES:
            issues.append(f"invalid_time_basis:{self.time_basis!r}")
        if self.field_regime not in _FIELD_REGIME_VALUES:
            issues.append(f"invalid_field_regime:{self.field_regime!r}")
        if self.source_mode not in _SOURCE_MODE_BASIS_VALUES:
            issues.append(f"invalid_source_mode:{self.source_mode!r}")
        if self.probe_basis not in _PROBE_BASIS_VALUES:
            issues.append(f"invalid_probe_basis:{self.probe_basis!r}")
        # Axis-space consistency checks
        active_axes = {a.name for a in self.axes if a.status == "active"}
        for ax in self.axes:
            v = ax.validate()
            if not v["valid"]:
                issues.extend([f"axis_{ax.name}:{i}" for i in v["issues"]])
        if self.space_basis == "collapsed" and active_axes:
            issues.append(f"collapsed_basis_must_not_have_active_axes:{sorted(active_axes)}")
        if self.space_basis == "xy" and "z" in active_axes:
            issues.append("xy_basis:z_must_not_be_active_unless_indexed")
        if self.space_basis == "xyz":
            missing = {"x", "y", "z"} - active_axes - {
                a.name for a in self.axes if a.status in ("active", "indexed")
            }
            if missing:
                issues.append(f"xyz_basis:missing_active_or_indexed_axes:{sorted(missing)}")
        if self.space_basis == "laminar_depth":
            z_ok = any(
                a.name == "z" and a.status in ("active", "indexed") for a in self.axes
            )
            if not z_ok:
                issues.append("laminar_depth_basis:z_must_be_active_or_indexed")
        return {
            "valid": not issues,
            "issues": issues,
            "space_basis": self.space_basis,
            "time_basis": self.time_basis,
            "field_regime": self.field_regime,
            "source_mode": self.source_mode,
            "source_mode_status": self.source_mode_status["status"],
            "source_mode_executable": self.source_mode_status["executable"],
            "probe_basis": self.probe_basis,
            "implemented": self.implemented,
            "future_regime": self.field_regime in _FUTURE_FIELD_REGIMES,
            "claim_allowed": self.claim_allowed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict including axis dimension_status."""
        dim_status = {a.name: a.status for a in self.axes}
        return {
            "space_basis": self.space_basis,
            "time_basis": self.time_basis,
            "field_regime": self.field_regime,
            "source_mode": self.source_mode,
            "source_mode_status": self.source_mode_status["status"],
            "source_mode_executable": self.source_mode_status["executable"],
            "probe_basis": self.probe_basis,
            "dimension_status": dim_status,
            "implemented": self.implemented,
            "future_regime": self.field_regime in _FUTURE_FIELD_REGIMES,
            "claim_allowed": self.claim_allowed,
        }


def default_basis_spec() -> BasisSpec:
    """Return the default BasisSpec matching the current laminar-proxy scaffold."""
    return BasisSpec()


def _default_basis_dict() -> dict[str, Any]:
    """Return JSON-safe basis metadata dict for manifest embedding."""
    return default_basis_spec().to_dict()


# ──────────────────────────────────────────────────────────────
def _normalize_manifest_readout(
    readout: Any,
) -> Optional[dict[str, Any]]:
    """Normalize any supported readout argument shape for :meth:`Model.manifest`.

    Accepted input forms:

    * ``None``                    → returns ``None``
    * ``dict``                    → returned unchanged (legacy shape)
    * :class:`ReadoutResult`      → wrapped in list, then converted
    * ``list/tuple`` of :class:`ReadoutResult` → converted to summary dict
    * ``list/tuple`` of ``dict``  → converted to summary dict

    The returned dict (when non-None) always contains:

    * ``readout_results``    – list of JSON-safe readout result dicts
    * ``requested_metrics``  – list of metric name strings
    * ``n_results``          – integer count
    * ``physical_amplitude_calibrated`` – always False
    """
    if readout is None:
        return None
    if isinstance(readout, dict):
        return readout
    # Normalize single ReadoutResult to a one-element list.
    if isinstance(readout, ReadoutResult):
        readout = [readout]
    if isinstance(readout, (list, tuple)):
        items: list[dict[str, Any]] = []
        metrics: list[str] = []
        for item in readout:
            if isinstance(item, ReadoutResult):
                items.append(item.to_dict())
                metrics.append(item.metric)
            elif isinstance(item, dict):
                items.append(json_safe(item))
                metrics.append(str(item.get("metric", "unknown")))
            else:
                items.append(json_safe({"raw": str(item)}))
                metrics.append("unknown")
        return {
            "readout_results": items,
            "requested_metrics": metrics,
            "n_results": len(items),
            "physical_amplitude_calibrated": False,
        }
    # Fallback: stringify unknown types rather than crash.
    return {"readout_results": [json_safe({"raw": str(readout)})], "n_results": 1,
            "physical_amplitude_calibrated": False}

