"""Core object model for :mod:`jaxfne`.

Design target: object-oriented public API, pure-JAX computational core.  The
current package is an honest TFNE scaffold: reduced emitters plus laminar proxy
source/readout status, not a full PDE field solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import EIGNetwork, IzhikevichParams, make_eig_network, simulate_eig_izhikevich
from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, manifest as build_manifest


def _default_operator_status() -> dict[str, str]:
    return {
        "E_theta": "prototype_api",
        "S_WDR": "prototype_api",
        "C_mu_nu": "specified_future_module",
        "Q_eta_alpha": "prototype_api",
        "F_field": "prototype_api",
        "P_probe": "prototype_api",
        "A_objective": "prototype_api",
        "O_optimizer": "specified_future_module",
        "C_constraints": "prototype_api",
    }


def _default_metadata() -> dict[str, Any]:
    return {
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "source_projection_mode": "proxy_no_field_solve",
        "source_decomposition": "proxy_reduced_emitter",
        "boundary_condition": "mean_zero_neumann",
        "gauge": "mean_zero",
        "csd_sign_convention": "proxy_positive_equals_extracellular_source_like",
        "field_solver_status": "laminar_proxy_no_pde",
        "manifest_schema_version": "0.0.4",
        "operator_status": _default_operator_status(),
    }


@dataclass(frozen=True)
class Configuration:
    """Declarative TFNE model configuration.

    It is the anatomical/model declaration, not the compiled model.  Methods
    return new objects, making the public API friendly while keeping mutation
    out of the construction path.
    """

    networks: list[dict[str, Any]] = field(default_factory=list)
    emitters: list[dict[str, Any]] = field(default_factory=list)
    fields: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=_default_metadata)

    def network(self, **kwargs: Any) -> "Configuration":
        return replace(self, networks=[*self.networks, dict(kwargs)])

    def emitter(self, **kwargs: Any) -> "Configuration":
        return replace(self, emitters=[*self.emitters, dict(kwargs)])

    def field(self, **kwargs: Any) -> "Configuration":
        return replace(self, fields=[*self.fields, dict(kwargs)])

    def probe(self, **kwargs: Any) -> "Configuration":
        return replace(self, probes=[*self.probes, dict(kwargs)])

    def update_metadata(self, **kwargs: Any) -> "Configuration":
        metadata = dict(self.metadata)
        metadata.update(kwargs)
        return replace(self, metadata=metadata)

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.networks:
            issues.append("no_networks_declared")
        if not self.emitters:
            issues.append("no_emitters_declared")
        if not self.fields:
            issues.append("no_field_declared")
        if not self.probes:
            issues.append("no_probes_declared")
        return {
            "valid": not issues,
            "issues": issues,
            "config_hash": config_hash(self),
            "truth_mode": self.metadata.get("truth_mode"),
            "claim_level": self.metadata.get("claim_level"),
        }


@dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime and dtype policy.

    ``dtype='float64'`` is honored only when JAX x64 is enabled.  The manifest
    always reports both requested and actual dtype policy.
    """

    backend: str = "auto"  # "auto" | "cpu" | "gpu" | "tpu"
    dtype: str = "float32"  # "float32" | "float64"
    jit: bool = False
    vmap: bool = False
    precision: str = "default"  # "default" | "high"
    seed: int = 0
    n_steps: int = 0
    # v0.0.3 compatibility names; if provided by old caller, they are folded in.
    device_type: Optional[str] = None
    dtype_primary: Optional[str] = None
    x64_enabled: Optional[bool] = None

    @property
    def requested_dtype(self) -> str:
        return self.dtype_primary or self.dtype

    @property
    def selected_backend(self) -> str:
        return self.device_type or self.backend

    @property
    def actual_dtype(self) -> str:
        if self.requested_dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
            return "float64"
        return "float32"

    @property
    def jnp_dtype(self) -> jnp.dtype:
        return jnp.float64 if self.actual_dtype == "float64" else jnp.float32

    def runtime_report(self) -> dict[str, Any]:
        devices = [str(device) for device in jax.devices()]
        return {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": _jaxlib_version(),
            "default_backend": jax.default_backend(),
            "available_devices": devices,
            "selected_backend": self.selected_backend,
            "backend": self.backend,
            "requested_dtype": self.requested_dtype,
            "actual_dtype": self.actual_dtype,
            "dtype": self.dtype,
            "jit": bool(self.jit),
            "vmap": bool(self.vmap),
            "precision": self.precision,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            # Compatibility keys from v0.0.3.
            "device_type": self.selected_backend,
            "dtype_primary": self.actual_dtype,
        }


def _jaxlib_version() -> str:
    try:
        import jaxlib  # type: ignore

        return str(jaxlib.__version__)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Simulation:
    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True
    runtime: RuntimeConfig | None = None

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
    name: str
    modes: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signals:
    """Simulation output container holding multiple arrays."""

    time_ms: jax.Array
    V_m: jax.Array
    spikes: jax.Array
    sources: jax.Array
    field: Optional[FieldOutput]
    metadata: dict[str, Any]


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
class ParadigmEvent:
    """Discrete event within a task trial: stimulus, behavioral code, or omission marker."""

    label: str
    onset_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    code: Optional[int] = None
    stimulus: Optional[str] = None
    is_omission: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "label": self.label,
            "onset_ms": self.onset_ms,
            "duration_ms": self.duration_ms,
            "code": self.code,
            "stimulus": self.stimulus,
            "is_omission": self.is_omission,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ParadigmCondition:
    """A specific trial condition: sequence of stimuli and associated events."""

    name: str
    sequence: tuple[str, str, str, str]
    omission_position: Optional[str] = None
    probability: Optional[float] = None
    condition_numbers: tuple[int, ...] = ()
    events: tuple[ParadigmEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_omission(self) -> bool:
        """Return True if this condition contains an omission."""
        return self.omission_position is not None

    def omitted_event_label(self) -> Optional[str]:
        """Return the label of the omitted event, or None if no omission."""
        if self.omission_position is None:
            return None
        for evt in self.events:
            if evt.label == self.omission_position:
                return evt.label
        return self.omission_position

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "sequence": self.sequence,
            "omission_position": self.omission_position,
            "probability": self.probability,
            "condition_numbers": self.condition_numbers,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class Paradigm:
    name: str = "none"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    conditions: tuple["ParadigmCondition", ...] = ()
    alignment_code: int = 101
    alignment_label: str = "p1"
    pre_stimulus_buffer_ms: float = 1000.0
    analysis_windows: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "baseline": (-500.0, 0.0),
        "event": (0.0, 500.0),
        "post_event": (500.0, 1000.0),
    })
    event_codes: dict[str, int] = field(default_factory=lambda: {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    })
    metadata: dict[str, Any] = field(default_factory=dict)

    def habituation(self, sequence: Sequence[str], n_trials: int) -> "Paradigm":
        return replace(
            self,
            blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}],
        )

    def main_block(self, **kwargs: Any) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0, condition_weights: Optional[dict[str, float]] = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_trials": n_trials,
            "seed": seed,
            "blocks": self.blocks,
            "condition_weights": condition_weights,
        }

    def condition(self, name: str) -> Optional["ParadigmCondition"]:
        """Return ParadigmCondition by name, or None if not found."""
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None

    def condition_names(self) -> list[str]:
        """Return list of condition names."""
        return [c.name for c in self.conditions]

    def omission_conditions(self) -> list["ParadigmCondition"]:
        """Return list of conditions containing omissions."""
        return [c for c in self.conditions if c.has_omission()]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "blocks": self.blocks,
            "conditions": [c.to_dict() for c in self.conditions],
            "alignment_code": self.alignment_code,
            "alignment_label": self.alignment_label,
            "pre_stimulus_buffer_ms": self.pre_stimulus_buffer_ms,
            "analysis_windows": self.analysis_windows,
            "event_codes": self.event_codes,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class DatasetSpec:
    """Manifest-safe dataset/alignment declaration for observed data.

    DatasetSpec is a schema object, not a loader.  It records how an external
    dataset is aligned and interpreted so objectives can reference data without
    hard-coding paths or claiming empirical validation.
    """

    name: str = "unnamed_dataset"
    modality: str = "unspecified"
    source_format: str = "unspecified"
    alignment_label: str = "p1"
    alignment_code: int = 101
    sampling_rate_hz: Optional[float] = None
    units: str = "unspecified"
    trial_filter: dict[str, Any] = field(default_factory=dict)
    condition_map: dict[str, list[int]] = field(default_factory=dict)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_condition_map(self, condition_map: Mapping[str, Sequence[int]]) -> "DatasetSpec":
        mapped = {str(k): [int(x) for x in v] for k, v in condition_map.items()}
        return replace(self, condition_map=mapped)

    def with_quality_gate(self, name: str, value: Any) -> "DatasetSpec":
        gates = dict(self.quality_gates)
        gates[str(name)] = value
        return replace(self, quality_gates=gates)

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.name:
            issues.append("dataset_name_missing")
        if self.alignment_code is None:
            issues.append("alignment_code_missing")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            issues.append("sampling_rate_hz_must_be_positive")
        return {
            "valid": not issues,
            "issues": issues,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        }

    def to_dict(self) -> dict[str, Any]:
        from .io import json_safe
        return json_safe({
            "name": self.name,
            "modality": self.modality,
            "source_format": self.source_format,
            "alignment_label": self.alignment_label,
            "alignment_code": self.alignment_code,
            "sampling_rate_hz": self.sampling_rate_hz,
            "units": self.units,
            "trial_filter": self.trial_filter,
            "condition_map": self.condition_map,
            "quality_gates": self.quality_gates,
            "metadata": self.metadata,
            "dataset_status": "schema_only_no_data_loaded",
            "empirical_validation_status": "not_empirically_validated",
        })


_KNOWN_METRICS = frozenset({
    "spike_rate_hz_mean",
    "spike_count_total",
    "mean_V_m",
    "source_proxy_abs_mean",
    "csd_proxy_abs_mean",
    "lfp_proxy_abs_mean",
})


def _finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _compute_all_metrics(signals: "Signals", readout: Optional[dict[str, Any]] = None) -> dict[str, Optional[float]]:
    """Compute all known scalar metrics from signals."""
    dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
    m: dict[str, Optional[float]] = {}
    m["spike_rate_hz_mean"] = _finite_or_none(float(jnp.mean(signals.spikes) * (1000.0 / dt_ms)))
    m["spike_count_total"] = _finite_or_none(float(jnp.sum(signals.spikes)))
    m["mean_V_m"] = _finite_or_none(float(jnp.mean(signals.V_m)))
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
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        if strict:
            result["status"] = msg
            result["value"] = None
            result["weighted_value"] = None
            warnings.append(msg)
            return result
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
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
        return result
    if metric not in _KNOWN_METRICS:
        msg = f"unknown_metric:{metric}"
        warnings.append(msg)
        result["value"] = None
        result["weighted_value"] = None
        result["status"] = msg
        return result
    value = metrics.get(metric)
    result["metric"] = metric
    result["value"] = value
    if value is None:
        result["weighted_value"] = None
        result["status"] = "metric_unavailable"
        return result
    target = float(spec.get("target", 0.0))
    raw = (value - target) ** 2
    weighted = float(spec.get("weight", 1.0)) * raw
    result["raw_regularizer"] = _finite_or_none(raw)
    result["weighted_value"] = _finite_or_none(weighted)
    result["status"] = "ok"
    return result


def _evaluate_gate_spec(
    spec: dict[str, Any],
    metrics: dict[str, Optional[float]],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    """Evaluate one gate spec; returns pass/fail."""
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


@dataclass(frozen=True)
class Model:
    cfg: Configuration
    params: dict[str, Any]
    static: dict[str, Any]

    def simulate(self, sim: Simulation, paradigm: Optional[Mapping[str, Any]] = None) -> Signals:
        """Run the default EIG/Izhikevich vertical slice."""

        runtime_cfg = sim.resolved_runtime
        key = jax.random.PRNGKey(sim.seed)
        emitter: IzhikevichParams = self.params["emitter"]
        voltages, spikes, sources = simulate_eig_izhikevich(
            emitter,
            sim.n_steps,
            sim.dt_ms,
            key,
            dtype=runtime_cfg.actual_dtype,
        )
        time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
            sim.dt_ms, dtype=runtime_cfg.jnp_dtype
        )
        positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
        field_output = None
        if sim.record_fields:
            field_output = project_laminar_sources(
                sources=sources,
                positions=positions,
                n_contacts=self.static.get("n_contacts", 16),
                dtype=runtime_cfg.actual_dtype,
            )
        metadata = {
            "config_hash": config_hash(self.cfg),
            "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
            "paradigm": dict(paradigm) if paradigm is not None else None,
            "plasticity_gain": sim.plasticity,
            "runtime": runtime_cfg.runtime_report(),
        }
        return Signals(
            time_ms=time_ms,
            V_m=voltages.astype(runtime_cfg.jnp_dtype),
            spikes=spikes,
            sources=sources.astype(runtime_cfg.jnp_dtype),
            field=field_output,
            metadata=metadata,
        )

    def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
        """Canonical TFNE readout method."""

        modes = list(modes or [])
        out: dict[str, Any] = {"requested_modes": modes}
        if "spikes" in modes:
            out["spikes"] = signals.spikes
        if "V_m" in modes:
            out["V_m"] = signals.V_m
        if "source" in modes or "sources" in modes:
            out["sources"] = signals.sources
        if signals.field is not None:
            out.update(probe_laminar_modes(signals.field, modes))
        return out

    def record(self, signals: Signals, modes: Sequence[str]) -> dict[str, Any]:
        """User-friendly alias for :meth:`probe`."""

        return self.probe(signals, modes)

    def evaluate(
        self,
        signals: Signals,
        objective: "Objective | str",
        readout: Optional[dict[str, Any]] = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Full objective/gate evaluation with JSON-safe report.

        Gate pass/fail is a computational diagnostic only.  It does not imply
        empirical validation, biological calibration, or mechanism proof.
        All truth gates from v0.0.4 are preserved in the report.
        """
        from .io import json_safe

        if isinstance(objective, str):
            objective = Objective(name=objective)

        cfg_meta = self.cfg.metadata
        warnings: list[str] = []

        computed_metrics = _compute_all_metrics(signals, readout)

        loss_results = []
        total_loss = 0.0
        has_loss_value = False
        for spec in objective.losses:
            r = _evaluate_loss_spec(spec, computed_metrics, warnings, strict)
            loss_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        reg_results = []
        for spec in objective.regularizers:
            r = _evaluate_regularizer_spec(spec, computed_metrics, warnings, strict)
            reg_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        gate_results = []
        all_gates_pass = True
        for spec in objective.gates:
            r = _evaluate_gate_spec(spec, computed_metrics, warnings, strict)
            gate_results.append(r)
            if not r.get("pass", True):
                all_gates_pass = False

        acceptance = "gates_pass" if all_gates_pass else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_v0.0.5",
            "objective_name": objective.name,
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "losses": loss_results,
            "regularizers": reg_results,
            "gates": gate_results,
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "warnings": warnings,
        })

    def tune(
        self,
        objective: "Objective",
        optimizer: Any = None,
        steps: int = 0,
        seed: int = 0,
        strategy: Optional[str] = None,
        strict: bool = False,
        simulation: Optional[Simulation] = None,
        parameter: str = "source_scale",
        bounds: tuple[float, float] = (0.25, 4.0),
    ) -> tuple["Model", dict[str, Any]]:
        """Run a small black-box tuning loop or guarded differentiable-path check.

        v0.0.6 adds a bounded metadata-safe black-box candidate loop for
        optimizers.  The loop searches one declared scalar parameter and uses
        Model.evaluate() as the scoring function.  This remains a computational
        scaffold: no biological calibration, no field-solver upgrade, and no
        optimizer-selected mechanism claim are made.
        """
        from .io import json_safe
        from .optim import _resolve_optimizer, propose_blackbox_candidates, require_optax

        cfg_meta = self.cfg.metadata
        spec = _resolve_optimizer(optimizer)
        sim = simulation or Simulation(duration_ms=10.0, dt_ms=0.1, seed=seed)
        n_steps = max(0, int(steps))
        base_report: dict[str, Any] = {
            "same_model_unchanged": True,
            "steps_requested": n_steps,
            "seed": int(seed),
            "strategy": strategy or spec.optimizer,
            "parameter": parameter,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "optimizer": spec.to_dict(),
            "objective_name": objective.name if not isinstance(objective, str) else objective,
            "losses_declared": len(objective.losses) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(objective.regularizers) if not isinstance(objective, str) else 0,
            "gates_declared": len(objective.gates) if not isinstance(objective, str) else 0,
            "truth_mode": cfg_meta.get("truth_mode", "truth_safe_unverified"),
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "laminar_proxy_no_pde"),
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }

        if spec.is_differentiable_path():
            if not spec.gradient_path_safe():
                report = {
                    **base_report,
                    "tuning_status": "blocked_non_differentiable_path",
                    "acceptance_decision": "REVISE",
                    "warnings": [
                        "optax_requires_differentiable_or_declared_surrogate",
                        "spiking_reset_not_differentiable_without_surrogate",
                    ],
                }
                return self, json_safe(report)
            try:
                require_optax()
                optax_status = "available"
            except ImportError:
                if strict:
                    raise
                optax_status = "unavailable"
            report = {
                **base_report,
                "tuning_status": "optax_guarded_path_no_loop_v0.0.8",
                "acceptance_decision": "REVISE" if optax_status == "unavailable" else "ACCEPT_CANDIDATE",
                "optax_status": optax_status,
                "same_model_unchanged": True,
                "warnings": ["differentiable_loop_not_enabled_for_spiking_reset_without_explicit_surrogate_kernel"],
            }
            return self, json_safe(report)

        if n_steps <= 0:
            report = {
                **base_report,
                "tuning_status": "metadata_only_no_steps_requested",
                "acceptance_decision": "REVISE",
                "candidate_history": [],
                "warnings": ["no_blackbox_steps_requested"],
            }
            return self, json_safe(report)

        candidates = propose_blackbox_candidates(
            optimizer=spec,
            n_steps=n_steps,
            seed=int(seed),
            bounds=(float(bounds[0]), float(bounds[1])),
        )
        best_model: Model = self
        best_loss: Optional[float] = None
        best_value: Optional[float] = None
        history: list[dict[str, Any]] = []
        warnings: list[str] = []
        for idx, candidate_value in enumerate(candidates):
            candidate_model = _model_with_scalar_parameter(self, parameter, float(candidate_value))
            candidate_signals = candidate_model.simulate(replace(sim, seed=int(seed) + idx))
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
            score = float(score)
            accepted = math.isfinite(score) and (best_loss is None or score < best_loss)
            if accepted:
                best_loss = score
                best_value = float(candidate_value)
                best_model = candidate_model
            history.append({
                "step": idx,
                "candidate_value": float(candidate_value),
                "score": _finite_or_none(score),
                "all_gates_pass": gates_pass,
                "accepted_as_best": bool(accepted),
                "evaluation_status": candidate_report.get("evaluation_status"),
            })
        if best_loss is None:
            warnings.append("no_finite_candidate_score")
            best_model = self
        report = {
            **base_report,
            "same_model_unchanged": best_model is self,
            "tuning_status": "blackbox_loop_v0.0.6",
            "acceptance_decision": "ACCEPT_CANDIDATE" if best_loss is not None else "REVISE",
            "best_parameter_value": best_value,
            "best_score": _finite_or_none(best_loss) if best_loss is not None else None,
            "candidate_history": history,
            "warnings": warnings + [
                "blackbox_loop_is_computational_scaffold_only",
                "optimizer_selected_candidate_is_not_biological_truth",
            ],
        }
        return best_model, json_safe(report)

    def manifest(
        self,
        signals: Optional[Signals] = None,
        readout: Optional[dict[str, Any]] = None,
        paradigm: Optional[dict[str, Any]] = None,
        objective: Optional[dict[str, Any]] = None,
        evaluation: Optional[dict[str, Any]] = None,
        tuning: Optional[dict[str, Any]] = None,
        dataset: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        runtime_cfg = None
        if signals is not None and "runtime" in signals.metadata:
            runtime_cfg = _RuntimeReportAdapter(signals.metadata["runtime"])
        return build_manifest(
            self.cfg,
            signals=signals,
            readout=readout,
            runtime_config=runtime_cfg,
            paradigm=paradigm,
            objective=objective,
            evaluation=evaluation,
            tuning=tuning,
            dataset=dataset,
        )


def _model_with_scalar_parameter(model: Model, parameter: str, value: float) -> Model:
    """Return a Model copy with one safe scalar emitter parameter changed."""
    emitter = model.params["emitter"]
    if parameter == "source_scale":
        new_emitter = replace(emitter, source_scale=jnp.asarray(value, dtype=emitter.source_scale.dtype))
    elif parameter == "drive_gain":
        new_emitter = replace(emitter, drive=emitter.drive * jnp.asarray(value, dtype=emitter.drive.dtype))
    else:
        raise ValueError(f"Unsupported tunable parameter: {parameter}")
    params = dict(model.params)
    params["emitter"] = new_emitter
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


@dataclass(frozen=True)
class _RuntimeReportAdapter:
    report: dict[str, Any]

    def runtime_report(self) -> dict[str, Any]:
        return self.report


def _mean_pairwise_corr_proxy(spikes: jax.Array) -> jax.Array:
    x = spikes.astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.std(x, axis=0, keepdims=True) + 1e-6
    z = x / denom
    corr = (z.T @ z) / jnp.maximum(1, z.shape[0] - 1)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    return jnp.sum(jnp.abs(corr) * mask) / jnp.maximum(1.0, jnp.sum(mask))


def configuration() -> Configuration:
    return Configuration()


def runtime(
    backend: str = "auto",
    dtype: str = "float32",
    jit: bool = False,
    vmap: bool = False,
    precision: str = "default",
    seed: int = 0,
    n_steps: int = 0,
    # v0.0.3 compatibility names.
    device_type: str | None = None,
    dtype_primary: str | None = None,
    x64_enabled: bool | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        backend=backend,
        dtype=dtype,
        jit=jit,
        vmap=vmap,
        precision=precision,
        seed=seed,
        n_steps=n_steps,
        device_type=device_type,
        dtype_primary=dtype_primary,
        x64_enabled=x64_enabled,
    )


def runtime_report(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    return (runtime_config or RuntimeConfig()).runtime_report()


def simulation(**kwargs: Any) -> Simulation:
    return Simulation(**kwargs)


def objective() -> Objective:
    return Objective()


def paradigm(name: str = "none") -> Paradigm:
    return Paradigm(name=name)


def construct(cfg: Configuration) -> Model:
    validation = cfg.validate()
    if not validation["valid"]:
        raise ValueError(f"Invalid jaxfne configuration: {validation['issues']}")
    net = cfg.networks[0]
    n = int(net.get("n", 100))
    cell_types = net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.1})
    network: EIGNetwork = make_eig_network(n=n, cell_type_fractions=cell_types)
    return Model(
        cfg=cfg,
        params={"emitter": network.params, "positions": network.positions},
        static={"n_contacts": 16, "operator_status": operator_status()},
    )


def operator_status() -> dict[str, str]:
    return _default_operator_status()


def standard_visual_omission() -> Paradigm:
    """Construct a Paradigm with standard visual oddball/omission task conditions.

    12 core conditions:
      - AAAB, AXAB, AAXB, AAAX (omission in p2, p3, p4, and p4 respectively)
      - BBBA, BXBA, BBXA, BBBX (omission in p2, p3, p4, and p4 respectively)
      - RRRR, RXRR, RRXR, RRRX (random-control stimuli, omissions in p2, p3, p4)

    Event codes:
      - fx: 10 (fixation)
      - p1: 101 (standard visual P1)
      - p2: 103 (standard visual P2)
      - p3: 105 (standard visual P3)
      - p4: 107 (standard visual P4)
      - rw: 96 (reward marker)

    Analysis windows:
      - baseline: -500 to 0 ms (pre-stimulus)
      - event: 0 to 500 ms (post-stimulus)
      - post_event: 500 to 1000 ms (post-stimulus)

    Alignment: P1 onset (code 101) at t=0.
    Pre-stimulus buffer: 1000 ms.
    """
    # Define event code mapping (immutable, hardcoded).
    event_codes = {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    }

    # Standard stimulus identifiers.
    std_A = "stimulus_A"
    std_B = "stimulus_B"
    std_X = "omitted_placeholder"
    std_R = "random_stimulus"

    # Define conditions with condition numbers and omission metadata.
    conditions = [
        # A-sequence (AAAB family): oddball in position 4.
        ParadigmCondition(
            name="AAAB",
            sequence=(std_A, std_A, std_A, std_B),
            omission_position=None,
            probability=None,
            condition_numbers=(1, 2),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AXAB",
            sequence=(std_A, std_X, std_A, std_B),
            omission_position="p2",
            probability=None,
            condition_numbers=(3,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AAXB",
            sequence=(std_A, std_A, std_X, std_B),
            omission_position="p3",
            probability=None,
            condition_numbers=(4,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_B),
            ),
        ),
        ParadigmCondition(
            name="AAAX",
            sequence=(std_A, std_A, std_A, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(5,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_A),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_A),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_A),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
            ),
        ),
        # B-sequence (BBBA family): oddball in position 4.
        ParadigmCondition(
            name="BBBA",
            sequence=(std_B, std_B, std_B, std_A),
            omission_position=None,
            probability=None,
            condition_numbers=(6, 7),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BXBA",
            sequence=(std_B, std_X, std_B, std_A),
            omission_position="p2",
            probability=None,
            condition_numbers=(8,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BBXA",
            sequence=(std_B, std_B, std_X, std_A),
            omission_position="p3",
            probability=None,
            condition_numbers=(9,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_A),
            ),
        ),
        ParadigmCondition(
            name="BBBX",
            sequence=(std_B, std_B, std_B, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(10,),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_B),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_B),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_B),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
            ),
        ),
        # R-sequence (random-control family): random stimulus identity, omissions in p2, p3, p4.
        ParadigmCondition(
            name="RRRR",
            sequence=(std_R, std_R, std_R, std_R),
            omission_position=None,
            probability=None,
            condition_numbers=tuple(range(11, 27)),  # [11-26]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RXRR",
            sequence=(std_R, std_X, std_R, std_R),
            omission_position="p2",
            probability=None,
            condition_numbers=tuple(range(27, 35)),  # [27-34]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RRXR",
            sequence=(std_R, std_R, std_X, std_R),
            omission_position="p3",
            probability=None,
            condition_numbers=(35, 37, 39, 41),
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_R),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
        ParadigmCondition(
            name="RRRX",
            sequence=(std_R, std_R, std_R, std_X),
            omission_position="p4",
            probability=None,
            condition_numbers=(36, 38, 40) + tuple(range(42, 51)),  # [36, 38, 40, 42-50]
            events=(
                ParadigmEvent(label="fx", onset_ms=0.0, code=event_codes["fx"]),
                ParadigmEvent(label="p1", onset_ms=100.0, code=event_codes["p1"], stimulus=std_R),
                ParadigmEvent(label="p2", onset_ms=200.0, code=event_codes["p2"], stimulus=std_R),
                ParadigmEvent(label="p3", onset_ms=300.0, code=event_codes["p3"], stimulus=std_R),
                ParadigmEvent(label="p4", onset_ms=400.0, code=event_codes["p4"], stimulus=std_X, is_omission=True),
                ParadigmEvent(label="rw", onset_ms=500.0, code=event_codes["rw"]),
            ),
        ),
    ]

    return Paradigm(
        name="standard_visual_omission",
        conditions=tuple(conditions),
        alignment_code=event_codes["p1"],
        alignment_label="p1",
        pre_stimulus_buffer_ms=1000.0,
        analysis_windows={
            "baseline": (-500.0, 0.0),
            "event": (0.0, 500.0),
            "post_event": (500.0, 1000.0),
        },
        event_codes=event_codes,
        metadata={
            "task_type": "visual_oddball_omission",
            "n_conditions": 12,
            "n_trials_per_condition": {c.name: len(c.condition_numbers) for c in conditions},
        },
    )


def dataset_spec(**kwargs: Any) -> DatasetSpec:
    """Return a DatasetSpec schema declaration."""
    return DatasetSpec(**kwargs)
