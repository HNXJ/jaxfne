"""Model: the immutable, runnable model built from a validated Configuration.

Split out of ``jaxfne/core.py`` (slice 4 of the core.py monolith split, see
``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports every symbol
here for backward compatibility -- import from ``jaxfne.core``, not this
module, unless you are working on core.py itself.

This is the biggest single integration surface in the split: Model reaches
into RuntimeConfig (jaxfne/_runtime_config.py) and Signals/Objective/metric
helpers (jaxfne/_signals.py) pervasively, and is itself the target of
``construct()`` (still in core.py, the genuine hub that pulls Configuration
+ Model together -- extracted last, after this module exists). A handful of
manifest/schema constants (``_JAXFNE_VERSION``, ``_RECEIPT_SCHEMA_VERSION``,
``_MANIFEST_SCHEMA_VERSION``, ``_SOURCE_PROXY_METADATA``,
``_KNOWN_READOUT_METRICS``) moved here too since Model is their heaviest
consumer; core.py imports them back for its own manifest()-area usage --
one-directional (core.py -> _model.py), not a cycle.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pathlib import Path
    from .optim import OptimizerSpec

from .emitters import (
    EdgeList,
    IzhikevichParams,
    make_edge_list_from_dense,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_homeostatic,
    simulate_edge_recurrent_izhikevich_hdp,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)
from .emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei
from .fields import probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, manifest as build_manifest
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN
from .experimental_hpc.contracts import SelectorSpec
from ._runtime_config import RuntimeConfig, _device_scope
from ._config import Configuration
from ._signals import (
    Signals,
    Objective,
    Simulation,
    StimulusSchedule,
    ReadoutSpec,
    ReadoutResult,
    ObjectiveReport,
    RunReceipt,
    TrialBatch,
    TrialResult,
    TrialBatchResult,
    ParadigmEvent,
    ParadigmCondition,
    _compute_all_metrics,
    _evaluate_gate_spec,
    _evaluate_loss_spec,
    _evaluate_regularizer_spec,
    _finite_or_none,
    _make_poisson_drive,
    _normalize_manifest_readout,
    _default_basis_dict,
)
@dataclass(frozen=True)
class MatrixParameterSpec:
    """Declarative specification for a tunable weight matrix parameter.

    Used in multi-parameter optimization to specify that a named parameter
    maps to a matrix (e.g., the synaptic weight matrix W) rather than a scalar.
    The mask field selects which matrix entries are subject to scaling:
    E_to_E, E_to_I, excitatory_to_all, or all.

    The name is always the dict key in the parameters argument to
    :func:; do **not** add a name field here.

    Attributes
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization scope; "current" means start from the
        model's existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.
    target : str
        Name of the emitter dataclass field to read/write, e.g. "W" (the
        default, the Izhikevich edge-weight matrix) or "G" (the
        homeostatic_ei emitter's conductance matrix). Backward compatible
        default keeps every existing caller's behavior unchanged.
    """

    mask: str
    bounds: tuple
    init: str = "current"
    trainable: bool = True
    target: str = "W"


def matrix_parameter(
    *,
    mask: str,
    bounds: tuple,
    init: str = "current",
    trainable: bool = True,
    target: str = "W",
) -> MatrixParameterSpec:
    """Create a matrix parameter specification for tuning weight matrices.

    Parameters
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization; "current" uses existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.
    target : str
        Name of the emitter dataclass field to read/write (default "W";
        e.g. "G0" for the homeostatic_ei emitter's conductance matrix).

    Returns
    -------
    MatrixParameterSpec
        Frozen specification object.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))
    >>> g_spec = jtfne.matrix_parameter(mask="all", bounds=(0.1, 5.0), target="G0")
    """
    return MatrixParameterSpec(mask=mask, bounds=bounds, init=init, trainable=trainable, target=target)


@dataclass
class TuneResult:
    """Result object returned by Model.tune() with multi-parameter optimization.

    This is a typed container for tuning results, with JSON-safe serialization
    via to_dict() method for reporting and logging.

    Attributes
    ----------
    best_parameters : dict[str, float]
        Optimized parameter values.
    best_score : float
        Best (lowest) objective score achieved.
    history : list[dict[str, Any]]
        Per-generation records with scores and parameter values.
    summary : dict[str, Any]
        High-level tuning summary (targets vs achieved, initial vs final scores, etc).
    model : Optional[Any]
        The model object (if returned by tuning; may be None for metadata-only runs).
    """

    best_parameters: dict[str, float]
    best_score: float
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    model: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary for serialization."""

        return json_safe({
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "history": self.history,
            "summary": self.summary,
        })

    def __iter__(self):
        """Support legacy tuple unpacking: ``model, report = tune(...)``.

        New code should use ``result.model`` and ``result.summary``.  The iterator
        remains to preserve existing notebooks and tests while surfacing a
        deprecation warning.
        """
        warnings.warn(
            "Tuple-unpacking TuneResult is deprecated; use result.model and result.summary.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield self.model
        yield self.summary


_JAXFNE_VERSION = "0.4.7"
_RECEIPT_SCHEMA_VERSION = "run_receipt_v0.0.21"
_MANIFEST_SCHEMA_VERSION = "manifest.v0.0.21"


# v0.0.21: explicit source proxy metadata.
# Documents what the current Izhikevich scaffold computes as the "source" field.
# Reading the edge/dense kernels: source_proxy = source_scale * (current_native
# + DEFAULT_SPIKE_IMPULSE_GAIN * spikes), where current_native = drive +
# recurrent_syn + noise. The gain constant lives in presets.py and is shared by
# every kernel variant in emitters.py (simulate_edge_recurrent_izhikevich and
# the dense variants) -- they must stay in sync or the double-count guard
# below breaks. No physical-amplitude claim is made; this remains an
# uncalibrated proxy. The double-count guard records that synaptic current
# enters the source only via the single proxy expression, not as a separate
# additive term.
_SOURCE_PROXY_METADATA: dict[str, Any] = {
    "source_model": "izhikevich_native_current_plus_spike_impulse_proxy",
    "source_mode": "native_current_plus_spike_impulse_proxy",
    "includes_native_current": True,
    "includes_drive_current": True,
    "includes_recurrent_synaptic_current": True,
    "includes_noise_current": True,
    "includes_spike_impulse": True,
    "spike_impulse_gain": DEFAULT_SPIKE_IMPULSE_GAIN,
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "physical_amplitude_calibrated": False,
    "double_count_synaptic_current_guard": (
        "single_proxy_expression_no_extra_synaptic_source"
    ),
}

_KNOWN_READOUT_METRICS = frozenset({
    "spike_rate_hz",
    "spike_count",
    "mean_V_m",
    "csd_abs_mean",
    "lfp_abs_mean",
    "source_abs_mean",
})


def stimulus_schedule(
    events: Sequence[Any],
    n_neurons: int,
    *,
    drive_amplitude: float = 5.0,
    event_duration_ms: float = 50.0,
) -> StimulusSchedule:
    """Build a :class:`StimulusSchedule` from a sequence of events.

    Each event may be a :class:`ParadigmEvent` or a dict-like with at least
    ``onset_ms``.  The ``drive_amplitude`` and ``event_duration_ms`` are the
    default values applied to all events that do not specify their own.

    Events that carry ``is_omission=True`` or an explicit ``amplitude=0`` inject
    zero drive (generic no-drive semantics, not cognitive omission logic).
    No calibrated-current or physical-amplitude claim is made.
    """
    ev_dicts: list[dict[str, Any]] = []
    for e in events:
        if isinstance(e, ParadigmEvent):
            amp = float(e.metadata.get("drive_amplitude", drive_amplitude))
            dur = float(e.metadata.get("event_duration_ms", event_duration_ms))
            is_drive = not e.is_omission and e.onset_ms is not None
            ev_dict = {
                "label": e.label,
                "onset_ms": float(e.onset_ms) if e.onset_ms is not None else 0.0,
                "duration_ms": dur,
                "amplitude": amp if is_drive else 0.0,
                "is_drive_event": is_drive,
            }
            if "target_indices" in e.metadata:
                ev_dict["target_indices"] = e.metadata["target_indices"]
            ev_dicts.append(ev_dict)
        else:
            d = dict(e)
            if "amplitude" not in d:
                d["amplitude"] = drive_amplitude
            if "duration_ms" not in d:
                d["duration_ms"] = event_duration_ms
            if "is_drive_event" not in d:
                d["is_drive_event"] = d.get("onset_ms") is not None and d["amplitude"] != 0.0
            ev_dicts.append(d)
    return StimulusSchedule(
        events=tuple(ev_dicts),
        n_neurons=int(n_neurons),
    )
@dataclass(frozen=True)
class _RuntimeReportAdapter:
    report: dict[str, Any]

    def runtime_report(self) -> dict[str, Any]:
        """Documented public function `runtime_report`."""
        return self.report



# --- Method implementations, split out (Phase 2 defragmentation, 2026-07-20)
# into sibling modules -- must be imported after the shared constants/classes
# above, since each submodule does `from ._model import ...` at ITS OWN
# module level for these names. Model class below delegates to these.
from ._model_simulate import (
    _simulate_arrays as _simulate_arrays_impl,
    _resolve_stimulus_schedule as _resolve_stimulus_schedule_impl,
    simulate as _simulate_impl,
    _simulate_homeostatic_ei as _simulate_homeostatic_ei_impl,
    last_homeostasis_diagnostics as _last_homeostasis_diagnostics_impl,
    last_hdp_diagnostics as _last_hdp_diagnostics_impl,
    simulate_condition as _simulate_condition_impl,
    simulate_batch as _simulate_batch_impl,
    run_trials as _run_trials_impl,
)
from ._model_readout import (
    run_receipt as _run_receipt_impl,
    compute_readout as _compute_readout_impl,
    probe as _probe_impl,
    record as _record_impl,
)
from ._model_evaluate import (
    evaluate as _evaluate_impl,
    evaluate_report as _evaluate_report_impl,
    _evaluate_group_rate_targets as _evaluate_group_rate_targets_impl,
    _evaluate_soft_rate_targets,
)
from ._model_tune import (
    tune as _tune_impl,
    _tune_multiparameter as _tune_multiparameter_impl,
    with_emitter_parameters as _with_emitter_parameters_method_impl,
    with_hdp_initial_state as _with_hdp_initial_state_impl,
    with_recurrent_coupling as _with_recurrent_coupling_impl,
    _model_with_scalar_parameter,
    _mask_for_parameter,
    _model_with_matrix_parameter,
    _model_with_parameters,
)
from ._model_manifest import manifest as _manifest_impl


@dataclass(frozen=True)
class Model:
    """Immutable, runnable model built from a validated :class:`Configuration`.

    Holds the source ``cfg``, the dynamic ``params`` pytree (arrays that may be
    tuned/traced), and ``static`` metadata (JIT-static, non-array). Construct via
    :func:`construct`; run via :func:`simulate` / :meth:`simulate`. Also exported
    as the alias ``Net``. The model is a computational scaffold — its field and
    probe outputs are proxy readouts, not calibrated physical signals.
    """

    cfg: Configuration
    params: dict[str, Any]
    static: dict[str, Any]

    def _require_izhikevich_emitter(self, method_name: str) -> "IzhikevichParams":
        """Guard for `Model` methods not yet generalized beyond the Izhikevich
        emitter family. Raises a clear, actionable error instead of an
        AttributeError from a missing Izhikevich-only field (e.g. `v0`/`W`)
        on a different emitter family's params dataclass (e.g.
        `HomeostaticEIParams`, which has no `v0`/`layer_labels`/`a`-`d`)."""
        emitter = self.params["emitter"]
        if not isinstance(emitter, IzhikevichParams):
            raise NotImplementedError(
                f"Model.{method_name}() is not yet supported for the "
                f"{type(emitter).__name__} emitter family -- only "
                "construct()/simulate()/probe()/evaluate()/tune() (matrix-parameter "
                "tuning) are supported for homeostatic_ei this pass."
            )
        return emitter

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe model metadata for notebook display."""
        emitter = self._require_izhikevich_emitter("summary")
        return json_safe({
            "config_hash": config_hash(self.cfg),
            "n_units": int(emitter.v0.shape[0]),
            "n_contacts": int(self.static.get("n_contacts", 16)),
            "claim_level": self.cfg.metadata.get("claim_level", "computational_scaffold"),
            "source_calibration_status": self.cfg.metadata.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "field_solver_status": self.cfg.metadata.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
        })


    def neuron_table(self) -> list[dict[str, Any]]:
        """Return declared neuron metadata rows for area/layer/cell-type grouping."""
        rows = self.static.get("neuron_metadata")
        if rows is not None:
            return [dict(row) for row in rows]
        emitter = self._require_izhikevich_emitter("neuron_table")
        layers = emitter.layer_labels or tuple("unspecified" for _ in emitter.labels)
        positions = self.params.get("positions")
        rows_out: list[dict[str, Any]] = []
        for idx, label in enumerate(emitter.labels):
            z_value = None
            try:
                z_value = float(positions[idx, 2]) if positions is not None else None
            except Exception:
                z_value = None
            rows_out.append({
                "neuron_id": int(idx),
                "area": "network",
                "layer": str(layers[idx]),
                "cell_type": str(label),
                "z": z_value,
            })
        return rows_out

    def checkpoint(self, path: str) -> "Path":
        """Persist the expensive construct() output (emitter/edge_list/positions
        arrays + their aux fields + static + cfg.metadata) for reload via
        :meth:`restore`, avoiding a second construct() call.

        Uses direct dataclass reconstruction (not JAX treedef unflatten) --
        the same verified-safe pattern as
        ``scripts/cortical_column_localized_workflow.py::save_column``/
        ``load_column`` (see ``skills/jaxfne-neural-tensor/SKILL.md``'s
        "Construct-once / checkpoint / reload" section for the two landmines
        this specifically avoids: a differently-sized dummy treedef silently
        producing a structurally-mismatched model, and reusing a pre-construct
        ``cfg.metadata`` that construct() has since mutated in place).
        """
        from pathlib import Path
        import json as _json
        import numpy as _np

        p = Path(path)
        emitter = self._require_izhikevich_emitter("checkpoint")
        edge_list: EdgeList = self.params["edge_list"]
        positions = self.params["positions"]
        _np.savez(
            p.with_suffix(".npz"),
            **{f"emitter_{f}": _np.asarray(getattr(emitter, f))
               for f in ("a", "b", "c", "d", "drive", "sign", "W", "v0", "u0", "source_scale")},
            **{f"edge_{f}": _np.asarray(getattr(edge_list, f))
               for f in ("pre", "post", "weight", "receptor_index", "tau_ms")},
            positions=_np.asarray(positions),
        )
        meta = {
            "schema": "model_checkpoint_v1",
            "emitter_labels": list(emitter.labels),
            "emitter_layer_labels": list(emitter.layer_labels) if emitter.layer_labels is not None else None,
            "emitter_source_calibration_status": emitter.source_calibration_status,
            "edge_source_calibration_status": edge_list.source_calibration_status,
            "static": json_safe(self.static),
            "cfg_metadata": json_safe(self.cfg.metadata),
        }
        p.with_suffix(".json").write_text(_json.dumps(meta, indent=2))
        return p

    @classmethod
    def restore(cls, path: str, cfg: Configuration) -> "Model":
        """Inverse of :meth:`checkpoint`.

        ``cfg`` must be a fresh, cheap (no ``construct()`` call) ``Configuration``
        built with the same kwargs used to build the checkpointed model -- its
        ``metadata`` is overwritten in place with the saved post-construct
        metadata (construct() mutates metadata; the checkpoint captured that
        mutated state, not the pre-construct declaration).
        """
        import dataclasses
        import json as _json
        import numpy as _np
        from pathlib import Path

        p = Path(path)
        meta = _json.loads(p.with_suffix(".json").read_text())
        if meta["schema"] != "model_checkpoint_v1":
            raise ValueError(f"Unknown checkpoint schema: {meta['schema']!r}")
        with _np.load(p.with_suffix(".npz")) as z:
            emitter = IzhikevichParams(
                a=jnp.array(z["emitter_a"]), b=jnp.array(z["emitter_b"]), c=jnp.array(z["emitter_c"]),
                d=jnp.array(z["emitter_d"]), drive=jnp.array(z["emitter_drive"]), sign=jnp.array(z["emitter_sign"]),
                W=jnp.array(z["emitter_W"]), v0=jnp.array(z["emitter_v0"]), u0=jnp.array(z["emitter_u0"]),
                source_scale=jnp.array(z["emitter_source_scale"]),
                labels=tuple(meta["emitter_labels"]),
                layer_labels=tuple(meta["emitter_layer_labels"]) if meta["emitter_layer_labels"] is not None else None,
                source_calibration_status=meta["emitter_source_calibration_status"],
            )
            edge_list = EdgeList(
                pre=jnp.array(z["edge_pre"]), post=jnp.array(z["edge_post"]), weight=jnp.array(z["edge_weight"]),
                receptor_index=jnp.array(z["edge_receptor_index"]), tau_ms=jnp.array(z["edge_tau_ms"]),
                source_calibration_status=meta["edge_source_calibration_status"],
            )
            positions = jnp.array(z["positions"])
        params = {"emitter": emitter, "positions": positions, "edge_list": edge_list}
        restored_cfg = dataclasses.replace(cfg, metadata=meta["cfg_metadata"])
        return cls(cfg=restored_cfg, params=params, static=meta["static"])

    def compile_connections(self, *, seed: int = 0, **kwargs: Any):
        """Compile this model's declared connection rules into sparse edges.

        Thin wrapper over :func:`jaxfne.compile_connection_rules`: resolves the
        rule selectors against this model's :meth:`neuron_table` and the
        ``metadata["circuit"]`` declarations. Declaration-only inputs in →
        finite sparse edge arrays out; no simulation numerics change.
        """
        from .connectivity import compile_connection_rules

        circuit = dict(self.cfg.metadata.get("circuit", {}))
        return compile_connection_rules(
            self.neuron_table(),
            circuit.get("connections", []),
            circuit.get("mechanisms", []),
            seed=seed,
            **kwargs,
        )

    def select(
        self,
        *,
        area: Optional[Any] = None,
        area_id: Optional[Any] = None,
        layer: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        ids: Optional[Sequence[int]] = None,
        allow_empty: bool = False,
    ) -> jax.Array:
        """Resolve semantic selectors to neuron row indices (does not mutate).

        Thin, non-mutating wrapper around :class:`SelectorSpec` over this model's
        :meth:`neuron_table`. Returns an int32 JAX array of row positions suitable
        for indexing the trailing (neuron) axis of V_m/spikes/sources. Empty
        matches raise ``ValueError`` unless ``allow_empty=True``. A requested
        field absent from the neuron table raises ``KeyError``.
        """
        spec = SelectorSpec(
            area=area,
            area_id=area_id,
            layer=layer,
            cell_type=cell_type,
            ids=tuple(ids) if ids is not None else None,
        )
        return spec.resolve(self.neuron_table(), allow_empty=allow_empty)


    def _simulate_arrays(self, *args, **kwargs):
        return _simulate_arrays_impl(self, *args, **kwargs)

    def _resolve_stimulus_schedule(self, *args, **kwargs):
        return _resolve_stimulus_schedule_impl(self, *args, **kwargs)

    def simulate(self, *args, **kwargs):
        return _simulate_impl(self, *args, **kwargs)

    def _simulate_homeostatic_ei(self, *args, **kwargs):
        return _simulate_homeostatic_ei_impl(self, *args, **kwargs)

    def last_homeostasis_diagnostics(self, *args, **kwargs):
        return _last_homeostasis_diagnostics_impl(self, *args, **kwargs)

    def last_hdp_diagnostics(self, *args, **kwargs):
        return _last_hdp_diagnostics_impl(self, *args, **kwargs)

    def simulate_condition(self, *args, **kwargs):
        return _simulate_condition_impl(self, *args, **kwargs)

    def simulate_batch(self, *args, **kwargs):
        return _simulate_batch_impl(self, *args, **kwargs)

    def run_trials(self, *args, **kwargs):
        return _run_trials_impl(self, *args, **kwargs)

    def run_receipt(self, *args, **kwargs):
        return _run_receipt_impl(self, *args, **kwargs)

    def compute_readout(self, *args, **kwargs):
        return _compute_readout_impl(self, *args, **kwargs)

    def probe(self, *args, **kwargs):
        return _probe_impl(self, *args, **kwargs)

    def record(self, *args, **kwargs):
        return _record_impl(self, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return _evaluate_impl(self, *args, **kwargs)

    def evaluate_report(self, *args, **kwargs):
        return _evaluate_report_impl(self, *args, **kwargs)

    def _evaluate_group_rate_targets(self, *args, **kwargs):
        return _evaluate_group_rate_targets_impl(self, *args, **kwargs)

    def tune(self, *args, **kwargs):
        return _tune_impl(self, *args, **kwargs)

    def _tune_multiparameter(self, *args, **kwargs):
        return _tune_multiparameter_impl(self, *args, **kwargs)

    def with_emitter_parameters(self, *args, **kwargs):
        return _with_emitter_parameters_method_impl(self, *args, **kwargs)

    def with_hdp_initial_state(self, *args, **kwargs):
        return _with_hdp_initial_state_impl(self, *args, **kwargs)

    def with_recurrent_coupling(self, *args, **kwargs):
        return _with_recurrent_coupling_impl(self, *args, **kwargs)

    def manifest(self, *args, **kwargs):
        return _manifest_impl(self, *args, **kwargs)



def _mean_pairwise_corr_proxy(spikes: jax.Array) -> jax.Array:
    x = spikes.astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.std(x, axis=0, keepdims=True) + 1e-6
    z = x / denom
    corr = (z.T @ z) / jnp.maximum(1, z.shape[0] - 1)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    return jnp.sum(jnp.abs(corr) * mask) / jnp.maximum(1.0, jnp.sum(mask))


def with_emitter_parameters(
    model: Model,
    *,
    a: "float | None" = None,
    b: "float | None" = None,
    c: "float | None" = None,
    d: "float | None" = None,
    drive_scale: "float | None" = None,
    a_per_neuron: "jax.Array | None" = None,
    b_per_neuron: "jax.Array | None" = None,
    c_per_neuron: "jax.Array | None" = None,
    d_per_neuron: "jax.Array | None" = None,
    drive_per_neuron: "jax.Array | None" = None,
) -> Model:
    """Functional wrapper for :meth:`Model.with_emitter_parameters`.

    Supports both scalar (uniform) and per-neuron (array) overrides.
    Per-neuron arrays take priority over scalars.
    Explicit None checks used — zero-valued JAX arrays handled correctly.
    """
    return model.with_emitter_parameters(
        a=a, b=b, c=c, d=d, drive_scale=drive_scale,
        a_per_neuron=a_per_neuron,
        b_per_neuron=b_per_neuron,
        c_per_neuron=c_per_neuron,
        d_per_neuron=d_per_neuron,
        drive_per_neuron=drive_per_neuron,
    )

