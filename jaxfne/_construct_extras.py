"""Paradigm/trial/receipt/manifest-adjacent module-level helpers: operator
status registry, the standard visual-omission paradigm, trial batching,
receipts, readout/dataset spec factories, and schema migration.

Split out of ``jaxfne/_construct.py`` (Phase 2 defragmentation, 2026-07-20,
part of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). ``jaxfne/_construct.py``
re-exports every symbol here for backward compatibility. This module has no
dependency on any other ``_construct_*`` submodule -- it is a leaf.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import jax

from ._config import _default_operator_status
from ._runtime_config import SurrogateConfig
from ._signals import (
    Simulation,
    Signals,
    DatasetSpec,
    TrialSpec,
    TrialBatch,
    TrialBatchResult,
    ReadoutSpec,
    RunReceipt,
    LaminarPopulation,
    LaminarSourceGeometry,
    ParadigmEvent,
    ParadigmCondition,
    Paradigm,
)
from ._model import Model, _JAXFNE_VERSION, _MANIFEST_SCHEMA_VERSION


def operator_status() -> dict[str, str]:
    """Return the current operator status registry for all declared operators.

    Returns a dict mapping operator symbol names to their readiness strings
    (e.g., ``"prototype_api"``, ``"not_implemented"``). This is a scaffold
    declaration; no operator has been empirically validated.

    Returns
    -------
    dict[str, str]
        Operator name to status string mapping.
    """
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

    Comparison: P1 onset (code 101) at t=0.
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
    std_X = "stimulus_omitted"
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
        comparison_code=event_codes["p1"],
        comparison_label="p1",
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


def trial_batch(
    conditions: Sequence[ParadigmCondition],
    n_reps: int = 1,
    seed: int = 0,
    seed_policy: str = "paired_by_replicate",
    batch_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TrialBatch:
    """Create a TrialBatch by repeating conditions.

    Correctly iterates reps then conditions to ensure deterministic ordering.
    Assigns unique trial_id in format "trial_{index:04d}_{condition_name}".

    Seed policy:
      - "paired_by_replicate" (default): seed = base_seed + replicate_index
      - "unique_per_trial": seed = base_seed + trial_index
    """
    if seed_policy not in {"unique_per_trial", "paired_by_replicate"}:
        raise ValueError(
            f"invalid_seed_policy: {seed_policy!r}; "
            "must be one of {'paired_by_replicate', 'unique_per_trial'}"
        )

    trials: list[TrialSpec] = []
    idx = 0
    for r in range(n_reps):
        for cond in conditions:
            t_id = f"trial_{idx:04d}_{cond.name}"
            if seed_policy == "unique_per_trial":
                trial_seed = seed + idx
            else:  # paired_by_replicate
                trial_seed = seed + r
            trials.append(
                TrialSpec(
                    trial_id=t_id,
                    condition=cond,
                    seed=trial_seed,
                    metadata={"rep": r},
                )
            )
            idx += 1
    return TrialBatch(
        trials=tuple(trials),
        batch_id=batch_id or f"batch_{seed}",
        metadata=metadata or {},
    )


def run_trials(
    model: Model, batch: TrialBatch, sim: Simulation, *, collect_errors: bool = False
) -> TrialBatchResult:
    """Execute a batch of trials using the model.

    Args:
        model: Model instance to run trials on.
        batch: TrialBatch with trial specifications.
        sim: Simulation parameters for each trial.
        collect_errors: If False (default), raise immediately on first trial failure.
                       If True, record failures in TrialResult and continue.

    Delegates to model.run_trials() for the actual execution.
    """
    return model.run_trials(batch, sim, collect_errors=collect_errors)


def run_receipt(
    model: "Model", signals: Signals, *, tags: Optional[dict[str, Any]] = None
) -> RunReceipt:
    """Build a RunReceipt for a completed simulation run.

    Convenience wrapper around Model.run_receipt().

    Args:
        model: Model that produced the signals.
        signals: Signals returned by model.simulate().
        tags: Optional user-supplied key-value metadata.

    Returns:
        RunReceipt with frozen truth gates and deterministic receipt_id.
    """
    return model.run_receipt(signals, tags=tags)


def provenance_receipt(
    branch: str = "unknown",
    sha: str = "unknown",
    dirty: bool = False,
) -> dict[str, Any]:
    """Capture release provenance atomically.

    Freezes branch, SHA, dirty flag, and jaxfne version metadata into a single
    JSON-safe dict for release auditing and reproducibility.

    Args:
        branch: Git branch name (e.g., "main", "feat/something"). Default: "unknown".
        sha: Git commit SHA (short or full). Default: "unknown".
        dirty: True if working tree has uncommitted changes. Default: False.

    Returns:
        dict[str, Any] (JSON-safe) with keys:
        - branch (str)
        - sha (str)
        - dirty (bool)
        - jaxfne_version (str, from _JAXFNE_VERSION)
        - config_schema_version (str, from _JAXFNE_CONFIG_SCHEMA_VERSION)
        - manifest_schema_version (str, from _MANIFEST_SCHEMA_VERSION)
        - timestamp (str, ISO-8601 format)

    Example:
        receipt = provenance_receipt(branch="main", sha="abc123def456", dirty=False)
        json.dumps(receipt, allow_nan=False)  # Always succeeds
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    receipt = {
        "branch": branch,
        "sha": sha,
        "dirty": dirty,
        "jaxfne_version": _JAXFNE_VERSION,
        "config_schema_version": _JAXFNE_CONFIG_SCHEMA_VERSION,
        "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
        "timestamp": timestamp,
    }

    # Verify JSON-safe by attempting serialization
    try:
        json.dumps(receipt, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"provenance_receipt produced non-JSON-safe dict: {e}"
        ) from e

    return receipt


def get_signal(obj: Any, key: str, **kwargs: Any) -> Any:
    """Thin free-function accessor that delegates to :meth:`Signals.get`.

    This is a convenience wrapper only; it does not implement any signal logic
    of its own. ``obj`` must be a :class:`Signals` instance.
    """
    if isinstance(obj, Signals):
        return obj.get(key, **kwargs)
    raise TypeError(
        f"get_signal expects a Signals instance, got {type(obj).__name__}"
    )


def readout_spec(
    name: str,
    metric: str,
    *,
    time_window_ms: Optional[tuple[float, float]] = None,
    n_contacts_slice: Optional[tuple[int, int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ReadoutSpec:
    """Build a ReadoutSpec for declarative feature extraction.

    Args:
        name: Unique label for this readout spec.
        metric: One of _KNOWN_READOUT_METRICS (spike_rate_hz, spike_count,
                mean_V_m, csd_abs_mean, lfp_abs_mean, source_abs_mean).
        time_window_ms: Optional (start_ms, end_ms) temporal slice.
        n_contacts_slice: Optional (start, end) contact-depth slice for field modes.
        metadata: Optional user-supplied metadata dict.

    Returns:
        ReadoutSpec (frozen, JSON-safe).
    """
    return ReadoutSpec(
        name=name,
        metric=metric,
        time_window_ms=time_window_ms,
        n_contacts_slice=n_contacts_slice,
        metadata=metadata or {},
    )


def dataset_spec(**kwargs: Any) -> DatasetSpec:
    """Return a DatasetSpec schema declaration."""
    return DatasetSpec(**kwargs)


def surrogate_config(**kwargs: Any) -> SurrogateConfig:
    """Return a SurrogateConfig declaration for an Optax gradient path."""
    return SurrogateConfig(**kwargs)


def laminar_source_geometry(
    populations: Sequence["LaminarPopulation"],
) -> "LaminarSourceGeometry":
    """Build a :class:`LaminarSourceGeometry` from an ordered population sequence.

    Depth overlap between populations is allowed; co-located cell types sharing
    a layer band are anatomically expected. Hard validation errors are raised only
    for invalid depth ranges, zero n_units, or empty population list.
    No physical-amplitude or calibration claim is made.
    """
    pops = tuple(populations)
    if not pops:
        raise ValueError("laminar_source_geometry requires at least one LaminarPopulation")
    issues: list[str] = []
    for p in pops:
        v = p.validate()
        if not v["valid"]:
            issues.extend([f"{p.name}:{i}" for i in v["issues"]])
    if issues:
        raise ValueError(f"Invalid LaminarPopulation(s): {issues}")
    n_total = sum(p.n_units for p in pops)
    return LaminarSourceGeometry(populations=pops, n_units_total=n_total)


def enable_x64() -> dict[str, Any]:
    """Enable JAX float64 mode before constructing arrays and report status."""
    jax.config.update("jax_enable_x64", True)
    return {"x64_enabled": bool(jax.config.read("jax_enable_x64")), "status": "enabled"}


# ──────────────────────────────────────────────────────────────
# v0.0.17 readout spec
# ──────────────────────────────────────────────────────────────

_OBJECTIVE_REPORT_SCHEMA_VERSION = "objective_report.v0.0.18"


# ──────────────────────────────────────────────────────────────
# v0.0.15 config foundation
# ───────────────────────────────────────────────��──────────────

_JAXFNE_CONFIG_SCHEMA_VERSION = "jaxfne.config.v0.0.16"

_REQUIRED_CONFIG_SECTIONS = frozenset(
    {"schema_version", "run", "truth", "network", "emitter", "field", "probes"}
)

_RECOGNIZED_OPTIONAL_CONFIG_SECTIONS = frozenset({
    "runtime",
    "geometry",
    "paradigm",
    "trials",
    "stimulus",
    "features",
    "objective",
    "targets",
    "validation",
    "output",
    "metadata",
})

# _CONSERVATIVE_TRUTH_DEFAULTS moved to jaxfne/_signals.py (only consumer is
# _evaluate_gate_spec, which lives there) and re-exported above.

#: Canonical field-solver-status values. ``linear_solver`` is the shipped laminar
#: proxy (a linear readout operator, no PDE); ``pde_solver`` is reserved for a future
#: elliptic/volume-conductor solve gated on boundary/gauge/residual/convergence tests.
_VALID_FIELD_SOLVER_STATUS = (None, "linear_solver", "pde_solver")


def migrate_schema(meta: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy truth/metadata dict to the canonical truth-gate schema.

    Pure, JSON-safe rewrite applied on load of older manifests/configs. Legacy key/
    value names are pre-v0.4.x field names, retired and unused by any current code
    path -- kept here as plain literals (not obfuscated) so a grep audit for these
    names still finds this migration path, not just live usage.

    - legacy physical-amplitude key -> ``physical_amplitude_calibrated`` (bool kept)
    - legacy laminar field-solver value -> ``linear_solver``
    - legacy proxy field-claim value -> ``proxy_readout``
    - drop the legacy truth-mode key

    Returns a new dict; the input is not mutated.
    """
    _amp = "physical_amplitude_claim_allowed"  # retired pre-v0.4.x key
    _solver = "laminar_proxy_no_pde"  # retired pre-v0.4.x value
    _claim = "proxy_readout_only"  # retired pre-v0.4.x value
    _tmode = "truth_mode"  # retired pre-v0.4.x key
    out: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if key == _tmode:
            continue
        if key == _amp:
            out["physical_amplitude_calibrated"] = value
            continue
        if key == "field_solver_status" and value == _solver:
            out[key] = "linear_solver"
            continue
        if key == "field_claim_level" and value == _claim:
            out[key] = "proxy_readout"
            continue
        out[key] = value
    from ._config import clamp_truth_gate_metadata

    return clamp_truth_gate_metadata(out)


