"""Top-level factory helpers (``configuration``, ``runtime``, ``simulation``,
``objective``, ``rate_targets``) and Suite No. 2 preset ``Configuration``
builders (``suite2_*_config``, ``suite2_run_bundle``).

Split out of ``jaxfne/_construct.py`` (Phase 2 defragmentation, 2026-07-20,
part of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). ``jaxfne/_construct.py``
re-exports every symbol here for backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from ._config import Configuration
from ._runtime_config import RuntimeConfig
from ._signals import Simulation, Signals, Objective
from ._model import Model, TuneResult
from .io import json_safe
from ._construct_population import (
    _SUITE2_LAYER_FRACTIONS,
    _SUITE2_LAYER_CELL_TYPES_V1,
    _SUITE2_LAYER_CELL_TYPES_V4,
    _SUITE2_PROXY_MODES,
    _default_layer_cell_types,
)


def configuration() -> Configuration:
    """Return a fresh, empty :class:`Configuration` builder.

    Entry point for the chainable configuration grammar
    (``configuration().network(...).emitter(...)...``). Equivalent to calling
    ``Configuration()`` directly.
    """
    return Configuration()


def runtime(
    backend: str = "auto",
    dtype: str = "float32",
    jit: bool = False,
    vmap: bool = False,
    precision: str = "default",
    seed: int = 0,
    n_steps: int = 0,
    recurrent_backend: str = "dense",
    synaptic_kernel: str = "exponential",
    # v0.0.3 compatibility names.
    device_type: str | None = None,
    dtype_primary: str | None = None,
    x64_enabled: bool | None = None,
) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` (JAX backend/dtype/jit/vmap policy).

    Thin keyword factory for :class:`RuntimeConfig`; ``device_type``,
    ``dtype_primary`` and ``x64_enabled`` are accepted as v0.0.3-compatibility
    aliases. This configures execution only — it makes no numerical or
    scientific claim about the simulation.
    """
    return RuntimeConfig(
        backend=backend,
        dtype=dtype,
        jit=jit,
        vmap=vmap,
        precision=precision,
        seed=seed,
        n_steps=n_steps,
        recurrent_backend=recurrent_backend,
        synaptic_kernel=synaptic_kernel,
        device_type=device_type,
        dtype_primary=dtype_primary,
        x64_enabled=x64_enabled,
    )


def runtime_report(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Return a JSON-safe dict describing the resolved JAX runtime.

    Reports the effective backend, dtype policy, and jit/vmap flags for the
    given :class:`RuntimeConfig` (or the default if ``None``). Use it to confirm
    ``actual_dtype`` and that jit/vmap are enabled before profiling.
    """
    return (runtime_config or RuntimeConfig()).runtime_report()


def simulation(**kwargs: Any) -> Simulation:
    """Build a :class:`Simulation` spec from keyword arguments.

    Thin factory for :class:`Simulation` (``duration_ms``, ``dt_ms``, ``seed``,
    ``plasticity``, recording flags, ...). Equivalent to ``Simulation(**kwargs)``.
    """
    return Simulation(**kwargs)


def objective() -> Objective:
    """Return a fresh, empty :class:`Objective` specification.

    Starting point for declaring losses, regularizers, and diagnostic gates.
    Equivalent to ``Objective()``.
    """
    return Objective()


def rate_targets(
    groups: dict[str, Any],
    targets_hz: dict[str, float],
    weights: Optional[dict[str, float]] = None,
    *,
    burn_in_ms: float = 0.0,
    window_end_ms: Optional[float] = None,
    epsilon: float = 1e-6,
) -> Objective:
    """Create a multi-group firing-rate objective.

    This factory creates an Objective with kind="group_rate_targets" that
    encodes group-wise firing-rate targets. When passed to Model.tune(),
    the optimization loop computes group-wise rates and minimizes
    squared-relative-error loss.

    Parameters
    ----------
    groups : dict[str, Any]
        Mapping from group names to neuron indices.
        E.g., {"first_half": np.arange(0, 24), "second_half": np.arange(24, 48)}.
    targets_hz : dict[str, float]
        Mapping from group names to target firing rates in Hz.
        E.g., {"first_half": 5.0, "second_half": 10.0}.
    weights : Optional[dict[str, float]]
        Mapping from group names to loss weights (default: 1.0 each).
    burn_in_ms : float
        Start of the measurement window in milliseconds.
    window_end_ms : float, optional
        End of the measurement window in milliseconds. ``None`` uses the
        simulation end.
    epsilon : float
        Positive denominator floor for relative rate loss.

    Returns
    -------
    Objective
        Objective with kind="group_rate_targets", storing groups and targets
        in metadata for use by optimization loops.

    Example
    -------
    >>> import numpy as np
    >>> import jaxfne as jtfne
    >>> objectives = jtfne.rate_targets(
    ...     groups={"first": np.arange(0, 24), "second": np.arange(24, 48)},
    ...     targets_hz={"first": 5.0, "second": 10.0},
    ... )
    >>> optimizer = jtfne.agsdr(parameters={"drive_scale_a": (0.3, 2.0)}, generations=8)
    >>> result = model.tune(objectives=objectives, optimizer=optimizer)
    >>> result.best_score
    """
    import numpy as np

    # Validate
    if not groups or not targets_hz:
        raise ValueError("groups and targets_hz must be non-empty")
    if set(groups.keys()) != set(targets_hz.keys()):
        raise ValueError("Group names must match between groups and targets_hz")

    if weights is None:
        weights = {name: 1.0 for name in groups.keys()}
    if not (math.isfinite(float(burn_in_ms)) and float(burn_in_ms) >= 0.0):
        raise ValueError("burn_in_ms must be finite and non-negative")
    if window_end_ms is not None and not math.isfinite(float(window_end_ms)):
        raise ValueError("window_end_ms must be finite when provided")
    if window_end_ms is not None and float(window_end_ms) <= float(burn_in_ms):
        raise ValueError("window_end_ms must be greater than burn_in_ms")
    if not (math.isfinite(float(epsilon)) and float(epsilon) > 0.0):
        raise ValueError("epsilon must be finite and positive")

    # Convert to JSON-safe lists
    groups_lists = {}
    for name, indices in groups.items():
        arr = np.asarray(indices, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Group '{name}' indices must be 1D")
        groups_lists[name] = arr.tolist()

    # Create objective with group metadata
    return Objective(
        name="rate_targets",
        kind="group_rate_targets",
        losses=[],
        regularizers=[],
        gates=[],
    ).gate(
        name="rate_targets_metadata",
        threshold=0,  # Threshold unused for optimizer-computed score
        criterion="below",
        metadata={
            "groups": groups_lists,
            "targets_hz": {k: float(v) for k, v in targets_hz.items()},
            "weights": {k: float(weights.get(k, 1.0)) for k in groups.keys()},
            "burn_in_ms": float(burn_in_ms),
            "window_end_ms": (
                float(window_end_ms) if window_end_ms is not None else None
            ),
            "epsilon": float(epsilon),
        },
    )


def suite2_celltype_presets() -> dict[str, dict[str, float | str]]:
    """Return compact E/PV/SST/VIP reduced-emitter preset metadata."""
    return {
        "E": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0, "drive": 5.0, "role": "excitatory_pyramidal_like"},
        "PV": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0, "drive": 3.0, "role": "fast_inhibitory_like"},
        "SST": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0, "drive": 3.5, "role": "somatostatin_like"},
        "VIP": {"a": 0.02, "b": -0.10, "c": -55.0, "d": 6.0, "drive": 3.0, "role": "disinhibitory_like"},
    }


def suite2_single_neuron_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, cell_type: str = "E") -> Configuration:
    """Build the Suite No. 2 one-emitter configuration."""
    fractions = {"E": 0.0, "PV": 0.0, "SST": 0.0, "VIP": 0.0}
    if cell_type not in fractions:
        raise ValueError(f"cell_type must be one of {tuple(fractions)}")
    fractions[cell_type] = 1.0
    return (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("single", layers=["uniform_3d"], n=1)
        .cell_types(fractions)
        .uniform3d(radius_mm=0.010, height_mm=0.010)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=4)
    )


def suite2_four_celltype_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> Configuration:
    """Build the Suite No. 2 four-emitter E/PV/SST/VIP configuration."""
    return (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("celltype_panel", layers=["uniform_3d"], n=4)
        .cell_types({"E": 0.25, "PV": 0.25, "SST": 0.25, "VIP": 0.25})
        .uniform3d(radius_mm=0.030, height_mm=0.10)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=4)
    )


def suite2_net1_config(
    *,
    seed: int = 7,
    n: int = 100,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    drives: Mapping[str, float] | None = None,
) -> Configuration:
    """Build net1: a uniformly sampled 3D E/PV/SST/VIP column."""
    cfg = (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("net1", layers=["uniform_3d"], n=int(n))
        .cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .uniform3d(radius_mm=0.25, height_mm=1.60)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45)
        .probes(_SUITE2_PROXY_MODES, n_contacts=16)
    )
    cfg = cfg.cell_type_drives(drives or {"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
    return cfg


def suite2_v1_v4_config(
    *,
    seed: int = 7,
    n_per_area: int = 400,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    v1_layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
    v4_layer_cell_types: Mapping[str, Mapping[str, float]] | None = None,
) -> Configuration:
    """Build the Suite No. 2 V1-V4 laminar scaffold with six layers per area."""
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]
    cfg = (
        Configuration()
        .runtime(seed=seed, dtype="float32", duration_ms=duration_ms, dt_ms=dt_ms)
        .column("V1", layers=layers, n=int(n_per_area))
        .column("V4", layers=layers, n=int(n_per_area))
        .cell_types({"E": 0.50, "PV": 0.20, "SST": 0.15, "VIP": 0.15})
        .layer_fractions(_SUITE2_LAYER_FRACTIONS, _default_layer_cell_types())
        .area_layer_cell_types("V1", v1_layer_cell_types or _SUITE2_LAYER_CELL_TYPES_V1)
        .area_layer_cell_types("V4", v4_layer_cell_types or _SUITE2_LAYER_CELL_TYPES_V4)
        .connectivity(
            within_area="all_to_all_uniform_random",
            within_gain=0.35,
            feedforward="V1_L2L3_E_to_V4_L4_E",
            feedback="V4_L2L3_E_to_V1_L5L6_L1_E",
            feedforward_gain=0.65,
            feedback_gain=0.50,
        )
        .suite2_interarea(True)
        .cell_type_drives({"E": 4.0, "PV": 2.0, "SST": 2.2, "VIP": 2.0})
        .probes(_SUITE2_PROXY_MODES, n_contacts=24)
    )
    return cfg


def suite2_simulation(
    *,
    seed: int = 7,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    noise_amplitude: float | None = None,
    noise_rate_hz: float = 2.0,
    target: str = "all",
    jit: bool = False,
    recurrent_backend: str = "dense",
) -> Simulation:
    """Create a Suite No. 2 simulation with deterministic runtime metadata."""
    runtime_cfg = RuntimeConfig(dtype="float32", seed=seed, jit=jit, recurrent_backend=recurrent_backend)
    poisson = None
    if noise_amplitude is not None:
        poisson = {"rate_hz": float(noise_rate_hz), "amplitude": float(noise_amplitude), "target": target, "seed": int(seed) + 7919}
    return Simulation(
        duration_ms=float(duration_ms),
        dt_ms=float(dt_ms),
        seed=int(seed),
        record_sources=True,
        record_fields=True,
        runtime=runtime_cfg,
        poisson_drive=poisson,
    )


def suite2_tune_noise_agsdr_adam(
    model: Model,
    *,
    simulation: Simulation | None = None,
    target_rate_hz: tuple[float, float] = (5.0, 10.0),
    amplitudes: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    poisson_rate_hz: float = 2.0,
    adam_steps: int = 2,
    learning_rate: float = 0.20,
    finite_difference_eps: float = 0.05,
    seed: int = 7,
) -> TuneResult:
    """Tune Poisson-drive amplitude toward a target mean firing-rate range.

    Multi-stage approach: outer stage evaluates an AGSDR-style candidate population
    (black-box sweep over noise amplitudes). Inner stage applies finite-difference
    Adam updates for local refinement. Current implementation: finite-difference Adam
    with black-box candidate sweep. Variance-balanced adaptive self-supervision is
    reserved for a future optimizer path with adaptive alpha/learning-rate schedules.
    This path tunes a solver-tuned drive parameter and preserves relative
    proxy-unit metadata.
    """
    sim0 = simulation or suite2_simulation(seed=seed, duration_ms=1000.0, dt_ms=0.1)
    lo, hi = float(target_rate_hz[0]), float(target_rate_hz[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError("target_rate_hz must be a positive (low, high) tuple")
    target_mid = 0.5 * (lo + hi)

    def run_amp(amp: float, run_seed: int) -> tuple[float, float, Signals]:
        """Documented public function `run_amp`."""
        sim = replace(
            sim0,
            seed=int(run_seed),
            poisson_drive={"rate_hz": float(poisson_rate_hz), "amplitude": float(max(0.0, amp)), "target": "all", "seed": int(run_seed) + 7919},
        )
        sig = model.simulate(sim)
        rate = float(sig.summary()["spike_rate_hz_mean"] or 0.0)
        if lo <= rate <= hi:
            loss = 0.0
        else:
            loss = (rate - target_mid) * (rate - target_mid)
        return loss, rate, sig

    history: list[dict[str, Any]] = []
    best_amp = float(amplitudes[0])
    best_loss, best_rate, best_sig = run_amp(best_amp, seed)
    history.append({"stage": "agsdr_population", "amplitude": best_amp, "rate_hz": best_rate, "loss": best_loss})
    for idx, amp in enumerate(amplitudes[1:], start=1):
        loss, rate, sig = run_amp(float(amp), seed + idx)
        history.append({"stage": "agsdr_population", "amplitude": float(amp), "rate_hz": rate, "loss": loss})
        if loss < best_loss:
            best_amp, best_loss, best_rate, _ = float(amp), loss, rate, sig

    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    amp = best_amp
    for step in range(1, int(adam_steps) + 1):
        loss0, rate0, _ = run_amp(amp, seed + 100 + step * 2)
        loss1, _, _ = run_amp(amp + finite_difference_eps, seed + 101 + step * 2)
        grad = (loss1 - loss0) / max(float(finite_difference_eps), 1e-6)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        amp = max(0.0, amp - float(learning_rate) * m_hat / (math.sqrt(v_hat) + eps))
        loss_new, rate_new, sig_new = run_amp(amp, seed + 200 + step)
        history.append({"stage": "finite_difference_adam", "step": step, "amplitude": amp, "rate_hz": rate_new, "loss": loss_new, "gradient_estimate": grad})
        if loss_new < best_loss:
            best_amp, best_loss, best_rate, _ = amp, loss_new, rate_new, sig_new

    summary = json_safe({
        "optimizer": "AGSDR_outer_finite_difference_Adam_inner",
        "target_rate_hz": [lo, hi],
        "best_noise_amplitude": best_amp,
        "best_rate_hz": best_rate,
        "best_loss": best_loss,
        "history_length": len(history),
        "tuned_parameter": "simulation.poisson_drive.amplitude",
        "units_or_status": "reduced_native_drive_units_relative_proxy",
        "field_solver_status": model.cfg.metadata.get("field_solver_status", "linear_solver"),
        "physical_amplitude_calibrated": False,
    })
    return TuneResult(
        best_parameters={"noise_amplitude": float(best_amp), "poisson_rate_hz": float(poisson_rate_hz)},
        best_score=float(best_loss),
        history=json_safe(history),
        summary=summary,
        model=model,
    )


def suite2_run_bundle(model: Model, *, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, noise_amplitude: float | None = None) -> dict[str, Any]:
    """Run simulation, readouts, manifest, and receipt for Suite No. 2 notebooks."""
    sim = suite2_simulation(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, noise_amplitude=noise_amplitude)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=list(_SUITE2_PROXY_MODES))
    manifest = model.manifest(signals=signals, readout=readout)
    return {"simulation": sim, "signals": signals, "readout": readout, "manifest": manifest, "neuron_table": model.neuron_table()}


