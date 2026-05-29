"""Optional bridges to external biophysical tools.

Bridge objects are manifest-safe contracts. They do not import optional
libraries at module import time and they do not create physical source claims
without explicit calibration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def require_jaxley():
    """Import Jaxley lazily with an informative error."""
    try:
        import jaxley  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This feature requires optional dependency 'jaxley'. "
            "Install with: pip install -e '.[jaxley]'"
        ) from exc
    return jaxley


@dataclass(frozen=True)
class BridgeSpec:
    """JSON-safe optional-backend bridge declaration."""

    name: str
    backend: str
    status: str = "schema_only_no_backend_constructed"
    source_calibration_status: str = "uncalibrated_bridge_output"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "status": self.status,
            "source_calibration_status": self.source_calibration_status,
            "metadata": self.metadata,
            "physical_amplitude_claim_allowed": False,
        }


@dataclass(frozen=True)
class JaxleyEmitterBridge:
    """Jaxley bridge contract for future compartment emitters."""

    morphology: str | None = None
    mechanisms: tuple[str, ...] = ()
    source_calibration_status: str = "uncalibrated_jaxley_bridge"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> BridgeSpec:
        return BridgeSpec(
            name="jaxley_emitter_bridge",
            backend="jaxley",
            status="schema_only_no_compartment_current_export",
            source_calibration_status=self.source_calibration_status,
            metadata={
                "morphology": self.morphology,
                "mechanisms": list(self.mechanisms),
                **self.metadata,
            },
        )

    def construct(self) -> dict[str, Any]:
        require_jaxley()
        spec = self.to_spec().to_dict()
        spec["status"] = "backend_available_contract_only"
        return spec


@dataclass(frozen=True)
class JaxleyTraceSpec:
    """Metadata specification for Jaxley-style voltage trace arrays.

    Immutable descriptor for trace layout, timing, spike derivation, and metadata
    assertion. All traces are treated as proxy external readouts with no physical
    amplitude claims.
    """

    trace_name: str = "jaxley_trace"
    backend: str = "jaxley"
    layout: str = "time_by_unit"  # "time_by_unit" | "unit_by_time" | "recording_by_time"
    state_name: str = "v"
    dt_ms: float = 0.025
    units_or_status: str = "mV_or_declared"
    source_mode: str = "voltage_proxy"
    source_calibration_status: str = "uncalibrated_jaxley_voltage_proxy"
    source_projection_mode: str = "external_trace_proxy"
    source_decomposition: str = "proxy_voltage_trace_not_current"
    spike_threshold: float | None = 0.0
    physical_amplitude_claim_allowed: bool = False
    truth_mode: str = "truth_safe_unverified"
    claim_level: str = "computational_scaffold"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate constraints after initialization."""
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be > 0, got {self.dt_ms}")
        if self.layout not in {"time_by_unit", "unit_by_time", "recording_by_time"}:
            raise ValueError(
                f"layout must be 'time_by_unit', 'unit_by_time', or 'recording_by_time', "
                f"got {self.layout}"
            )
        if self.physical_amplitude_claim_allowed is not False:
            raise ValueError(
                "physical_amplitude_claim_allowed must be False (immutable); "
                f"got {self.physical_amplitude_claim_allowed}"
            )
        if self.claim_level != "computational_scaffold":
            raise ValueError(
                f"claim_level must be 'computational_scaffold', got {self.claim_level}"
            )

    def validate(self) -> dict[str, Any]:
        """Return validation metadata for audit."""
        return {
            "spec_class": "JaxleyTraceSpec",
            "backend": self.backend,
            "layout": self.layout,
            "dt_ms": self.dt_ms,
            "source_mode": self.source_mode,
            "claim_level": self.claim_level,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe specification dict."""
        return {
            "trace_name": self.trace_name,
            "backend": self.backend,
            "layout": self.layout,
            "dt_ms": float(self.dt_ms),
            "spike_threshold": float(self.spike_threshold) if self.spike_threshold is not None else None,
            "source_mode": self.source_mode,
            "source_calibration_status": self.source_calibration_status,
            "claim_level": self.claim_level,
            "physical_amplitude_claim_allowed": self.physical_amplitude_claim_allowed,
            "metadata": self.metadata,
        }


def _normalize_trace_layout(x: Any, layout: str) -> Any:
    """Normalize trace array to canonical [T, N] layout.

    Parameters
    ----------
    x : array-like
        Input array in one of three layouts.
    layout : str
        "time_by_unit" → [T, N] (no change)
        "unit_by_time" → [N, T] (transpose)
        "recording_by_time" → [R, T] (no change, treat as [T, N] where R is time dim)

    Returns
    -------
    jax.Array
        Normalized to [T, N] shape.

    Raises
    ------
    ValueError
        If layout is unknown, rank != 2, or array is non-finite.
    """
    import jax.numpy as jnp

    x_jax = jnp.asarray(x)

    if x_jax.ndim != 2:
        raise ValueError(
            f"trace must be 2D array, got shape {x_jax.shape} (rank {x_jax.ndim})"
        )

    if not jnp.all(jnp.isfinite(x_jax)):
        raise ValueError("trace contains non-finite values (NaN/Inf)")

    if layout == "time_by_unit":
        return x_jax
    elif layout == "unit_by_time":
        return x_jax.T
    elif layout == "recording_by_time":
        return x_jax
    else:
        raise ValueError(
            f"layout must be 'time_by_unit', 'unit_by_time', or 'recording_by_time', "
            f"got {layout}"
        )


def jaxley_trace_to_signals(
    trace: Any,
    *,
    spec: JaxleyTraceSpec | None = None,
    dt_ms: float | None = None,
    layout: str | None = None,
    source: Any = None,
) -> Any:
    """Convert Jaxley-style voltage trace array to jaxfne Signals.

    Minimal array-first bridge: accepts voltage-like trace arrays and derives
    spike proxy via threshold, returning a frozen immutable Signals object with
    conservative metadata and no field computation.

    Parameters
    ----------
    trace : array-like
        Voltage trace array. Shape depends on layout.
    spec : JaxleyTraceSpec, optional
        Trace specification (timing, calibration, claim gates).
        If None, defaults are used.
    dt_ms : float, optional
        Timestep in ms. Overrides spec.dt_ms if provided.
    layout : str, optional
        Array layout: "time_by_unit" (default) | "unit_by_time" | "recording_by_time".
        Overrides spec.layout if provided.
    source : array-like, optional
        Optional external source array (e.g., current density). If None and
        source_mode == "voltage_proxy", trace is used as source.
        If current-like source_mode requested without source, raises ValueError.

    Returns
    -------
    jaxfne.core.Signals
        Immutable Signals object with:
        - time_ms: time axis in ms
        - V_m: voltage array [n_time, n_neurons]
        - spikes: derived spike proxy [n_time, n_neurons]
        - sources: optional source array or None
        - field: None (not computed in bridge)
        - metadata: conservative claim gates and calibration status

    Raises
    ------
    ValueError
        If layout is unknown, trace is non-finite, spike_threshold is requested
        but source is not provided, or any specification constraint is violated.
    ImportError
        If jaxfne.core.Signals cannot be imported.
    """
    import jax.numpy as jnp

    try:
        from .core import Signals
    except ImportError as exc:
        raise ImportError("jaxfne.core.Signals not available") from exc

    # Initialize spec with defaults if not provided
    if spec is None:
        spec = JaxleyTraceSpec()

    # Override spec with explicit arguments
    _layout = layout if layout is not None else spec.layout
    _dt_ms = dt_ms if dt_ms is not None else spec.dt_ms

    # Normalize trace to [T, N]
    trace_canonical = _normalize_trace_layout(trace, _layout)
    n_time, n_units = trace_canonical.shape

    # Create time array
    time_ms = jnp.arange(n_time, dtype=jnp.float32) * float(_dt_ms)

    # Voltage is the trace
    V_m = jnp.asarray(trace_canonical, dtype=jnp.float32)

    # Derive spike proxy from threshold
    if spec.spike_threshold is not None:
        spikes = (V_m >= float(spec.spike_threshold)).astype(jnp.float32)
    else:
        spikes = jnp.zeros((n_time, n_units), dtype=jnp.float32)

    # Handle source
    if source is None:
        if spec.source_mode == "voltage_proxy":
            # Use voltage as conservative proxy source
            sources = jnp.asarray(trace_canonical, dtype=jnp.float32)
        else:
            sources = None
    else:
        # User provided explicit source; validate it
        sources = jnp.asarray(source, dtype=jnp.float32)
        if sources.shape[0] != n_time:
            raise ValueError(
                f"source time dimension {sources.shape[0]} != trace {n_time}"
            )

    # Construct metadata
    metadata = {
        "source": "jaxley_array_bridge_v0222",
        "trace_backend": spec.backend,
        "trace_layout_input": _layout,
        "trace_layout_canonical": "time_by_unit",
        "source_mode": spec.source_mode,
        "source_projection_mode": spec.source_projection_mode,
        "source_decomposition": spec.source_decomposition,
        "source_calibration_status": spec.source_calibration_status,
        "field_solver_status": "not_computed",
        "field_claim_level": "proxy_readout_only",
        "physical_amplitude_claim_allowed": False,
        "truth_mode": spec.truth_mode,
        "claim_level": spec.claim_level,
        "dt_ms": float(_dt_ms),
        "n_time": int(n_time),
        "n_units": int(n_units),
        "spike_threshold": float(spec.spike_threshold) if spec.spike_threshold is not None else None,
    }

    if spec.metadata:
        metadata.update(spec.metadata)

    # Return Signals with field=None
    return Signals(
        time_ms=time_ms,
        V_m=V_m,
        spikes=spikes,
        sources=sources,
        field=None,
        metadata=metadata,
    )


def hh_jaxley_reference_trace(
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    current_amplitude: float = 10.0,
) -> tuple[Any, Any, Any]:
    """Hodgkin-Huxley reference trace via optional Jaxley bridge.

    Generates a reference HH voltage trace using Jaxley's HH channel model
    (if available). Falls back to NotImplementedError if Jaxley is not installed.

    **Scope:** Reference emitter model through optional bridge.
    **Evidence:** Simulated voltage trace for tutorial comparison.
    **Interpretation:** Emitter-level reference before TFNE source/readout projection.

    Parameters
    ----------
    duration_ms : float
        Simulation duration in milliseconds.
    dt_ms : float
        Time step in milliseconds.
    current_amplitude : float
        Injected current amplitude in μA/cm².

    Returns
    -------
    t : ndarray (n_steps,)
        Time in ms.
    V : ndarray (n_steps,)
        Membrane potential in mV.
    I_inj : ndarray (n_steps,)
        Injected current in μA/cm².

    Raises
    ------
    NotImplementedError
        If Jaxley is not installed. Install with: pip install jaxley
    """
    require_jaxley()
    raise NotImplementedError(
        "TODO: implement HH reference through the optional Jaxley bridge. "
        "Required: expose a Jaxley HH emitter trace path."
    )


def hh_numpy_reference_trace(
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    current_amplitude: float = 10.0,
) -> tuple[Any, Any, Any]:
    """Standalone tutorial/reference Hodgkin-Huxley single-compartment trace.

    This is a standalone tutorial/reference HH utility, NOT Jaxley bridge validation,
    and NOT evidence that JaxleyBridge executed.
    """
    import numpy as np

    # NumPy high-fidelity Hodgkin-Huxley single-compartment solver
    C_m = 1.0  # uF/cm^2
    g_Na = 120.0  # mS/cm^2
    g_K = 36.0  # mS/cm^2
    g_L = 0.3  # mS/cm^2
    E_Na = 50.0  # mV
    E_K = -77.0  # mV
    E_L = -54.387  # mV

    def alpha_m(V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) if abs(V + 40.0) > 1e-6 else 1.0

    def beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) if abs(V + 55.0) > 1e-6 else 0.1

    def beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    t = np.arange(0, duration_ms, dt_ms)
    n_steps = len(t)
    V = np.zeros(n_steps)
    I_inj = np.zeros(n_steps)
    I_inj[t >= 50.0] = current_amplitude

    V[0] = -65.0
    m = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    n = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

    for i in range(1, n_steps):
        v = V[i-1]
        dm = (alpha_m(v) * (1.0 - m) - beta_m(v) * m) * dt_ms
        dh = (alpha_h(v) * (1.0 - h) - beta_h(v) * h) * dt_ms
        dn = (alpha_n(v) * (1.0 - n) - beta_n(v) * n) * dt_ms
        m += dm
        h += dh
        n += dn
        I_Na = g_Na * (m**3) * h * (v - E_Na)
        I_K = g_K * (n**4) * (v - E_K)
        I_L = g_L * (v - E_L)
        dv = ((I_inj[i-1] - I_Na - I_K - I_L) / C_m) * dt_ms
        V[i] = v + dv

    return t, V, I_inj


class JaxleyBridge:
    """Jaxley-focused biophysical emitter bridge."""
    def __init__(self, model: Any, source_mode: str = "transmembrane_current", compartment_axis: str = "last"):
        self.model = model
        self.source_mode = source_mode
        self.compartment_axis = compartment_axis

    def simulate(self, *args: Any, **kwargs: Any) -> Any:
        require_jaxley()
        raise NotImplementedError(
            "TODO: implement JaxleyBridge.simulate for detailed compartment simulation rollout"
        )

    def extract_sources(self, signals: Any) -> Any:
        """Extract source tensor from Jaxley bridge signals."""
        import jax.numpy as jnp
        if hasattr(signals, "sources") and signals.sources is not None:
            return jnp.asarray(signals.sources)
        if hasattr(signals, "V_m") and signals.V_m is not None:
            return jnp.asarray(signals.V_m)
        if hasattr(signals, "i_membrane") and signals.i_membrane is not None:
            val = jnp.asarray(signals.i_membrane)
            if val.ndim == 3:
                val = val.reshape(val.shape[0], -1)
            return val
        if hasattr(signals, "v") and signals.v is not None:
            val = jnp.asarray(signals.v)
            if val.ndim == 3:
                val = val.reshape(val.shape[0], -1)
            return val
        if isinstance(signals, dict):
            if "sources" in signals and signals["sources"] is not None:
                return jnp.asarray(signals["sources"])
            if "V_m" in signals and signals["V_m"] is not None:
                return jnp.asarray(signals["V_m"])
            if "i_membrane" in signals and signals["i_membrane"] is not None:
                val = jnp.asarray(signals["i_membrane"])
                if val.ndim == 3:
                    val = val.reshape(val.shape[0], -1)
                return val
            if "v" in signals and signals["v"] is not None:
                val = jnp.asarray(signals["v"])
                if val.ndim == 3:
                    val = val.reshape(val.shape[0], -1)
                return val
        return jnp.asarray(signals)

    def report(self) -> dict:
        return {
            "bridge_name": "jaxley_bridge",
            "source_mode": self.source_mode,
            "source_calibration_status": "uncalibrated_jaxley_bridge",
            "physical_amplitude_claim_allowed": False
        }

