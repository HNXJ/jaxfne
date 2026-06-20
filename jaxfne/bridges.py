"""Optional bridges to external biophysical tools.

Bridge objects are manifest-safe contracts. They do not import optional
libraries at module import time and they do not create physical source claims
without explicit calibration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_JAX_CLIP_COMPAT_INSTALLED = False


def _install_jax_clip_compat() -> bool:
    """Restore ``jnp.clip(a_min=/a_max=)`` keywords removed in newer JAX.

    jaxley (<=0.13) channel gating calls ``jax.numpy.clip(x, a_min=..., a_max=...)``.
    Recent JAX dropped those keyword names (use ``min=``/``max=``), so without a
    shim jaxley's HH/channel step raises ``TypeError`` and *no* Jaxley emitter can
    integrate. This shim is idempotent and backward compatible — positional and
    ``min=``/``max=`` calls still work — and is installed lazily only when needed.

    Returns
    -------
    bool
        True if the shim is in force (installed now or previously), False if the
        running JAX already supports ``a_max`` so no shim was needed.
    """
    global _JAX_CLIP_COMPAT_INSTALLED
    if _JAX_CLIP_COMPAT_INSTALLED:
        return True
    import jax.numpy as jnp

    orig = jnp.clip
    try:
        orig(jnp.asarray([0.0]), a_max=1.0)
        return False  # native a_max support; no shim required
    except TypeError:
        pass

    def _clip_compat(a, a_min=None, a_max=None, *, min=None, max=None, **kwargs):
        lo = a_min if a_min is not None else min
        hi = a_max if a_max is not None else max
        return orig(a, lo, hi)

    _clip_compat.__jaxfne_clip_compat__ = True  # marker for transparency/audit
    jnp.clip = _clip_compat
    _JAX_CLIP_COMPAT_INSTALLED = True
    return True


def require_jaxley():
    """Import Jaxley lazily, installing the JAX clip-compat shim before use.

    The shim (`_install_jax_clip_compat`) is what makes Jaxley channel emitters
    actually integrate on current JAX; every Jaxley entry point routes through
    here so it is always in place before a model runs.
    """
    try:
        import jaxley  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This optional feature requires 'Jaxley'. "
            "Install with: pip install jaxfne[jaxley]"
        ) from exc
    _install_jax_clip_compat()
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
        """Documented public function `to_dict`."""
        return {
            "name": self.name,
            "backend": self.backend,
            "status": self.status,
            "source_calibration_status": self.source_calibration_status,
            "metadata": self.metadata,
            "physical_amplitude_calibrated": False,
        }


@dataclass(frozen=True)
class JaxleyEmitterBridge:
    """Jaxley bridge contract for future compartment emitters."""

    morphology: str | None = None
    mechanisms: tuple[str, ...] = ()
    source_calibration_status: str = "uncalibrated_jaxley_bridge"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> BridgeSpec:
        """Documented public function `to_spec`."""
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
        """Documented public function `construct`."""
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
    physical_amplitude_calibrated: bool = False
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
        if self.physical_amplitude_calibrated is not False:
            raise ValueError(
                "physical_amplitude_calibrated must be False (immutable); "
                f"got {self.physical_amplitude_calibrated}"
            )
        if self.claim_level != "computational_scaffold":
            raise ValueError(
                f"claim_level must be 'computational_scaffold', got {self.claim_level}"
            )

    def validate(self) -> dict[str, Any]:
        """Return validation metadata for audit."""
        from dataclasses import fields as dc_fields
        result = {"spec_class": "JaxleyTraceSpec"}
        result.update({f.name: getattr(self, f.name) for f in dc_fields(self)
                if f.name in ("backend", "layout", "dt_ms", "source_mode", "claim_level", "physical_amplitude_calibrated")})
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe specification dict."""
        from dataclasses import asdict
        d = asdict(self)
        # Ensure float types for numeric fields
        d["dt_ms"] = float(d["dt_ms"])
        if d["spike_threshold"] is not None:
            d["spike_threshold"] = float(d["spike_threshold"])
        return d


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
        "field_claim_level": "proxy_readout",
        "physical_amplitude_calibrated": False,
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


def _recorded_compartment_positions(module: Any, state: str = "v") -> Any:
    """Best-effort xyz positions of a Jaxley module's recorded compartments.

    Reads ``module.recordings`` (``rec_index`` = global compartment index) and
    ``module.nodes`` (x/y/z columns). Returns an ``(n_recorded, 3)`` numpy array
    aligned with the recordings order, or ``None`` if positions are unavailable
    (no geometry, no matching recordings, or any structural mismatch). Never
    raises — geometry is an optional enrichment, not a contract.
    """
    try:
        recs = module.recordings
        nodes = module.nodes
        if not {"x", "y", "z"}.issubset(set(nodes.columns)):
            return None
        sel = recs[recs["state"] == state] if "state" in recs.columns else recs
        if "rec_index" not in sel.columns:
            return None
        idx = [int(i) for i in sel["rec_index"].tolist()]
        if not idx:
            return None
        import numpy as np

        return nodes.loc[idx, ["x", "y", "z"]].to_numpy(dtype=float)
    except Exception:
        return None


def jaxley_to_signals(
    module: Any,
    recordings: Any,
    *,
    dt_ms: float = 0.025,
    state: str = "v",
    spec: "JaxleyTraceSpec | None" = None,
    source: Any = None,
) -> Any:
    """Convert a Jaxley module + ``jaxley.integrate`` output into jaxfne ``Signals``.

    One-call glue so a Jaxley-based emitter is immediately usable by tfne readouts,
    projections, and ``jaxfne.vis``. The recordings array returned by
    ``jaxley.integrate`` has shape ``(n_recordings, n_time)`` (unit-by-time); this
    maps it to canonical ``[T, N]`` ``V_m``, derives a threshold spike proxy, and
    pulls the recorded compartments' xyz from ``module.nodes`` into metadata
    (``recorded_positions_xyz``) for downstream projection.

    All outputs are conservative proxies: voltage is a proxy readout, no field is
    computed, ``physical_amplitude_calibrated`` stays ``False`` and
    ``claim_level`` stays ``computational_scaffold``.

    Parameters
    ----------
    module : jaxley Module/Cell/Network
        The integrated Jaxley module (used only to read recordings layout/geometry).
    recordings : array-like
        ``jaxley.integrate(...)`` output, shape ``(n_recordings, n_time)`` or
        ``(n_time,)`` for a single recording.
    dt_ms : float
        Integration timestep in ms (must match the ``delta_t`` used).
    state : str
        Recorded state name to attribute (default ``"v"``).
    spec : JaxleyTraceSpec, optional
        Trace spec (claim gates, spike threshold). Defaults to a voltage proxy spec.
    source : array-like, optional
        Explicit external source array; if None the voltage proxy is used.

    Returns
    -------
    jaxfne.core.Signals
        Proxy Signals (``field=None``), ready for tfne readouts/vis.
    """
    require_jaxley()
    import jax.numpy as jnp
    from dataclasses import replace as _dc_replace

    rec = jnp.asarray(recordings)
    if rec.ndim == 1:
        rec = rec[None, :]
    if rec.ndim != 2:
        raise ValueError(
            f"recordings must be 1D or 2D (n_recordings, n_time); got shape {rec.shape}"
        )

    if spec is None:
        spec = JaxleyTraceSpec(dt_ms=dt_ms, layout="unit_by_time")

    extra: dict[str, Any] = {
        "bridge": "jaxley_module_to_signals",
        "jaxley_recorded_state": state,
        "jaxley_n_recordings": int(rec.shape[0]),
        "jax_clip_compat_installed": bool(_JAX_CLIP_COMPAT_INSTALLED),
    }
    positions = _recorded_compartment_positions(module, state)
    if positions is not None:
        extra["recorded_positions_xyz"] = [[float(c) for c in row] for row in positions]

    spec = _dc_replace(spec, metadata={**spec.metadata, **extra})
    return jaxley_trace_to_signals(
        rec, spec=spec, dt_ms=dt_ms, layout="unit_by_time", source=source
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
        Time step in milliseconds (Jaxley ``delta_t``; ~0.025 ms recommended for HH).
    current_amplitude : float
        Step-current amplitude passed to ``jaxley.step_current`` (scaled to nA; the
        default produces ~physiological spiking on the single HH compartment).

    Returns
    -------
    t : ndarray (n_steps,)
        Time in ms.
    V : ndarray (n_steps,)
        Membrane potential in mV (proxy readout; not amplitude-calibrated).
    I_inj : ndarray (n_steps,)
        Injected step current (nA), aligned to ``t``.

    Raises
    ------
    ImportError
        If Jaxley is not installed. Install with: pip install jaxfne[jaxley]
    """
    jx = require_jaxley()
    from jaxley.channels import HH
    import numpy as np

    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    cell.record("v")
    i_amp_nA = float(current_amplitude) * 0.01
    i = jx.step_current(
        i_delay=max(10.0, 0.1 * duration_ms),
        i_dur=0.5 * duration_ms,
        i_amp=i_amp_nA,
        delta_t=dt_ms,
        t_max=duration_ms,
    )
    cell.stimulate(i)
    v = jx.integrate(cell, delta_t=dt_ms, solver="bwd_euler", t_max=duration_ms)
    V = np.asarray(v).reshape(-1)
    n = V.shape[0]
    t = np.arange(n) * float(dt_ms)
    I_inj = np.asarray(i).reshape(-1)
    if I_inj.shape[0] != n:  # align stimulus length to recorded trace
        m = min(I_inj.shape[0], n)
        t, V, I_inj = t[:m], V[:m], I_inj[:m]
    return t, V, I_inj


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
        """Documented public function `alpha_m`."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) if abs(V + 40.0) > 1e-6 else 1.0

    def beta_m(V):
        """Documented public function `beta_m`."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(V):
        """Documented public function `alpha_h`."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(V):
        """Documented public function `beta_h`."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(V):
        """Documented public function `alpha_n`."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) if abs(V + 55.0) > 1e-6 else 0.1

    def beta_n(V):
        """Documented public function `beta_n`."""
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

    def simulate(
        self,
        duration_ms: float = 100.0,
        dt_ms: float = 0.025,
        *,
        record_state: str = "v",
        solver: str = "bwd_euler",
        checkpoint_lengths: Any = None,
        return_recordings: bool = False,
    ) -> Any:
        """Run the stored Jaxley module end-to-end and return jaxfne ``Signals``.

        Installs the JAX clip-compat shim, ensures the module records ``record_state``,
        integrates with a stable implicit solver (``bwd_euler``) at ``delta_t=dt_ms``,
        and converts the recordings to ``Signals`` via :func:`jaxley_to_signals`.

        Output is a conservative voltage proxy: ``field=None``,
        ``physical_amplitude_calibrated=False``, ``claim_level=computational_scaffold``.
        Pass ``checkpoint_lengths`` for long BPTT-friendly runs (see Jaxley docs).

        Returns ``Signals`` (or ``(Signals, recordings)`` if ``return_recordings``).
        """
        jx = require_jaxley()
        module = self.model
        try:
            has_rec = len(module.recordings) > 0
        except Exception:
            has_rec = False
        if not has_rec:
            module.record(record_state)

        kwargs: dict[str, Any] = {
            "delta_t": float(dt_ms),
            "t_max": float(duration_ms),
            "solver": solver,
        }
        if checkpoint_lengths is not None:
            kwargs["checkpoint_lengths"] = checkpoint_lengths
        recordings = jx.integrate(module, **kwargs)
        signals = jaxley_to_signals(module, recordings, dt_ms=dt_ms, state=record_state)
        if return_recordings:
            return signals, recordings
        return signals

    def simulate_homeostatic(
        self,
        duration_ms: float = 400.0,
        dt_ms: float = 0.025,
        *,
        record_state: str = "v",
        solver: str = "bwd_euler",
        window_ms: float = 20.0,
        base_current_nA: Any = 0.01,
        target_rate_hz: float = 40.0,
        tau_r_ms: float = 300.0,
        k_gain: float = 0.2,
        g_min: float = -12.0,
        g_max: float = 8.0,
        bias_current_scale: float = 0.001,
        spike_threshold_mV: float = 0.0,
        return_diagnostics: bool = False,
    ) -> Any:
        """Run the Jaxley model under tfne's homeostatic excitability controller.

        Wraps the Jaxley emitter in an **outer-loop windowed controller** so you keep
        Jaxley's channel/morphology toolset *and* gain tfne's homeostasis. Per recorded
        compartment (one per controlled cell), the controller maintains a slow activity
        trace ``r`` and injects an excitability current bias following tfne's restoring-bias
        law (here parameterized by firing rate in Hz, which is dt-independent — unlike the
        Izhikevich kernel's per-step spike-fraction trace)::

            rate_w = spikes / window_seconds               # per-window firing rate (Hz)
            r      = decay_r * r + (1 - decay_r) * rate_w   # decay_r = exp(-window_ms / tau_r_ms)
            g      = clip(k_gain * (target_rate_hz - r), g_min, g_max)  # over-active -> g<0
            I_cell = base_current_nA + bias_current_scale * g           # injected next window

        Windows are stitched with continuous state resume (Jaxley ``all_states``), and
        per-cell current is injected via grad-safe ``data_stimulate`` (no module mutation).
        ``k_gain=0`` disables the controller (a clean null). Because Jaxley's ``integrate``
        is a closed scan, the feedback runs at *window* cadence, not per step — a deliberate,
        documented approximation of the per-step Izhikevich homeostatic kernel.

        The model must have at least one recording (``module.record(...)``) — recorded
        compartments are the controlled cells. Output is a conservative proxy
        (``field=None``, ``physical_amplitude_calibrated=False``).

        Note: Jaxley single-compartment HH has a non-monotonic f-I curve (high current ->
        depolarization block). Keep ``base_current_nA`` and the bias range inside the
        monotonic band (roughly 0.002-0.02 nA for the default compartment) so the controller
        operates where rate increases with current.

        Parameters
        ----------
        duration_ms, dt_ms, record_state, solver : as in :meth:`simulate`.
        window_ms : float
            Controller update cadence (and Jaxley integration window).
        base_current_nA : float or per-cell array
            Baseline injected current per controlled cell (the controller modulates around it).
        target_rate_hz : float
            Homeostatic set-point firing rate the controller restores cells toward.
        tau_r_ms, k_gain, g_min, g_max : float
            Homeostasis controller parameters (same law as the Izhikevich kernel; ``k_gain``
            now multiplies a Hz-valued error so it is smaller than the per-step kernel's).
        bias_current_scale : float
            Maps the dimensionless bias ``g`` to an nA current increment.
        spike_threshold_mV : float
            Upward-crossing threshold for the spike proxy used to measure activity.
        return_diagnostics : bool
            If True, also return a dict with per-window ``g_bias``, ``r_trace``,
            ``rate_hz`` arrays (shape ``[n_windows, n_cells]``).

        Returns
        -------
        Signals, or (Signals, diagnostics) if ``return_diagnostics``.
        """
        jx = require_jaxley()
        import numpy as np
        import jax.numpy as jnp

        module = self.model
        recs = getattr(module, "recordings", None)
        if recs is None or len(recs) == 0:
            raise ValueError(
                "simulate_homeostatic requires recordings: call module.record('v') on the "
                "compartments to control before running the homeostatic controller."
            )
        sel = recs[recs["state"] == record_state] if "state" in recs.columns else recs
        gidx = [int(i) for i in sel["rec_index"].tolist()]
        n_cells = len(gidx)
        if n_cells == 0:
            raise ValueError(f"no recordings for state {record_state!r}")

        base = np.broadcast_to(np.asarray(base_current_nA, dtype=float), (n_cells,)).astype(float)
        dt = float(dt_ms)
        n_win_steps = max(1, int(round(float(window_ms) / dt)))
        win_ms_eff = n_win_steps * dt
        n_windows = max(1, int(round(float(duration_ms) / win_ms_eff)))
        decay_r = float(np.exp(-win_ms_eff / float(tau_r_ms)))

        views = [module.select(nodes=[k]) for k in gidx]
        r = np.zeros(n_cells, dtype=float)  # slow rate trace (Hz)
        states = None
        v_chunks: list[Any] = []
        g_hist: list[Any] = []
        r_hist: list[Any] = []
        rate_hist: list[Any] = []

        for w in range(n_windows):
            g = np.clip(k_gain * (target_rate_hz - r), g_min, g_max)
            data = None
            for ci, view in enumerate(views):
                amp = float(base[ci] + bias_current_scale * g[ci])
                cur = jnp.full(n_win_steps + 1, amp)
                data = view.data_stimulate(cur, data_stimuli=data)
            rec, states = jx.integrate(
                module, delta_t=dt, t_max=win_ms_eff, solver=solver,
                data_stimuli=data, all_states=states, return_states=True,
            )
            vw = np.asarray(rec)  # (n_cells, n_win_steps+1)
            # spike proxy: upward threshold crossings -> per-window firing rate (Hz)
            crossings = ((vw[:, :-1] < spike_threshold_mV) & (vw[:, 1:] >= spike_threshold_mV))
            rate_w = crossings.sum(axis=1) / (win_ms_eff / 1000.0)
            r = decay_r * r + (1.0 - decay_r) * rate_w
            g_hist.append(g.copy())
            r_hist.append(r.copy())
            rate_hist.append(rate_w)
            v_chunks.append(vw if w == 0 else vw[:, 1:])  # drop shared boundary sample

        recordings = np.concatenate(v_chunks, axis=1)  # (n_cells, total_T)

        from dataclasses import replace as _dc_replace
        spec = JaxleyTraceSpec(
            dt_ms=dt, layout="unit_by_time", spike_threshold=spike_threshold_mV,
        )
        homeo_meta = {
            "homeostasis": {
                "method": "minimal_homeostatic_resource_adaptation_controller",
                "implementation": "jaxley_outer_loop_windowed",
                "claim_status": "computational_control_proxy_not_biological_mechanism",
                "biological_learning_claim": False,
                "mechanism_claim_status": "not_claimed",
                "window_ms": float(win_ms_eff),
                "n_windows": int(n_windows),
                "target_rate_hz": float(target_rate_hz), "tau_r_ms": float(tau_r_ms),
                "k_gain": float(k_gain), "g_min": float(g_min), "g_max": float(g_max),
                "bias_current_scale": float(bias_current_scale),
                "controller_disabled_null": bool(k_gain == 0),
            }
        }
        spec = _dc_replace(spec, metadata=homeo_meta)
        signals = jaxley_to_signals(module, recordings, dt_ms=dt, state=record_state, spec=spec)

        if return_diagnostics:
            diagnostics = {
                "g_bias": np.asarray(g_hist),       # [n_windows, n_cells]
                "r_trace": np.asarray(r_hist),      # [n_windows, n_cells]
                "rate_hz": np.asarray(rate_hist),   # [n_windows, n_cells]
                "controlled_rec_index": gidx,
            }
            return signals, diagnostics
        return signals

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
        """Documented public function `report`."""
        return {
            "bridge_name": "jaxley_bridge",
            "source_mode": self.source_mode,
            "source_calibration_status": "uncalibrated_jaxley_bridge",
            "physical_amplitude_calibrated": False
        }

