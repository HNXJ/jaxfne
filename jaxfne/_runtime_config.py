"""JAX runtime/dtype/device policy: :class:`RuntimeConfig` and backend helpers.

Split out of ``jaxfne/core.py`` (first slice of the core.py monolith split,
see ``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports every
symbol here for backward compatibility -- import from ``jaxfne.core`` or
``jaxfne.runtime``, not this module, unless you are working on core.py itself.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp

_ALLOWED_SYNAPTIC_KERNELS = ("exponential", "receptor_exponential")
_ALLOWED_DTYPES = ("float32", "float64", "bfloat16")


@dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime, dtype policy, and emitter control.

    Ownership (2026-08-22 W4 scoping): this class is the single canonical owner
    of execution-policy semantics — backend, dtype, jit/vmap, recurrent/synaptic
    kernel selection — for the Configuration construction path. The tensor
    workflow uses :class:`jaxfne.neuronal_tensor.RuntimeConfiguration`, which
    ``construct(tensor, runtime=...)`` bridges into these same resolved
    semantics; the two types are path-scoped views, not competing policies.

    ``dtype='float64'`` is honored only when JAX x64 is enabled.  The manifest
    always reports both requested and actual dtype policy.

    Homeostasis: when ``enable_homeostasis=True``, emitters use per-neuron
    activity-trace feedback to balance firing rates. Pass ``homeostasis_params``
    as a dict with keys {r_star, tau_r_ms, alpha, k_gain, g_min, g_max, r_max};
    k_gain=0 disables the feedback.

    HDP (Hidden-state Dependent Plasticity): when ``enable_hdp=True``, emitters
    use a finite-dimensional RBS representation (public API: ``h_state_*``)
    driving adaptive updates to biophysical parameter coordinates (see
    ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp``).
    RBS is the latent carrier within the HDP formulation; it is not
    synonymous with HDP itself and not intrinsically homeostatic.

    ``hdp_params`` is a **compatibility transport dict**, not the conceptual
    API. Keys are grouped semantically (see ``jaxfne.public_surface``):

    - **RBS (``h_state_*``)**: ``h_state_dim``, ``h_state_locality`` (``node`` | ``population``),
      ``h_state_readout``, ``h_state_coupling``
    - **H-dynamics**: ``hdp_rule``, income/spending/barrier coefficients,
      ``K_HDP``, ``K_ctrl``, ``tau_0_ms``, ...
    - **Theta-adaptation**: ``controller_*``, channel masks/bounds for adaptive
      coordinates (Θ distinct from synaptic weight storage W)

    Internal dispatch rule identifiers (e.g. population restoring controllers)
    must not be treated as public vocabulary; use ``h_state_locality`` and the
    grouped coefficients instead. See ``validate_hdp_params_semantics`` in
    ``jaxfne.public_surface`` / ``validate_runtime_config``.
    """

    backend: str = "auto"  # "auto" | "cpu" | "gpu" | "tpu"
    dtype: str = "float32"  # "float32" | "float64"
    jit: bool | str = False
    vmap: bool | str = False
    precision: str = "default"  # "default" | "high"
    seed: int = 0
    n_steps: int = 0
    recurrent_backend: str = "dense"  # "dense" | "edge_list"
    synaptic_kernel: str = "exponential"  # "exponential" | "receptor_exponential"
    recompilation_guard: str = "warning"  # "warning" | "exception" | "off"
    enable_homeostasis: bool = False  # Enable per-neuron homeostatic resource
    homeostasis_params: dict = field(default_factory=lambda: {
        "r_star": 0.05,
        "tau_r_ms": 300.0,
        "alpha": 1.0,
        "k_gain": 1.0,
        "g_min": -12.0,
        "g_max": 8.0,
        "r_max": 1.0,
    })  # Homeostasis parameters; k_gain=0 disables, default 1.0 is a gentle in-band nudge
    enable_hdp: bool = False  # Enable Hidden-state Dependent Plasticity (per-neuron
    # RBS H_i driving excitatory/inhibitory weight ODEs); see
    # jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp and
    # jaxfne.hdp_network.DEFAULT_HDP / DEFAULT_HDP_DESYNC for tuned presets.
    # Mutually exclusive with enable_homeostasis (distinct controllers).
    hdp_params: dict = field(default_factory=lambda: {
        "K_HDP": 1.0, "tau_0_ms": 100.0, "alpha": 0.0, "beta": 0.0,
        "gamma": 0.0, "delta": 0.0, "C_spike": 0.0, "K_ctrl": 0.0,
        "barrier_c": 0.0, "barrier_d": 0.0,
        "h_state_dim": 1, "h_state_readout": None, "h_state_coupling": None,
    })  # All income/spending/control gains default to 0.0 -- the documented
    # null control (H_i pinned at 1.0, weight term identically zero).
    # v0.0.3 compatibility names; if provided by old caller, they are folded in.
    device_type: Optional[str] = None
    dtype_primary: Optional[str] = None
    x64_enabled: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.synaptic_kernel not in _ALLOWED_SYNAPTIC_KERNELS:
            raise ValueError(
                f"synaptic_kernel must be one of {_ALLOWED_SYNAPTIC_KERNELS}; "
                f"got {self.synaptic_kernel!r}"
            )
        if self.recompilation_guard not in {"warning", "exception", "off"}:
            raise ValueError(
                f"recompilation_guard must be one of {{'warning', 'exception', 'off'}}; "
                f"got {self.recompilation_guard!r}"
            )
        # Validate jit and vmap options
        jit_str = str(self.jit).lower()
        if jit_str not in {"true", "false", "auto", "1", "0"}:
            raise ValueError(
                f"jit must be a boolean or 'auto'; got {self.jit!r}"
            )
        vmap_str = str(self.vmap).lower()
        if vmap_str not in {"true", "false", "auto", "1", "0"}:
            raise ValueError(
                f"vmap must be a boolean or 'auto'; got {self.vmap!r}"
            )
        if self.enable_homeostasis and self.enable_hdp:
            raise ValueError(
                "enable_homeostasis and enable_hdp are mutually exclusive "
                "controllers; enable only one."
            )
        requested = self.dtype_primary or self.dtype
        if requested not in _ALLOWED_DTYPES:
            raise ValueError(
                f"dtype must be one of {_ALLOWED_DTYPES}; got {requested!r}. "
                f"Previously any unrecognized value silently fell back to float32 -- "
                f"that silent fallback is now an explicit error."
            )
        if self.backend not in {"auto", "cpu", "gpu", "tpu"}:
            raise ValueError(
                f"backend must be one of {{'auto', 'cpu', 'gpu', 'tpu'}}; "
                f"got {self.backend!r}."
            )

    def resolve_jit(self, n_steps: int, n_units: int, batch: int = 1) -> bool:
        """Resolve the effective JIT compilation status based on policy and parameters."""
        if isinstance(self.jit, bool):
            return self.jit
        jit_str = str(self.jit).lower()
        if jit_str == "auto":
            return (n_steps * n_units * batch) > 50000
        return jit_str in {"true", "1"}

    def resolve_vmap(self, batch: int) -> bool:
        """Resolve the effective vmap vectorization status based on policy and batch size."""
        if isinstance(self.vmap, bool):
            return self.vmap
        vmap_str = str(self.vmap).lower()
        if vmap_str == "auto":
            return batch > 1
        return vmap_str in {"true", "1"}

    @property
    def requested_dtype(self) -> str:
        """Documented public function `requested_dtype`."""
        return self.dtype_primary or self.dtype

    @property
    def selected_backend(self) -> str:
        """Documented public function `selected_backend`."""
        return self.device_type or self.backend

    @property
    def actual_dtype(self) -> str:
        """Documented public function `actual_dtype`."""
        if self.requested_dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
            return "float64"
        if self.requested_dtype == "bfloat16":
            return "bfloat16"
        return "float32"

    @property
    def jnp_dtype(self) -> jnp.dtype:
        """Documented public function `jnp_dtype`."""
        if self.actual_dtype == "float64":
            return jnp.float64
        if self.actual_dtype == "bfloat16":
            return jnp.bfloat16
        return jnp.float32

    def runtime_report(self) -> dict[str, Any]:
        """Compile a summary of JAX system runtime, backend, and device topology.

        Detects requested vs actual execution backends (CPU, GPU, or TPU) and
        validates system properties.

        Returns
        -------
        dict[str, Any]
            System configuration report metadata.
        """
        from .io import json_safe

        devices = [str(device) for device in jax.devices()]
        default_backend = jax.default_backend()
        requested_backend = self.selected_backend  # alias for selected (user-requested)

        # Compute actual_backend / backend_enforced / backend_warning.
        # Conservative policy (CPU-only environment is the supported scaffold target):
        # - "auto": actual_backend = JAX default; enforced = True (whatever JAX picks).
        # - "cpu" : actual_backend = "cpu"; enforced = True iff JAX default is cpu.
        # - "gpu"/"tpu": if JAX default is the requested device, enforced = True;
        #   otherwise enforced = False with a clean warning (no false claim).
        # v0.3.40: simulate() pins execution to the requested backend device via
        # ``_device_scope`` (jax.default_device) when that device is available, so
        # a non-"auto" backend is genuinely enforced — not merely reported. The
        # report mirrors that: enforced iff the requested backend resolves to an
        # available device. Unavailable devices are an honest downgrade.
        backend_warning: Optional[str] = None
        if requested_backend == "auto":
            actual_backend = default_backend
            backend_enforced = True
        elif requested_backend in ("cpu", "gpu", "tpu"):
            device_available = _resolve_backend_device(requested_backend) is not None
            if device_available:
                actual_backend = requested_backend
                backend_enforced = True
            else:
                # Honest downgrade: do not falsely report a device that is absent.
                actual_backend = default_backend
                backend_enforced = False
                backend_warning = (
                    f"requested_backend_unavailable:requested={requested_backend!r}"
                    f"_actual={default_backend!r}"
                )
        else:
            actual_backend = default_backend
            backend_enforced = False
            backend_warning = f"unknown_requested_backend:{requested_backend!r}"

        return {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": _jaxlib_version(),
            "default_backend": default_backend,
            "available_devices": devices,
            "selected_backend": self.selected_backend,
            "backend": self.backend,
            # v0.0.21 explicit requested-vs-actual reporting.
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "backend_enforced": bool(backend_enforced),
            "backend_warning": backend_warning,
            "requested_dtype": self.requested_dtype,
            "actual_dtype": self.actual_dtype,
            "dtype": self.dtype,
            "jit": bool(self.jit),
            "vmap": bool(self.vmap),
            "precision": self.precision,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "recurrent_backend": self.recurrent_backend,
            "synaptic_kernel": self.synaptic_kernel,
            "enable_homeostasis": bool(self.enable_homeostasis),
            "homeostasis_params": dict(self.homeostasis_params) if self.enable_homeostasis else None,
            "enable_hdp": bool(self.enable_hdp),
            "hdp_params": json_safe(dict(self.hdp_params)) if self.enable_hdp else None,
            # Compatibility keys from v0.0.3.
            "device_type": self.selected_backend,
            "dtype_primary": self.actual_dtype,
        }


def _resolve_backend_device(backend: Optional[str]):
    """Resolve a RuntimeConfig backend string to a concrete JAX device or ``None``.

    ``None``/``"auto"`` returns ``None`` (let JAX keep its default device).
    ``"cpu"``/``"gpu"``/``"tpu"`` return that backend's first device when it
    exists, else ``None`` — an honest fallback that never crashes and never
    falsely claims placement on an absent device.
    """
    if not backend or backend == "auto":
        return None
    try:
        devices = jax.devices(backend)
    except (RuntimeError, ValueError):
        return None
    return devices[0] if devices else None


def _device_scope(backend: Optional[str]):
    """Return a context manager pinning execution to the requested backend device.

    Yields ``jax.default_device(dev)`` when the backend resolves to an available
    device, else a no-op context (``"auto"`` or an unavailable backend) so JAX
    keeps its default device. Compiling and running inside this scope is what
    actually honors ``RuntimeConfig.backend`` for cpu/gpu/tpu placement; on a
    single-device host it is a no-op.
    """
    device = _resolve_backend_device(backend)
    if device is not None:
        return jax.default_device(device)
    return contextlib.nullcontext()


def _jaxlib_version() -> str:
    try:
        import jaxlib  # type: ignore

        return str(jaxlib.__version__)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class SurrogateConfig:
    """Declared surrogate-gradient metadata for discontinuous emitter paths.

    The current implementation records the declaration only; it does not alter
    the Izhikevich reset dynamics.  Optax paths may reference this object to
    distinguish explicit surrogate intent from accidental differentiation
    through hard spike resets.
    """

    method: str = "none"  # "none" | "straight_through" | "sigmoid_beta"
    beta: float = 10.0
    applies_to: str = "izhikevich_reset"
    status: str = "declaration_only_v0.0.8"

    def gradient_path_status(self) -> str:
        """Documented public function `gradient_path_status`."""
        if self.method in {"straight_through", "sigmoid_beta"}:
            return "declared_surrogate"
        return "required_but_missing"

    def to_dict(self) -> dict[str, Any]:
        """Documented public function `to_dict`."""
        from .io import json_safe
        return json_safe({
            "method": self.method,
            "beta": float(self.beta),
            "applies_to": self.applies_to,
            "status": self.status,
            "gradient_path_status": self.gradient_path_status(),
            "mechanism_claim_status": "not_claimed",
        })
