"""Runtime configuration and execution-report helpers for jaxfne.

Docs: ``docs/api/runtime.md`` (https://jaxfne.readthedocs.io/en/latest/api/runtime/) —
update that page when this module's public API changes.

This module provides a stable import surface for runtime-related public
objects. Runtime behavior is implemented in the core package; this module
re-exports the public runtime contracts for user convenience.

Example:
    >>> from jaxfne.runtime import RuntimeConfig, runtime, runtime_report
    >>> from jaxfne.runtime import get_jax_backend_report, set_precision_policy
    >>> from jaxfne.runtime import safe_jit, safe_vmap
    >>> cfg = RuntimeConfig()
    >>> report = runtime_report(cfg)
    >>> backend = runtime()
    >>> backend_info = get_jax_backend_report()
"""

from typing import Any, Callable, TypeVar
import jax
import jax.numpy as jnp

from .core import RuntimeConfig, runtime, runtime_report

__all__ = [
    "RuntimeConfig",
    "runtime",
    "runtime_report",
    "get_jax_backend_report",
    "set_precision_policy",
    "safe_jit",
    "safe_vmap",
]


def get_jax_backend_report() -> dict[str, Any]:
    """Get JAX backend and device information.

    Returns a dict with:
        - available_devices: list of available JAX devices
        - default_backend: default JAX backend (cpu/gpu/tpu)
        - x64_enabled: whether float64 is enabled
        - dtype_default: default dtype (float32 or float64)

    Example:
        >>> report = get_jax_backend_report()
        >>> print(report["available_devices"])
    """
    try:
        devices = jax.devices()
        default_backend = devices[0].platform if devices else "unknown"
    except Exception:
        devices = []
        default_backend = "unknown"

    return {
        "available_devices": [str(d) for d in devices],
        "default_backend": default_backend,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "dtype_default": "float64" if jax.config.read("jax_enable_x64") else "float32",
    }


def set_precision_policy(dtype: str = "float32", enable_x64: bool = False) -> dict[str, Any]:
    """Set global JAX dtype precision policy.

    Args:
        dtype: Target dtype ("float32" or "float64"). Note: float64 requires enable_x64=True.
        enable_x64: Whether to enable float64 mode. If False, dtype is ignored for x64.

    Returns:
        dict with keys:
            - requested_dtype: what was requested
            - actual_dtype: what was set
            - x64_enabled: whether x64 mode is active
            - status: "set" or "unchanged"

    Example:
        >>> result = set_precision_policy(dtype="float32", enable_x64=False)
        >>> print(result["actual_dtype"])
    """
    import os
    import warnings
    is_strict = os.environ.get("JAXFNE_STRICT", "0") in ("1", "true", "True")
    
    if enable_x64:
        try:
            jax.config.update("jax_enable_x64", True)
            actual_x64 = jax.config.read("jax_enable_x64")
            actual_dtype = "float64" if actual_x64 else "float32"
        except Exception as e:
            actual_x64 = False
            actual_dtype = "float32"
            if is_strict:
                raise RuntimeError(f"Failed to set precision policy to x64: {e}") from e
            warnings.warn(f"Failed to enable x64 mode. Fallback to float32. Reason: {e}", RuntimeWarning)
    else:
        actual_x64 = False
        actual_dtype = "float32"

    if dtype == "float64" and not actual_x64:
        msg = "float64 requested but x64 mode is not active/enabled."
        if is_strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning)

    return {
        "requested_dtype": dtype,
        "actual_dtype": actual_dtype,
        "x64_enabled": actual_x64,
        "status": "set",
    }


T = TypeVar("T")


def safe_jit(fn: Callable[..., T], strict: bool = False, **jit_kwargs: Any) -> Callable[..., T]:
    """Wrap a function with JAX jit, with fallback to no-op if jit is not available.

    Args:
        fn: Function to jit-compile.
        strict: If True, raise exception on jit failure rather than falling back.
        **jit_kwargs: Additional keyword arguments to pass to jax.jit (e.g., static_argnums).

    Returns:
        jit-compiled function if jit is available; otherwise returns the original function.
    """
    import os
    import warnings
    is_strict = strict or os.environ.get("JAXFNE_STRICT", "0") in ("1", "true", "True")
    try:
        return jax.jit(fn, **jit_kwargs)
    except Exception as e:
        if is_strict:
            raise RuntimeError(f"JIT compilation failed in strict mode: {e}") from e
        warnings.warn(f"JIT compilation failed, falling back to eager execution. Reason: {e}", RuntimeWarning)
        return fn


def safe_vmap(fn: Callable[..., T], in_axes: int | None = 0, strict: bool = False, **vmap_kwargs: Any) -> Callable[..., T]:
    """Wrap a function with JAX vmap, with fallback to no-op if vmap is not available.

    Args:
        fn: Function to vectorize.
        in_axes: Which axis to map over (default 0, the batch axis).
        strict: If True, raise exception on vmap failure rather than falling back.
        **vmap_kwargs: Additional keyword arguments to pass to jax.vmap.

    Returns:
        vmap-vectorized function if vmap is available; otherwise returns the original function.
    """
    import os
    import warnings
    is_strict = strict or os.environ.get("JAXFNE_STRICT", "0") in ("1", "true", "True")
    try:
        return jax.vmap(fn, in_axes=in_axes, **vmap_kwargs)
    except Exception as e:
        if is_strict:
            raise RuntimeError(f"VMAP vectorization failed in strict mode: {e}") from e
        warnings.warn(f"VMAP vectorization failed, falling back to loop execution. Reason: {e}", RuntimeWarning)
        return fn
