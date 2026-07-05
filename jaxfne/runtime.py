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
    """Wrap a function with JAX jit, falling back to the eager function on WRAP-TIME failure.

    ``jax.jit(fn, **jit_kwargs)`` itself is lazy -- it almost never raises, since
    tracing/compilation only happens on the *first call* of the returned wrapper.
    This guard therefore only catches failures at wrap time (e.g. an invalid
    ``jit_kwargs`` value such as a malformed ``static_argnums``), not trace-time
    failures like a shape mismatch or unsupported control flow inside ``fn`` --
    those surface uncaught on the first call of the jitted function this returns.
    Do not rely on this to make an arbitrary function "safe to call" under jit.

    Args:
        fn: Function to jit-compile.
        strict: If True, raise exception on wrap-time jit failure rather than falling back.
        **jit_kwargs: Additional keyword arguments to pass to jax.jit (e.g., static_argnums).

    Returns:
        jit-wrapped function if wrapping succeeded; otherwise returns the original function.
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
    """Wrap a function with JAX vmap, falling back to the eager function on WRAP-TIME failure.

    ``jax.vmap(fn, ...)`` itself is lazy -- like ``safe_jit``, it almost never
    raises at wrap time, since tracing only happens on the *first call* of the
    returned wrapper. This guard therefore only catches wrap-time failures (e.g.
    invalid ``vmap_kwargs``), not trace-time failures like an axis-shape
    mismatch inside ``fn`` -- those surface uncaught on the first call. Do not
    rely on this to make an arbitrary function "safe to call" under vmap.

    Args:
        fn: Function to vectorize.
        in_axes: Which axis to map over (default 0, the batch axis).
        strict: If True, raise exception on wrap-time vmap failure rather than falling back.
        **vmap_kwargs: Additional keyword arguments to pass to jax.vmap.

    Returns:
        vmap-wrapped function if wrapping succeeded; otherwise returns the original function.
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
