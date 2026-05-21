"""Runtime configuration and execution-report helpers for jaxfne.

This module provides a stable import surface for runtime-related public
objects. Runtime behavior is implemented in the core package; this module
re-exports the public runtime contracts for user convenience.

Example:
    >>> from jaxfne.runtime import RuntimeConfig
    >>> cfg = RuntimeConfig()
"""

from .core import RuntimeConfig

__all__ = ["RuntimeConfig"]
