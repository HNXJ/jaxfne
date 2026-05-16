"""Optional bridges to external biophysical tools."""

from __future__ import annotations

from dataclasses import dataclass
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
class JaxleyEmitterBridge:
    """Jaxley bridge skeleton.

    This object intentionally does not construct compartments yet.  It marks the
    future integration seam where Jaxley compartment currents can become TFNE
    source traces.
    """

    morphology: str | None = None
    mechanisms: tuple[str, ...] = ()

    def construct(self) -> dict[str, Any]:
        require_jaxley()
        return {
            "status": "specified_future_module",
            "morphology": self.morphology,
            "mechanisms": list(self.mechanisms),
        }
