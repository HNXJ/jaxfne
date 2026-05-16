"""Optional bridges to external tools."""

from __future__ import annotations


def require_jaxley():
    try:
        import jaxley  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This feature requires optional dependency 'jaxley'. Install with: pip install -e '.[jaxley]'"
        ) from exc
    return jaxley


class JaxleyEmitterBridge:
    """Placeholder bridge from Jaxley compartment currents into jaxfne sources."""

    def __init__(self, morphology=None, mechanisms=None):
        self.morphology = morphology
        self.mechanisms = mechanisms or []

    def construct(self):
        jaxley = require_jaxley()
        return {"backend": jaxley, "morphology": self.morphology, "mechanisms": self.mechanisms}
