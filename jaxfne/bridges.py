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
