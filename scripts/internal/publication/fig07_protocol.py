"""Figure 7 E-integration publication protocol."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
FIG07_SPEC_PATH = _REPO / "artifacts" / "publication" / "fig07_integration_spec.json"
FIG07_AUDIT_PATH = _REPO / "artifacts" / "publication" / "fig07_semantic_audit.json"
FIG07_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig07_generation_receipt.json"
FIG07_PATH = _REPO / "figures" / "publication" / "fig07_e_integration.png"


def load_fig07_spec(path: Path | None = None) -> dict:
    return json.loads((path or FIG07_SPEC_PATH).read_text())


def load_fig07_semantic_audit(path: Path | None = None) -> dict:
    return json.loads((path or FIG07_AUDIT_PATH).read_text())


def load_fig07_generation_receipt(path: Path | None = None) -> dict:
    return json.loads((path or FIG07_RECEIPT_PATH).read_text())


def validate_fig07_spec(spec: dict | None = None) -> None:
    spec = spec or load_fig07_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("fig07 spec must be FROZEN")
    if len(spec.get("pec_panel_ids", [])) != 6:
        raise ValueError("fig07 requires six PEC panel bindings")


def validate_fig07_semantic_audit(audit: dict | None = None) -> None:
    audit = audit or load_fig07_semantic_audit()
    if audit.get("status") != "PASSED":
        raise ValueError("fig07 audit must be PASSED")
    checks = audit["checks"]
    if checks.get("e5_classification") != "HIERARCHICAL_PROPAGATION":
        raise ValueError("E5 must classify as HIERARCHICAL_PROPAGATION")
    if not checks.get("n0_equals_n1_all_seeds"):
        raise ValueError("N0==N1 must be preserved")


def validate_fig07_generation_receipt(receipt: dict | None = None) -> None:
    receipt = receipt or load_fig07_generation_receipt()
    if receipt.get("status") != "CLOSED":
        raise ValueError("fig07 receipt must be CLOSED")
    if not FIG07_PATH.is_file():
        raise FileNotFoundError(str(FIG07_PATH))
