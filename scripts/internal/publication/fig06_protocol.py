"""Figure 6 H/W/D publication protocol."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
FIG06_SPEC_PATH = _REPO / "artifacts" / "publication" / "fig06_hwd_spec.json"
FIG06_AUDIT_PATH = _REPO / "artifacts" / "publication" / "fig06_semantic_audit.json"
FIG06_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig06_generation_receipt.json"
FIG06_PATH = _REPO / "figures" / "publication" / "fig06_rbs_hdp_ladder.png"


def load_fig06_spec(path: Path | None = None) -> dict:
    return json.loads((path or FIG06_SPEC_PATH).read_text())


def load_fig06_semantic_audit(path: Path | None = None) -> dict:
    return json.loads((path or FIG06_AUDIT_PATH).read_text())


def load_fig06_generation_receipt(path: Path | None = None) -> dict:
    return json.loads((path or FIG06_RECEIPT_PATH).read_text())


def validate_fig06_spec(spec: dict | None = None) -> None:
    spec = spec or load_fig06_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("fig06 spec must be FROZEN")
    if "E5" not in spec.get("excluded_content", []):
        raise ValueError("fig06 must exclude E5")


def validate_fig06_semantic_audit(audit: dict | None = None) -> None:
    audit = audit or load_fig06_semantic_audit()
    if audit.get("status") != "PASSED":
        raise ValueError("fig06 audit must be PASSED")
    checks = audit["checks"]
    if not checks.get("w3b_unresolved_not_negative"):
        raise ValueError("W3b must remain unresolved not negative")
    if not checks.get("h4_remains_negative"):
        raise ValueError("H4 must remain negative")


def validate_fig06_generation_receipt(receipt: dict | None = None) -> None:
    receipt = receipt or load_fig06_generation_receipt()
    if receipt.get("status") != "CLOSED":
        raise ValueError("fig06 receipt must be CLOSED")
    if not FIG06_PATH.is_file():
        raise FileNotFoundError(str(FIG06_PATH))
