"""Figure 1 grammar generation protocol — load, validate, audit."""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
FIG01_SPEC_PATH = _REPO_ROOT / "artifacts" / "publication" / "fig01_grammar_spec.json"
FIG01_AUDIT_PATH = _REPO_ROOT / "artifacts" / "publication" / "fig01_semantic_audit.json"
FIG01_RECEIPT_PATH = _REPO_ROOT / "artifacts" / "publication" / "fig01_generation_receipt.json"
FIG01_FIGURE_PATH = _REPO_ROOT / "artifacts" / "figures" / "publication" / "fig01_tfne_grammar.png"

_CLAIM_LEVELS = frozenset(
    {"DEMONSTRATED", "MECHANISTICALLY_SUPPORTED", "REPRESENTATIONAL", "PROSPECTIVE", "MIXED"}
)
_ARROW_STYLES = frozenset({"solid", "dashed", "containment", "none"})
_PANELS = frozenset({"A", "B", "C", "D", "E", "F", "banner", "legend"})


def load_fig01_spec(path: Path | None = None) -> dict:
    return json.loads((path or FIG01_SPEC_PATH).read_text())


def load_fig01_semantic_audit(path: Path | None = None) -> dict:
    return json.loads((path or FIG01_AUDIT_PATH).read_text())


def load_fig01_generation_receipt(path: Path | None = None) -> dict:
    return json.loads((path or FIG01_RECEIPT_PATH).read_text())


def validate_fig01_spec(spec: dict | None = None) -> None:
    spec = spec or load_fig01_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("fig01 spec must be FROZEN")
    if spec.get("pec_panel_id") != "Fig01.grammar":
        raise ValueError("fig01 spec pec_panel_id must be Fig01.grammar")
    elements = spec.get("semantic_elements", [])
    if len(elements) < 30:
        raise ValueError("fig01 spec requires comprehensive semantic_elements")
    for el in elements:
        if el.get("claim_level") not in _CLAIM_LEVELS:
            raise ValueError(f"invalid claim_level on {el.get('id')}")
        if el.get("arrow_style") not in _ARROW_STYLES:
            raise ValueError(f"invalid arrow_style on {el.get('id')}")
        if el.get("panel") not in _PANELS:
            raise ValueError(f"invalid panel on {el.get('id')}")
        if not el.get("doctrine_ref"):
            raise ValueError(f"missing doctrine_ref on {el.get('id')}")
    for key in "ABCDEF":
        if key not in spec.get("panels", {}):
            raise ValueError(f"missing panel {key} in spec")


def validate_fig01_semantic_audit(audit: dict | None = None, *, spec: dict | None = None) -> None:
    audit = audit or load_fig01_semantic_audit()
    spec = spec or load_fig01_spec()
    if audit.get("status") != "PASSED":
        raise ValueError("semantic audit must be PASSED")
    if audit.get("empirical_results_excluded") is not True:
        raise ValueError("fig01 must exclude empirical results")
    if audit.get("epistemic_separation_verified") is not True:
        raise ValueError("epistemic separation must be verified")
    spec_ids = {e["id"] for e in spec["semantic_elements"]}
    audit_ids = {e["id"] for e in audit["elements"]}
    if spec_ids != audit_ids:
        missing = spec_ids - audit_ids
        extra = audit_ids - spec_ids
        raise ValueError(f"audit/spec element mismatch missing={missing} extra={extra}")


def validate_fig01_generation_receipt(receipt: dict | None = None) -> None:
    receipt = receipt or load_fig01_generation_receipt()
    if receipt.get("status") != "CLOSED":
        raise ValueError("fig01 generation receipt must be CLOSED")
    if receipt.get("pec_panel_id") != "Fig01.grammar":
        raise ValueError("receipt pec_panel_id mismatch")
    if not FIG01_FIGURE_PATH.is_file():
        raise FileNotFoundError(f"missing figure: {FIG01_FIGURE_PATH}")
    if receipt.get("semantic_audit_status") != "PASSED":
        raise ValueError("receipt requires PASSED semantic audit")
