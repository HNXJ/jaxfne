"""Figure 5 Protocol C publication protocol."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
FIG05_SPEC_PATH = _REPO / "artifacts" / "publication" / "fig05_wave_spec.json"
FIG05_AUDIT_PATH = _REPO / "artifacts" / "publication" / "fig05_semantic_audit.json"
FIG05_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig05_generation_receipt.json"
FIG05_PATH = _REPO / "figures" / "publication" / "fig05_traveling_wave_no_wave.png"


def load_fig05_spec(path: Path | None = None) -> dict:
    return json.loads((path or FIG05_SPEC_PATH).read_text())


def load_fig05_semantic_audit(path: Path | None = None) -> dict:
    return json.loads((path or FIG05_AUDIT_PATH).read_text())


def load_fig05_generation_receipt(path: Path | None = None) -> dict:
    return json.loads((path or FIG05_RECEIPT_PATH).read_text())


def validate_fig05_spec(spec: dict | None = None) -> None:
    spec = spec or load_fig05_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("fig05 spec must be FROZEN")
    if spec.get("polarity") != "NEGATIVE":
        raise ValueError("fig05 polarity must be NEGATIVE")
    if spec.get("claim_level") != "DEMONSTRATED":
        raise ValueError("fig05 claim_level must be DEMONSTRATED")
    fq = spec["frozen_quantities"]
    if fq["N_TW"] != 0 or fq["N_U"] != 0:
        raise ValueError("frozen quantities require N_TW=0 and N_U=0")


def validate_fig05_semantic_audit(audit: dict | None = None) -> None:
    audit = audit or load_fig05_semantic_audit()
    if audit.get("status") != "PASSED":
        raise ValueError("fig05 semantic audit must be PASSED")
    checks = audit.get("checks", {})
    if not checks.get("zero_unresolved"):
        raise ValueError("fig05 requires zero UNRESOLVED")
    if not checks.get("all_60_c3_cells_accounted"):
        raise ValueError("fig05 requires all 60 C3 cells")


def validate_fig05_generation_receipt(receipt: dict | None = None) -> None:
    receipt = receipt or load_fig05_generation_receipt()
    if receipt.get("status") != "CLOSED":
        raise ValueError("fig05 receipt must be CLOSED")
    if receipt.get("polarity") != "NEGATIVE":
        raise ValueError("fig05 receipt polarity must be NEGATIVE")
    if not FIG05_PATH.is_file():
        raise FileNotFoundError(f"missing {FIG05_PATH}")
