"""Protocol D closure at D3 — milestone interpretation receipt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
D_CLOSURE_RECEIPT_PATH = (
    REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d_closure_interpretation_receipt.json"
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def load_d_closure_receipt() -> dict[str, Any]:
    return json.loads(D_CLOSURE_RECEIPT_PATH.read_text())


def validate_d_closure_receipt(receipt: dict[str, Any] | None = None) -> None:
    receipt = receipt or load_d_closure_receipt()
    if not receipt.get("protocol_d_closed"):
        raise ValueError("protocol_d_closed must be true")
    if receipt.get("closed_at_checkpoint") != "D3":
        raise ValueError("Protocol D must close at D3")
    if receipt.get("D4_status") != "not_authorized":
        raise ValueError("D4 must remain not_authorized")
    if receipt.get("next_milestone") != "0.4.17-E":
        raise ValueError("next milestone must be 0.4.17-E")
