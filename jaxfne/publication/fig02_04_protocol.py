"""Figures 2–4 coordinated Experiment A publication protocol."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
FIG02_04_SPEC_PATH = _REPO / "artifacts" / "publication" / "fig02_04_experiment_a_spec.json"
FIG02_04_AUDIT_PATH = _REPO / "artifacts" / "publication" / "fig02_04_cross_figure_audit.json"
FIG02_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig02_generation_receipt.json"
FIG03_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig03_generation_receipt.json"
FIG04_RECEIPT_PATH = _REPO / "artifacts" / "publication" / "fig04_generation_receipt.json"
FIG02_PATH = _REPO / "figures" / "publication" / "fig02_emitter_source.png"
FIG03_PATH = _REPO / "figures" / "publication" / "fig03_local_observation.png"
FIG04_PATH = _REPO / "figures" / "publication" / "fig04_multiscale_boundary.png"
CANONICAL_NPZ = _REPO / "artifacts" / "etudes" / "experiment_a" / "canonical_source.npz"


def load_fig02_04_spec(path: Path | None = None) -> dict:
    return json.loads((path or FIG02_04_SPEC_PATH).read_text())


def load_fig02_04_cross_audit(path: Path | None = None) -> dict:
    return json.loads((path or FIG02_04_AUDIT_PATH).read_text())


def load_fig02_receipt() -> dict:
    return json.loads(FIG02_RECEIPT_PATH.read_text())


def load_fig03_receipt() -> dict:
    return json.loads(FIG03_RECEIPT_PATH.read_text())


def load_fig04_receipt() -> dict:
    return json.loads(FIG04_RECEIPT_PATH.read_text())


def validate_fig02_04_spec(spec: dict | None = None) -> None:
    spec = spec or load_fig02_04_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("fig02_04 spec must be FROZEN")
    for key in ("fig02", "fig03", "fig04"):
        if key not in spec.get("figures", {}):
            raise ValueError(f"missing figure spec {key}")


def validate_fig02_04_cross_audit(audit: dict | None = None) -> None:
    audit = audit or load_fig02_04_cross_audit()
    if audit.get("status") != "PASSED":
        raise ValueError("cross-figure audit must be PASSED")
    if not audit.get("cross_figure_q_invariant"):
        raise ValueError("Q hash must be invariant across figures 2–4")
    if audit.get("neural_rerun_detected"):
        raise ValueError("neural rerun must not occur")
    if not audit.get("analysis_only_visible_in_fig04"):
        raise ValueError("Fig04 must mark EEG/MEG analysis-only visibly")
    q_hashes = audit.get("q_hashes_by_figure", {})
    if len(set(q_hashes.values())) != 1:
        raise ValueError("all figures must share one Q hash")


def validate_fig02_04_receipts() -> None:
    for path, fig_path in (
        (FIG02_RECEIPT_PATH, FIG02_PATH),
        (FIG03_RECEIPT_PATH, FIG03_PATH),
        (FIG04_RECEIPT_PATH, FIG04_PATH),
    ):
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "CLOSED":
            raise ValueError(f"{path.name} must be CLOSED")
        if receipt.get("neural_rerun") is not False:
            raise ValueError(f"{path.name} must record neural_rerun=false")
        if not fig_path.is_file():
            raise FileNotFoundError(f"missing {fig_path}")
