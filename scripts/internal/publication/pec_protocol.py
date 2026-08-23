"""Publication Evidence Consolidation (PEC) — frozen index authority."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
PEC_SPEC_PATH = _REPO_ROOT / "artifacts" / "publication" / "pec_consolidation_spec.json"
PUBLICATION_EVIDENCE_INDEX_PATH = (
    _REPO_ROOT / "artifacts" / "publication" / "publication_evidence_index.json"
)
PEC_CONSOLIDATION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "publication" / "pec_consolidation_receipt.json"
)

_CLAIM_LEVELS = (
    "DEMONSTRATED",
    "MECHANISTICALLY_SUPPORTED",
    "REPRESENTATIONAL",
    "PROSPECTIVE",
)
_POLARITIES = ("POSITIVE", "NEGATIVE", "UNRESOLVED")


def load_pec_spec(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or PEC_SPEC_PATH).read_text())


def load_publication_evidence_index(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or PUBLICATION_EVIDENCE_INDEX_PATH).read_text())


def panel_ids(index: dict[str, Any] | None = None) -> tuple[str, ...]:
    idx = index or load_publication_evidence_index()
    return tuple(str(p["panel_id"]) for p in idx["panels"])


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()


def _resolve_receipt_path(ref: str | None) -> Path | None:
    if ref is None:
        return None
    if "#" in ref:
        ref = ref.split("#", 1)[0]
    path = _REPO_ROOT / ref
    return path if path.is_file() else None


def validate_pec_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_pec_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("PEC spec must be FROZEN")
    if spec.get("checkpoint") != "PEC":
        raise ValueError("PEC checkpoint id must be PEC")
    if not spec["execution_authorization"].get("consolidation_only"):
        raise ValueError("PEC must be consolidation_only at freeze")
    if spec["execution_authorization"].get("figure_rendering_authorized") is not False:
        raise ValueError("figure rendering must not be authorized at PEC freeze")


def validate_publication_evidence_index(index: dict[str, Any] | None = None) -> None:
    index = index or load_publication_evidence_index()
    spec = load_pec_spec()
    required = set(spec["panel_record_required_fields"])

    if index.get("status") != "FROZEN":
        raise ValueError("publication evidence index must be FROZEN")

    for panel in index["panels"]:
        missing = required - set(panel)
        if missing:
            raise ValueError(f"panel {panel.get('panel_id')} missing fields: {missing}")
        if panel["claim_level"] not in _CLAIM_LEVELS:
            raise ValueError(f"invalid claim_level on {panel['panel_id']}")
        if panel["polarity"] not in _POLARITIES:
            raise ValueError(f"invalid polarity on {panel['panel_id']}")
        raw = panel.get("raw_receipt")
        interp = panel.get("interpretation_receipt")
        if raw is not None and _resolve_receipt_path(raw) is None:
            raise ValueError(f"missing raw receipt for {panel['panel_id']}: {raw}")
        if interp is not None and _resolve_receipt_path(interp) is None:
            raise ValueError(f"missing interpretation receipt for {panel['panel_id']}: {interp}")

    figures = {int(p["figure"]) for p in index["panels"]}
    if not figures.issuperset({1, 5, 6, 7}):
        raise ValueError("index must cover figures 1, 5, 6, 7 at minimum")


def write_pec_consolidation_receipt(*, artifact_commit_sha: str | None = None) -> dict[str, Any]:
    validate_pec_spec()
    index = load_publication_evidence_index()
    validate_publication_evidence_index(index)
    sha = artifact_commit_sha or _git_head()
    receipt = {
        "schema": "jaxfne.publication.pec_consolidation_receipt.v1",
        "checkpoint": "PEC",
        "status": "FROZEN",
        "write_once": True,
        "artifact_commit_sha": sha,
        "spec": "artifacts/publication/pec_consolidation_spec.json",
        "index": "artifacts/publication/publication_evidence_index.json",
        "panel_count": len(index["panels"]),
        "panel_ids": list(panel_ids(index)),
        "feature_freeze": index["feature_freeze"],
        "next_checkpoint": "figure_1_generation",
        "figure_rendering_authorized": False,
    }
    PEC_CONSOLIDATION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_pec_consolidation_receipt() -> dict[str, Any]:
    return json.loads(PEC_CONSOLIDATION_RECEIPT_PATH.read_text())
