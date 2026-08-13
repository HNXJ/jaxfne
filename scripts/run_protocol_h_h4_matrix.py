#!/usr/bin/env python3
"""Execute Protocol H4 matrix and freeze prospective receipt (no post-hoc tuning)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaxfne.h4_matrix import H4ProtocolConfig, run_h4_matrix
from jaxfne.io import config_hash, json_safe, save_json, sha256_file


def main() -> int:
    out_dir = REPO / "artifacts" / "protocol_h_rbd" / "h4_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "h4_matrix_receipt.json"
    if receipt_path.exists():
        print(f"Refusing to overwrite frozen receipt: {receipt_path}", file=sys.stderr)
        return 2

    cfg = H4ProtocolConfig()
    receipt = run_h4_matrix(cfg, rng_seed=0)
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
        .decode()
        .strip()
    )
    receipt["repository"] = {"branch": "dev", "sha": sha}
    receipt["config_hash"] = config_hash(cfg)
    receipt["runner"] = "scripts/run_protocol_h_h4_matrix.py"
    save_json(receipt, receipt_path)
    receipt["receipt_sha256"] = sha256_file(receipt_path)

    manifest = {
        "schema": "protocol_h_rbd_h4_manifest.v1",
        "protocol_id": "protocol_h_rbd_memory_h4",
        "status": "FROZEN_PROSPECTIVE",
        "repository": receipt["repository"],
        "config_hash": receipt["config_hash"],
        "artifacts": {
            "receipt": "h4_matrix_receipt.json",
        },
        "reproduce": "python scripts/run_protocol_h_h4_matrix.py",
        "note": "Receipt is write-once. Delete manually only to re-run under a new protocol revision.",
    }
    save_json(manifest, out_dir / "manifest.json")
    print(f"Wrote {receipt_path}")
    print(f"config_hash={receipt['config_hash']}")
    print(f"cell_M_X_area={receipt['cell_M_X_area']}")
    print(f"factorial={receipt['factorial']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
