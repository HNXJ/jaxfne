#!/usr/bin/env python3
"""Experiment A (0.4.17-B) — canonical multiscale observation runner.

See docs/etudes/experiment_a.md. Freezes B1 canonical arrays then B2/B3
observation receipts without manuscript figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jaxfne.experiment_a.receipt import write_b3_bundle

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    bundle = write_b3_bundle()
    receipt = bundle["receipt"]
    verification = bundle["verification"]
    metrics = json.loads((ROOT / receipt["metrics"]).read_text())
    hashes_ok = all(verification["canonical_hashes"].values())
    print(
        json.dumps(
            {
                "levels": metrics["levels"],
                "protocol": metrics["protocol"],
                "package_head": metrics["package_head"],
                "canonical_hashes": verification["canonical_hashes"],
                "receipt_status": verification["tracked_receipts"]["b3_experiment_a_receipt.json"]["status"],
                "receipt": str(ROOT / "artifacts/etudes/experiment_a/b3_experiment_a_receipt.json"),
            },
            indent=2,
        )
    )
    return 0 if (metrics["levels"]["A"] and metrics["levels"]["B"] and hashes_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
