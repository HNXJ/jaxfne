"""C2 prospective delay-state continuation validation runner and receipt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from jaxfne.io import json_safe

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "artifacts" / "protocol_c"
RECEIPT_PATH = BUNDLE_ROOT / "c2_delay_continuation_receipt.json"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_c2_receipt() -> dict[str, Any]:
    return json.loads(RECEIPT_PATH.read_text())


def write_c2_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    import pytest

    exit_code = pytest.main(
        [
            "-q",
            "tests/test_v0417_c2_delay_continuation.py",
            "-k",
            "not frozen_receipt",
            "tests/test_continuation_contract.py",
        ]
    )
    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_c.c2_delay_continuation_receipt.v1",
        "checkpoint": "C2",
        "package_head": package_head or _git_head(),
        "c2_pass": exit_code == 0,
        "scope": "canonical_model_simulate_delay_state_continuation",
        "equivalence_contract": "Sim(T1+T2)(X0) == Sim(T2)(Sim(T1)(X0)) bit-exact on X,H,B_t,synaptic state",
        "public_state_name": "delay_state",
        "legacy_alias": "spike_history",
        "tests": [
            "tests/test_v0417_c2_delay_continuation.py",
            "tests/test_continuation_contract.py",
        ],
        "pytest_exit_code": int(exit_code),
    }
    RECEIPT_PATH.write_text(json.dumps(json_safe(receipt), indent=2) + "\n")
    return receipt
