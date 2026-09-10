"""W8 benchmark harness structure and invariant tests (no exact RSS/timing equality)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts" / "audit" / "w8_baseline.json"
HARNESS = ROOT / "scripts" / "perf" / "w8_benchmark_harness.py"

W7_ANCHOR_PROFILES = frozenset({
    "dense_all_to_all",
    "bounded_degree_dual_storage",
    "dense_masked",
    "sparse_direct",
})


def test_w8_single_cell_runs():
    proc = subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--cell",
            "dense_all_to_all_N1000",
            "--duration-ms",
            "4",
            "--dt-ms",
            "0.5",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    assert rec["status"] == "ok"
    assert rec["M_persistent_bytes"] > 0
    assert rec["M_construction_peak_bytes"] >= 0
    assert rec["M_recording_bytes"] >= 0
    assert rec["M_simulation_state_derived_bytes"] >= 0
    assert "derived RSS upper bound" in rec["M_simulation_state_note"]
    assert rec["backend_realized"] == "dense"
    assert rec["t_compile_s"] >= 0
    assert rec["t_step_s"] > 0
    assert math.isfinite(rec["t_compile_s"]) and math.isfinite(rec["t_step_s"])


def test_w8_baseline_schema_and_invariants():
    assert BASELINE.is_file(), "run: python scripts/perf/w8_benchmark_harness.py"
    data = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert data["w8_verdict"] == "BASELINE_RECORDED"
    assert data["schema"] == "jaxfne.w8.baseline.v1"
    assert len(data["cells"]) == 8

    prov = data["provenance"]
    for key in (
        "git_head",
        "git_branch",
        "python",
        "platform",
        "jax",
        "jaxlib",
        "jax_default_backend",
        "dtype",
        "recording_mode",
    ):
        assert key in prov and prov[key]

    assert data["duration_ms"] > 0
    assert data["dt_ms"] > 0
    assert "volatile_fields" in data
    assert "generated_utc" in data["volatile_fields"]

    profiles = {c.get("profile") for c in data["cells"]}
    assert W7_ANCHOR_PROFILES <= profiles

    sweep = [c for c in data["cells"] if c.get("sweep") == "standard_N"]
    assert len(sweep) == 4
    assert {c["config"]["n"] for c in sweep} == {1, 10, 100, 1000}

    dual = next(c for c in data["cells"] if c["profile"] == "bounded_degree_dual_storage")
    w = next(a for a in dual["nxn_arrays"] if ".W" in a["array"])
    assert tuple(w["shape"]) == (1000, 1000)
    assert w["R_k"] > 1000  # measurement evidence only; not a removability proof

    sparse = next(c for c in data["cells"] if c["profile"] == "sparse_direct")
    assert sparse["nxn_arrays"] == []

    for cell in data["cells"]:
        assert cell["status"] == "ok"
        assert cell["M_persistent_bytes"] > 0
        assert cell["M_construction_peak_bytes"] >= 0
        assert cell["M_recording_bytes"] >= 0
        assert cell["M_simulation_state_derived_bytes"] >= 0
        assert "derived RSS upper bound" in cell["M_simulation_state_note"]
        assert cell["t_compile_s"] >= 0
        assert cell["t_step_s"] > 0
        assert cell["config"]["n"] > 0
        assert "backend_realized" in cell
        assert "E" in cell
