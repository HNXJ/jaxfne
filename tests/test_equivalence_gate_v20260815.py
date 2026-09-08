"""Phase-A seam-equivalence gate: the Figure 1-7 build_figure*() refactor
authorized no scientific change (exact decoded-pixel identity vs frozen PNGs)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "publication_figures" / "equivalence_gate.py"
TRACKED_REPORT = ROOT / "artifacts" / "publication" / "equivalence_report.json"
FROZEN_MANIFEST = ROOT / "artifacts/publication/frozen_manifest.json"


def _run_gate(*, report_dir: Path) -> dict:
    render = report_dir / "render"
    report = report_dir / "equivalence_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--render-dir",
            str(render),
            "--report",
            str(report),
        ],
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(report.read_text(encoding="utf-8"))


def test_equivalence_gate_tracked_report_exists_and_schema():
    assert TRACKED_REPORT.is_file()
    report = json.loads(TRACKED_REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == "jaxfne.harness.seam_equivalence.v1"
    assert len(report["cases"]) == 7
    assert all(c["decoded_pixel_equal"] for c in report["cases"])
    for c in report["cases"]:
        assert not c["temp_png"].startswith("/")


def test_equivalence_gate_reproducible_7_of_7(tmp_path):
    assert FROZEN_MANIFEST.is_file()
    report = _run_gate(report_dir=tmp_path)
    assert len(report["cases"]) == 7
    for c in report["cases"]:
        assert c["H_equal"] and c["W_equal"]
        if report["byte_identity_pinned"]:
            # Freeze platform (macOS, matplotlib 3.10.9): the refactor must
            # reproduce the frozen art exactly, pixels AND bytes.
            assert c["decoded_pixel_equal"]
            assert c["byte_sha_equal"]
        else:
            # Other platforms (CI Linux): PNG rendering is not byte/pixel
            # deterministic across font stacks; the gate still runs and
            # dimension-equality is enforced, while pixel/byte identity is
            # recorded as informational (see equivalence_gate.py's
            # byte_identity_pinned()). The paper's reproducibility claims
            # rest on frozen receipts and tracked SHAs, not PNG bytes.
            pass


def test_equivalence_gate_tracked_report_matches_fresh_run(tmp_path):
    fresh = _run_gate(report_dir=tmp_path)
    if not fresh["byte_identity_pinned"]:
        # Fresh non-freeze-platform runs report byte_sha_equal=False by
        # design; the frozen tracked report encodes freeze-platform runs
        # only, so cross-platform comparison is undefined here.
        pytest.skip("byte identity is only pinned on the freeze platform")
    tracked = json.loads(TRACKED_REPORT.read_text(encoding="utf-8"))
    keys = {"figure", "frozen_png", "H_equal", "W_equal", "RGBA_equal",
            "decoded_pixel_equal", "byte_sha_equal", "sha256_frozen", "sha256_post"}
    fresh_cases = [{k: c[k] for k in keys} for c in fresh["cases"]]
    tracked_cases = [{k: c[k] for k in keys} for c in tracked["cases"]]
    assert fresh_cases == tracked_cases


@pytest.mark.skipif(not FROZEN_MANIFEST.is_file(), reason="frozen manifest missing")
def test_equivalence_gate_frozen_manifest_selfcheck(tmp_path):
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    expected = [
        "artifacts/figures/publication/fig01_tfne_grammar.png",
        "artifacts/figures/publication/fig02_emitter_source.png",
        "artifacts/figures/publication/fig03_local_observation.png",
        "artifacts/figures/publication/fig04_multiscale_boundary.png",
        "artifacts/figures/publication/fig05_traveling_wave_no_wave.png",
        "artifacts/figures/publication/fig06_rbs_hdp_ladder.png",
        "artifacts/figures/publication/fig07_e_integration.png",
    ]
    for rel in expected:
        assert rel in files, f"{rel} not in artifacts/publication/frozen_manifest.json"

# ── W12: renderer version is part of frozen-figure provenance ────────────────
def test_frozen_manifest_declares_renderer_provenance():
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    renderer = manifest.get("renderer")
    assert renderer, "frozen manifest must declare the renderer it was frozen under"
    assert renderer["requirement"].startswith("matplotlib")
    assert renderer["verified_compatible"], "at least one verified-compatible renderer"
    assert "3.11.1" in renderer["verified_incompatible"]


def test_dev_extra_pins_renderer_below_311():
    """The dev extra runs the equivalence gate, so it must pin <3.11; viz must not."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev = next(ln for ln in text.splitlines() if ln.startswith("dev = ["))
    viz = next(ln for ln in text.splitlines() if ln.startswith("viz = ["))
    assert "matplotlib>=3.10.9,<3.11" in dev, "dev extra must pin the frozen-figure renderer"
    assert "matplotlib>=3.10.9,<3.12" in viz, "viz users must not be constrained by the freeze"


def test_gate_refuses_unverified_renderer_with_a_named_version():
    """A renderer mismatch must fail with the version named, not as a width mismatch."""
    spec = importlib.util.spec_from_file_location("_equivalence_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_equivalence_gate"] = module
    sys.path.insert(0, str(GATE.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(GATE.parent))
    import matplotlib

    with mock.patch.object(matplotlib, "__version__", "3.11.1"):
        with pytest.raises(SystemExit) as excinfo:
            module.check_renderer()
    message = str(excinfo.value)
    assert "3.11.1" in message and "known-incompatible" in message
