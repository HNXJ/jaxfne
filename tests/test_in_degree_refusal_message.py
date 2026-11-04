"""Non-integer in-degree refusal hint and changelog dry-run UTF-8 (todo 30)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import subprocess
import sys
from pathlib import Path

import pytest

from jaxfne.connectivity import _validate_in_degree, compile_connection_rules

NEURONS = [
    {"neuron_id": i, "area": "X", "layer": "L", "cell_type": "E"}
    for i in range(10)
]
MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]


@pytest.mark.parametrize("k", [2.5, 0.5])
def test_float_in_degree_names_multinomial_sampling(k):
    with pytest.raises(ValueError) as excinfo:
        _validate_in_degree("r", k)
    message = str(excinfo.value)
    assert "must be a positive integer" in message
    assert 'sampling="multinomial"' in message or "sampling='multinomial'" in message


def test_float_in_degree_default_sampling_names_multinomial():
    rule = {
        "name": "rec",
        "source": {"area": "X"},
        "target": {"area": "X"},
        "weight": 0.1,
        "mechanism": "ampa",
        "allow_self_connections": True,
        "in_degree": 2.5,
    }
    with pytest.raises(ValueError) as excinfo:
        compile_connection_rules(NEURONS, [rule], MECHS, seed=0)
    message = str(excinfo.value)
    assert 'sampling="multinomial"' in message or "sampling='multinomial'" in message


def test_assemble_changelog_dry_run_writes_utf8_bytes(tmp_path):
    frag_dir = tmp_path / "frags"
    frag_dir.mkdir()
    (frag_dir / "cafe.fixed.md").write_bytes(
        "- Fixed caf\u00e9 handling \u2603 in the dry run.\n".encode("utf-8")
    )
    changelog = tmp_path / "changelog.md"
    changelog.write_bytes(b"# Changelog\n\n## [Unreleased]\n\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "assemble_changelog.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run(
        [sys.executable, str(script), "--version", "9.9.9", "--date", "2026-11-03",
         "--fragments", str(frag_dir), "--changelog", str(changelog), "--dry-run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")
    assert "caf\u00e9" in text
    assert "\u2603" in text
