"""B1 M3 signed-grid contract: defaults must not silently select a pilot grid."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT / "artifacts/runs/benchmark/b1/b1_jaxley.py"


def test_m3_default_grid_matches_signed_prereg():
    tree = ast.parse(BASELINE.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "GRID_DURATION_MS", "GRID_DT_MS"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    assert constants == {"GRID_DURATION_MS": 1000.0, "GRID_DT_MS": 0.1}
