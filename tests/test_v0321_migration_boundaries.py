"""Advanced Decoupling Invariants tests.

Ensures visual rendering and plotting packages (matplotlib, plotly) are completely
decoupled from the core JAX simulation and objective engines.
"""
from __future__ import annotations

import sys
import pytest


def test_simulation_engine_has_zero_graphics_overhead():
    """Enforces a strict architectural barrier verifying that loading the primary
    computational engine triggers zero visual library attachments.
    """
    import subprocess
    import sys

    # Run in a fresh, isolated Python subprocess to guarantee absolute cleanliness
    code = """
import sys
import jaxfne.core as jtfne_core
import jaxfne.objectives as jtfne_objectives
import jaxfne.runtime as jtfne_runtime

# Check that matplotlib or plotly were not loaded
has_matplotlib = any("matplotlib" in k for k in sys.modules)
has_plotly = any("plotly" in k for k in sys.modules)

if has_matplotlib or has_plotly:
    loaded = [k for k in sys.modules if "matplotlib" in k or "plotly" in k]
    print(f"FAILED: Graphics modules loaded: {loaded}")
    sys.exit(1)
else:
    print("SUCCESS")
    sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Graphics isolation failed: {result.stdout}\nError: {result.stderr}"
