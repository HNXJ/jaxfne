"""Public API regression tests for jaxfne.vis network 3D visualization.

Purpose: ensure that visualize_network_3d is always reachable through every
documented import path and that the installed jaxfne[viz] extra correctly
wires Plotly.  These tests exist as a sentinel — if this public API ever
disappears from the package, CI fails loudly rather than silently.

Truth posture: proxy/computational-scaffold diagnostics only.
"""
from __future__ import annotations

import importlib

import pytest


# ── export contract ────────────────────────────────────────────────────────────


def test_visualize_network_3d_is_exported_from_jtfne_vis() -> None:
    """jtfne.vis.visualize_network_3d must be callable."""
    import jaxfne as jtfne

    assert hasattr(jtfne, "vis"), "jaxfne.vis subpackage not attached to jtfne"
    assert hasattr(jtfne.vis, "visualize_network_3d"), (
        "jaxfne.vis is missing visualize_network_3d — export contract violated"
    )
    assert callable(jtfne.vis.visualize_network_3d)


def test_visualize_network_3d_is_importable_from_network3d_module() -> None:
    """jaxfne.vis.network3d.visualize_network_3d must be importable directly."""
    mod = importlib.import_module("jaxfne.vis.network3d")
    assert hasattr(mod, "visualize_network_3d"), (
        "jaxfne.vis.network3d missing visualize_network_3d"
    )
    assert callable(mod.visualize_network_3d)


def test_visualize_network_3d_same_object_via_both_paths() -> None:
    """Both import paths must resolve to the same callable object."""
    import jaxfne as jtfne
    mod = importlib.import_module("jaxfne.vis.network3d")
    assert jtfne.vis.visualize_network_3d is mod.visualize_network_3d


# ── minimal smoke with Plotly ─────────────────────────────────────────────────


def test_visualize_network_3d_writes_plotly_html_when_plotly_available(tmp_path) -> None:
    """With plotly installed, writing an HTML artifact must succeed."""
    pytest.importorskip("plotly")

    import jaxfne as jtfne

    output_html = tmp_path / "network.html"
    rows = [
        {
            "node_id": 0,
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "cell_type": "E",
            "layer": "L4",
            "area": "V1",
        },
        {
            "node_id": 1,
            "x_m": 1.0e-5,
            "y_m": 0.0,
            "z_m": 1.0e-5,
            "cell_type": "PV",
            "layer": "L4",
            "area": "V1",
        },
    ]

    fig, node_table = jtfne.vis.visualize_network_3d(
        rows,
        output_html=output_html,
        title="Public API smoke - proxy geometry (truth_safe_unverified)",
        coordinate_unit="m",
        display_unit="um",
        return_node_table=True,
    )

    assert fig is not None, "visualize_network_3d returned None figure"
    assert len(node_table) == 2, f"expected 2 nodes, got {len(node_table)}"
    assert output_html.exists(), f"HTML artifact not written to {output_html}"
    assert output_html.stat().st_size > 0, "HTML artifact is empty"


def test_visualize_network_3d_no_plotly_raises_import_error() -> None:
    """Without plotly, the function must raise ImportError (not AttributeError or silent fail)."""
    import sys
    import types

    import jaxfne as jtfne

    # Stub out plotly so _require_plotly() fires
    fake_plotly = None
    original = sys.modules.get("plotly")
    try:
        sys.modules["plotly"] = None  # type: ignore[assignment]
        with pytest.raises((ImportError, ModuleNotFoundError)):
            jtfne.vis.visualize_network_3d(
                [{"node_id": 0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                  "cell_type": "E", "layer": "L4", "area": "V1"}]
            )
    finally:
        if original is None:
            sys.modules.pop("plotly", None)
        else:
            sys.modules["plotly"] = original
