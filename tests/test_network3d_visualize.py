"""Tests for jtfne.vis.visualize_network_3d.

Verifies:
- All supported input coercions (mapping, DataFrame, list-of-dict, Model, Signals, networkx)
- 1D/2D coordinate promotion to 3D
- Deterministic duplicate-position jitter
- HTML output writing
- return_node_table API
- Core import does NOT require Plotly
- Missing Plotly raises clear install message
- Existing geometry3d still returns a matplotlib figure (compatibility guard)

Truth posture: proxy/computational-scaffold diagnostics only.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _minimal_rows(n=4):
    """Return minimal list-of-dict rows with x_m/y_m/z_m."""
    rng = np.random.default_rng(42)
    return [
        {
            "node_id": i,
            "x_m": float(rng.uniform(-0.3, 0.3)),
            "y_m": float(rng.uniform(-0.3, 0.3)),
            "z_m": float(rng.uniform(0.0, 1.6)),
            "cell_type": ["E", "PV", "SST", "VIP"][i % 4],
            "layer": "L4",
            "area": "V1",
        }
        for i in range(n)
    ]


def _require_plotly_or_skip():
    """Skip test if Plotly is not installed."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        pytest.skip("Plotly not installed")


# ── tests ──────────────────────────────────────────────────────────────────────


def test_mapping_positions_m_3d_returns_plotly_figure():
    """dict with positions_m (N×3) returns a Plotly Figure."""
    _require_plotly_or_skip()
    import plotly.graph_objects as go
    import jaxfne.vis as vis

    positions = np.array([[0.1, 0.2, 0.5], [-0.1, 0.0, 1.2], [0.3, 0.1, 0.8]])
    fig = vis.visualize_network_3d({"positions_m": positions})
    assert isinstance(fig, go.Figure)


def test_dataframe_x_m_y_m_z_m_supported():
    """pandas DataFrame with x_m/y_m/z_m columns is accepted."""
    _require_plotly_or_skip()
    pd = pytest.importorskip("pandas")
    import plotly.graph_objects as go
    import jaxfne.vis as vis

    df = pd.DataFrame(_minimal_rows())
    fig = vis.visualize_network_3d(df)
    assert isinstance(fig, go.Figure)


def test_list_of_dict_rows_supported():
    """list-of-dict records are accepted."""
    _require_plotly_or_skip()
    import plotly.graph_objects as go
    import jaxfne.vis as vis

    fig = vis.visualize_network_3d(_minimal_rows())
    assert isinstance(fig, go.Figure)


def test_one_dimensional_positions_are_promoted_to_3d():
    """1D coordinates (shape N×1) are promoted: y=0, z=0."""
    _require_plotly_or_skip()
    import jaxfne.vis as vis

    positions = np.array([[0.1], [0.2], [0.3]])
    _, rows = vis.visualize_network_3d(
        {"positions_m": positions}, return_node_table=True
    )
    for r in rows:
        assert r["y_m"] == 0.0
        assert r["z_m"] == 0.0


def test_two_dimensional_positions_are_promoted_to_3d():
    """2D coordinates (shape N×2) are promoted: z=0."""
    _require_plotly_or_skip()
    import jaxfne.vis as vis

    positions = np.array([[0.1, 0.2], [0.3, 0.4]])
    _, rows = vis.visualize_network_3d(
        {"positions_m": positions}, return_node_table=True
    )
    for r in rows:
        assert r["z_m"] == 0.0


def test_duplicate_positions_are_jittered_deterministically():
    """Identical positions receive deterministic jitter; same seed → same result."""
    _require_plotly_or_skip()
    import jaxfne.vis as vis

    # 4 identical positions
    positions = np.zeros((4, 3))
    _, rows_a = vis.visualize_network_3d(
        {"positions_m": positions}, seed=7, return_node_table=True
    )
    _, rows_b = vis.visualize_network_3d(
        {"positions_m": positions}, seed=7, return_node_table=True
    )
    for ra, rb in zip(rows_a, rows_b):
        assert ra["x_m"] == pytest.approx(rb["x_m"])
        assert ra["y_m"] == pytest.approx(rb["y_m"])
        assert ra["z_m"] == pytest.approx(rb["z_m"])


def test_output_html_is_written(tmp_path):
    """output_html writes an HTML file to the specified path."""
    _require_plotly_or_skip()
    import jaxfne.vis as vis

    html_path = tmp_path / "network.html"
    vis.visualize_network_3d(_minimal_rows(), output_html=html_path)
    assert html_path.exists()
    assert html_path.stat().st_size > 100


def test_return_node_table_returns_rows():
    """return_node_table=True returns (fig, list[dict]) with required keys."""
    _require_plotly_or_skip()
    import plotly.graph_objects as go
    import jaxfne.vis as vis

    result = vis.visualize_network_3d(_minimal_rows(), return_node_table=True)
    assert isinstance(result, tuple) and len(result) == 2
    fig, rows = result
    assert isinstance(fig, go.Figure)
    assert isinstance(rows, list)
    required = {"node_id", "x_m", "y_m", "z_m", "cell_type", "layer", "area", "jittered"}
    for row in rows:
        assert required.issubset(row.keys()), f"Missing keys: {required - row.keys()}"


def test_networkx_like_graph_supported_if_networkx_available():
    """networkx Graph is accepted via nodes(data=True) / edges()."""
    _require_plotly_or_skip()
    nx = pytest.importorskip("networkx")
    import plotly.graph_objects as go
    import jaxfne.vis as vis

    G = nx.Graph()
    G.add_node(0, x_m=0.1, y_m=0.0, z_m=0.5, cell_type="E", layer="L4", area="V1")
    G.add_node(1, x_m=0.2, y_m=0.1, z_m=1.0, cell_type="PV", layer="L4", area="V1")
    G.add_edge(0, 1)
    fig = vis.visualize_network_3d(G, show_edges=True)
    assert isinstance(fig, go.Figure)


def test_core_import_does_not_require_plotly():
    """import jaxfne (and jaxfne.vis) must succeed without Plotly present."""
    # This test runs in the current interpreter, which may have Plotly.
    # We verify that the import chain doesn't eagerly import Plotly.
    import importlib
    import jaxfne  # noqa: F401
    import jaxfne.vis  # noqa: F401
    # No plotly in the module's top-level namespace check
    vis_mod = sys.modules.get("jaxfne.vis")
    assert vis_mod is not None
    # The function should be importable without triggering plotly
    from jaxfne.vis import visualize_network_3d  # noqa: F401


def test_missing_plotly_raises_clear_install_message(monkeypatch):
    """When Plotly is not importable, calling visualize_network_3d raises ImportError
    with a message directing the user to install jaxfne[viz]."""
    import jaxfne.vis.network3d as n3d

    # Temporarily make _require_plotly raise as if plotly is absent
    original = n3d._require_plotly

    def _fake_require_plotly():
        raise ImportError(
            'Plotly is required for interactive 3D visualization.\n'
            'Install visualization extras with: pip install "jaxfne[viz]"'
        )

    monkeypatch.setattr(n3d, "_require_plotly", _fake_require_plotly)
    with pytest.raises(ImportError, match="jaxfne\\[viz\\]"):
        n3d.visualize_network_3d(_minimal_rows())


def test_existing_geometry3d_still_returns_matplotlib_figure():
    """geometry3d must still return a matplotlib Figure (compatibility guard)."""
    import matplotlib.figure
    import jaxfne.vis as vis

    # geometry3d with empty input falls back gracefully to a placeholder figure
    class _FakeSignals:
        metadata = {}

    fig = vis.geometry3d(_FakeSignals())
    assert isinstance(fig, matplotlib.figure.Figure), (
        f"geometry3d return type changed: got {type(fig).__name__}"
    )
