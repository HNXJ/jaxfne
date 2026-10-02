"""Network views: E/I edge split in plot_network_3d and the area graph."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("plotly")

from jaxfne.vis.plotly.network import plot_area_graph, plot_network_3d  # noqa: E402


def _model(areas, cell_types, pre, post):
    rows = [{"area": a, "cell_type": c, "layer": "L1", "x": float(i), "y": 0.0, "z": 0.0}
            for i, (a, c) in enumerate(zip(areas, cell_types))]
    edges = SimpleNamespace(pre=np.array(pre), post=np.array(post))
    return SimpleNamespace(neuron_table=lambda: rows, params={"edge_list": edges})


def _segments(trace):
    return sum(v is None for v in trace.x)


def test_source_class_splits_edges_by_presynaptic_class():
    m = _model(["A"] * 4, ["E", "E", "PV", "SST"], pre=[0, 1, 2, 3, 0], post=[2, 3, 0, 1, 1])
    fig = plot_network_3d(m, show_edges=True, edge_color_by="source_class")
    by_name = {t.name: t for t in fig.data}
    assert _segments(by_name["edges from E* types"]) == 3
    assert _segments(by_name["edges from other types"]) == 2


def test_default_edges_stay_one_grey_trace():
    m = _model(["A"] * 2, ["E", "PV"], pre=[0], post=[1])
    names = [t.name for t in plot_network_3d(m, show_edges=True).data]
    assert names == ["E", "PV", "edges"]


def test_edge_color_by_is_validated():
    m = _model(["A"] * 2, ["E", "PV"], pre=[0], post=[1])
    with pytest.raises(ValueError):
        plot_network_3d(m, edge_color_by="source_class")
    with pytest.raises(ValueError):
        plot_network_3d(m, show_edges=True, edge_color_by="weight")


def test_area_graph_counts_directed_inter_area_edges():
    # A->B twice, B->A once, C->A once, A->A (within-area, not drawn)
    m = _model(["A", "A", "B", "C"], ["E"] * 4, pre=[0, 1, 2, 3, 0], post=[2, 2, 0, 1, 1])
    fig = plot_area_graph(m, n_width_classes=2)
    hover = [t for t in fig.data if t.hovertemplate == "%{text}<extra></extra>" and t.name != "areas"][0]
    assert sorted(hover.text) == sorted(["A→B: 2<br>B→A: 1", "A→C: 0<br>C→A: 1"])
    nodes = [t for t in fig.data if t.name == "areas"][0]
    assert list(nodes.text) == ["A", "B", "C"] and list(nodes.customdata) == [2, 1, 1]


def test_area_graph_needs_two_areas():
    with pytest.raises(ValueError):
        plot_area_graph(_model(["A", "A"], ["E", "E"], pre=[0], post=[1]))


@pytest.mark.parametrize("n", [0, -1, True, 2.0])
def test_area_graph_refuses_non_positive_int_width_classes(n):
    m = _model(["A", "B"], ["E", "E"], pre=[0], post=[1])
    with pytest.raises(ValueError, match="n_width_classes"):
        plot_area_graph(m, n_width_classes=n)
