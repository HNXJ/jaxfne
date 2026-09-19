"""Acceptance tests for jaxfne.vis.network_inspect.

Stand-ins expose only the public surface the renderers use: neuron_table() and
params["edge_list"]. Nothing here assumes a cortical ontology.
"""

from __future__ import annotations

import ast
import pathlib
import types

import numpy as np
import pytest

from jaxfne.vis import network_inspect as JV

MODULE = pathlib.Path(JV.__file__)


def model(rows, edges):
    if edges:
        pre, post, w, rec = (np.array(c) for c in zip(*edges))
    else:
        pre = post = np.zeros(0, dtype=int)
        w, rec = np.zeros(0), np.zeros(0, dtype=int)
    el = types.SimpleNamespace(pre=pre.astype(int), post=post.astype(int),
                               weight=w.astype(float), receptor_index=rec.astype(int))
    return types.SimpleNamespace(neuron_table=lambda: rows, params={"edge_list": el})


def grid(areas, layers, classes, counts):
    rows, gid, nid = [], {}, 0
    for a in areas:
        for L in layers:
            for c, n in zip(classes, counts):
                gid[(a, L, c)] = (nid, nid + n)
                rows += [{"neuron_id": i, "area": a, "layer": L, "cell_type": c}
                         for i in range(nid, nid + n)]
                nid += n
    return rows, gid


def link(gid, src, dst, receptor=0, weight=0.5):
    (s0, s1), (d0, d1) = gid[src], gid[dst]
    return [(p, q, weight, receptor) for p in range(s0, s1) for q in range(d0, d1)]


# An ontology deliberately unlike any cortical one: no E/PV/SST/VIP, no L1..L6.
ROWS, GID = grid(["Alpha", "Beta", "Gamma"], ["top", "mid", "bot"], ["Fast", "Slow"], (4, 2))
EDGES = (link(GID, ("Alpha", "bot", "Fast"), ("Beta", "mid", "Fast"))
         + link(GID, ("Beta", "bot", "Fast"), ("Gamma", "mid", "Fast"))
         + link(GID, ("Gamma", "top", "Slow"), ("Beta", "top", "Fast"), receptor=1))
M = model(ROWS, EDGES)


# ------------------------------------------------------------------ portability


def test_module_imports_nothing_from_jomission_or_tfne():
    """The packet is copied verbatim into jaxfne, so it may not reach back here."""
    tree = ast.parse(MODULE.read_text("utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    bad = [m for m in imported if m.split(".")[0] in ("jomission", "tfne")]
    assert bad == [], f"portable module imports {bad}"


def test_module_hardcodes_no_area_or_layer_names():
    src = MODULE.read_text("utf-8")
    for name in ("V1_1", "V4_1", "FEF", "PFC", "L2.E", "retina", "Retina"):
        assert name not in src, f"{name!r} is hardcoded in a portable renderer"


# -------------------------------------------------------------------- structure


def test_describe_uses_whatever_ontology_the_table_declares():
    d = JV.describe(M)
    assert d["areas"] == ["Alpha", "Beta", "Gamma"]
    assert set(d["populations"]["Alpha"]) == {"bot", "mid", "top"}
    assert d["populations"]["Alpha"]["bot"] == {"Fast": 4, "Slow": 2}
    assert d["n_neurons"] == len(ROWS)


def test_layers_sort_naturally_not_lexically():
    rows, gid = grid(["A"], ["L1", "L2", "L10"], ["E"], (1,))
    d = JV.describe(model(rows, []))
    assert list(d["populations"]["A"]) == ["L1", "L2", "L10"]


def test_no_individual_neuron_appears_in_the_description():
    d = JV.describe(M)
    for p in d["projections"]:
        assert "ids" not in p and "pre" not in p and "post" not in p


def test_sign_comes_from_the_declared_inhibitory_receptor():
    d = JV.describe(M)
    signs = {(p["source_area"], p["target_area"]): p["sign"] for p in d["projections"]}
    assert signs[("Gamma", "Beta")] == "inhibitory"
    assert signs[("Alpha", "Beta")] == "excitatory"
    # The receptor that means inhibition is the caller's to declare, not ours to assume.
    flipped = JV.describe(M, inhibitory_receptors=(0,))
    s2 = {(p["source_area"], p["target_area"]): p["sign"] for p in flipped["projections"]}
    assert s2[("Alpha", "Beta")] == "inhibitory"
    assert s2[("Gamma", "Beta")] == "excitatory"


def test_local_edges_are_counted_but_not_drawn_as_projections():
    rows, gid = grid(["A", "B"], ["x"], ["E"], (2,))
    m = model(rows, link(gid, ("A", "x", "E"), ("A", "x", "E")) +
              link(gid, ("A", "x", "E"), ("B", "x", "E")))
    d = JV.describe(m)
    assert d["n_edges_local"] == 4 and d["n_edges_long_range"] == 4
    assert len(d["projections"]) == 1


# --------------------------------------------------------------- stage inference


def test_stages_are_inferred_from_the_projection_graph():
    d = JV.describe(M)
    assert d["stage_of"] == {"Alpha": 0, "Beta": 1, "Gamma": 2}


def test_sparser_feedback_does_not_reverse_the_inferred_order():
    """The heavier direction wins, which is what makes a feedback loop survivable."""
    rows, gid = grid(["P", "Q"], ["x"], ["E"], (4,))
    fwd = link(gid, ("P", "x", "E"), ("Q", "x", "E"))
    back = fwd[:2]
    back = [(q, p, w, r) for p, q, w, r in back]
    assert JV.describe(model(rows, fwd + back))["stage_of"] == {"P": 0, "Q": 1}


def test_a_symmetric_pair_carries_no_ordering_and_stays_level():
    rows, gid = grid(["P", "Q"], ["x"], ["E"], (3,))
    fwd = link(gid, ("P", "x", "E"), ("Q", "x", "E"))
    back = [(q, p, w, r) for p, q, w, r in fwd]
    assert JV.describe(model(rows, fwd + back))["stage_of"] == {"P": 0, "Q": 0}


def test_a_model_with_no_projections_is_one_column():
    rows, _ = grid(["Solo"], ["x"], ["E"], (3,))
    d = JV.describe(model(rows, []))
    assert d["stage_of"] == {"Solo": 0}
    assert d["projections"] == []


# ------------------------------------------------------------------- rendering


def test_hspice_renders_a_single_neuron_model(tmp_path):
    m = model([{"neuron_id": 0, "area": "u", "layer": "-", "cell_type": "-"}], [])
    out = tmp_path / "n1.png"
    info = JV.network_hspice(m, path=out)
    assert out.is_file() and out.stat().st_size > 0
    assert info["n_neurons"] == 1 and info["n_projections"] == 0


def test_hspice_renders_with_no_projections_at_all(tmp_path):
    rows, _ = grid(["A", "B"], ["x", "y"], ["E", "I"], (2, 1))
    info = JV.network_hspice(model(rows, []), path=tmp_path / "e.png")
    assert info["n_edges_total"] == 0 and info["n_projections"] == 0


def test_raster_orders_rows_area_then_layer_then_class():
    order, groups = JV.unit_order(M, areas=["Gamma", "Alpha", "Beta"])
    assert [(g["area"], g["layer"], g["cell_type"]) for g in groups][:4] == [
        ("Gamma", "bot", "Fast"), ("Gamma", "bot", "Slow"),
        ("Gamma", "mid", "Fast"), ("Gamma", "mid", "Slow")]
    assert sorted(order.tolist()) == [r["neuron_id"] for r in ROWS]


def test_raster_accepts_a_signals_object_or_a_bare_array(tmp_path):
    spikes = np.zeros((50, len(ROWS)), dtype=np.int8)
    spikes[10, 0] = 1
    a = JV.network_raster(M, spikes, path=tmp_path / "a.png")
    b = JV.network_raster(M, types.SimpleNamespace(spikes=spikes), path=tmp_path / "b.png")
    assert a["n_spikes"] == b["n_spikes"] == 1


def test_raster_window_outside_the_recording_raises(tmp_path):
    spikes = np.zeros((50, len(ROWS)), dtype=np.int8)
    with pytest.raises(ValueError, match="outside the recorded"):
        JV.network_raster(M, spikes, window_ms=(0.0, 500.0), path=tmp_path / "r.png")


def test_raster_reports_thinning_rather_than_dropping_silently(tmp_path):
    spikes = np.ones((40, len(ROWS)), dtype=np.int8)
    info = JV.network_raster(M, spikes, path=tmp_path / "r.png", max_points=50)
    assert info["thinned"] is True
    assert info["n_spikes"] == 40 * len(ROWS)


# ----------------------------------------------------------------------- themes


@pytest.mark.parametrize("fig", ["hspice", "raster"])
def test_theme_changes_presentation_only(tmp_path, fig):
    """The packet's central design rule, as an executable assertion."""
    spikes = np.zeros((60, len(ROWS)), dtype=np.int8)
    spikes[5, 1] = spikes[30, 7] = 1
    out = {}
    for theme in ("light", "dark"):
        if fig == "hspice":
            out[theme] = JV.network_hspice(M, theme=theme, path=tmp_path / f"{theme}.png")
        else:
            out[theme] = JV.network_raster(M, spikes, theme=theme,
                                           path=tmp_path / f"{theme}.png")
    a = {k: v for k, v in out["light"].items() if k not in ("path", "theme")}
    b = {k: v for k, v in out["dark"].items() if k not in ("path", "theme")}
    assert a == b, "theme altered the information, not only the presentation"
    assert out["light"]["theme"] == "light" and out["dark"]["theme"] == "dark"
    # And the pixels must actually differ, or the theme did nothing.
    assert (tmp_path / "light.png").read_bytes() != (tmp_path / "dark.png").read_bytes()


def test_unknown_theme_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown theme"):
        JV.resolve_theme("solarized")


def test_a_theme_object_is_accepted_directly(tmp_path):
    custom = JV.Theme(**{**JV.THEMES["light"].__dict__, "name": "custom",
                         "background": "#fff8e7"})
    info = JV.network_hspice(M, theme=custom, path=tmp_path / "c.png")
    assert info["theme"] == "custom"


def test_unknown_cell_classes_get_distinct_colours():
    """Two populations must never share a colour just because neither is in the palette."""
    th = JV.THEMES["light"]
    colours = th.class_colors(["Fast", "Slow", "Mod"])
    assert len(set(colours.values())) == 3, colours
    # Named classes keep their palette entry even alongside unknown ones.
    mixed = th.class_colors(["E", "Fast", "PV", "Slow"])
    assert mixed["E"] == th.classes["E"] and mixed["PV"] == th.classes["PV"]
    assert len(set(mixed.values())) == 4, mixed


def test_rendering_is_deterministic(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    JV.network_hspice(M, path=a)
    JV.network_hspice(M, path=b)
    assert a.read_bytes() == b.read_bytes()
