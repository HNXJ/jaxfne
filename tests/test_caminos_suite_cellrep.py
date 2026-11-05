"""Lane B: Cell Reports CAMINOS suite panels, notebook rules and fixtures.

Tiny synthetic fixtures (a few hundred steps, a few dozen neurons, 2 JSON
rows) stand in for the warehouse receipts; every panel function must return
a matplotlib Figure with the expected number of axes, the raster axis must
hold points for each population, the colour maps must hold exactly one row
per fraction and share one colour scale, no-oscillation f_pk points must be
masked to open floor markers, and the notebook must obey the repo's cell
rules (code cells at most 10 logical lines, no two adjacent code cells) and
display each figure exactly once.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

SUITE_DIR = (Path(__file__).resolve().parents[1]
             / "artifacts" / "etudes" / "papers" / "caminos" / "suite")
NOTEBOOK_PATH = SUITE_DIR / "cellrep_suite.ipynb"
sys.path.insert(0, str(SUITE_DIR))

import cellrep_suite as cs  # noqa: E402

FREQS = [0.0, 10.0, 25.0, 30.0, 35.0, 45.0, 60.0]
MAGS = [0.0, 1.0, 3.0, 5.0, 6.0, 3.0, 1.0]


@pytest.fixture(autouse=True)
def _clear_map_registry():
    cs._MAP_IMAGE_REGISTRY.clear()
    yield
    cs._MAP_IMAGE_REGISTRY.clear()


def make_rows(swept_key, values, seeds=(1, 2), with_delay=True, silent=()):
    """Two-row-shaped run-JSON rows: one row per (value, seed).

    Values listed in ``silent`` get band power None and E rate 0, i.e. the
    "no oscillation" condition the f_pk curves must mask.
    """
    rows = []
    for value in values:
        for seed in seeds:
            silent_row = value in silent
            rows.append({
                "seed": seed,
                swept_key: value,
                "rates_Hz": {"E": 0.0 if silent_row else 2.0 + value + 0.1 * seed,
                             "PV": 7.0 + value + 0.1 * seed,
                             "SOM": 5.0 + value + 0.1 * seed},
                "f_pk_hz": None if silent_row else 30.0 + value + 0.1 * seed,
                "band_power_25_45": None if silent_row else 100.0 + value + seed,
                "max_e_rate_hz": 5.0 + value + seed,
                "phase_delay": {"delay_ms": (6.0 + value) if with_delay else None},
                "psd_freqs_hz": list(FREQS),
                "psd_mag": [m * (1.0 + value) for m in MAGS],
            })
    return rows


def make_ppc_rows(seeds=(1, 2, 3)):
    rows = []
    for seed in seeds:
        rows.append({
            "seed": seed,
            "ppc_by_type": {"by_type": {"E": 0.01 * seed, "PV": 0.05, "SOM": 0.02}},
            "preferred_phase_by_type": {"E": 0.9, "PV": 0.5, "SOM": 1.6},
        })
    return rows


def make_trace(tmp_path, n_steps=400, n_neurons=(20, 10, 10), with_lfp=False,
               with_t0=False, name="baseline.npz", manifest_t_s=None):
    """A shared-format trace .npz (and, when asked, the run manifest.json)."""
    n_e, n_pv, n_som = n_neurons
    n_total = n_e + n_pv + n_som
    rng = np.random.default_rng(0)
    raster = rng.random((n_steps, n_total)) < 0.05
    trace = {
        "raster": raster,
        "type_slices": np.array([[0, n_e], [n_e, n_e + n_pv],
                                 [n_e + n_pv, n_total]], dtype=int),
        "rate_hz": rng.random((n_steps // 10, 3)).astype(float) * 5.0,
        "f_pk_hz": 33.7,
        "band_power_25_45": 1234.5,
        "max_e_rate_hz": 4.7,
    }
    if with_lfp:
        trace["lfp"] = rng.standard_normal(n_steps)
    if with_t0:
        trace["t0_s"] = 11.5
    path = tmp_path / "seed1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **trace)
    if manifest_t_s is not None:
        with open(tmp_path / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump({"T_s": manifest_t_s}, fh)
    return path


def raster_axis(fig):
    return fig.axes[0]


def raster_artists(fig):
    return [line for line in raster_axis(fig).get_lines()
            if line.get_label() in ("E", "PV", "SOM")]


def map_mesh(fig):
    meshes = [c for c in fig.axes[0].collections if hasattr(c, "get_array")]
    assert meshes, "the colour map axis must hold a pcolormesh"
    return meshes[0]


def mesh_data(mesh):
    data = np.ma.filled(np.ma.asanyarray(mesh.get_array(), dtype=float), np.nan)
    return np.atleast_2d(data)


def test_fig2a_top_f_pk_and_delay():
    fig = cs.fig2a_top(make_rows("rate_hz", [2.0, 5.0, 8.0]), source="fig2a.json")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # frequency (Hz) and delay (ms) on separate y axes
    assert "Hz" in fig.axes[0].get_ylabel()
    assert "ms" in fig.axes[1].get_ylabel()
    labels = {line.get_label() for ax in fig.axes for line in ax.get_lines()}
    assert {"f_pk", "PV-to-SOM delay"} <= labels
    assert any("fig2a.json" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_fig2a_masks_no_oscillation():
    rows = make_rows("rate_hz", [2.0, 5.0], silent=(2.0,))
    fig = cs.fig2a_top(rows)
    ax_f = fig.axes[0]
    fpk = next(line for line in ax_f.get_lines() if line.get_label() == "f_pk")
    ydata = np.asarray(fpk.get_ydata(), dtype=float)
    assert np.isnan(ydata[0]), "the silent drive's f_pk must be masked out"
    open_markers = [line for line in ax_f.get_lines()
                    if line.get_markerfacecolor() == "none"]
    assert open_markers, "the silent drive must appear as an open floor marker"
    assert any("no oscillation" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_fig2b_top_rates_one_line_per_population():
    fig = cs.fig2b_top(make_rows("tdrate_hz", [0.0, 1.0]))
    assert len(fig.axes) == 2
    rate_lines = [line for line in fig.axes[0].get_lines()
                  if line.get_label() in ("E", "PV", "SOM")]
    assert {line.get_label() for line in rate_lines} == {"E", "PV", "SOM"}
    assert len(rate_lines[0].get_xdata()) == 2
    plt.close(fig)


@pytest.mark.parametrize(
    "func,swept,n_axes",
    [(cs.fig5a, "value_nS", 2), (cs.fig5b, "value_nS", 2), (cs.fig6a_bottom, "ratio", 1)])
def test_fig5_6_panels(func, swept, n_axes):
    fig = func(make_rows(swept, [0.0, 2.0]))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == n_axes
    plt.close(fig)


@pytest.mark.parametrize("func", [cs.fig3a, cs.fig3b, cs.fig4a, cs.fig4b])
def test_colour_map_panels(func):
    rows = make_rows("fraction", [0.0, 0.5], with_delay=True)
    fig = func(rows, source="map.json")
    assert len(fig.axes) == 4  # map + colourbar + band power + f_pk
    mesh = map_mesh(fig)
    data = mesh_data(mesh)
    assert data.shape[0] == 2  # one equal-height map row per swept fraction
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["0%", "50%"]  # y ticks are the fractions as percent
    lo, hi = fig.axes[0].get_xlim()
    assert (lo, hi) == (0.0, 100.0)  # x = 0-100 Hz
    cbar = next(ax for ax in fig.axes if "log10 |X|" in ax.get_ylabel())
    assert "seed mean" in cbar.get_ylabel()
    assert any("map.json" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_colour_map_one_row_per_fraction():
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig = cs.fig3a(make_rows("fraction", fractions))
    data = mesh_data(map_mesh(fig))
    assert data.shape[0] == len(fractions)  # exactly one row per fraction
    plt.close(fig)


def test_colour_maps_share_limits():
    funcs = (cs.fig3a, cs.fig3b, cs.fig4a, cs.fig4b)
    figs = [f(make_rows("fraction", [0.0, 0.5])) for f in funcs]
    cs.share_map_colours()
    clims = {tuple(map_mesh(fig).get_clim()) for fig in figs}
    assert len(clims) == 1, "the four maps must share one colour scale"
    assert next(iter(clims))[0] < next(iter(clims))[1]
    for fig in figs:
        plt.close(fig)


def test_fig3c_two_stacked_axes():
    fig = cs.fig3c(make_rows("fraction", [0.0, 0.5]),
                   make_rows("fraction", [0.0, 0.7]))
    assert len(fig.axes) == 2
    assert len(fig.axes[0].get_lines()[0].get_xdata()) == 2
    plt.close(fig)


def test_fig1c_ppc_and_phase():
    fig = cs.fig1c(make_ppc_rows())
    assert len(fig.axes) == 2
    for ax in fig.axes:
        points = [line for line in ax.get_lines() if len(line.get_xdata())]
        assert points, "each Fig 1C axis must hold plotted points"
        colours = {line.get_color() for line in points}
        assert set(cs.POP_COLOURS.values()) <= colours, (
            "each population must carry its fixed colour")
    plt.close(fig)


def test_fig_s1_rates_and_spectrum():
    fig = cs.fig_s1(make_rows("p", [0.02, 0.04]))
    assert len(fig.axes) == 2
    rate_lines = [line for line in fig.axes[1].get_lines()
                  if line.get_label() in ("E", "PV", "SOM")]
    assert {line.get_label() for line in rate_lines} == {"E", "PV", "SOM"}
    plt.close(fig)


def test_raster_figure_with_lfp_three_axes(tmp_path):
    path = make_trace(tmp_path, with_lfp=True, with_t0=True)
    fig = cs.fig1b_raster(path, source="baseline.npz")
    assert len(fig.axes) == 3
    artists = raster_artists(fig)
    assert {line.get_label() for line in artists} == {"E", "PV", "SOM"}
    assert all(len(line.get_xdata()) > 0 for line in artists)
    footnote = " ".join(t.get_text() for t in fig.texts)
    assert "20 E / 10 PV / 10 SOM" in footnote  # subsample stated in the footnote
    assert "baseline.npz" in footnote
    plt.close(fig)


def test_raster_figure_without_lfp_two_axes_and_manifest_t0(tmp_path):
    path = make_trace(tmp_path, manifest_t_s=12.0)  # no lfp, no t0_s
    fig = cs.fig4_raster(path)
    assert len(fig.axes) == 2
    loaded = cs.load_trace(path)
    assert loaded["t0_s"] == 11.5  # 12 s minus the 0.5 s window
    assert loaded["lfp"] is None
    plt.close(fig)


def test_load_trace_or_none_missing(tmp_path):
    assert cs.load_trace_or_none(tmp_path / "seed1" / "absent.npz") is None


def test_fig5c_raster_accepts_loaded_trace(tmp_path):
    path = make_trace(tmp_path, with_lfp=True, with_t0=True)
    fig = cs.fig5c(cs.load_trace(path))
    assert len(fig.axes) == 3
    plt.close(fig)


def test_subsample_caps():
    idx = cs._subsample_indices(0, 8000, 200)
    assert len(idx) == 200 and idx[0] == 0 and idx[-1] == 7999
    assert len(cs._subsample_indices(5, 15, 50)) == 10  # below the cap: all kept


# ------------------------------------------------------------- notebook rules

def _logical_lines(source):
    return [line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_cell_rules():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)
    cells = notebook.cells
    assert any(cell.cell_type == "code" for cell in cells)
    previous_code = None
    for cell in cells:
        if cell.cell_type == "markdown":
            previous_code = None
            continue
        if cell.cell_type != "code":
            continue
        assert previous_code is None, "two adjacent code cells in the notebook"
        assert len(_logical_lines(cell.source)) <= 10, (
            f"code cell over 10 logical lines: {cell.source[:60]!r}")
        previous_code = cell
    code_sources = "".join(cell.source for cell in _code_cells(notebook))
    assert "WAREHOUSE" in code_sources and "cellrep_suite" in code_sources


def test_notebook_panels_display_once():
    """A top-level bare ``cs.fig*`` call makes the figure the cell's execute
    result, which the inline backend shows in addition to its own end-of-cell
    flush - the double display. Every panel call must suppress the return
    value (``_ = ...``, a trailing ``;``, or sit inside an if/else block)."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)
    bare = []
    for cell in _code_cells(notebook):
        for line in cell.source.splitlines():
            if re.match(r"cs\.fig\w+\(", line) and not line.rstrip().endswith(";"):
                bare.append(line.strip())
    assert not bare, f"these calls display the figure twice: {bare}"


def test_trace_from_spikes_keeps_the_window_and_rates_per_type():
    # 30 ms run, 4 E + 2 PV + 2 SOM; E neuron 0 spikes every ms, PV neuron 4 once in the kept window.
    steps = 300
    spikes = np.zeros((steps, 8), dtype=bool)
    spikes[::10, 0] = True
    spikes[250, 4] = True
    tr = cs.trace_from_spikes(spikes, (4, 2, 2), window_s=0.005)
    assert tr["raster"].shape == (50, 8)
    assert tr["t0_s"] == pytest.approx(0.025, rel=1e-12)
    assert tr["type_slices"].tolist() == [[0, 4], [4, 6], [6, 8]]
    assert tr["rate_hz"].shape == (5, 3)
    np.testing.assert_allclose(tr["rate_hz"][:, 0], 250.0, rtol=1e-12)  # 1 spike/ms over 4 cells
    np.testing.assert_allclose(tr["rate_hz"][:, 1], [500.0, 0, 0, 0, 0], rtol=1e-12)
    assert tr["rate_hz"][:, 2].sum() == 0
    with pytest.raises(ValueError):
        cs.trace_from_spikes(spikes, (4, 2, 3))
    with pytest.raises(ValueError):
        cs.trace_from_spikes(spikes, (4, 2, 2), window_s=1.0)