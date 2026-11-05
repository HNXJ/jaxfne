"""Tests for the CAMINOS PLoS CB figure suite (Lane A).

Tiny synthetic fixtures in ``tmp_path``: a few hundred steps, a few dozen
neurons, 2 rows of run-JSON. Each panel function must return a Figure with the
expected axes and the raster axis must hold points for each population. The
notebook must parse and follow the repo's cell rules (code cell <= 10 logical
lines, no two adjacent code cells).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_SUITE = Path(__file__).resolve().parents[1] / "artifacts" / "etudes" / "papers" / "caminos" / "suite"
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

import plos_suite as ps  # noqa: E402

NB_PATH = Path("artifacts") / "etudes" / "papers" / "caminos" / "suite" / "plos_suite.ipynb"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _trace_npz(path, with_lfp=False, t0_s=None):
    rng = np.random.default_rng(7)
    steps, slices = 500, [(0, 20), (20, 25), (25, 30)]
    raster = np.zeros((steps, 30), dtype=bool)
    for a, b in slices:
        cols = rng.integers(a, b, size=40)
        raster[rng.integers(0, steps, size=40), cols] = True
    data = {
        "raster": raster,
        "type_slices": np.array(slices, dtype=int),
        "rate_hz": rng.uniform(0.0, 5.0, size=(steps, 3)),
        "f_pk_hz": 33.0,
        "band_power_25_45": 1.2,
        "max_e_rate_hz": 9.0,
    }
    if with_lfp:
        data["lfp"] = rng.normal(size=steps)
        data["t0_s"] = t0_s if t0_s is not None else 11.5
    np.savez(path, **data)


@pytest.fixture
def run_dir(tmp_path):
    """A minimal plosr-style run dir: one seed, one silenced and one randomized condition."""
    _trace_npz(tmp_path / "trace.npz")
    np.savez(tmp_path / "seed1_pv_randomized_0.4.npz",
             **dict(np.load(tmp_path / "trace.npz")))
    np.savez(tmp_path / "meanfield_pv_silenced_0.4.npz",
             t_s=np.linspace(11.5, 12.0, 500), nu_hz=np.tile([1.0, 8.0, 4.0], (500, 1)))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "T_s": 12.0, "seeds": [1],
        "conditions": {"1": {
            "pv_silenced_0.4": {"file": "trace.npz", "sha256": "x"},
            "pv_silenced_0.9": {"file": "trace.npz", "sha256": "x"},
            "som_silenced_0.9": {"file": "trace.npz", "sha256": "x"},
            "pv_randomized_0.4": {"file": "seed1_pv_randomized_0.4.npz", "sha256": "x"}}},
        "meanfield": {"pv_silenced_0.4": {"file": "meanfield_pv_silenced_0.4.npz", "sha256": "x"}},
    }))
    assert "t0_s" not in np.load(tmp_path / "trace.npz").files  # PLoS trace: no t0_s
    return tmp_path


@pytest.fixture
def sweep_json(tmp_path):
    freqs = np.linspace(0.0, 100.0, 101)
    rows = []
    for frac in (0.0, 0.4):
        for seed in (1, 2):
            rows.append({
                "seed": seed, "fraction": frac,
                "rates_Hz": {"E": 1.0 + frac, "PV": 8.0, "SOM": 4.0},
                "f_pk_hz": 30.0 + 5.0 * frac, "band_power_25_45": 1.0 + 0.5 * frac,
                "max_e_rate_hz": 8.0 + 10.0 * frac,
                "psd_freqs_hz": freqs.tolist(),
                "psd_mag": list(np.exp(-0.5 * ((freqs - 33.0) / 4.0) ** 2) * (1.0 + frac)),
            })
    path = tmp_path / "silence_pv.json"
    path.write_text(json.dumps({"what": "test", "rows": rows}))
    return path


@pytest.fixture
def counts_dir(tmp_path):
    rng = np.random.default_rng(5)
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"file": "counts_seed%d_p%g.npy" % (s, p), "seed": s, "p": p, "dt": 1e-4,
         "sizes": {"E": 8, "PV": 3, "SOM": 3}, "T": 12.0}
        for p in (0.02, 0.04) for s in (1, 2)]}))
    for p in (0.02, 0.04):
        for s in (1, 2):
            counts = rng.poisson(0.02, size=(30000, 3)).astype("int32")  # 3 s: 2 s discard leaves data
            np.save(tmp_path / ("counts_seed%d_p%g.npy" % (s, p)), counts)
    return tmp_path


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def test_load_trace_t0_from_manifest(run_dir):
    t = ps.load_trace(run_dir / "trace.npz", run_dir)
    assert t["t0_s"] == 11.5  # T_s 12.0 from the manifest minus the 0.5 s window
    assert t["lfp"] is None


def test_load_trace_t0_from_file(tmp_path):
    _trace_npz(tmp_path / "trace.npz", with_lfp=True, t0_s=9.25)
    t = ps.load_trace(tmp_path / "trace.npz")
    assert t["t0_s"] == 9.25
    assert t["lfp"].shape == (500,)


def test_mean_rates_and_smoothing():
    counts = np.zeros((12000, 3))
    counts[::10, 0] = 1  # spikes every 10th step
    rates = ps.mean_rates(counts, 1e-4, (20, 5, 5), discard_s=0.2)
    assert rates[0] == pytest.approx(50.0, rel=1e-3)  # 0.1 counts/step / 1e-4 s / 20 cells
    assert rates[1] == 0.0 and rates[2] == 0.0
    sm = ps.smooth_rate(counts[:, 0], 1e-4, 20)
    assert sm.shape == (12000,) and sm.max() > 0


# ---------------------------------------------------------------------------
# panels return Figures with the expected axes and populations
# ---------------------------------------------------------------------------
def _labeled_pops(ax):
    return {label for label in ax.get_legend_handles_labels()[1]}


def _raster_points_per_label(ax):
    """label -> number of plotted points, from the scatter handles with labels."""
    out = {}
    for handle, label in zip(*ax.get_legend_handles_labels()):
        offsets = handle.get_offsets() if hasattr(handle, "get_offsets") else None
        arr = np.asarray(offsets) if offsets is not None else np.empty((0, 2))
        out[label] = int(arr.shape[0]) if arr.ndim == 2 else int(arr.size // 2)
    return out


def test_fig_2a_sweep_panel(sweep_json):
    fig = ps.fig_plos_2a(sweep_json)
    # 2x2 scalar/spectrum panels plus a colorbar axis (two swept values)
    assert len(fig.axes) == 5
    assert any("band power" in a.get_ylabel() for a in fig.axes)
    # smoothed spectra (method named in the axis label) and the seed-SD footnote
    assert any("moving average" in a.get_ylabel() for a in fig.axes)
    assert any("mean ± SD over 2 seeds" in t.get_text() for t in fig.texts)
    # title: paper panel label only, no file name
    assert fig._suptitle.get_text() == "Fig 2A: PV silencing"
    assert not any("silence_pv" in a.get_title() for a in fig.axes)
    plt.close(fig)


def test_fig_4a_ratio_key(sweep_json, tmp_path):
    rows = json.loads(sweep_json.read_text())["rows"]
    for r in rows:
        r["ratio"] = r.pop("fraction") + 0.5  # swept key "ratio", reference 0.5
    path = tmp_path / "fig6a.json"
    path.write_text(json.dumps({"what": "test", "rows": rows}))
    fig = ps.fig_plos_4a(path)
    assert len(fig.axes) == 5
    plt.close(fig)


def test_fig_2c_raster_with_meanfield(run_dir):
    fig = ps.fig_plos_2c(run_dir)
    assert len(fig.axes) == 3  # MF trace, raster, rate
    raster_ax = fig.axes[1]
    assert _labeled_pops(raster_ax) == set(ps.POP_NAMES)
    points = _raster_points_per_label(raster_ax)
    assert all(points[name] > 0 for name in ps.POP_NAMES)
    # one time axis in ms shared by all three rows (the MF row was in s before)
    assert fig.axes[0].get_xlim() == fig.axes[2].get_xlim()
    assert all(l.get_xdata().max() > 1000.0 for l in fig.axes[0].get_lines())
    # markers visible at print size (>= 2 px stroke)
    sizes = [s for c in raster_ax.collections for s in np.atleast_1d(c.get_sizes())]
    assert sizes and min(sizes) >= 4
    # population legend once, on the top row; rate row legend network vs mean field
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[2].get_legend() is not None
    assert raster_ax.get_legend() is None
    # title: paper panel label only; the data file lives in the grey footnote
    assert raster_ax.get_title() == ""
    assert fig.axes[0].get_title() == "Fig 2C: PV silenced 40%"
    assert not any("plosr" in a.get_title() for a in fig.axes)
    assert any(t.get_text().startswith("data:") for t in fig.texts)
    plt.close(fig)


def test_fig_3c_raster_without_meanfield(run_dir):
    fig = ps.fig_plos_3c(run_dir)
    assert len(fig.axes) == 2  # raster, rate
    assert _labeled_pops(fig.axes[0]) == set(ps.POP_NAMES)
    assert all(v > 0 for v in _raster_points_per_label(fig.axes[0]).values())
    assert fig.axes[0].get_legend() is not None  # the one population legend
    assert fig.axes[0].get_title() == "Fig 3C: PV randomized 40%"
    assert any(t.get_text().startswith("data:") for t in fig.texts)
    plt.close(fig)


def test_fig_2f_t0_fallback(run_dir):
    fig = ps.fig_plos_2f(run_dir)
    assert len(fig.axes) == 2  # no mean-field file for som_silenced_0.9
    plt.close(fig)


def test_fig_1e(counts_dir):
    fig = ps.fig_plos_1e(counts_dir, critical_p=0.04, seeds=(1, 2))
    assert len(fig.axes) == 2
    assert _labeled_pops(fig.axes[0]) == set(ps.POP_NAMES)
    # example panel: 1 s of steady state (2 s discard dropped), all three populations
    ax2 = fig.axes[1]
    assert ax2.get_xlim() == (ps.DISCARD_S, ps.DISCARD_S + 1.0)
    assert len(ax2.get_lines()) == 3
    assert _labeled_pops(ax2) == set(ps.POP_NAMES)
    plt.close(fig)


def test_fig_1d_meanfield():
    pytest.importorskip("scipy")
    fig = ps.fig_plos_1d(t_s=0.5)  # short run: fast, still exercises the code path
    assert len(fig.axes) == 1
    plt.close(fig)


def test_fig_6a_6b_6c():
    pytest.importorskip("scipy")
    assert len(ps.fig_plos_6a().axes) == 1
    assert len(ps.fig_plos_6b().axes) == 1
    assert len(ps.fig_plos_6c().axes) == 2
    plt.close("all")


# ---------------------------------------------------------------------------
# notebook rules
# ---------------------------------------------------------------------------
def _logical_lines(source):
    return [ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_notebook_cell_rules():
    nbformat = pytest.importorskip("nbformat")
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(str(root / NB_PATH), as_version=4)
    nbformat.validate(nb)
    cells = nb.cells
    codes = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    assert codes, "notebook has no code cells"
    for i, j in zip(codes, codes[1:]):
        between = cells[i + 1:j]
        assert any(c.cell_type == "markdown" for c in between), \
            "no markdown cell between code cells %d and %d" % (i, j)
    for i in codes:
        n = len(_logical_lines(cells[i].source))
        assert n <= 10, "code cell %d has %d logical lines" % (i, n)
    assert "CAMINOS_WAREHOUSE" in cells[codes[0]].source


def test_notebook_single_display_per_panel():
    """Each panel cell must show its figure exactly once: display(fig) then close it.

    With the inline backend an open figure is rendered again at end of cell, so
    a trailing ``fig`` expression plus the backend flush would show it twice.
    The chosen mechanism: exactly one explicit ``display(fig)`` and the cell's
    last statement closes the figure, so the backend flush finds nothing open.
    """
    nbformat = pytest.importorskip("nbformat")
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(str(root / NB_PATH), as_version=4)
    panels = [c for c in nb.cells if c.cell_type == "code" and "ps.fig_" in c.source]
    assert len(panels) >= 14, "expected every paper panel to have a cell"
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code" or "ps.fig_" not in c.source:
            continue
        lines = [ln.strip() for ln in c.source.splitlines() if ln.strip()]
        assert sum(1 for ln in lines if ln.startswith("display(")) == 1, \
            "code cell %d must call display(fig) exactly once" % i
        assert lines[-1] == "plt.close(fig)", \
            "code cell %d must end with plt.close(fig) (no trailing figure expression)" % i
        assert not any(ln == "fig" for ln in lines), \
            "code cell %d must not return the figure (inline backend would redisplay it)" % i