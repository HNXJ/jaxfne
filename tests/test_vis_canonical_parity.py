"""Backend-independent canonical numerical contracts (Rc repair checkpoint).

Tests NUMBERS, not pixels: the matplotlib and Plotly renderers behind
``jaxfne.vis.canonical`` must draw the same underlying quantity for
``plot_population_rate`` (ungrouped) and ``plot_band_power``, and the
reproduced P2-P4/P7-P8 defects must stay closed.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from test_vis_smoke_all import _FakeModel, _FakeSignals

DT_MS = 1.0


def _signals(n_steps=200, n_neurons=8, n_contacts=6, with_time=True, with_field=True):
    s = _FakeSignals(n_steps, n_neurons, n_contacts, with_field=with_field)
    s.time_ms = np.arange(n_steps) * DT_MS
    s.metadata = {"dt_ms": DT_MS}
    if not with_time:
        del s.time_ms
    return s


# P5: shared helper identity -------------------------------------------------


def test_rate_helper_identity():
    from jaxfne.vis.core import binned_population_rate_hz
    from jaxfne.vis.plotly._common import population_rate_hz

    spk = (np.random.default_rng(1).random((100, 6)) < 0.05).astype(float)
    c1, r1 = binned_population_rate_hz(spk, 0.5, 10.0)
    c2, r2 = population_rate_hz(spk, 0.5, 10.0)
    np.testing.assert_allclose(c1, c2)
    np.testing.assert_allclose(r1, r2)


def test_rate_helper_rejects_nonpositive_dt():
    from jaxfne.vis.core import binned_population_rate_hz

    with pytest.raises(ValueError):
        binned_population_rate_hz(np.zeros((10, 2)), 0.0, 10.0)


# P5: rate parity (ungrouped) ------------------------------------------------


def test_rate_parity_ungrouped():
    from jaxfne.vis import traces as T
    from jaxfne.vis.plotly.raster import plot_population_rates as PL

    s = _signals()
    mpl_fig = T.rate(s, bin_ms=10.0)
    pl_fig = PL(s, model=None, bin_ms=10.0)
    mpl_y = np.asarray(mpl_fig.axes[0].lines[0].get_ydata(), dtype=float)
    mpl_x = np.asarray(mpl_fig.axes[0].lines[0].get_xdata(), dtype=float)
    pl_y = np.asarray(pl_fig.data[0].y, dtype=float)
    pl_x = np.asarray(pl_fig.data[0].x, dtype=float)
    np.testing.assert_allclose(mpl_y, pl_y, rtol=1e-10)
    np.testing.assert_allclose(mpl_x, pl_x, rtol=1e-10)


# P5: band-power parity -------------------------------------------------------


def test_band_power_parity_alpha():
    from jaxfne.vis import fields as F
    from jaxfne.vis.plotly.spectra import plot_band_power as PL

    s = _signals(n_steps=256)
    mpl_fig = F.bandpower(s)  # first panel: alpha/beta 8-25 Hz
    pl_fig = PL(s)  # first trace: alpha_beta (8, 25)
    mpl_widths = np.array([p.get_width() for p in mpl_fig.axes[0].patches], dtype=float)
    pl_y = np.asarray(pl_fig.data[0].y, dtype=float)
    assert mpl_widths.shape == pl_y.shape
    np.testing.assert_allclose(mpl_widths, pl_y, rtol=1e-8)


# P2: opacity encoding ----------------------------------------------------------


def test_network3d_rate_alpha_encoded():
    from jaxfne.vis.plotly.network import plot_network_3d

    model = _FakeModel(n_neurons=4)
    hot = _FakeSignals(n_steps=20, n_neurons=4, with_field=False)
    hot.spikes = np.array([[5, 0, 0, 0]] * 20, dtype=float)  # neuron 0 hot
    fig = plot_network_3d(model, hot, color_by="cell_type")
    colors = [c for tr in fig.data for c in tr.marker.color]
    assert colors and all(str(c).startswith("rgba(") for c in colors)
    alphas = [float(str(c).rsplit(",", 1)[-1].rstrip(")")) for c in colors]
    assert max(alphas) > min(alphas)  # hotter neuron -> higher alpha


def test_network3d_no_signals_plain_hex():
    from jaxfne.vis.plotly.network import plot_network_3d

    fig = plot_network_3d(_FakeModel(n_neurons=4), None, color_by="cell_type")
    colors = [c for tr in fig.data for c in tr.marker.color]
    assert colors and all(str(c).startswith("#") for c in colors)


# P3: honest time labels ----------------------------------------------------------


def test_raster_label_index_without_time():
    from jaxfne.vis.rasters import raster

    s = _signals(with_time=False)
    fig = raster(s)
    assert fig.axes[0].get_xlabel() == "Time step index"


def test_raster_label_ms_with_time():
    from jaxfne.vis.rasters import raster

    fig = raster(_signals())
    assert fig.axes[0].get_xlabel() == "Time (ms)"


# P7: square raster demands explicit axis -------------------------------------------


def test_square_raster_raises_without_axis():
    import jax.numpy as jnp
    from jaxfne.vis.rasters import plot_spike_rasters

    with pytest.raises(ValueError, match="n_neurons_axis"):
        plot_spike_rasters(jnp.eye(10), {})


# P4: synthetic geometry labeled ----------------------------------------------------


def test_synthetic_geometry_labels():
    from jaxfne.vis.core import geometry3d_from_config

    class _Cfg:
        metadata = {"columns": [{"name": "col", "n": 6}], "cell_types": {}}

    fig = geometry3d_from_config(_Cfg())
    ax = fig.axes[0]
    assert "NOT declared" in ax.get_title()
    assert "synthetic" in ax.get_xlabel()


# P8: route distinction ---------------------------------------------------------------


def test_field_proxy_requires_declared_probes():
    from jaxfne.vis.plotly._common import field_proxy

    with pytest.raises(ValueError):
        field_proxy(_FakeSignals(with_field=False), "lfp")


def test_constructed_probe_synthesizes_contacts():
    import jax.numpy as jnp
    from jaxfne.fields.probes import lfp_proxy_probe

    # 0.5.2 item 5: synthesis without declared field contacts requires the
    # explicit opt-in and is labeled in the report, never silent.
    out = lfp_proxy_probe(
        jnp.ones((5, 4)),
        contact_depths=jnp.array([0.2, 0.8]),
        allow_synthesized_field_contacts=True,
    )
    assert out.report["method"] == "depth_interpolation_on_phi_e_proxy"
    assert out.report["synthesized_field_contacts"] is True


def test_visualization_proxy_needs_only_sources():
    from jaxfne.vis import traces as T

    s = _FakeSignals(with_field=False)
    s.sources = np.random.default_rng(2).normal(size=(s.spikes.shape[0], 4))
    fig = T.eeg(s, n_channels=2)  # works without any field: display-only route
    assert len(fig.axes) == 1


# Depth-profile divergence stays declared ----------------------------------------------


def test_depth_profile_divergence_declared():
    from jaxfne.vis import canonical as C

    assert "BACKENDS DIFFER" in (C.plot_depth_profile.__doc__ or "")
