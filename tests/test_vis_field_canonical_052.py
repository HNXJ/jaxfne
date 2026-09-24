"""Canonical field-visualization contract — 0.5.2 ENGINE item 8.

Field visualization consumes canonical Q/Phi only; the declared-STOP
``plot_depth_profile`` backend divergence is KEPT (see
``jaxfne.vis.canonical.plot_depth_profile`` for the recorded rationale).

- Plotly canonical field entry points refuse source-only signals: Q is never
  synthesized into Phi on the visualization path (``field_proxy`` gate).
- The matplotlib canonical field entry points draw the declared Phi numbers
  (heatmap/bar data equal ``signals.field`` holders).
- The Q-side display route draws ``signals.sources`` numbers.
- Divergence record: the matplotlib depth renderer is invariant to field
  data (it draws declared neuron geometry, not Phi) while the Plotly depth
  renderer requires declared Phi — different quantities under one name,
  which is why the STOP is kept rather than resolved.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from test_vis_smoke_all import _FakeSignals

DT_MS = 1.0


def _signals(n_steps=256, n_neurons=8, n_contacts=6, with_field=True):
    s = _FakeSignals(n_steps, n_neurons, n_contacts, with_field=with_field)
    s.time_ms = np.arange(n_steps) * DT_MS
    s.metadata = {"dt_ms": DT_MS}
    return s


def _sources_only():
    s = _signals(with_field=False)
    s.sources = np.random.default_rng(2).normal(size=(s.spikes.shape[0], 4))
    return s


# No Q -> Phi synthesis on the visualization path --------------------------------


@pytest.mark.parametrize(
    "fn_name",
    ["plot_lfp", "plot_csd", "plot_psd", "plot_spectrogram", "plot_band_power"],
)
def test_canonical_plotly_field_refuses_source_only_signals(fn_name):
    from jaxfne.vis import canonical as C

    fn = getattr(C, fn_name)
    with pytest.raises(ValueError):
        fn(_sources_only(), backend="plotly")


def test_canonical_plotly_depth_profile_requires_declared_phi():
    from jaxfne.vis import canonical as C

    with pytest.raises(ValueError):
        C.plot_depth_profile(_sources_only(), backend="plotly")


# Declared Phi in, same Phi numbers out -------------------------------------------


def test_canonical_mpl_lfp_draws_declared_phi():
    from jaxfne.vis import canonical as C

    s = _signals()
    fig = C.plot_lfp(s, backend="matplotlib")
    drawn = np.asarray(fig.axes[0].images[0].get_array(), dtype=float)
    np.testing.assert_allclose(drawn, np.asarray(s.field.lfp_proxy).T, rtol=1e-10)


def test_canonical_mpl_csd_draws_declared_phi():
    from jaxfne.vis import canonical as C

    s = _signals()
    fig = C.plot_csd(s, backend="matplotlib")
    drawn = np.asarray(fig.axes[0].images[0].get_array(), dtype=float)
    np.testing.assert_allclose(drawn, np.asarray(s.field.csd_proxy).T, rtol=1e-10)


def test_canonical_mpl_bandpower_dispatches_on_declared_phi():
    from jaxfne.vis import canonical as C
    from jaxfne.vis import fields as F

    s = _signals()
    via_canonical = C.plot_band_power(s, backend="matplotlib")
    direct = F.bandpower(s)
    w_canonical = np.array([p.get_width() for p in via_canonical.axes[0].patches], dtype=float)
    w_direct = np.array([p.get_width() for p in direct.axes[0].patches], dtype=float)
    np.testing.assert_allclose(w_canonical, w_direct, rtol=1e-12)


def test_q_side_display_draws_declared_sources():
    from jaxfne.vis import traces as T

    s = _sources_only()
    fig = T.source(s)
    drawn = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
    np.testing.assert_allclose(drawn, np.asarray(s.sources)[:, 0], rtol=1e-10)


# Divergence record: different quantities, STOP kept --------------------------------


def _signals_with_geometry(n_steps=200, n_contacts=6, with_field=True):
    s = _signals(n_steps=n_steps, n_contacts=n_contacts, with_field=with_field)
    s.metadata["neuron_metadata"] = [
        {"z": float(i) / 8, "cell_type": "E" if i % 2 == 0 else "I"} for i in range(8)
    ]
    return s


def test_mpl_depth_renderer_invariant_to_field_data():
    from jaxfne.vis import fields as F

    with_field = _signals_with_geometry(with_field=True)
    without_field = _signals_with_geometry(with_field=False)
    widths_with = np.array(
        [p.get_width() for p in F.laminar_profile(with_field).axes[0].patches],
        dtype=float,
    )
    widths_without = np.array(
        [p.get_width() for p in F.laminar_profile(without_field).axes[0].patches],
        dtype=float,
    )
    # Neuron-count histogram: declared geometry in, geometry out — Phi irrelevant.
    np.testing.assert_allclose(widths_with, widths_without, rtol=1e-12)


def test_depth_profile_divergence_still_declared():
    from jaxfne.vis import canonical as C

    assert "BACKENDS DIFFER" in (C.plot_depth_profile.__doc__ or "")
    assert "KEEP" in (C.plot_depth_profile.__doc__ or "")
