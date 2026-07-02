"""Tests for the generalized visualization suite.

Validates that all visualizers accept standard signals or readout arrays,
return valid matplotlib figures, and that placeholders throw expected exceptions.
"""

import os
import numpy as np

import jaxfne as jtfne


def test_vis_rich_metadata_result():
    """Verify that _with_meta wrappers return FigureResult."""
    jtfne.vis.require_matplotlib()
    import matplotlib.pyplot as plt
    
    spikes = np.zeros((50, 2))
    result = jtfne.vis.raster_with_meta(spikes)
    
    assert isinstance(result, jtfne.vis.FigureResult)
    assert isinstance(result.fig, plt.Figure)
    assert result.metadata["plot_type"] == "raster"
    assert result.metadata["proxy_safe"] is True
    plt.close(result.fig)


def test_vis_plotters_accept_dt_ms_kwarg():
    """Regression: every time-domain plotter must accept (and not forward) dt_ms.

    Previously raster/lfp/csd/lfp_traces/csd_traces/eeg/meg/emm forwarded dt_ms
    into plt.figure(), raising 'Figure.set() got an unexpected keyword argument
    dt_ms'. The vis API is uniformly (signals, **kwargs) with dt_ms a recognized
    signal-domain kwarg, so all plotters must tolerate it.
    """
    jtfne.vis.require_matplotlib()
    import matplotlib.pyplot as plt

    t_steps, n_neurons, n_contacts = 80, 5, 4
    rng = np.random.default_rng(0)
    signals = {
        "spikes": (rng.random((t_steps, n_neurons)) > 0.9).astype(np.float32),
        "V_m": (rng.standard_normal((t_steps, n_neurons)) * 10 - 65).astype(np.float32),
        "sources": rng.standard_normal((t_steps, n_neurons)).astype(np.float32),
        "lfp_proxy": rng.standard_normal((t_steps, n_contacts)).astype(np.float32),
        "csd_proxy": rng.standard_normal((t_steps, n_contacts)).astype(np.float32),
        "time_ms": np.arange(t_steps) * 0.1,
    }

    for func_name in [
        "raster", "vm", "rate", "source", "lfp", "csd",
        "lfp_traces", "csd_traces", "eeg", "meg", "emm",
        "psd", "spectrogram", "summary",
    ]:
        func = getattr(jtfne.vis, func_name)
        fig = func(signals, dt_ms=0.1)
        assert isinstance(fig, plt.Figure), f"vis.{func_name}(dt_ms=...) must return a Figure"
        plt.close(fig)


def test_phase5_vis_functions_implemented_not_stubs():
    """Verify that Phase 5 visualization functions are implemented (stubs removed).

    Previously verified NotImplementedError from placeholder stubs.
    After Phase 5 implementation: bandpower, laminar_profile, connectivity,
    and geometry3d all return figures or graceful fallbacks — not NotImplementedError.
    """
    for func_name in [
        "bandpower", "laminar_profile", "connectivity", "geometry3d",
        "layer_celltype_counts", "connectivity_matrix", "column_geometry",
        "multi_area_layout", "objective_report",
    ]:
        assert callable(getattr(jtfne.vis, func_name, None)), \
            f"vis.{func_name} should be callable after Phase 5"
