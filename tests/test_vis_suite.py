"""Tests for the generalized visualization suite.

Validates that all visualizers accept standard signals or readout arrays,
return valid matplotlib figures, and that placeholders throw expected exceptions.
"""

import os
import pytest
import numpy as np

import jaxfne as jtfne


def test_vis_suite_plot_generations():
    """Verify that plotting functions return valid matplotlib figures."""
    jtfne.vis.require_matplotlib()
    import matplotlib.pyplot as plt
    
    # Generate mock signals
    t_steps = 100
    n_neurons = 5
    spikes = (np.random.rand(t_steps, n_neurons) > 0.9).astype(np.float32)
    V_m = np.random.randn(t_steps, n_neurons).astype(np.float32) * 10 - 65
    sources = np.random.randn(t_steps, n_neurons).astype(np.float32)
    
    signals = {
        "spikes": spikes,
        "V_m": V_m,
        "sources": sources,
        "time_ms": np.arange(t_steps) * 0.1
    }
    
    # 1. Raster
    fig_raster = jtfne.vis.raster(signals)
    assert isinstance(fig_raster, plt.Figure)
    plt.close(fig_raster)
    
    # 2. Vm
    fig_vm = jtfne.vis.vm(signals)
    assert isinstance(fig_vm, plt.Figure)
    plt.close(fig_vm)
    
    # 3. Rate
    fig_rate = jtfne.vis.rate(signals)
    assert isinstance(fig_rate, plt.Figure)
    plt.close(fig_rate)
    
    # 4. Source
    fig_source = jtfne.vis.source(signals)
    assert isinstance(fig_source, plt.Figure)
    plt.close(fig_source)
    
    # 5. LFP and CSD
    lfp_data = np.random.randn(t_steps, 4).astype(np.float32)
    fig_lfp = jtfne.vis.lfp(lfp_data)
    assert isinstance(fig_lfp, plt.Figure)
    plt.close(fig_lfp)
    
    fig_csd = jtfne.vis.csd(lfp_data)
    assert isinstance(fig_csd, plt.Figure)
    plt.close(fig_csd)


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


def test_reserved_vis_functions_throw():
    """Verify that reserved/placeholder visualization functions throw NotImplementedError."""
    spikes = np.zeros((50, 2))

    # Note: eeg, meg, emm are now implemented via Suite No. 2; only these remain reserved
    reserved_funcs = [
        jtfne.vis.bandpower,
        jtfne.vis.laminar_profile,
        jtfne.vis.connectivity,
        jtfne.vis.geometry3d,
    ]

    for func in reserved_funcs:
        with pytest.raises(NotImplementedError):
            func(spikes)
