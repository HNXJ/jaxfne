"""Tests for proxy-safe title enforcement on all visualization figures.

Asserts that no title uses words like "real", "empirical", or "calibrated" without
explicit proxy context, and that simulated/proxy vocabulary is correctly present.
"""

import pytest
import numpy as np

import jaxfne as jtfne


def test_vis_titles_proxy_safe():
    """Verify that plotting function titles do not claim empirical/calibrated reality."""
    jtfne.vis.require_matplotlib()
    
    # Generate mock signals
    spikes = np.zeros((10, 2))
    
    # List of implemented visualization functions
    vis_funcs = [
        jtfne.vis.raster,
        jtfne.vis.vm,
        jtfne.vis.rate,
        jtfne.vis.source,
        jtfne.vis.lfp,
        jtfne.vis.csd,
        jtfne.vis.psd,
        jtfne.vis.spectrogram,
        jtfne.vis.summary
    ]
    
    forbidden = {"real ", "calibrated ", "biological", "empirical", "empirical "}
    required_any = {"simulated", "proxy", "summary"}
    
    for func in vis_funcs:
        fig = func(spikes)
        
        # Collect all text labels/titles in the figure
        texts = []
        if fig.texts:
            texts.extend([t.get_text() for t in fig.texts])
        for ax in fig.axes:
            if ax.get_title():
                texts.append(ax.get_title())
            if ax.title:
                texts.append(ax.title.get_text())
                
        full_text = " ".join(texts).lower()
        
        # Verify no forbidden words
        for word in forbidden:
            assert word not in full_text, f"Forbidden word '{word}' found in {func.__name__} title: '{full_text}'"
            
        # Verify at least one required proxy word
        assert any(req in full_text for req in required_any), \
            f"No proxy-safe terminology found in {func.__name__} title: '{full_text}'"
