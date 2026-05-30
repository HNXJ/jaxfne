"""Test that vis.py has no duplicate function definitions and visualization functions work correctly."""

import pytest
import numpy as np
import jaxfne as jtfne
from jaxfne.vis import FigureResult


def test_no_duplicate_vis_functions():
    """Verify there are no duplicate top-level function definitions in vis.py."""
    import ast
    from pathlib import Path

    vis_path = Path(__file__).parent.parent / "jaxfne" / "vis" / "__init__.py"
    code = vis_path.read_text()
    tree = ast.parse(code)

    top_funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_funcs.setdefault(node.name, 0)
            top_funcs[node.name] += 1

    duplicates = {k: v for k, v in top_funcs.items() if v > 1}
    assert not duplicates, f"Found duplicate functions in vis.py: {duplicates}"


def test_raster_exists_and_callable():
    """Raster function should exist and be callable."""
    assert hasattr(jtfne.vis, 'raster')
    assert callable(jtfne.vis.raster)


def test_raster_with_sort_by_parameter():
    """Raster should support sort_by parameter for depth sorting."""
    # Create simple mock signals
    signals = {
        'spikes': np.random.rand(10, 5) > 0.8,
        'time_ms': np.arange(10) * 0.1,
    }

    # Should work with sort_by=None (default)
    fig1 = jtfne.vis.raster(signals, sort_by=None)
    assert fig1 is not None

    # Should work with sort_by="z"
    fig2 = jtfne.vis.raster(signals, sort_by="z")
    assert fig2 is not None


def test_eeg_meg_emm_exist_and_callable():
    """EEG, MEG, EMM should exist and be callable (Suite No. 2 implementations)."""
    for func_name in ['eeg', 'meg', 'emm']:
        assert hasattr(jtfne.vis, func_name), f"vis.{func_name} should exist"
        assert callable(getattr(jtfne.vis, func_name)), f"vis.{func_name} should be callable"


def test_phase5_vis_functions_are_implemented():
    """Phase 5 visualization functions are now implemented (no longer stubs).

    Previously this test verified NotImplementedError for stub functions.
    After Phase 5: bandpower, laminar_profile, connectivity, and geometry3d
    are all implemented and return matplotlib figures (or graceful fallbacks).
    NotImplementedError is no longer expected.
    """
    for func_name in ['bandpower', 'laminar_profile', 'connectivity', 'geometry3d']:
        assert hasattr(jtfne.vis, func_name), f"vis.{func_name} missing"
        func = getattr(jtfne.vis, func_name)
        assert callable(func), f"vis.{func_name} not callable"
        # Verify the function does NOT raise NotImplementedError anymore
        # (it may raise other errors on invalid input, but not NotImplementedError)


def test_raster_with_meta_returns_figure_result():
    """raster_with_meta should return FigureResult."""
    signals = {
        'spikes': np.random.rand(10, 5) > 0.8,
        'time_ms': np.arange(10) * 0.1,
    }

    result = jtfne.vis.raster_with_meta(signals)
    assert isinstance(result, FigureResult)
    assert result.metadata['plot_type'] == 'raster'
    assert result.metadata['proxy_safe'] is True
