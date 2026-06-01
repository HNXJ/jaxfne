"""Tests for jaxfne.vis.tutorial_panels (visualization suites)."""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from jaxfne.tutorial_utils import (
    make_laminar_column_config,
    build_laminar_column,
    simulate_laminar_trials,
    spectrolaminar_from_trials,
    summarize_spectrolaminar_similarity,
)
from jaxfne.vis.tutorial_panels import (
    visualize_laminar_column_3d,
    activity_trace_suite,
    spectrolaminar_suite_3panel,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return make_laminar_column_config(
        areas=("V1", "V4"),
        cell_types=("E", "PV"),
        n_neuron_per_column=50,
        n_contacts=16,
    )


@pytest.fixture
def model(config):
    """Build test model."""
    return build_laminar_column(config)


@pytest.fixture
def trials(model, config):
    """Generate test trials."""
    return simulate_laminar_trials(model, config, n_trials=2)


class TestVisualizeLaminarColumn3D:
    """Test 3D network visualization."""

    def test_returns_figure(self, model, config):
        """Test function returns matplotlib Figure."""
        fig = visualize_laminar_column_3d(model, config)
        assert fig is not None
        # Check it's a matplotlib figure
        assert hasattr(fig, 'axes') or hasattr(fig, 'get_axes')

    def test_returns_node_table(self, model, config):
        """Test function returns node table when requested."""
        fig, node_table = visualize_laminar_column_3d(
            model, config,
            return_node_table=True
        )
        assert node_table is not None
        assert len(node_table) == len(model['neurons'])
        assert 'x_m' in node_table.columns

    def test_saves_png(self, model, config, tmp_path):
        """Test PNG output is written."""
        png_path = tmp_path / "test.png"
        visualize_laminar_column_3d(
            model, config,
            output_png=png_path
        )
        assert png_path.exists()
        assert png_path.stat().st_size > 1000

    def test_plotly_html_optional(self, model, config, tmp_path):
        """Test Plotly HTML output is optional (doesn't fail if Plotly missing)."""
        html_path = tmp_path / "test.html"
        # Should not raise even if Plotly not available
        try:
            visualize_laminar_column_3d(
                model, config,
                output_html=html_path
            )
            # HTML might or might not exist depending on Plotly availability
        except ImportError:
            pass  # Expected if Plotly not installed


class TestActivityTraceSuite:
    """Test activity trace visualization."""

    def test_returns_figure(self, trials, config):
        """Test function returns Figure."""
        fig = activity_trace_suite(trials, config)
        assert fig is not None

    def test_saves_png(self, trials, config, tmp_path):
        """Test PNG output is written."""
        png_path = tmp_path / "activity.png"
        activity_trace_suite(
            trials, config,
            output_png=png_path
        )
        assert png_path.exists()
        assert png_path.stat().st_size > 1000

    def test_has_four_subplots(self, trials, config):
        """Test figure has 4 subplots (raster, LFP, CSD, PSD)."""
        fig = activity_trace_suite(trials, config)
        # Matplotlib figure should have subplots
        axes = fig.get_axes()
        # Should have at least 4 subplots
        assert len(axes) >= 4

    def test_duration_window(self, trials, config):
        """Test duration_window_ms parameter works."""
        fig = activity_trace_suite(
            trials, config,
            duration_window_ms=(100.0, 500.0)
        )
        assert fig is not None

    def test_psd_parameters(self, trials, config):
        """Test PSD-related parameters."""
        fig = activity_trace_suite(
            trials, config,
            psd_freq_range_hz=(1.0, 150.0),
            psd_log_x=True,
            psd_smooth_sigma=1.5,
        )
        assert fig is not None

    def test_lfp_gain_parameter(self, trials, config):
        """Test LFP gain parameter."""
        fig = activity_trace_suite(
            trials, config,
            lfp_gain=10.0,
        )
        assert fig is not None


class TestSpectrolaminarSuite3Panel:
    """Test spectrolaminar 3-panel visualization."""

    def test_returns_dict(self, model, config, trials):
        """Test function returns dict mapping area -> Figure."""
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs = spectrolaminar_suite_3panel(specs, model, config)
        assert isinstance(figs, dict)
        assert len(figs) > 0

    def test_one_fig_per_area(self, model, config, trials):
        """Test one figure per area."""
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs = spectrolaminar_suite_3panel(specs, model, config)
        assert len(figs) == len(config.areas)

    def test_saves_png_per_area(self, model, config, trials, tmp_path):
        """Test PNG files are written for each area."""
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs = spectrolaminar_suite_3panel(
            specs, model, config,
            output_dir=tmp_path
        )
        for area in config.areas:
            png_pattern = f"spectrolaminar_*_{area}.png"
            png_files = list(tmp_path.glob(png_pattern))
            assert len(png_files) > 0

    def test_three_subplots_per_figure(self, model, config, trials):
        """Test each figure has 3 main subplots (A, B, C) plus optional colorbar."""
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs = spectrolaminar_suite_3panel(specs, model, config)
        for area, fig in figs.items():
            axes = fig.get_axes()
            # Should have at least 3 main axes (may have colorbar as extra)
            assert len(axes) >= 3

    def test_custom_parameters(self, model, config, trials):
        """Test custom visualization parameters."""
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs = spectrolaminar_suite_3panel(
            specs, model, config,
            density_bins=25,
            density_smooth_sigma=2.0,
            power_vmin=0.4,
            power_vmax=0.95,
        )
        assert len(figs) > 0


# Integration test
class TestFullPipeline:
    """Integration test of full visualization pipeline."""

    def test_full_visualization_pipeline(self, model, config, trials, tmp_path):
        """Test complete visualization pipeline."""
        # Network visualization
        fig_net = visualize_laminar_column_3d(
            model, config,
            output_png=tmp_path / "network.png"
        )
        assert (tmp_path / "network.png").exists()

        # Activity suite
        fig_activity = activity_trace_suite(
            trials, config,
            output_png=tmp_path / "activity.png"
        )
        assert (tmp_path / "activity.png").exists()

        # Spectrolaminar suite
        scores, specs = summarize_spectrolaminar_similarity(trials, config)
        figs_spectro = spectrolaminar_suite_3panel(
            specs, model, config,
            output_dir=tmp_path
        )
        assert len(figs_spectro) == len(config.areas)

        # Verify PNG files exist
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) >= 2  # At least network, activity, and one spectrolaminar
