"""Tests for Phase 5 visualization API expansion.

Covers:
- bandpower
- laminar_profile / layer_celltype_counts
- connectivity / connectivity_matrix
- geometry3d / column_geometry
- multi_area_layout
- objective_report
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import vis


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent figure-count warnings."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_signals(n_neurons: int = 8, n_steps: int = 50, n_contacts: int = 8):
    """Return a minimal Signals object from a real smoke simulation."""
    cfg = jtfne.suite2_net1_config(seed=7, n=n_neurons, duration_ms=5.0, dt_ms=0.1)
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.1, seed=7)
    return model.simulate(sim)


def _make_configuration():
    """Return a multi-area Configuration with 2 columns."""
    return jtfne.default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=50)


# ---------------------------------------------------------------------------
# bandpower
# ---------------------------------------------------------------------------

class TestBandpower:
    def test_returns_figure(self):
        signals = _make_minimal_signals()
        fig = vis.bandpower(signals)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_custom_bands(self):
        signals = _make_minimal_signals()
        fig = vis.bandpower(signals, band_definitions={"gamma": (40.0, 80.0)})
        assert fig is not None

    def test_proxy_title_present(self):
        signals = _make_minimal_signals()
        fig = vis.bandpower(signals)
        titles = [ax.get_title() for ax in fig.axes]
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "proxy" in (suptitle + " ".join(titles)).lower()

    def test_finite_data_required(self):
        """bandpower does not raise on signals with finite LFP."""
        signals = _make_minimal_signals()
        fig = vis.bandpower(signals)
        assert fig is not None


# ---------------------------------------------------------------------------
# laminar_profile / layer_celltype_counts
# ---------------------------------------------------------------------------

class TestLaminarProfile:
    def test_returns_figure_from_signals(self):
        signals = _make_minimal_signals()
        fig = vis.laminar_profile(signals)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_alias_layer_celltype_counts(self):
        signals = _make_minimal_signals()
        fig = vis.layer_celltype_counts(signals)
        assert fig is not None

    def test_proxy_safe_title(self):
        signals = _make_minimal_signals()
        fig = vis.laminar_profile(signals)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "proxy" in titles.lower() or "declared" in titles.lower()

    def test_no_neuron_metadata_fallback(self):
        """Returns a placeholder figure when neuron_metadata unavailable."""
        from jaxfne.core import Signals
        import jax.numpy as jnp
        # Minimal Signals with no neuron rows in metadata
        sparse_signals = Signals(
            time_ms=jnp.arange(10, dtype=jnp.float32),
            V_m=jnp.zeros((8, 10)),
            spikes=jnp.zeros((8, 10), dtype=bool),
            sources=jnp.zeros((8, 10)),
            field=None,
            metadata={},  # no neuron_metadata key
        )
        fig = vis.laminar_profile(sparse_signals)
        assert fig is not None


# ---------------------------------------------------------------------------
# connectivity / connectivity_matrix
# ---------------------------------------------------------------------------

class TestConnectivity:
    def test_from_raw_matrix(self):
        W = np.random.randn(20, 20).astype(np.float32)
        fig = vis.connectivity(W)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_alias_connectivity_matrix(self):
        W = np.random.randn(10, 10).astype(np.float32)
        fig = vis.connectivity_matrix(W)
        assert fig is not None

    def test_with_cell_type_labels(self):
        W = np.random.randn(4, 4).astype(np.float32)
        labels = ["E", "PV", "SST", "VIP"]
        fig = vis.connectivity(W, cell_type_labels=labels)
        assert fig is not None

    def test_no_weight_matrix_fallback(self):
        """Returns placeholder figure when weight matrix is not accessible."""
        class FakeModel:
            params = {}  # no 'W' key
        fig = vis.connectivity(FakeModel())
        assert fig is not None

    def test_proxy_title(self):
        W = np.eye(5, dtype=np.float32)
        fig = vis.connectivity(W)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "proxy" in titles.lower() or "weight" in titles.lower()


# ---------------------------------------------------------------------------
# geometry3d / column_geometry
# ---------------------------------------------------------------------------

class TestGeometry3d:
    def test_from_configuration(self):
        cfg = _make_configuration()
        fig = vis.geometry3d(cfg)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_alias_column_geometry(self):
        cfg = _make_configuration()
        fig = vis.column_geometry(cfg)
        assert fig is not None

    def test_proxy_safe_title(self):
        cfg = _make_configuration()
        fig = vis.geometry3d(cfg)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "proxy" in titles.lower() or "declared" in titles.lower()

    def test_from_signals(self):
        signals = _make_minimal_signals()
        fig = vis.geometry3d(signals)
        assert fig is not None

    def test_area_filter(self):
        cfg = _make_configuration()
        fig = vis.geometry3d(cfg, areas=["V1"])
        assert fig is not None


# ---------------------------------------------------------------------------
# multi_area_layout
# ---------------------------------------------------------------------------

class TestMultiAreaLayout:
    def test_from_configuration(self):
        cfg = _make_configuration()
        fig = vis.multi_area_layout(cfg)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_proxy_title(self):
        cfg = _make_configuration()
        fig = vis.multi_area_layout(cfg)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        combined = (suptitle + " " + titles).lower()
        assert "proxy" in combined or "declared" in combined or "metadata" in combined

    def test_no_columns_fallback(self):
        """Returns placeholder when column metadata absent."""
        from jaxfne.core import Signals
        import jax.numpy as jnp
        sparse_signals = Signals(
            time_ms=jnp.arange(10, dtype=jnp.float32),
            V_m=jnp.zeros((4, 10)),
            spikes=jnp.zeros((4, 10), dtype=bool),
            sources=jnp.zeros((4, 10)),
            field=None,
            metadata={},  # no columns key
        )
        fig = vis.multi_area_layout(sparse_signals)
        assert fig is not None

    def test_area_filter(self):
        cfg = _make_configuration()
        fig = vis.multi_area_layout(cfg, areas=["V1"])
        assert fig is not None


# ---------------------------------------------------------------------------
# objective_report
# ---------------------------------------------------------------------------

class TestObjectiveReport:
    def test_from_list(self):
        history = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42]
        fig = vis.objective_report(history)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_from_dict(self):
        data = {"score_history": [2.0, 1.5, 1.2, 0.9, 0.7]}
        fig = vis.objective_report(data)
        assert fig is not None

    def test_no_history_fallback(self):
        """Returns placeholder when history not extractable."""
        fig = vis.objective_report(None)
        assert fig is not None

    def test_proxy_safe_suptitle(self):
        fig = vis.objective_report([1.0, 0.5, 0.3])
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "surrogate" in suptitle.lower() or "proxy" in suptitle.lower()

    def test_finite_only(self):
        """objective_report handles NaN gracefully."""
        history = [1.0, 0.8, float("nan"), 0.5, 0.4]
        fig = vis.objective_report(history)
        assert fig is not None


# ---------------------------------------------------------------------------
# Public API discovery
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_all_phase5_functions_callable_from_vis(self):
        for fn_name in [
            "bandpower", "laminar_profile", "layer_celltype_counts",
            "connectivity", "connectivity_matrix",
            "geometry3d", "column_geometry",
            "multi_area_layout", "objective_report",
        ]:
            assert callable(getattr(vis, fn_name, None)), f"vis.{fn_name} is not callable"

    def test_vis_accessible_from_jtfne_root(self):
        assert hasattr(jtfne, "vis")
        assert callable(jtfne.vis.bandpower)
        assert callable(jtfne.vis.connectivity_matrix)
        assert callable(jtfne.vis.layer_celltype_counts)
        assert callable(jtfne.vis.column_geometry)
        assert callable(jtfne.vis.multi_area_layout)
        assert callable(jtfne.vis.objective_report)
