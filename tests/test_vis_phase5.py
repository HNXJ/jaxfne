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
from matplotlib.figure import Figure

import jaxfne as jtfne
from jaxfne import vis
from jaxfne.builders import Configuration


def _assert_real_figure(fig):
    """A returned vis figure must be a real matplotlib Figure with content."""
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0
    assert hasattr(fig, "savefig")


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


def _default_spectrolaminar_config(
    areas=None,
    n_per_area: int = 100,
    seed=None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Local replacement for the removed jtfne.default_spectrolaminar_config.

    default_spectrolaminar_config was removed from jaxfne/builders.py
    (archived as legacy-schema JSON at
    jaxfne/configs/legacy/spectrolaminar_default.json). Reproduced here
    verbatim from the pre-removal source (commit 1715ec2^) using the
    still-public Configuration fluent API.
    """
    if areas is None:
        areas = ["V1", "V4"]

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .areas(areas)
    )

    for area in areas:
        cfg = cfg.column(area, layers=["L1", "L2/3", "L4", "L5", "L6"], n=n_per_area)

    cfg = (
        cfg.cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .area_layer_cell_types(
            "V1",
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )
    )

    if len(areas) > 1:
        cfg = cfg.area_layer_cell_types(
            areas[1],
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )

    cfg = (
        cfg.uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.35, edge_seed=seed or 42)
    )

    if len(areas) >= 2:
        cfg = cfg.inter_column_connectivity(
            source_area=areas[0],
            target_area=areas[1],
            mode="sparse",
            p_feedforward=0.3,
            p_feedback=0.2,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
        )

    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD", "EEG", "MEG", "EMM"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .objective(
            firing_rate_target={"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0},
            band_definitions={"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)},
        )
    )
    return cfg


def _make_configuration():
    """Return a multi-area Configuration with 2 columns."""
    return _default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=50)


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
        _assert_real_figure(fig)

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
        _assert_real_figure(fig)


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
        _assert_real_figure(fig)

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
        _assert_real_figure(fig)


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
        _assert_real_figure(fig)

    def test_with_cell_type_labels(self):
        W = np.random.randn(4, 4).astype(np.float32)
        labels = ["E", "PV", "SST", "VIP"]
        fig = vis.connectivity(W, cell_type_labels=labels)
        _assert_real_figure(fig)

    def test_no_weight_matrix_fallback(self):
        """Returns placeholder figure when weight matrix is not accessible."""
        class FakeModel:
            params = {}  # no 'W' key
        fig = vis.connectivity(FakeModel())
        _assert_real_figure(fig)

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
        _assert_real_figure(fig)

    def test_proxy_safe_title(self):
        cfg = _make_configuration()
        fig = vis.geometry3d(cfg)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "proxy" in titles.lower() or "declared" in titles.lower()

    def test_from_signals(self):
        signals = _make_minimal_signals()
        fig = vis.geometry3d(signals)
        _assert_real_figure(fig)

    def test_area_filter(self):
        cfg = _make_configuration()
        fig = vis.geometry3d(cfg, areas=["V1"])
        _assert_real_figure(fig)


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
        _assert_real_figure(fig)

    def test_area_filter(self):
        cfg = _make_configuration()
        fig = vis.multi_area_layout(cfg, areas=["V1"])
        _assert_real_figure(fig)


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
        _assert_real_figure(fig)

    def test_no_history_fallback(self):
        """Returns placeholder when history not extractable."""
        fig = vis.objective_report(None)
        _assert_real_figure(fig)

    def test_proxy_safe_suptitle(self):
        fig = vis.objective_report([1.0, 0.5, 0.3])
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "surrogate" in suptitle.lower() or "proxy" in suptitle.lower()

    def test_finite_only(self):
        """objective_report handles NaN gracefully."""
        history = [1.0, 0.8, float("nan"), 0.5, 0.4]
        fig = vis.objective_report(history)
        _assert_real_figure(fig)


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
