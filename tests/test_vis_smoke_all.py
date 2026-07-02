"""Smoke-test coverage for every public plotting function across 18
previously-untested jaxfne.vis modules.

This is a SMOKE test, not a correctness/regression test: each function is
called with small synthetic data (a handful of neurons/timesteps) and we
assert only that (1) no exception is raised, and (2) the return value is a
non-empty, meaningful figure object of the type its docstring promises
(matplotlib Figure with content, plotly go.Figure with >=1 trace, or a
dict/tuple/None per that function's documented contract).

Modules covered:
- jaxfne/vis/exporters.py
- jaxfne/vis/hdp_diagnostics.py
- jaxfne/vis/plasticity_viz.py
- jaxfne/vis/plotly/{connectivity,csd,dashboard,lfp,metrics,raster,spectra}.py
- jaxfne/vis/plotly/manuscript/{__init__,figure0..4}.py
- jaxfne/vis/raster_arrays.py
- jaxfne/vis/report_plots.py
- jaxfne/vis/script_reports.py
- jaxfne/vis/tutorial_array_plots.py

Functions that genuinely cannot be smoke-tested cheaply are marked with
pytest.mark.skip and a concrete, documented reason (see SKIPPED_ENTRIES).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

N_NEURONS = 8
N_STEPS = 40
N_CONTACTS = 6
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Shared synthetic-fixture factories
# ---------------------------------------------------------------------------

def _t_ms(n=N_STEPS, dt=1.0):
    return np.arange(n) * dt


def _spike_matrix(n_steps=N_STEPS, n_neurons=N_NEURONS, p=0.1):
    return (RNG.random((n_steps, n_neurons)) < p).astype(float)


def _voltage_matrix(n_steps=N_STEPS, n_neurons=N_NEURONS):
    return RNG.normal(-65.0, 5.0, size=(n_steps, n_neurons))


def _weight_matrix(n=N_NEURONS):
    return RNG.normal(0.0, 1.0, size=(n, n))


def _lfp_matrix(n_steps=N_STEPS, n_contacts=N_CONTACTS):
    return RNG.normal(0.0, 1.0, size=(n_steps, n_contacts))


def _contact_depths(n_contacts=N_CONTACTS):
    return np.linspace(0.0, 1.0, n_contacts)


def _departures(keys):
    return {k: None for k in keys}


class _FakeEdges:
    """Duck-typed edge_list: pre/post/weight arrays."""

    def __init__(self, n_neurons=N_NEURONS, n_edges=12):
        self.pre = RNG.integers(0, n_neurons, size=n_edges)
        self.post = RNG.integers(0, n_neurons, size=n_edges)
        self.weight = RNG.normal(0.0, 1.0, size=n_edges)


class _FakeModel:
    """Duck-typed Model: only what jaxfne.vis.plotly.* actually reads
    (neuron_table() rows + params['edge_list'])."""

    def __init__(self, n_neurons=N_NEURONS):
        self.n_neurons = n_neurons
        cell_types = ["E", "PV", "SST", "VIP"]
        self._rows = [
            {
                "neuron_id": i,
                "area": "V1",
                "layer": f"L{(i % 4) + 1}",
                "cell_type": cell_types[i % len(cell_types)],
                "x": float(RNG.uniform(-1, 1)),
                "y": float(RNG.uniform(-1, 1)),
                "z": float(i) / n_neurons,
            }
            for i in range(n_neurons)
        ]
        self.params = {"edge_list": _FakeEdges(n_neurons)}

    def neuron_table(self):
        return self._rows


class _FakeField:
    """Duck-typed signals.field: lfp_proxy/csd_proxy/contact_depths."""

    def __init__(self, n_steps=N_STEPS, n_contacts=N_CONTACTS):
        self.lfp_proxy = _lfp_matrix(n_steps, n_contacts)
        self.csd_proxy = _lfp_matrix(n_steps, n_contacts)
        self.phi_e_proxy = _lfp_matrix(n_steps, n_contacts)
        self.source_proxy = _lfp_matrix(n_steps, n_contacts)
        self.contact_depths = _contact_depths(n_contacts) / 1.0e3  # stored in meters, *1e3 -> mm


class _FakeSignals:
    """Duck-typed Signals: spikes/V_m/time_ms/field, matching what
    jaxfne.vis.plotly._common helpers read."""

    def __init__(self, n_steps=N_STEPS, n_neurons=N_NEURONS, n_contacts=N_CONTACTS, with_field=True):
        self.spikes = _spike_matrix(n_steps, n_neurons)
        self.V_m = _voltage_matrix(n_steps, n_neurons)
        self.time_ms = _t_ms(n_steps)
        self.field = _FakeField(n_steps, n_contacts) if with_field else None


def _fake_model_signals():
    return _FakeModel(), _FakeSignals()


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_mpl_figure(fig):
    assert isinstance(fig, matplotlib.figure.Figure), f"expected matplotlib Figure, got {type(fig)}"
    assert len(fig.axes) >= 1, "figure has no axes"


def _assert_plotly_figure(fig):
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure), f"expected plotly Figure, got {type(fig)}"
    assert len(fig.data) >= 1, "plotly figure has no traces"


def _assert_figure_bundle(bundle):
    from jaxfne.vis.exporters import FigureBundle
    assert isinstance(bundle, FigureBundle)
    assert len(list(bundle.keys())) >= 1
    for _name, fig in bundle:
        assert fig is not None


# ---------------------------------------------------------------------------
# Per-function synthetic-input factories: name -> (import_path, attr, kwargs_factory, checker)
# ---------------------------------------------------------------------------

def _mpl_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


# --- exporters.py -----------------------------------------------------------

def _case_export_figure():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "fig")
        from jaxfne.vis.exporters import export_figure
        written = export_figure(fig, out_path, formats=("png",))
        assert "png" in written and os.path.exists(written["png"])
    plt.close(fig)


def _case_export_figures():
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    fig_mpl, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig_plotly = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    with tempfile.TemporaryDirectory() as d:
        from jaxfne.vis.exporters import export_figures
        manifest = export_figures({"a": fig_mpl, "b": fig_plotly}, d, formats=("png",))
        assert set(manifest.keys()) == {"a", "b"}
        assert os.path.exists(manifest["a"]["png"])
        assert os.path.exists(manifest["b"]["png"])
    plt.close(fig_mpl)


def _case_figure_bundle():
    import matplotlib.pyplot as plt
    from jaxfne.vis.exporters import FigureBundle
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    bundle = FigureBundle({"only": fig})
    _assert_figure_bundle(bundle)
    with tempfile.TemporaryDirectory() as d:
        manifest = bundle.export(d, formats=("png",))
        assert os.path.exists(manifest["only"]["png"])
    plt.close(fig)


# --- hdp_diagnostics.py -------------------------------------------------

def _case_synaptic_channel_decomposition_trace():
    from jaxfne.vis.hdp_diagnostics import synaptic_channel_decomposition_trace
    t = _t_ms()
    series = {"E->PV": RNG.normal(size=N_STEPS), "PV->E": RNG.normal(size=N_STEPS)}
    fig = synaptic_channel_decomposition_trace(
        t, series, _departures(series), window_start_ms=0.0, window_end_ms=float(t[-1]), bin_ms=1.0,
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_dH_component_trace_panels():
    from jaxfne.vis.hdp_diagnostics import dH_component_trace_panels
    t = _t_ms()
    series = lambda: RNG.normal(size=N_STEPS)
    fig = dH_component_trace_panels(
        t, series(), series(), series(), series(), series(), series(), series(),
        _departures(["H_mean", "var_H", "var_w", "dH_income", "dH_rate"]),
        window_start_ms=0.0, window_end_ms=float(t[-1]), bin_ms=1.0,
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_bifurcation_state_trace_panels():
    from jaxfne.vis.hdp_diagnostics import bifurcation_state_trace_panels
    t = _t_ms()
    rate_by_type = {"E": RNG.uniform(0, 10, N_STEPS), "PV": RNG.uniform(0, 10, N_STEPS)}
    fig = bifurcation_state_trace_panels(
        t, rate_by_type, RNG.uniform(0, 1, N_STEPS), RNG.normal(size=N_STEPS), RNG.normal(size=N_STEPS),
        RNG.normal(1.0, 0.1, N_STEPS), RNG.uniform(0, 1, N_STEPS),
        _departures(["kappa_synchrony", "weight_variance", "H_variance", "SST_rate"]),
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_gain_ratio_sweep_panels():
    from jaxfne.vis.hdp_diagnostics import gain_ratio_sweep_panels
    windows = ["short", "full"]
    K = np.array([0.0, 0.1, 1.0, 10.0])
    by_window = lambda: {w: RNG.normal(size=K.shape[0]) for w in windows}
    fig = gain_ratio_sweep_panels(
        {w: K for w in windows}, by_window(), by_window(), by_window(),
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_balance_sweep_by_network_size():
    from jaxfne.vis.hdp_diagnostics import balance_sweep_by_network_size
    K_grid = np.array([1.0, 10.0, 100.0])
    sizes = [10, 50]
    fig = balance_sweep_by_network_size(
        K_grid,
        {n: RNG.uniform(0.5, 1.5, K_grid.shape[0]) for n in sizes},
        {n: RNG.uniform(0.0, 0.5, K_grid.shape[0]) for n in sizes},
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_lfp_traces_stacked_by_depth():
    from jaxfne.vis.hdp_diagnostics import lfp_traces_stacked_by_depth
    fig = lfp_traces_stacked_by_depth(_t_ms(), _lfp_matrix(), _contact_depths())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_etude2_extended_panel_grid():
    from jaxfne.vis.hdp_diagnostics import etude2_extended_panel_grid
    layer_order = ["L1", "L2", "L4", "L6"]
    cell_types = ["E", "PV"]
    n_layers = len(layer_order)
    n_freq = 16
    n_contacts = N_CONTACTS
    freq_hz = np.linspace(1, 100, n_freq)
    n_spikes = 20
    fig = etude2_extended_panel_grid(
        layer_order=layer_order,
        cell_types_present=cell_types,
        ct_colors={"E": "#1f77b4", "PV": "#d62728"},
        layer_density_fracs={"E": np.full(n_layers, 0.6), "PV": np.full(n_layers, 0.4)},
        freq_hz=freq_hz,
        relative_power=RNG.uniform(0, 1, size=(n_freq, n_contacts)),
        pos_from_l4_mm=np.linspace(-1, 1, n_contacts),
        alpha_beta_profile=RNG.uniform(0, 1, n_contacts),
        gamma_profile=RNG.uniform(0, 1, n_contacts),
        time_win_ms=_t_ms(20),
        lfp_win=_lfp_matrix(20, n_contacts),
        contact_depths=_contact_depths(n_contacts),
        spike_times_ms=RNG.uniform(0, 20, n_spikes),
        spike_neuron_rank=RNG.integers(0, N_NEURONS, n_spikes),
        n_neurons_zsorted=N_NEURONS,
        rate_table_hz=RNG.uniform(0, 20, size=(2, n_layers)),
        k_hdp_used=1.0,
        last_window_ms=20.0,
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_voltage_trace_with_spikes_by_celltype():
    from jaxfne.vis.hdp_diagnostics import voltage_trace_with_spikes_by_celltype
    cell_types = ["E", "PV"]
    t = _t_ms()
    voltages = {ct: RNG.normal(-65, 5, N_STEPS) for ct in cell_types}
    spikes = {ct: (RNG.random(N_STEPS) < 0.1).astype(float) for ct in cell_types}
    params = {ct: {"drive": 1.0, "noise": 0.1, "spike_count": 5} for ct in cell_types}
    fig = voltage_trace_with_spikes_by_celltype(t, voltages, spikes, params)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


# --- plasticity_viz.py --------------------------------------------------

def _case_plot_stdp_adaptation_suite():
    from jaxfne.vis.plasticity_viz import plot_stdp_adaptation_suite
    n_neurons = N_NEURONS
    n_steps_down = 30
    trajectories = {
        "vm": _voltage_matrix(n_steps_down, n_neurons),
        "spk": _spike_matrix(n_steps_down, n_neurons),
    }
    W_before = _weight_matrix(n_neurons)
    W_after = _weight_matrix(n_neurons)
    stimulus = RNG.normal(size=n_steps_down * 10)
    with tempfile.TemporaryDirectory() as d:
        result = plot_stdp_adaptation_suite(trajectories, W_before, W_after, stimulus, d, prefix="t_")
        assert result is None  # documented: writes PNGs, returns None
        pngs = [f for f in os.listdir(d) if f.endswith(".png")]
        assert len(pngs) == 4, f"expected 4 PNGs, found {pngs}"


# --- plotly/connectivity.py, csd.py, dashboard.py, lfp.py, metrics.py, raster.py, spectra.py ---

def _case_plot_connectivity():
    from jaxfne.vis.plotly.connectivity import plot_connectivity
    model = _FakeModel()
    fig = plot_connectivity(model, max_neurons=100)
    _assert_plotly_figure(fig)


def _case_plot_csd():
    from jaxfne.vis.plotly.csd import plot_csd
    _, signals = _fake_model_signals()
    fig = plot_csd(signals)
    _assert_plotly_figure(fig)


def _case_create_dashboard():
    from jaxfne.vis.plotly.dashboard import create_dashboard
    model, signals = _fake_model_signals()
    fig = create_dashboard(signals, model)
    _assert_plotly_figure(fig)


def _case_plot_lfp():
    from jaxfne.vis.plotly.lfp import plot_lfp
    _, signals = _fake_model_signals()
    fig = plot_lfp(signals)
    _assert_plotly_figure(fig)


def _case_plot_objective_history():
    from jaxfne.vis.plotly.metrics import plot_objective_history
    fig = plot_objective_history({"history": list(RNG.uniform(0, 1, 20))})
    _assert_plotly_figure(fig)


def _case_plot_raster():
    from jaxfne.vis.plotly.raster import plot_raster
    model, signals = _fake_model_signals()
    fig = plot_raster(signals, model)
    _assert_plotly_figure(fig)


def _case_plot_population_rates():
    from jaxfne.vis.plotly.raster import plot_population_rates
    model, signals = _fake_model_signals()
    fig = plot_population_rates(signals, model)
    _assert_plotly_figure(fig)


def _case_plot_membrane_potentials():
    from jaxfne.vis.plotly.raster import plot_membrane_potentials
    model, signals = _fake_model_signals()
    fig = plot_membrane_potentials(signals, model)
    _assert_plotly_figure(fig)


def _case_plot_psd():
    from jaxfne.vis.plotly.spectra import plot_psd
    _, signals = _fake_model_signals()
    fig = plot_psd(signals)
    _assert_plotly_figure(fig)


def _case_plot_spectrogram():
    from jaxfne.vis.plotly.spectra import plot_spectrogram
    _, signals = _fake_model_signals()
    fig = plot_spectrogram(signals, contact=0)
    _assert_plotly_figure(fig)


def _case_plot_band_power():
    from jaxfne.vis.plotly.spectra import plot_band_power
    _, signals = _fake_model_signals()
    fig = plot_band_power(signals)
    _assert_plotly_figure(fig)


def _case_plot_depth_profile():
    from jaxfne.vis.plotly.spectra import plot_depth_profile
    _, signals = _fake_model_signals()
    fig = plot_depth_profile(signals)
    _assert_plotly_figure(fig)


# --- plotly/manuscript ----------------------------------------------------

def _case_manuscript_figure0():
    from jaxfne.vis.plotly.manuscript import figure0
    model, signals = _fake_model_signals()
    fig = figure0.build(signals, model)
    _assert_plotly_figure(fig)


def _case_manuscript_figure1():
    from jaxfne.vis.plotly.manuscript import figure1
    model, signals = _fake_model_signals()
    fig = figure1.build(signals, model)
    _assert_plotly_figure(fig)


def _case_manuscript_figure2():
    from jaxfne.vis.plotly.manuscript import figure2
    model, signals = _fake_model_signals()
    fig = figure2.build(signals, model)
    _assert_plotly_figure(fig)


def _case_manuscript_figure3():
    from jaxfne.vis.plotly.manuscript import figure3
    model, signals = _fake_model_signals()
    fig = figure3.build(signals, model)
    _assert_plotly_figure(fig)
    # also exercise the multi-condition dict path
    _, signals2 = _fake_model_signals()
    fig2 = figure3.build({"baseline": signals, "PV_suppressed": signals2}, model)
    _assert_plotly_figure(fig2)


def _case_manuscript_figure4():
    from jaxfne.vis.plotly.manuscript import figure4
    model, signals = _fake_model_signals()
    fig = figure4.build(signals, model)
    _assert_plotly_figure(fig)
    _, signals2 = _fake_model_signals()
    fig2 = figure4.build({"baseline": signals, "FF_boosted": signals2}, model)
    _assert_plotly_figure(fig2)


def _case_manuscript_figures():
    from jaxfne.vis.plotly.manuscript import manuscript_figures
    model, signals = _fake_model_signals()
    bundle = manuscript_figures(signals, model)
    _assert_figure_bundle(bundle)
    assert set(bundle.keys()) == {
        "figure0_background", "figure1_model_validation", "figure2_spectrolaminar_motif",
        "figure3_celltype_perturbations", "figure4_hierarchical_manipulations",
    }


# --- raster_arrays.py --------------------------------------------------

def _case_raster_from_arrays():
    from jaxfne.vis.raster_arrays import raster_from_arrays
    n_spikes = 30
    spike_times = RNG.uniform(0, 100, n_spikes)
    spike_ids = RNG.integers(0, N_NEURONS, n_spikes)
    fig = raster_from_arrays(spike_times, spike_ids, _t_ms(100))
    _assert_mpl_figure(fig)
    _mpl_close(fig)


# --- report_plots.py -----------------------------------------------------

def _case_dual_raster_comparison():
    from jaxfne.vis.report_plots import dual_raster_comparison
    fig = dual_raster_comparison(_spike_matrix(), _spike_matrix(), duration_s=1.0)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_optimization_progress_line():
    from jaxfne.vis.report_plots import optimization_progress_line
    fig = optimization_progress_line(list(range(10)), list(RNG.uniform(0, 1, 10)))
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_spike_grid_heatmap():
    from jaxfne.vis.report_plots import spike_grid_heatmap
    fig = spike_grid_heatmap(_spike_matrix(), duration_s=1.0)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_signed_heatmap_with_colorbar():
    from jaxfne.vis.report_plots import signed_heatmap_with_colorbar
    fig = signed_heatmap_with_colorbar(_lfp_matrix())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_grouped_bar_comparison():
    from jaxfne.vis.report_plots import grouped_bar_comparison
    categories = ["A", "B", "C"]
    fig = grouped_bar_comparison(categories, {"g1": [1, 2, 3], "g2": [3, 2, 1]})
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_agsdr_rate_tuning_panel_grid():
    from jaxfne.vis.report_plots import agsdr_rate_tuning_panel_grid
    gains = list(np.linspace(0.1, 2.0, 5))
    fig = agsdr_rate_tuning_panel_grid(
        gains=gains,
        scores=list(RNG.uniform(0, 1, 5)),
        mean_rates=list(RNG.uniform(1, 10, 5)),
        best_score=0.9,
        best_gain=1.0,
        target_rate_hz=5.0,
        rate_tolerance_hz=1.0,
        baseline_vals=[3.0, 1.0],
        tuned_vals=[5.0, 2.0],
        min_neuron_rate_gate_hz=1.0,
        tuned_rates_histogram=RNG.uniform(0, 10, N_NEURONS),
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_open_pdf_pdf_text_pdf_figure_pages():
    """Exercise open_pdf_pages + pdf_text_page + pdf_figure_page together
    (they're a cohesive triad: writer handle -> pages -> save)."""
    from jaxfne.vis.report_plots import open_pdf_pages, pdf_text_page, pdf_figure_page
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as d:
        # a source PNG for pdf_figure_page to embed
        img_path = os.path.join(d, "src.png")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(img_path)
        plt.close(fig)

        pdf_path = os.path.join(d, "report.pdf")
        pdf = open_pdf_pages(pdf_path)
        try:
            pdf_text_page(pdf, ["## Section", "line one", "line two"], title="Report")
            pdf_figure_page(pdf, img_path, "a caption")
        finally:
            pdf.close()
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0


def _case_celltype_sweep_heatmap_grid():
    from jaxfne.vis.report_plots import celltype_sweep_heatmap_grid
    n_noise, n_drive = 3, 4
    grids = {ct: RNG.integers(0, 20, size=(n_noise, n_drive)) for ct in ("E", "PV", "SST", "VIP")}
    fig = celltype_sweep_heatmap_grid(
        grids, title="sweep", ylabel="noise", xlabel="drive",
        xticklabels=list(np.linspace(0, 1, n_drive)), yticklabels=list(np.linspace(0, 1, n_noise)),
    )
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_gain_matrix_heatmap():
    from jaxfne.vis.report_plots import gain_matrix_heatmap
    n_areas = 4
    fig = gain_matrix_heatmap(RNG.uniform(0, 3, size=(n_areas, n_areas)), [f"area{i}" for i in range(n_areas)])
    _assert_mpl_figure(fig)
    _mpl_close(fig)


# --- script_reports.py --------------------------------------------------

def _case_column_network_3d_scatter():
    from jaxfne.vis.script_reports import column_network_3d_scatter
    n = N_NEURONS
    fig = column_network_3d_scatter(
        RNG.uniform(-1, 1, n), RNG.uniform(-1, 1, n), RNG.uniform(0, 1, n),
        RNG.uniform(0, 20, n), [f"n{i}" for i in range(n)],
    )
    _assert_plotly_figure(fig)


def _case_column_readout_panels():
    from jaxfne.vis.script_reports import column_readout_panels
    t = _t_ms()
    layer_names = ["L1", "L4"]
    fig = column_readout_panels(
        t, RNG.normal(size=N_STEPS), t, RNG.uniform(0, 20, N_STEPS), RNG.normal(size=N_STEPS),
        layer_names, {ln: RNG.normal(-65, 5, N_STEPS) for ln in layer_names},
        {"L1": "#1f77b4", "L4": "#d62728"},
    )
    _assert_plotly_figure(fig)


def _case_rate_histogram_by_celltype():
    from jaxfne.vis.script_reports import rate_histogram_by_celltype
    fig = rate_histogram_by_celltype({"E": 5.0, "PV": 12.0}, band=(4.0, 8.0))
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_sweep_rate_lines():
    from jaxfne.vis.script_reports import sweep_rate_lines
    xs = [0.1, 1.0, 10.0]
    rate_by_type_per_point = [{"E": 5.0, "PV": 10.0} for _ in xs]
    fig = sweep_rate_lines(xs, rate_by_type_per_point, band=(4.0, 8.0), xlabel="K_HDP")
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_spectrolaminar_motif_heatmap():
    from jaxfne.vis.script_reports import spectrolaminar_motif_heatmap
    fig = spectrolaminar_motif_heatmap(RNG.uniform(0, 1, size=(N_CONTACTS, 16)), _contact_depths())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_spike_triggered_waveforms_plot():
    from jaxfne.vis.script_reports import spike_triggered_waveforms_plot
    waveforms = {"E": RNG.normal(-65, 2, 20), "PV": RNG.normal(-65, 2, 20)}
    widths = {"E": 1.5, "PV": 1.2}
    fig = spike_triggered_waveforms_plot(waveforms, dt_ms=0.5, widths_ms=widths)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_hdp_state_trajectories():
    from jaxfne.vis.script_reports import hdp_state_trajectories
    per_type = {
        "E": {"t_ms": _t_ms(), "mean_H_t": RNG.normal(1.0, 0.1, N_STEPS)},
        "PV": {"t_ms": _t_ms(), "mean_H_t": RNG.normal(1.0, 0.1, N_STEPS)},
    }
    fig = hdp_state_trajectories(per_type)
    _assert_mpl_figure(fig)
    _mpl_close(fig)
    fig2 = hdp_state_trajectories(per_type, mode="pressure")
    _assert_mpl_figure(fig2)
    _mpl_close(fig2)


def _case_rate_vs_pressure_scatter():
    from jaxfne.vis.script_reports import rate_vs_pressure_scatter
    per_type = {
        "E": {"pressure_binned": RNG.normal(size=10), "rate_binned_hz": RNG.uniform(0, 10, 10)},
        "PV": {"pressure_binned": RNG.normal(size=10), "rate_binned_hz": RNG.uniform(0, 10, 10)},
    }
    fig = rate_vs_pressure_scatter(per_type)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_hdp_H_occupancy_histogram():
    from jaxfne.vis.script_reports import hdp_H_occupancy_histogram
    fig = hdp_H_occupancy_histogram({"E": RNG.normal(1.0, 0.1, 100), "PV": RNG.normal(1.0, 0.1, 100)})
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_hdp_stability_map():
    from jaxfne.vis.script_reports import hdp_stability_map
    k_grid = [0.1, 1.0, 10.0]
    tau0_grid = [10.0, 20.0]
    shape = (len(k_grid), len(tau0_grid))
    grids = {
        "mean_rate_hz": RNG.uniform(0, 20, size=shape),
        "weight_saturation_fraction": RNG.uniform(0, 1, size=shape),
        "max_var_H": RNG.uniform(0, 1, size=shape),
    }
    fig = hdp_stability_map(grids, k_grid, tau0_grid)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_spike_raster():
    from jaxfne.vis.script_reports import tutorial_spike_raster
    fig = tutorial_spike_raster(_spike_matrix())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_voltage_traces():
    from jaxfne.vis.script_reports import tutorial_voltage_traces
    fig = tutorial_voltage_traces(_voltage_matrix(), n_display=4)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_matrix_heatmap():
    from jaxfne.vis.script_reports import tutorial_matrix_heatmap
    fig = tutorial_matrix_heatmap(_lfp_matrix())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_lfp_proxy_trace():
    from jaxfne.vis.script_reports import tutorial_lfp_proxy_trace
    fig = tutorial_lfp_proxy_trace(RNG.normal(size=N_STEPS))
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_conservation_bar():
    from jaxfne.vis.script_reports import tutorial_conservation_bar
    fig = tutorial_conservation_bar({"metric_a": 1.2, "metric_b": 3.4})
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_contact_depths_barh():
    from jaxfne.vis.script_reports import tutorial_contact_depths_barh
    fig = tutorial_contact_depths_barh(_contact_depths())
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_status_text_panel():
    from jaxfne.vis.script_reports import tutorial_status_text_panel
    fig = tutorial_status_text_panel("Gate A: PASS\nGate B: PASS\n## Section\nline")
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_tutorial_spectral_summary():
    from jaxfne.vis.script_reports import tutorial_spectral_summary
    freqs = np.linspace(0.1, 50, 30)
    fig = tutorial_spectral_summary(freqs, RNG.uniform(0, 1, 30))
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_spike_raster_plotly():
    from jaxfne.vis.script_reports import spike_raster_plotly
    n_spikes = 20
    fig = spike_raster_plotly(list(RNG.uniform(0, 100, n_spikes)), list(RNG.integers(0, N_NEURONS, n_spikes)))
    _assert_plotly_figure(fig)


def _case_spectrolaminar_profile_bars_plotly():
    from jaxfne.vis.script_reports import spectrolaminar_profile_bars_plotly
    layers = ["L1", "L4", "L6"]
    fig = spectrolaminar_profile_bars_plotly(layers, [0.3, 0.5, 0.2], [0.1, 0.4, 0.5])
    _assert_plotly_figure(fig)


def _case_tutorial_spike_raster_vlines_plotly():
    from jaxfne.vis.script_reports import tutorial_spike_raster_vlines_plotly
    spike_times = [[1.0, 5.0], [2.0], [3.0, 4.0]]
    fig = tutorial_spike_raster_vlines_plotly(spike_times)
    # This function only ever calls add_vline (shapes/annotations), never
    # add_trace -- so fig.data is legitimately empty by design. Assert on
    # layout content instead of trace count.
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) >= 1, "expected at least one vline shape"


def _case_tutorial_voltage_trace_plotly():
    from jaxfne.vis.script_reports import tutorial_voltage_trace_plotly
    t = list(_t_ms())
    fig = tutorial_voltage_trace_plotly(t, {0: list(RNG.normal(-65, 5, N_STEPS)), 1: list(RNG.normal(-65, 5, N_STEPS))})
    _assert_plotly_figure(fig)


def _case_tutorial_firing_rate_plotly():
    from jaxfne.vis.script_reports import tutorial_firing_rate_plotly
    t = list(_t_ms())
    fig = tutorial_firing_rate_plotly(t, list(RNG.uniform(0, 20, N_STEPS)))
    _assert_plotly_figure(fig)


# --- tutorial_array_plots.py --------------------------------------------

def _case_plot_population_raster():
    from jaxfne.vis.tutorial_array_plots import plot_population_raster
    n_spikes = 15
    times_list = [RNG.uniform(0, 100, n_spikes)]
    ids_list = [RNG.integers(0, N_NEURONS, n_spikes)]
    fig = plot_population_raster(times_list, ids_list, _t_ms(100), show=False)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_plot_population_rate_array():
    from jaxfne.vis.tutorial_array_plots import plot_population_rate_array
    t = np.linspace(0, 500, 500)
    spikes = (RNG.random(500) < 0.05).astype(float)
    fig, rates = plot_population_rate_array(t, spikes, bin_ms=25.0, dt_ms=1.0, show=False)
    _assert_mpl_figure(fig)
    assert len(rates) > 0
    _mpl_close(fig)


def _case_plot_voltage_samples_array():
    from jaxfne.vis.tutorial_array_plots import plot_voltage_samples_array
    fig = plot_voltage_samples_array(_t_ms(), _voltage_matrix(), max_neurons=5, stacked=True, show=False)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_plot_connectivity_matrix_array():
    from jaxfne.vis.tutorial_array_plots import plot_connectivity_matrix_array
    fig = plot_connectivity_matrix_array(_weight_matrix(), show=False)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


def _case_plot_laminar_readout_array():
    from jaxfne.vis.tutorial_array_plots import plot_laminar_readout_array
    t = _t_ms()
    lfp = _lfp_matrix()
    csd = _lfp_matrix()
    fig = plot_laminar_readout_array(t, lfp, csd_proxy=csd, show=False)
    _assert_mpl_figure(fig)
    _mpl_close(fig)
    fig2 = plot_laminar_readout_array(t, lfp, csd_proxy=None, show=False)
    _assert_mpl_figure(fig2)
    _mpl_close(fig2)


def _case_plot_spectrolaminar_power_array():
    from jaxfne.vis.tutorial_array_plots import plot_spectrolaminar_power_array
    t = np.linspace(0, 100, 200)
    signal = RNG.normal(size=(200, 3))
    fig = plot_spectrolaminar_power_array(t, signal, n_freqs=64, show=False)
    _assert_mpl_figure(fig)
    _mpl_close(fig)


# ---------------------------------------------------------------------------
# Parametrize table: (id, case_fn) — grouped roughly by source module/order
# above so failures map back to file location easily.
# ---------------------------------------------------------------------------

CASES = [
    # exporters.py
    ("exporters.export_figure", _case_export_figure),
    ("exporters.export_figures", _case_export_figures),
    ("exporters.FigureBundle", _case_figure_bundle),
    # hdp_diagnostics.py (8 functions)
    ("hdp_diagnostics.synaptic_channel_decomposition_trace", _case_synaptic_channel_decomposition_trace),
    ("hdp_diagnostics.dH_component_trace_panels", _case_dH_component_trace_panels),
    ("hdp_diagnostics.bifurcation_state_trace_panels", _case_bifurcation_state_trace_panels),
    ("hdp_diagnostics.gain_ratio_sweep_panels", _case_gain_ratio_sweep_panels),
    ("hdp_diagnostics.balance_sweep_by_network_size", _case_balance_sweep_by_network_size),
    ("hdp_diagnostics.lfp_traces_stacked_by_depth", _case_lfp_traces_stacked_by_depth),
    ("hdp_diagnostics.etude2_extended_panel_grid", _case_etude2_extended_panel_grid),
    ("hdp_diagnostics.voltage_trace_with_spikes_by_celltype", _case_voltage_trace_with_spikes_by_celltype),
    # plasticity_viz.py
    ("plasticity_viz.plot_stdp_adaptation_suite", _case_plot_stdp_adaptation_suite),
    # plotly/connectivity.py, csd.py, dashboard.py, lfp.py, metrics.py, raster.py, spectra.py
    ("plotly.connectivity.plot_connectivity", _case_plot_connectivity),
    ("plotly.csd.plot_csd", _case_plot_csd),
    ("plotly.dashboard.create_dashboard", _case_create_dashboard),
    ("plotly.lfp.plot_lfp", _case_plot_lfp),
    ("plotly.metrics.plot_objective_history", _case_plot_objective_history),
    ("plotly.raster.plot_raster", _case_plot_raster),
    ("plotly.raster.plot_population_rates", _case_plot_population_rates),
    ("plotly.raster.plot_membrane_potentials", _case_plot_membrane_potentials),
    ("plotly.spectra.plot_psd", _case_plot_psd),
    ("plotly.spectra.plot_spectrogram", _case_plot_spectrogram),
    ("plotly.spectra.plot_band_power", _case_plot_band_power),
    ("plotly.spectra.plot_depth_profile", _case_plot_depth_profile),
    # plotly/manuscript
    ("plotly.manuscript.figure0.build", _case_manuscript_figure0),
    ("plotly.manuscript.figure1.build", _case_manuscript_figure1),
    ("plotly.manuscript.figure2.build", _case_manuscript_figure2),
    ("plotly.manuscript.figure3.build", _case_manuscript_figure3),
    ("plotly.manuscript.figure4.build", _case_manuscript_figure4),
    ("plotly.manuscript.manuscript_figures", _case_manuscript_figures),
    # raster_arrays.py
    ("raster_arrays.raster_from_arrays", _case_raster_from_arrays),
    # report_plots.py (11 functions)
    ("report_plots.dual_raster_comparison", _case_dual_raster_comparison),
    ("report_plots.optimization_progress_line", _case_optimization_progress_line),
    ("report_plots.spike_grid_heatmap", _case_spike_grid_heatmap),
    ("report_plots.signed_heatmap_with_colorbar", _case_signed_heatmap_with_colorbar),
    ("report_plots.grouped_bar_comparison", _case_grouped_bar_comparison),
    ("report_plots.agsdr_rate_tuning_panel_grid", _case_agsdr_rate_tuning_panel_grid),
    ("report_plots.open_pdf_pages+pdf_text_page+pdf_figure_page", _case_open_pdf_pdf_text_pdf_figure_pages),
    ("report_plots.celltype_sweep_heatmap_grid", _case_celltype_sweep_heatmap_grid),
    ("report_plots.gain_matrix_heatmap", _case_gain_matrix_heatmap),
    # script_reports.py (22 functions)
    ("script_reports.column_network_3d_scatter", _case_column_network_3d_scatter),
    ("script_reports.column_readout_panels", _case_column_readout_panels),
    ("script_reports.rate_histogram_by_celltype", _case_rate_histogram_by_celltype),
    ("script_reports.sweep_rate_lines", _case_sweep_rate_lines),
    ("script_reports.spectrolaminar_motif_heatmap", _case_spectrolaminar_motif_heatmap),
    ("script_reports.spike_triggered_waveforms_plot", _case_spike_triggered_waveforms_plot),
    ("script_reports.hdp_state_trajectories", _case_hdp_state_trajectories),
    ("script_reports.rate_vs_pressure_scatter", _case_rate_vs_pressure_scatter),
    ("script_reports.hdp_H_occupancy_histogram", _case_hdp_H_occupancy_histogram),
    ("script_reports.hdp_stability_map", _case_hdp_stability_map),
    ("script_reports.tutorial_spike_raster", _case_tutorial_spike_raster),
    ("script_reports.tutorial_voltage_traces", _case_tutorial_voltage_traces),
    ("script_reports.tutorial_matrix_heatmap", _case_tutorial_matrix_heatmap),
    ("script_reports.tutorial_lfp_proxy_trace", _case_tutorial_lfp_proxy_trace),
    ("script_reports.tutorial_conservation_bar", _case_tutorial_conservation_bar),
    ("script_reports.tutorial_contact_depths_barh", _case_tutorial_contact_depths_barh),
    ("script_reports.tutorial_status_text_panel", _case_tutorial_status_text_panel),
    ("script_reports.tutorial_spectral_summary", _case_tutorial_spectral_summary),
    ("script_reports.spike_raster_plotly", _case_spike_raster_plotly),
    ("script_reports.spectrolaminar_profile_bars_plotly", _case_spectrolaminar_profile_bars_plotly),
    ("script_reports.tutorial_spike_raster_vlines_plotly", _case_tutorial_spike_raster_vlines_plotly),
    ("script_reports.tutorial_voltage_trace_plotly", _case_tutorial_voltage_trace_plotly),
    ("script_reports.tutorial_firing_rate_plotly", _case_tutorial_firing_rate_plotly),
    # tutorial_array_plots.py (6 functions)
    ("tutorial_array_plots.plot_population_raster", _case_plot_population_raster),
    ("tutorial_array_plots.plot_population_rate_array", _case_plot_population_rate_array),
    ("tutorial_array_plots.plot_voltage_samples_array", _case_plot_voltage_samples_array),
    ("tutorial_array_plots.plot_connectivity_matrix_array", _case_plot_connectivity_matrix_array),
    ("tutorial_array_plots.plot_laminar_readout_array", _case_plot_laminar_readout_array),
    ("tutorial_array_plots.plot_spectrolaminar_power_array", _case_plot_spectrolaminar_power_array),
]

CASE_IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("name,case_fn", CASES, ids=CASE_IDS)
def test_vis_smoke(name, case_fn):
    """Call each public plotting function with small synthetic data; assert
    no exception and a non-empty, correctly-typed figure result."""
    case_fn()


# ---------------------------------------------------------------------------
# Documented skips: functions that cannot be cheaply smoke-tested here.
#
# NOTE: every function named in the task spec is covered above except the
# ones listed below. None were silently dropped.
# ---------------------------------------------------------------------------

def test_no_undocumented_skips_placeholder():
    """Placeholder acknowledging there are currently zero hard-skip entries.

    Every one of the 18 target files' public functions turned out to be
    cheaply smoke-testable with small synthetic/duck-typed fixtures (no
    live file dialog, no real PDF-viewer backend, no GPU-only path was
    required) — see CASES above for the full enumerated list. This test
    exists so a future contributor who adds a genuinely-unskippable
    function has an obvious place to add a `pytest.mark.skip(reason=...)`
    entry rather than silently omitting coverage.
    """
    assert len(CASES) == 69
