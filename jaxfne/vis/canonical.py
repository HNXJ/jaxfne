"""Canonical, backend-dispatching visualization entry points.

One function per logical plot category (network geometry, raster, population
rate, membrane potentials, LFP/CSD proxy, PSD, spectrogram, band power,
laminar depth profile, connectivity, objective history). Each accepts
``backend="plotly"`` (default) or ``backend="matplotlib"`` and dispatches to
the corresponding implementation already present in :mod:`jaxfne.vis.plotly`
or the legacy matplotlib submodules (:mod:`jaxfne.vis.network3d`,
:mod:`jaxfne.vis.rasters`, :mod:`jaxfne.vis.traces`, :mod:`jaxfne.vis.spectra`,
:mod:`jaxfne.vis.fields`).

These are additive: the pre-existing matplotlib-only names (``vis.lfp``,
``vis.csd``, ``vis.raster``, ...) are untouched and keep returning matplotlib
figures unconditionally. Use the names here when you want one call site that
can produce either backend's figure.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

_BACKENDS = ("plotly", "matplotlib")


def _check_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")


def _apply_layout(
    fig: Any,
    backend: str,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Any:
    """Apply universal, defaulted layout overrides on top of a backend figure.

    Every override defaults to "leave the underlying implementation's choice
    alone" (``None``/``True``) so this never changes a figure unless the
    caller explicitly asks for it. Anti-overlap layout itself is handled by
    :func:`relative_layout` — this only sets labels/legend/colors/size.
    """
    if fig is None:
        return fig
    if backend == "plotly":
        layout_kwargs: dict = {"showlegend": legend}
        if title is not None:
            layout_kwargs["title"] = title
        if width is not None:
            layout_kwargs["width"] = width
        if height is not None:
            layout_kwargs["height"] = height
        fig.update_layout(**layout_kwargs)
        if xlabel is not None:
            fig.update_xaxes(title_text=xlabel)
        if ylabel is not None:
            fig.update_yaxes(title_text=ylabel)
        if colors:
            try:
                fig.update_traces(marker=dict(color=colors[0]) if len(colors) == 1 else None)
                for i, trace in enumerate(fig.data):
                    trace.update(marker_color=colors[i % len(colors)], line_color=colors[i % len(colors)])
            except Exception:
                pass
        return fig
    # matplotlib
    axes = list(getattr(fig, "axes", []) or [])
    if title is not None and axes:
        fig.suptitle(title)
    for ax in axes:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if colors:
            for i, line in enumerate(ax.get_lines()):
                line.set_color(colors[i % len(colors)])
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()
        elif not legend and ax.get_legend() is not None:
            ax.get_legend().remove()
    if width is not None or height is not None:
        w, h = fig.get_size_inches()
        fig.set_size_inches(width / 100.0 if width else w, height / 100.0 if height else h)
    return fig


def plot_network_3d(
    model,
    signals=None,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Network/column geometry, optionally with recurrent edges.

    ``backend="matplotlib"`` requires ``signals.metadata['neuron_metadata']``
    (passes ``signals`` to :func:`jaxfne.vis.network3d.geometry3d`); the
    ``model`` argument is only used by the plotly backend. ``title``/``xlabel``/
    ``ylabel``/``legend``/``colors``/``width``/``height`` are optional cosmetic
    overrides applied uniformly across both backends; leave at default (``None``/
    ``True``) to keep the underlying implementation's own choices.
    """
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.network import plot_network_3d as _impl
        fig = _impl(model, signals, **kwargs)
    else:
        from .network3d import geometry3d
        fig = geometry3d(signals if signals is not None else model, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_raster(
    signals,
    model=None,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "Neuron index",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Spike raster (time vs neuron id). See :func:`plot_network_3d` for the
    shared meaning of the cosmetic override kwargs.

    NAME COLLISION NOTE: ``jaxfne.tutorial_utils.plot_raster`` is a DIFFERENT
    function with a different signature (raw ``spike_times_list``/
    ``spike_neuron_ids_list``/``t`` arrays, not ``Signals``/``Model``) -- it's
    a thin forwarding wrapper to ``vis.plot_population_raster``, not this
    function. This is the root-exported ``jtfne.plot_raster``.
    """
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.raster import plot_raster as _impl
        fig = _impl(signals, model, **kwargs)
    else:
        from .rasters import raster
        fig = raster(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_population_rate(
    signals,
    model=None,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "Rate (Hz)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Binned population firing rate, optionally split by group."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.raster import plot_population_rates as _impl
        fig = _impl(signals, model, **kwargs)
    else:
        from .traces import rate
        fig = rate(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_membrane_potentials(
    signals,
    model=None,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "V_m proxy (mV)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Membrane potential (V_m, proxy) traces for a subset of neurons."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.raster import plot_membrane_potentials as _impl
        fig = _impl(signals, model, **kwargs)
    else:
        from .traces import vm
        fig = vm(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_lfp(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "Depth (stacked, proxy units)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """LFP proxy traces, stacked by depth."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.lfp import plot_lfp as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .traces import lfp
        fig = lfp(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_csd(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "Contact (depth)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """CSD proxy heatmap (depth x time)."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.csd import plot_csd as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .traces import csd
        fig = csd(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_psd(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Frequency (Hz)",
    ylabel: Optional[str] = "Power (proxy)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Power spectral density of a laminar proxy signal."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.spectra import plot_psd as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .spectra import psd
        fig = psd(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_spectrogram(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Time (ms)",
    ylabel: Optional[str] = "Frequency (Hz)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Time x frequency spectrogram of a laminar proxy signal."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.spectra import plot_spectrogram as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .spectra import spectrogram
        fig = spectrogram(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_band_power(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Depth (contact index)",
    ylabel: Optional[str] = "Relative band power",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Relative band power vs depth."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.spectra import plot_band_power as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .fields import bandpower
        fig = bandpower(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_depth_profile(
    signals,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Relative power",
    ylabel: Optional[str] = "Depth (contact index)",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Single-band relative power vs laminar depth (spectrolaminar-style readout)."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.spectra import plot_depth_profile as _impl
        fig = _impl(signals, **kwargs)
    else:
        from .fields import laminar_profile
        fig = laminar_profile(signals, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_connectivity(
    model,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Sending neuron",
    ylabel: Optional[str] = "Receiving neuron",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Recurrent connectivity (weight matrix) heatmap."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.connectivity import plot_connectivity as _impl
        fig = _impl(model, **kwargs)
    else:
        from .fields import connectivity_matrix
        fig = connectivity_matrix(model, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


def plot_objective_history(
    tune_result,
    *,
    backend: str = "plotly",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Iteration",
    ylabel: Optional[str] = "Objective value",
    legend: bool = True,
    colors: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs,
) -> Any:
    """Optimizer/objective-value history trajectory."""
    _check_backend(backend)
    if backend == "plotly":
        from .plotly.metrics import plot_objective_history as _impl
        fig = _impl(tune_result, **kwargs)
    else:
        from .fields import objective_report
        fig = objective_report(tune_result, **kwargs)
    return _apply_layout(fig, backend, title=title, xlabel=xlabel, ylabel=ylabel,
                          legend=legend, colors=colors, width=width, height=height)


__all__ = [
    "plot_network_3d",
    "plot_raster",
    "plot_population_rate",
    "plot_membrane_potentials",
    "plot_lfp",
    "plot_csd",
    "plot_psd",
    "plot_spectrogram",
    "plot_band_power",
    "plot_depth_profile",
    "plot_connectivity",
    "plot_objective_history",
]
