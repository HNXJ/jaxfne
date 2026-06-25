"""Manuscript figure suite built on top of the generic Plotly primitives.

Figures 0-2 use a single (typically baseline) :class:`jaxfne.Signals` run.
Figures 3-4 accept either a single ``Signals`` (shown as one "baseline"
condition) or a ``{condition_label: Signals}`` dict once multi-condition
sweeps (cell-type suppression, FF/FB gain) have been generated.
"""
from __future__ import annotations

from . import figure0, figure1, figure2, figure3, figure4
from ..exporters import FigureBundle

__all__ = ["manuscript_figures", "figure0", "figure1", "figure2", "figure3", "figure4"]


def manuscript_figures(signals, model, *, perturbation_signals=None, hierarchy_signals=None) -> FigureBundle:
    """Build the full manuscript figure set as a :class:`FigureBundle`.

    ``signals``/``model`` — the baseline run, used for Figures 0-2.
    ``perturbation_signals`` — optional ``{condition: Signals}`` dict for
    Figure 3 (cell-type suppression); falls back to baseline-only if omitted.
    ``hierarchy_signals`` — optional ``{condition: Signals}`` dict for
    Figure 4 (FF/FB manipulation); falls back to baseline-only if omitted.
    """
    figures = {
        "figure0_background": figure0.build(signals, model),
        "figure1_model_validation": figure1.build(signals, model),
        "figure2_spectrolaminar_motif": figure2.build(signals, model),
        "figure3_celltype_perturbations": figure3.build(perturbation_signals or signals, model),
        "figure4_hierarchical_manipulations": figure4.build(hierarchy_signals or signals, model),
    }
    return FigureBundle(figures)
