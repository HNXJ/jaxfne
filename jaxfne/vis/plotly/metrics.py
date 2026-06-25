"""Objective/optimizer history visualization (Plotly)."""
from __future__ import annotations

from ._common import require_plotly


def plot_objective_history(tune_result, *, title: str = "Objective history"):
    """Plot optimizer loss/objective trajectory from a jaxfne TuneResult-like object.

    Accepts either a ``TuneResult`` (reads ``.history``/``.loss_history``) or a
    plain dict/list of per-step objective values.
    """
    require_plotly()
    import plotly.graph_objects as go
    import numpy as np

    if hasattr(tune_result, "history"):
        history = tune_result.history
    elif hasattr(tune_result, "loss_history"):
        history = tune_result.loss_history
    elif isinstance(tune_result, dict):
        history = tune_result.get("history") or tune_result.get("loss_history")
    else:
        history = tune_result

    history = np.asarray(history, dtype=float)
    fig = go.Figure(go.Scatter(x=np.arange(history.shape[0]), y=history, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="optimization step", yaxis_title="objective value")
    return fig
