"""Shared semantic visual grammar for publication figures (aligned with Figure 1)."""

from __future__ import annotations

# Typed computational dependency
SOLID = {"linewidth": 1.4, "linestyle": "-", "color": "#1A4A8A"}
DASHED = {"linewidth": 1.1, "linestyle": (0, (4, 3)), "color": "#666666"}

# Semantic status styles (Experiment A)
NATIVE = {"color": "#1A4A8A", "label": "native neural state", "alpha": 1.0}
CANONICAL_Q = {"color": "#0B6E4F", "label": "canonical relative source Q", "alpha": 1.0}
RELATIVE_PROXY = {"color": "#B85C00", "label": "relative proxy", "alpha": 1.0}
ANALYSIS_ONLY = {
    "color": "#888888",
    "label": "ANALYSIS_ONLY",
    "alpha": 0.55,
    "facecolor": "#F3F3F3",
    "edgecolor": "#888888",
    "linestyle": "dashed",
}
CALIBRATED = {
    "color": "#6A0DAD",
    "label": "calibrated physical (not claimed)",
    "alpha": 1.0,
    "facecolor": "#F5E6FF",
    "edgecolor": "#6A0DAD",
}

DEMONSTRATED_BOX = {"edgecolor": "#1A4A8A", "facecolor": "#E8F0FE", "linestyle": "solid"}
ANALYSIS_ONLY_BOX = {"edgecolor": "#888888", "facecolor": "#F3F3F3", "linestyle": "dashed"}
