"""Shared helpers for jaxfne evidence figure generators.

Dependency-light manifest helpers delegate to ``jaxfne.vis.evidence_manifest``.
Matplotlib save/close delegates to ``jaxfne.vis.evidence_export``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure
from jaxfne.vis.evidence_manifest import (
    ensure_evidence_dirs,
    evidence_checklist_path,
    jaxfne_version,
    manifest_path_for,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

__all__ = [
    "_finite_array",
    "_finite_status",
    "_load_checklist",
    "_read_json_if_exists",
    "_two_neuron_config",
    "ensure_evidence_dirs",
    "evidence_checklist_path",
    "jaxfne_version",
    "manifest_path_for",
    "repo_root",
    "repo_sha",
    "save_figure_manifest",
    "save_matplotlib_figure",
    "sha256_file",
    "truth_gates",
    "utc_now_iso",
    "write_json_strict",
]


def _finite_array(data) -> bool:
    if data is None or not hasattr(data, "shape"):
        return True
    return bool(jnp.all(jnp.isfinite(jnp.asarray(data))))


def _finite_status(signals, readout: dict) -> bool:
    checks = [jnp.all(jnp.isfinite(signals.V_m)), jnp.all(jnp.isfinite(signals.spikes))]
    for key in ("CSD", "LFP", "csd_proxy", "lfp_proxy"):
        if key in readout and hasattr(readout[key], "shape"):
            checks.append(jnp.all(jnp.isfinite(readout[key])))
    return bool(all(checks))


def _load_checklist() -> dict:
    path = evidence_checklist_path()
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _two_neuron_config() -> jtfne.Configuration:
    return (
        jtfne.configuration()
        .network(
            name="two_neuron_ei",
            kind="coupled_neurons",
            n=2,
            cell_types={"E": 0.5, "I": 0.5},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(
            name="multimodal_two_neuron_ei",
            modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"],
        )
    )


def save_matplotlib_figure(fig, path, *, dpi: int = 150) -> str:
    """Save and close a matplotlib figure via ``jaxfne.vis`` (grammar rule 2 bridge)."""
    return save_matplotlib_evidence_figure(
        fig, path, dpi=dpi, bbox_inches="tight", facecolor="white", close=True,
    )
