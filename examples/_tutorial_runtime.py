"""Runtime knobs the tutorial runner passes down to the example scripts.

``scripts/run_all_tutorials.py`` invokes each example as a separate process with
no arguments, so its ``--out-root`` / ``--smoke`` / ``--write-figures`` flags had
nowhere to go and were accepted and ignored -- configuration stored but never
consumed. These three environment variables are the channel; each example reads
them through the helpers below, so a flag that changes nothing is a test
failure rather than a silent no-op.

Defaults reproduce the historical behaviour exactly: ``outputs/<name>``, full
duration, figures written. Running an example by hand with no environment set
is unchanged.
"""

from __future__ import annotations

import os
import pathlib

ENV_OUT_ROOT = "JAXFNE_TUTORIAL_OUT_ROOT"
ENV_SMOKE = "JAXFNE_TUTORIAL_SMOKE"
ENV_WRITE_FIGURES = "JAXFNE_TUTORIAL_WRITE_FIGURES"

DEFAULT_OUT_ROOT = "outputs"

# Smoke mode divides the simulated duration by this and nothing else. A fixed
# integer divisor keeps the reduction deterministic and reproducible -- a
# fraction of wall-clock, or a duration tuned per example, would not be.
SMOKE_DIVISOR = 10
# Below this the run is too short to produce the spikes the output contract
# expects, so the reduction stops here rather than emptying the figures.
SMOKE_MIN_MS = 10.0


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean; use one of 1/0, true/false, yes/no, on/off"
    )


def out_dir(name: str) -> pathlib.Path:
    """``<out-root>/<name>``, created. ``out-root`` defaults to ``outputs``."""
    path = pathlib.Path(os.environ.get(ENV_OUT_ROOT) or DEFAULT_OUT_ROOT) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def smoke() -> bool:
    return _flag(ENV_SMOKE, False)


def write_figures() -> bool:
    return _flag(ENV_WRITE_FIGURES, True)


def duration_ms(full_ms: float) -> float:
    """The simulated duration, reduced deterministically under ``--smoke``."""
    if not smoke():
        return full_ms
    return max(SMOKE_MIN_MS, float(full_ms) / SMOKE_DIVISOR)


def manifest_model_status(manifest: dict) -> str:
    """Map canonical ``claim_level`` to the tutorial validation_report key.

    ``model_status`` is a tutorial-contract field only (runner checks
    ``validation_report.json``). The manifest schema exposes ``claim_level``;
    read that here rather than treating ``model_status`` as canonical.
    """
    if "model_status" in manifest and manifest["model_status"] is not None:
        return manifest["model_status"]
    return manifest.get("claim_level", "computational_scaffold")


def manifest_field_model_status(manifest: dict) -> str:
    """Map canonical ``field_claim_level`` to tutorial ``field_model_status``."""
    if "field_model_status" in manifest and manifest["field_model_status"] is not None:
        return manifest["field_model_status"]
    return manifest.get("field_claim_level", "proxy_readout")


def manifest_amplitude_status(manifest: dict) -> bool:
    """Map canonical ``physical_amplitude_calibrated`` to tutorial contract."""
    if "amplitude_status" in manifest and manifest["amplitude_status"] is not None:
        return bool(manifest["amplitude_status"])
    return bool(manifest.get("physical_amplitude_calibrated", False))
