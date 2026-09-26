"""Per-AT in-memory data bundles (0.5.5 ENGINE item 2b).

Human decision (2026-09-25): Atlas figures need each simulation's
trajectories, but the AT runners return scalar summaries and drop
``model``/``signals``. Bundles live in memory: each canonical runner in
``artifacts/atlas/at_manifest.py:REGISTRY`` accepts ``keep_bundle=True``
and returns its executed ``model``/``signals`` objects beside the
unchanged summary under ``"bundle"`` (``{arm_name: {"model", "signals"}}``,
no extra simulations, nothing large persisted). This module is the single
consumer entry point over that convention.

Bundle arms per runner: AT-01 ``reduced`` (the reduced jaxfne arm only;
the Jaxley arm is not a jaxfne Model); AT-02 ``delayed``/``zero``/
``absent``; AT-03 ``pair``; AT-04 one entry per geometry arm in
``AT04_ARMS``; AT-05 one entry per arm in ``AT05_ARMS``; AT-06
``column``; AT-07 ``hebbian``/``fixed``/``noisy``/``clamp`` plus the
``repro`` RNG-domain check and the ``budgeted`` decimation run; AT-04-R2
``baseline``/``perturbed``/``disabled_baseline``/``disabled_perturbed``;
AT-08 ``fixed``/``adapt_full``/``adapt_local`` (one shared ensemble
Model); AT-09 ``plastic_cross``/``plastic_member``/``frozen`` (one shared
ensemble Model); AT-10 ``main``.

Import rule (enforced by tests/test_atlas_firewall.py): atlas files
import only the top-level ``jaxfne`` package plus stdlib / numpy. This
file reads the sibling registry module, which is not a
``jaxfne.<submodule>`` import.
"""

from __future__ import annotations

from typing import Any

from artifacts.atlas.at_manifest import REGISTRY, resolve_runner

__all__ = ["BUNDLE_FIELDS", "bundle", "BUNDLE_ARM_NAMES"]

# What a figure consumer may read from each bundle arm. Shapes are
# neuron-major ``(T, N)``; ``field`` is None where the runner recorded no
# field probes. Every entry cites where the quantity lives.
BUNDLE_FIELDS: dict[str, str] = {
    "signals.spikes": (
        "(T, N) spike raster; jaxfne/_signals.py:141 (Signals.spikes; V_m is (T, N) at :140)"
    ),
    "signals.V_m": "(T, N) membrane trajectories; jaxfne/_signals.py:140",
    "signals.sources": (
        "(T, N) source representation, or None where the runner recorded "
        "no sources; jaxfne/_signals.py:142"
    ),
    "signals.field": (
        "FieldOutput proxy readout (e.g. .lfp_proxy), or None where the "
        "runner recorded no field probes (every canonical arm probed so far "
        "records LFP: AT-01/03/06/08/10, 2026-09-25); "
        "jaxfne/_signals.py:143"
    ),
    "hdp": (
        "per-arm {H_final, w_final, H_trace, w_trace} captured right after that "
        "arm simulated (AT-07, AT-04-R2, AT-08, AT-09 HDP arms); None for "
        "fixed-W arms and absent for non-HDP runners (H/W stay OMITTED there). "
        "Never read model.last_hdp_diagnostics() from a bundle: arms can share "
        "one Model (AT-08/09) and it reflects only the latest run."
    ),
}

# Arm names each canonical runner reports under keep_bundle=True.
BUNDLE_ARM_NAMES: dict[str, tuple[str, ...]] = {
    "AT-01": ("reduced",),
    "AT-02": ("delayed", "zero", "absent"),
    "AT-03": ("pair",),
    "AT-04": ("stacked", "swapped", "overlap"),
    "AT-05": ("n8_s11", "n8_s7", "n4_s7", "n2_s7"),
    "AT-06": ("column",),
    "AT-07": ("hebbian", "fixed", "noisy", "clamp", "repro", "budgeted"),
    "AT-04-R2": ("baseline", "perturbed", "disabled_baseline", "disabled_perturbed"),
    "AT-08": ("fixed", "adapt_full", "adapt_local"),
    "AT-09": ("plastic_cross", "plastic_member", "frozen"),
    "AT-10": ("main",),
}


def bundle(at_id: str) -> dict[str, Any]:
    """Return the in-memory data bundle for one canonical AT id.

    Runs ``resolve_runner(at_id)(keep_bundle=True)`` and returns its
    ``"bundle"`` entry: ``{arm_name: {"model", "signals"}}`` holding the
    executed objects (no extra simulations, nothing persisted).
    Raises KeyError for an unknown id (from the registry lookup).
    """
    if at_id not in REGISTRY:
        raise KeyError(f"unknown AT id {at_id!r}; want one of {sorted(REGISTRY)}")
    out = resolve_runner(at_id)(keep_bundle=True)
    return out["bundle"]
