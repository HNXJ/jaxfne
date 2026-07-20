"""Model.run_receipt()/compute_readout()/probe()/record() -- receipt
generation and post-hoc readout extraction from ``Signals``.

Split out of ``jaxfne/_model.py`` (Phase 2 defragmentation, 2026-07-20, part
of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import jax.numpy as jnp

from .fields import probe_laminar_modes
from .io import config_hash, json_safe
from ._signals import ReadoutResult, ReadoutSpec, RunReceipt, Signals
from ._model import _JAXFNE_VERSION, _KNOWN_READOUT_METRICS, _RECEIPT_SCHEMA_VERSION


def run_receipt(self, signals: Signals, *, tags: Optional[dict[str, Any]] = None) -> RunReceipt:
    """Build a RunReceipt capturing this run for audit and reproducibility.

    **Canonical v0.1 workflow method.**  Prefer this over :meth:`manifest`
    for recording completed simulation runs.

    Args:
        signals: Signals returned by self.simulate().
        tags: Optional user-supplied key-value metadata (condition, paper, etc.).

    Returns:
        RunReceipt with frozen truth gates and deterministic receipt_id.

    Note:
        ``receipt_id`` is deterministic for the same
        ``(config_hash, seed, _JAXFNE_VERSION)`` triple.  Upgrading the
        package version changes the ID even when config and seed are
        identical, because the computational kernel may have changed.
        IDs are audit identifiers; they are not empirical claims.
    """
    from .io import sha256_text

    cfg_h = config_hash(self.cfg)
    # Seed is stored inside the runtime sub-dict (via RuntimeConfig.runtime_report)
    seed = int(signals.metadata.get("runtime", {}).get("seed", signals.metadata.get("seed", 0)))

    sim_meta = signals.metadata
    sim_summary: dict[str, Any] = {
        "duration_ms": sim_meta.get("duration_ms"),
        "dt_ms": sim_meta.get("dt_ms"),
        "seed": seed,
        "n_steps": int(signals.time_ms.shape[0]),
        "record_sources": sim_meta.get("record_sources"),
        "record_fields": sim_meta.get("record_fields"),
    }

    # Deterministic receipt_id based on config, version, simulation, and key runtime metadata
    receipt_payload = {
        "config_hash": cfg_h,
        "jaxfne_version": _JAXFNE_VERSION,
        "simulation": sim_summary,
        "runtime": sim_meta.get("runtime"),
        "condition_name": sim_meta.get("condition_name"),
        "stimulus_schedule": sim_meta.get("stimulus_schedule"),
        "recurrent_backend": sim_meta.get("recurrent_backend"),
        "synaptic_kernel": sim_meta.get("synaptic_kernel"),
        "source_model": sim_meta.get("source_model"),
    }
    receipt_id = sha256_text(
        json.dumps(json_safe(receipt_payload), sort_keys=True, allow_nan=False)
    )[:16]

    truth: dict[str, Any] = {
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "field_solver_status": "linear_solver",
        "field_claim_level": "proxy_readout",
        "physical_amplitude_calibrated": False,
        "empirical_validation_status": "not_empirically_validated",
        "mechanism_claim_status": "not_claimed",
    }

    claim_labels: dict[str, Any] = {
        "receipt_status": _RECEIPT_SCHEMA_VERSION,
        "empirical_validation_status": "not_empirically_validated",
        "mechanism_claim_status": "not_claimed",
        "physical_amplitude_calibrated": False,
    }

    backend: dict[str, Any] = {
        "recurrent_backend": signals.metadata.get("recurrent_backend", "dense"),
        "synaptic_kernel": signals.metadata.get("synaptic_kernel", "exponential"),
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "physical_amplitude_calibrated": False,
        "source_model": signals.metadata.get("source_model"),
        "source_bookkeeping": signals.metadata.get("source_bookkeeping"),
    }
    if "edge_list" in self.params:
        edges = self.params["edge_list"]
        backend["edge_list_n_edges"] = int(edges.n_edges)
        backend["edge_list_backend"] = "edge_list_recurrent_v0.0.9"

    return RunReceipt(
        receipt_id=receipt_id,
        jaxfne_version=_JAXFNE_VERSION,
        config_hash=cfg_h,
        simulation=sim_summary,
        signals_summary=signals.summary(),
        truth=truth,
        claim_labels=claim_labels,
        backend=backend,
        tags=dict(tags or {}),
    )


def compute_readout(
    self,
    signals: Signals,
    specs: "Sequence[ReadoutSpec]",
) -> "list[ReadoutResult]":
    """Compute scalar features from Signals according to a list of ReadoutSpecs.

    **Canonical v0.1 workflow method.**  Prefer this over :meth:`probe`
    for declarative, typed feature extraction.

    Args:
        signals: Signals returned by self.simulate().
        specs: Sequence of ReadoutSpec objects declaring what to extract.

    Returns:
        List of ReadoutResult objects in the same order as specs.
        Values are None when not applicable (missing field, unknown metric).

    No physical-amplitude, empirical-validation, or mechanism claim is
    introduced.  All values are proxy/native-current scaffold outputs.
    """
    results: list[ReadoutResult] = []
    for spec in specs:
        if spec.metric not in _KNOWN_READOUT_METRICS:
            results.append(ReadoutResult(
                spec_name=spec.name,
                metric=spec.metric,
                value=None,
                status="unknown_metric",
            ))
            continue

        dt_ms = (
            float(signals.time_ms[1] - signals.time_ms[0])
            if signals.time_ms.shape[0] > 1
            else 1.0
        )

        # Time slice (optional); negative start is treated as empty window.
        if spec.time_window_ms is not None:
            start_ms, end_ms = spec.time_window_ms
            t0 = max(0, int(start_ms / dt_ms))
            t1 = min(int(signals.time_ms.shape[0]), int(end_ms / dt_ms))
            if t0 >= t1:
                results.append(ReadoutResult(
                    spec_name=spec.name,
                    metric=spec.metric,
                    value=None,
                    status="empty_time_window",
                ))
                continue
            V_m_sl = signals.V_m[t0:t1]
            sp_sl = signals.spikes[t0:t1]
            src_sl = signals.sources[t0:t1] if signals.sources is not None else None
            field_t0, field_t1 = t0, t1
        else:
            V_m_sl = signals.V_m
            sp_sl = signals.spikes
            src_sl = signals.sources
            field_t0, field_t1 = 0, int(signals.time_ms.shape[0])

        if spec.metric == "spike_rate_hz":
            value = float(jnp.mean(sp_sl) * (1000.0 / dt_ms))
        elif spec.metric == "spike_count":
            value = float(jnp.sum(sp_sl))
        elif spec.metric == "mean_V_m":
            value = float(jnp.mean(V_m_sl))
        elif spec.metric == "source_abs_mean":
            if src_sl is None:
                results.append(ReadoutResult(
                    spec_name=spec.name,
                    metric=spec.metric,
                    value=None,
                    status="missing_sources",
                ))
                continue
            value = float(jnp.mean(jnp.abs(src_sl)))
        elif spec.metric in ("csd_abs_mean", "lfp_abs_mean"):
            if signals.field is None:
                results.append(ReadoutResult(
                    spec_name=spec.name,
                    metric=spec.metric,
                    value=None,
                    status="no_field",
                ))
                continue
            arr = signals.field.csd if spec.metric == "csd_abs_mean" else signals.field.lfp
            # Apply time-window slice first, then contact slice.
            arr = arr[field_t0:field_t1]
            if spec.n_contacts_slice is not None:
                c0, c1 = spec.n_contacts_slice
                arr = arr[:, c0:c1]
            value = float(jnp.mean(jnp.abs(arr)))
        else:
            value = None

        results.append(ReadoutResult(
            spec_name=spec.name,
            metric=spec.metric,
            value=value,
            status="computed",
        ))
    return results

def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
    """Extract named arrays from Signals by mode.

    Compatibility alias retained from v0.0.3–v0.0.14.  For typed,
    declarative feature extraction in the canonical v0.1 workflow, prefer
    :meth:`compute_readout` with :class:`ReadoutSpec` objects.
    """

    modes = list(modes or [])
    out: dict[str, Any] = {"requested_modes": modes}
    if "spikes" in modes:
        out["spikes"] = signals.spikes
    if "V_m" in modes:
        out["V_m"] = signals.V_m
    if "source" in modes or "sources" in modes:
        out["sources"] = signals.sources
    if signals.field is not None:
        out.update(probe_laminar_modes(signals.field, modes))
    return out

def record(self, signals: Signals, modes: Sequence[str]) -> dict[str, Any]:
    """User-friendly alias for :meth:`probe`."""

    return self.probe(signals, modes)

