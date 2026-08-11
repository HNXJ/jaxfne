"""Model.manifest() -- the JSON-safe run summary combining ``Signals``,
readouts, tuning reports, and truth-gate metadata.

Split out of ``jaxfne/_model.py`` (Phase 2 defragmentation, 2026-07-20, part
of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1).
"""

from __future__ import annotations

from typing import Any, Optional

from .io import manifest as build_manifest
from ._signals import Signals, _default_basis_dict, _normalize_manifest_readout
from ._model import _MANIFEST_SCHEMA_VERSION, _RuntimeReportAdapter, _SOURCE_PROXY_METADATA


def manifest(
    self,
    signals: Optional[Signals] = None,
    readout: Optional[Any] = None,
    paradigm: Optional[dict[str, Any]] = None,
    objective: Optional[dict[str, Any]] = None,
    evaluation: Optional[dict[str, Any]] = None,
    tuning: Optional[dict[str, Any]] = None,
    dataset: Optional[dict[str, Any]] = None,
    trials: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a JSON-safe run manifest dict.

    Compatibility method retained from v0.0.4–v0.0.14.  For the canonical
    v0.1 workflow, prefer :meth:`run_receipt` (typed, immutable, with
    deterministic receipt ID) and :meth:`evaluate_report` (typed objective
    evaluation).  This method remains supported and is not scheduled for
    removal.

    The ``readout`` argument accepts any of:

    * ``None`` — no readout section included.
    * ``dict`` — passed through to the manifest as-is (legacy shape).
    * ``list`` or ``tuple`` of :class:`ReadoutResult` objects — the canonical
      output of :meth:`compute_readout`.  Converted to a JSON-safe readout
      summary dict with ``readout_results`` and ``requested_metrics`` keys.
    * ``list`` or ``tuple`` of ``dict`` — same conversion applied to each element.
    * Single :class:`ReadoutResult` — wrapped in a list and handled as above.
    """
    readout_normalized = _normalize_manifest_readout(readout)
    runtime_cfg = None
    if signals is not None and "runtime" in signals.metadata:
        runtime_cfg = _RuntimeReportAdapter(signals.metadata["runtime"])
    source_model = dict(_SOURCE_PROXY_METADATA)
    if signals is not None:
        source_mode_class = signals.metadata.get("source_mode_class")
        if source_mode_class == "specialized":
            source_model = {
                "source_model": signals.metadata.get("source_mode"),
                "source_mode": signals.metadata.get("source_mode"),
                "source_mode_class": source_mode_class,
                "source_decomposition": signals.metadata.get("source_decomposition"),
                "source_contract": signals.metadata.get("source_contract"),
                "source_calibration_status": signals.metadata.get(
                    "source_calibration_status"
                ),
                "representation": signals.metadata.get("representation", "relative"),
                "calibration_transform": signals.metadata.get(
                    "calibration_transform", "explicit_boundary_transform"
                ),
                "physical_amplitude_calibrated": signals.metadata.get(
                    "physical_amplitude_calibrated", False
                ),
            }
        else:
            source_model.update(
                {
                    key: value
                    for key, value in {
                        "source_calibration_status": signals.metadata.get(
                            "source_calibration_status"
                        ),
                        "source_mode": signals.metadata.get("source_mode"),
                        "source_mode_class": signals.metadata.get("source_mode_class"),
                        "source_contract": signals.metadata.get("source_contract"),
                    }.items()
                    if value is not None
                }
            )
    res = build_manifest(
        self.cfg,
        signals=signals,
        readout=readout_normalized,
        runtime_config=runtime_cfg,
        paradigm=paradigm,
        objective=objective,
        evaluation=evaluation,
        tuning=tuning,
        dataset=dataset,
    )
    if trials is not None:
        res["trials"] = trials
    # If readout was provided as ReadoutResult list (canonical v0.1 workflow),
    # surface the normalized readout summary in the manifest under "readout_results".
    # Dict-shaped readouts are already surfaced via build_manifest's field_diagnostics
    # logic; non-dict shapes are added here only.
    if readout_normalized is not None and isinstance(readout_normalized, dict):
        if "readout_results" in readout_normalized:
            res["readout_results"] = readout_normalized
    # Backend metadata: distinguish executed backend from available infrastructure.
    used_backend = "dense"
    used_kernel = "exponential"
    if signals is not None:
        used_backend = signals.metadata.get("recurrent_backend", "dense")
        used_kernel = signals.metadata.get("synaptic_kernel", "exponential")
    elif "edge_list" in self.params:
        used_backend = "unknown_not_run"
    backend_meta: dict[str, Any] = {
        "used_recurrent_backend": used_backend,
        "used_synaptic_kernel": used_kernel,
        "available_edge_list": "edge_list" in self.params,
        # Renamed 2026-07-05 from "manifest_schema_version" -- that name
        # collided (same key, different meaning/value) with the manifest
        # ROOT's own "manifest_schema_version" (Configuration's
        # _default_metadata, "0.0.4"). Both coexisted without silently
        # overwriting each other (different nesting depth), but the
        # shared name was a real trap for a future reader. This key
        # versions only this backend_metadata block, not the manifest.
        "backend_metadata_schema_version": _MANIFEST_SCHEMA_VERSION,
        "source_model": source_model,
    }
    # v0.2.0: Field admissibility metadata
    if signals is not None and signals.field is not None:
        from .validation import build_field_admissibility_report
        field_admissibility = build_field_admissibility_report(
            field_output=signals.field,
            cfg_metadata=dict(self.cfg.metadata or {}),
        )
        backend_meta["field_admissibility"] = field_admissibility
        if "field_admissibility" in signals.field.diagnostics:
            backend_meta["field_admissibility_diagnostics"] = signals.field.diagnostics.get(
                "field_admissibility"
            )
    if "edge_list" in self.params:
        edges = self.params["edge_list"]
        backend_meta["edge_count"] = int(edges.n_edges)
        backend_meta["receptor_indexed"] = True
        backend_meta["edge_list_source_calibration_status"] = edges.source_calibration_status
        backend_meta["edge_list_physical_amplitude_calibrated"] = False
        # v0.0.21: explicitly document which tau source each kernel uses.
        # simulate_edge_recurrent_izhikevich → edges.tau_ms (per-edge field)
        # simulate_receptor_exponential_izhikevich → standard_receptor_tau_table
        #   (receptor_index → standard catalog). Current standard table agrees
        #   with make_edge_list_from_dense for receptor_index ∈ {0, 1}, so
        #   these are numerically equivalent in the default scaffold flow.
        backend_meta["receptor_tau_source"] = {
            "exponential_kernel_uses": "edges.tau_ms",
            "receptor_exponential_kernel_uses": "standard_receptor_tau_table_by_receptor_index",
            "consistent_for_receptor_index_in": [0, 1],
        }
        # v0.0.21: surface receptor spec metadata so manifest documents
        # the receptor labels/taus the kernel can index. The actual per-edge
        # tau_ms lives on EdgeList; this is the catalog.
        from .emitters import standard_receptor_specs
        backend_meta["receptor_specs"] = {
            name: {
                "name": spec.name,
                "receptor_index": spec.receptor_index,
                "sign": spec.sign,
                "tau_ms": spec.tau_ms,
                "reversal_mV": spec.reversal_mV,
                "source_calibration_status": spec.source_calibration_status,
            }
            for name, spec in standard_receptor_specs().items()
        }
    # v0.0.21: explicit source model in manifest.
    res["source_model"] = source_model
    res["backend_metadata"] = backend_meta
    if "geometry" in self.static:
        res["source_geometry"] = self.static["geometry"]
    # v0.2.26: computation-basis block
    res["basis"] = _default_basis_dict()
    # v0.2.27: conservation-inspired proxy diagnostics
    if signals is not None and signals.field is not None:
        from .fields import compute_conservation_proxy_diagnostics
        _src_cal = (
            signals.metadata.get("source_calibration_status",
                                 "uncalibrated_izhikevich_native_current")
        )
        res["conservation_proxy_diagnostics"] = compute_conservation_proxy_diagnostics(
            field_solution=signals.field,
            source_calibration_status=_src_cal,
            field_solver_status="linear_solver",
            field_claim_level="proxy_readout",
        )
    return res


