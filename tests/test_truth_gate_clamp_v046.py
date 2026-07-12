"""Truth-gate clamp: update_metadata / manifest cannot escalate claim surfaces."""

from __future__ import annotations

import jaxfne as jtfne


def test_update_metadata_cannot_escalate_amplitude_or_claim_level():
    cfg = jtfne.suite2_single_neuron_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    escalated = cfg.update_metadata(
        physical_amplitude_calibrated=True,
        claim_level="validated",
        field_claim_level="physical_field",
        field_solver_status="made_up_solver",
    )
    meta = escalated.metadata
    assert meta["physical_amplitude_calibrated"] is False
    assert meta["claim_level"] == "computational_scaffold"
    assert meta["field_claim_level"] == "proxy_readout"
    assert meta["field_solver_status"] == "linear_solver"


def test_manifest_clamps_escalated_cfg_metadata():
    cfg = jtfne.suite2_single_neuron_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    raw = dict(cfg.metadata)
    raw["physical_amplitude_calibrated"] = True
    raw["claim_level"] = "empirically_validated"
    raw["field_claim_level"] = "calibrated_pde"
    from dataclasses import replace

    hostile = replace(cfg, metadata=raw)
    man = jtfne.manifest(hostile)
    assert man["physical_amplitude_calibrated"] is False
    assert man["claim_level"] == "computational_scaffold"
    assert man["field_claim_level"] == "proxy_readout"


def test_model_manifest_and_run_receipt_agree_on_amplitude():
    cfg = jtfne.suite2_single_neuron_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    model = jtfne.construct(cfg.update_metadata(physical_amplitude_calibrated=True))
    signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
    man = model.manifest(signals=signals)
    receipt = model.run_receipt(signals)
    assert man["physical_amplitude_calibrated"] is False
    assert receipt.truth["physical_amplitude_calibrated"] is False
    assert receipt.claim_labels["physical_amplitude_calibrated"] is False


def test_clamp_truth_gate_metadata_helper_is_public_via_core():
    clamped = jtfne.core.clamp_truth_gate_metadata(
        {"physical_amplitude_calibrated": True, "claim_level": "validated"}
    )
    assert clamped["physical_amplitude_calibrated"] is False
    assert clamped["claim_level"] == "computational_scaffold"
