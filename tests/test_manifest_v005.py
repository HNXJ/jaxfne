"""Tests for v0.0.5-P4 manifest extensions and example smoke runs."""

import json
import subprocess
import sys
from pathlib import Path

import jaxfne as jtfne


def _model_signals_readout(n=8, duration_ms=5.0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    return model, signals, readout


def test_manifest_with_paradigm_json_safe():
    """manifest() with paradigm kwarg produces JSON-safe output."""
    model, signals, readout = _model_signals_readout()
    paradigm = jtfne.standard_visual_omission()

    mf = model.manifest(signals=signals, readout=readout, paradigm=paradigm.to_dict())
    assert "paradigm" in mf
    assert mf["paradigm"]["name"] == "standard_visual_omission"
    assert "v005_claim_labels" in mf

    json_str = json.dumps(mf, allow_nan=False)
    assert isinstance(json_str, str)


def test_manifest_with_objective_evaluation_json_safe():
    """manifest() with objective + evaluation produces JSON-safe output."""
    model, signals, readout = _model_signals_readout()
    obj = (
        jtfne.Objective(name="test_obj")
        .loss("rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .gate("rate_gate", threshold=500.0, criterion="below", metric="spike_rate_hz_mean")
    )
    eval_report = model.evaluate(signals, obj)
    obj_dict = {"name": obj.name, "losses": obj.losses, "regularizers": obj.regularizers, "gates": obj.gates}

    mf = model.manifest(signals=signals, readout=readout, objective=obj_dict, evaluation=eval_report)
    assert "objective" in mf
    assert "evaluation" in mf
    assert "v005_claim_labels" in mf
    assert mf["evaluation"]["evaluation_status"] == "objective_evaluate_v0.0.5"

    json_str = json.dumps(mf, allow_nan=False)
    assert isinstance(json_str, str)


def test_manifest_with_tuning_report_json_safe():
    """manifest() with tuning report produces JSON-safe output."""
    model, signals, readout = _model_signals_readout()
    obj = jtfne.Objective(name="test_tune_obj")
    _, tune_report = model.tune(obj, optimizer="GSDR", steps=0)

    mf = model.manifest(signals=signals, readout=readout, tuning=tune_report)
    assert "tuning" in mf
    assert "v005_claim_labels" in mf
    assert mf["tuning"]["tuning_status"] == "metadata_only_no_steps_requested"

    json_str = json.dumps(mf, allow_nan=False)
    assert isinstance(json_str, str)


def test_manifest_preserves_truth_gates_with_v005_metadata():
    """All v0.0.4 truth gates are present even when v0.0.5 metadata is added."""
    model, signals, readout = _model_signals_readout()
    paradigm = jtfne.standard_visual_omission()
    obj = jtfne.objective()
    eval_report = model.evaluate(signals, obj)
    _, tune_report = model.tune(obj, optimizer="AGSDR", steps=0)

    mf = model.manifest(
        signals=signals,
        readout=readout,
        paradigm=paradigm.to_dict(),
        objective={"name": obj.name, "losses": [], "regularizers": [], "gates": []},
        evaluation=eval_report,
        tuning=tune_report,
    )

    # v0.0.4 gates
    assert mf["truth_mode"] == "truth_safe_unverified"
    assert mf["claim_level"] == "computational_scaffold"
    assert mf["source_calibration_status"] == "uncalibrated_izhikevich_native_current"
    assert mf["field_solver_status"] == "laminar_proxy_no_pde"
    assert mf["source_field_status"]["field_claim_level"] == "proxy_readout_only"
    assert mf["source_field_status"]["physical_amplitude_claim_allowed"] is False

    # v0.0.5 claim labels (static label in io.py, not derived from tune report)
    labels = mf["v005_claim_labels"]
    assert labels["objective_status"] == "computational_diagnostic"
    assert "tuning_status" in labels
    assert labels["empirical_validation_status"] == "not_empirically_validated"
    assert labels["mechanism_claim_status"] == "not_claimed"
    assert labels["physical_amplitude_claim_allowed"] is False


def _run_example(example_path: str) -> subprocess.CompletedProcess:
    import os
    env = os.environ.copy()
    repo_root = str(Path(__file__).parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_examples_02_omission_scaffold_runs():
    """examples/02_omission_scaffold.py runs to completion without error."""
    example = str(Path(__file__).parent.parent / "examples" / "02_omission_scaffold.py")
    result = _run_example(example)
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
    assert "standard_visual_omission" in result.stdout
    assert "AAAX" in result.stdout
    assert "not_claimed" in result.stdout


def test_examples_03_objective_and_tune_smoke_runs():
    """examples/03_objective_and_tune_smoke.py runs to completion without error."""
    example = str(Path(__file__).parent.parent / "examples" / "03_objective_and_tune_smoke.py")
    result = _run_example(example)
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
    assert "metadata_only_no_steps_requested" in result.stdout
    assert "not_empirically_validated" in result.stdout


def test_readme_no_overclaim_language():
    """README.md must not contain forbidden overclaim terms."""
    readme_path = Path(__file__).parent.parent / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8").lower()

    forbidden = [
        "mechanism_proven",
        "active_inference_validated",
        "prediction_error_proven",
        "calibrated amplitude",
        "biological truth",
        "proves that",
        "validated mechanism",
    ]
    for term in forbidden:
        assert term not in readme_text, f"Overclaim term found in README: {term!r}"


def test_manifest_no_empirical_or_mechanism_claim():
    """Full manifest with v0.0.5 metadata must not assert empirical/mechanism claims."""
    model, signals, readout = _model_signals_readout()
    paradigm = jtfne.standard_visual_omission()
    obj = jtfne.Objective(name="smoke").loss("rate", target=20.0, metric="spike_rate_hz_mean")
    eval_report = model.evaluate(signals, obj)
    _, tune_report = model.tune(obj, optimizer="GSDR", steps=0)

    mf = model.manifest(
        signals=signals, readout=readout,
        paradigm=paradigm.to_dict(),
        evaluation=eval_report,
        tuning=tune_report,
    )

    # Serialize and scan JSON text for overclaim language.
    # Use word-boundary-style checks: the value "not_empirically_validated" is safe;
    # a standalone value of "empirically_validated" (without negation prefix) would be unsafe.
    mf_str = json.dumps(mf, allow_nan=False)
    for bad_term in ["mechanism_proven", "biological_truth",
                     "prediction_error_proven", "active_inference_validated"]:
        assert bad_term not in mf_str.lower(), f"Overclaim term in manifest JSON: {bad_term!r}"
    # Ensure "empirically_validated" only appears inside the negation form.
    import re
    # Only flag if "empirically_validated" appears NOT preceded by "not_"
    bad_ev = re.search(r'(?<!not_)empirically_validated', mf_str)
    assert bad_ev is None, f"Unsafely bare 'empirically_validated' found in manifest JSON"

    # Claim labels must be present and correct
    labels = mf["v005_claim_labels"]
    assert labels["empirical_validation_status"] == "not_empirically_validated"
    assert labels["mechanism_claim_status"] == "not_claimed"
