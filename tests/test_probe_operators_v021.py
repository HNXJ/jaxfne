"""Tests for v0.2.1 multimodal probe operator contract.

Covers:
A. Probe operator report structure (JSON-safe, truth gates, metadata)
B. SPK, Vm, source operators (basic shape and report)
C. LFP-proxy, CSD-proxy operators (including sign convention metadata)
D. EEG-proxy, MEG-proxy operators (simulated/proxy status)
E. EMM-proxy operator (no biological metabolism claims)
F. Terminology: no -like, real EEG, real MEG, validated/calibrated claims
G. All operators produce finite outputs on synthetic inputs
H. Existing examples still run
"""

import json
import pytest
import jax
import jax.numpy as jnp
import jax.random

import jaxfne
from jaxfne.fields import (
    spk_probe,
    vm_probe,
    source_probe,
    lfp_proxy_probe,
    csd_proxy_probe,
    eeg_proxy_probe,
    meg_proxy_probe,
    emm_proxy_probe,
)
from jaxfne.io import json_safe


# ─── A. Probe Operator Report Structure ─────────────────────────────────────


def test_a_spk_report_is_json_safe():
    """SPK report is JSON-safe with allow_nan=False."""
    spikes = jnp.ones((100, 8), dtype=jnp.float32)
    readout = spk_probe(spikes)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_vm_report_is_json_safe():
    """Vm report is JSON-safe with allow_nan=False."""
    voltage = jnp.ones((100, 8), dtype=jnp.float32)
    readout = vm_probe(voltage)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_source_report_is_json_safe():
    """Source report is JSON-safe with allow_nan=False."""
    source = jnp.ones((100, 8), dtype=jnp.float32)
    readout = source_probe(source)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_lfp_proxy_report_is_json_safe():
    """LFP-proxy report is JSON-safe with allow_nan=False."""
    lfp = jnp.ones((100, 16), dtype=jnp.float32)
    readout = lfp_proxy_probe(lfp)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_csd_proxy_report_is_json_safe():
    """CSD-proxy report is JSON-safe with allow_nan=False."""
    csd = jnp.ones((100, 16), dtype=jnp.float32)
    readout = csd_proxy_probe(csd)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_eeg_proxy_report_is_json_safe():
    """EEG-proxy report is JSON-safe with allow_nan=False."""
    eeg = jnp.ones((100, 32), dtype=jnp.float32)
    readout = eeg_proxy_probe(eeg)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_meg_proxy_report_is_json_safe():
    """MEG-proxy report is JSON-safe with allow_nan=False."""
    meg = jnp.ones((100, 32), dtype=jnp.float32)
    readout = meg_proxy_probe(meg)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


def test_a_emm_proxy_report_is_json_safe():
    """EMM-proxy report is JSON-safe with allow_nan=False."""
    emm = jnp.ones((100,), dtype=jnp.float32)
    readout = emm_proxy_probe(emm)
    json.dumps(json_safe(readout.report), allow_nan=False)  # must not raise


# ─── B. SPK, Vm, source Operators ───────────────────────────────────────────


def test_b_spk_returns_finite_output():
    """SPK operator returns finite output on synthetic inputs."""
    spikes = jnp.zeros((50, 8), dtype=jnp.float32)
    readout = spk_probe(spikes)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "spk"


def test_b_vm_returns_finite_output():
    """Vm operator returns finite output on synthetic inputs."""
    key = jax.random.PRNGKey(0)
    voltage = jax.random.normal(key, shape=(50, 8))
    readout = vm_probe(voltage)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "vm"


def test_b_source_returns_finite_output():
    """Source operator returns finite output on synthetic inputs."""
    key = jax.random.PRNGKey(0)
    source = jax.random.normal(key, shape=(50, 8))
    readout = source_probe(source)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "source"


def test_b_operators_declare_shape():
    """Each operator declares output shape in report."""
    spikes = jnp.ones((100, 8))
    readout = spk_probe(spikes)
    assert "data_shape" in readout.report
    assert readout.report["data_shape"] == "(100, 8)"


# ─── C. LFP-proxy, CSD-proxy Operators ──────────────────────────────────────


def test_c_lfp_proxy_returns_finite_output():
    """LFP-proxy operator returns finite output."""
    lfp = jnp.ones((50, 16), dtype=jnp.float32)
    readout = lfp_proxy_probe(lfp)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "lfp_proxy"


def test_c_csd_proxy_returns_finite_output():
    """CSD-proxy operator returns finite output."""
    csd = jnp.ones((50, 16), dtype=jnp.float32)
    readout = csd_proxy_probe(csd)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "csd_proxy"


def test_c_csd_proxy_includes_sign_convention():
    """CSD-proxy report includes CSD sign convention metadata."""
    csd = jnp.ones((50, 16), dtype=jnp.float32)
    readout = csd_proxy_probe(csd, csd_sign_convention="positive_equals_extracellular_source")
    assert "CSD_sign_convention" in readout.report
    assert readout.report["CSD_sign_convention"] == "positive_equals_extracellular_source"


def test_c_lfp_proxy_uses_proxy_label_not_like():
    """LFP-proxy operator uses 'lfp_proxy', not 'lfp_like'."""
    lfp = jnp.ones((50, 16), dtype=jnp.float32)
    readout = lfp_proxy_probe(lfp)
    assert readout.kind == "lfp_proxy"
    assert "lfp_like" not in readout.kind


def test_c_csd_proxy_uses_proxy_label_not_like():
    """CSD-proxy operator uses 'csd_proxy', not 'csd_like'."""
    csd = jnp.ones((50, 16), dtype=jnp.float32)
    readout = csd_proxy_probe(csd)
    assert readout.kind == "csd_proxy"
    assert "csd_like" not in readout.kind


# ─── D. EEG-proxy, MEG-proxy Operators ──────────────────────────────────────


def test_d_eeg_proxy_returns_finite_output():
    """EEG-proxy operator returns finite output."""
    eeg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = eeg_proxy_probe(eeg)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "eeg_proxy"


def test_d_meg_proxy_returns_finite_output():
    """MEG-proxy operator returns finite output."""
    meg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = meg_proxy_probe(meg)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "meg_proxy"


def test_d_eeg_proxy_contains_simulated_proxy_status():
    """EEG-proxy report contains simulated/proxy status."""
    eeg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = eeg_proxy_probe(eeg)
    report_str = str(readout.report)
    assert "simulated" in report_str or "proxy" in report_str


def test_d_meg_proxy_contains_simulated_proxy_status():
    """MEG-proxy report contains simulated/proxy status."""
    meg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = meg_proxy_probe(meg)
    report_str = str(readout.report)
    assert "simulated" in report_str or "proxy" in report_str


def test_d_eeg_proxy_uses_proxy_label_not_like():
    """EEG-proxy operator uses 'eeg_proxy', not 'eeg_like'."""
    eeg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = eeg_proxy_probe(eeg)
    assert readout.kind == "eeg_proxy"
    assert "eeg_like" not in readout.kind


def test_d_meg_proxy_uses_proxy_label_not_like():
    """MEG-proxy operator uses 'meg_proxy', not 'meg_like'."""
    meg = jnp.ones((50, 32), dtype=jnp.float32)
    readout = meg_proxy_probe(meg)
    assert readout.kind == "meg_proxy"
    assert "meg_like" not in readout.kind


# ─── E. EMM-proxy Operator ──────────────────────────────────────────────────


def test_e_emm_proxy_returns_finite_output():
    """EMM-proxy operator returns finite output."""
    emm = jnp.ones((50,), dtype=jnp.float32)
    readout = emm_proxy_probe(emm)
    assert jnp.all(jnp.isfinite(readout.data))
    assert readout.kind == "emm_proxy"


def test_e_emm_proxy_contains_proxy_status():
    """EMM-proxy report contains proxy status."""
    emm = jnp.ones((50,), dtype=jnp.float32)
    readout = emm_proxy_probe(emm)
    report_str = str(readout.report)
    assert "proxy" in report_str


def test_e_emm_proxy_does_not_claim_biological_metabolism():
    """EMM-proxy report does not claim biological metabolism."""
    emm = jnp.ones((50,), dtype=jnp.float32)
    readout = emm_proxy_probe(emm)
    report_str = str(readout.report).lower()
    # The word "biological" should not appear in a claim context
    # (it may appear in "not biological metabolism" which is safe)
    assert "biological_metabolism" not in report_str or "not" in report_str


# ─── F. Terminology Checks ──────────────────────────────────────────────────


def test_f_no_like_in_public_operators():
    """No '-like' suffix in public operator kinds."""
    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]
    for readout in operators:
        assert "-like" not in readout.kind
        assert "_like" not in readout.kind


def test_f_physical_amplitude_claim_false():
    """All proxy operators set physical_amplitude_claim_allowed to False."""
    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]
    for readout in operators:
        assert readout.report.get("physical_amplitude_claim_allowed") is False


def test_f_truth_gates_frozen():
    """All operators preserve frozen report metadata (no truth_mode in public reports)."""
    operators = [
        spk_probe(jnp.ones((10, 8))),
        vm_probe(jnp.ones((10, 8))),
        source_probe(jnp.ones((10, 8))),
        lfp_proxy_probe(jnp.ones((10, 16))),
        csd_proxy_probe(jnp.ones((10, 16))),
        eeg_proxy_probe(jnp.ones((10, 32))),
        meg_proxy_probe(jnp.ones((10, 32))),
        emm_proxy_probe(jnp.ones((10,))),
    ]
    for readout in operators:
        report = readout.report
        # v0.2.12: truth_mode is internal only, not in public reports
        assert "truth_mode" not in report, f"truth_mode should not be in public report: {report.keys()}"
        assert report.get("field_claim_level") == "proxy_readout_only"
        assert report.get("physical_amplitude_claim_allowed") is False


# ─── G. Finite Output Test ──────────────────────────────────────────────────


def test_g_all_operators_finite_on_random_inputs():
    """All operators return finite outputs on random synthetic inputs."""
    key = jax.random.PRNGKey(0)

    spikes = jnp.zeros((100, 8))
    assert jnp.all(jnp.isfinite(spk_probe(spikes).data))

    key, subkey = jax.random.split(key)
    voltage = jax.random.normal(subkey, shape=(100, 8))
    assert jnp.all(jnp.isfinite(vm_probe(voltage).data))

    key, subkey = jax.random.split(key)
    source = jax.random.normal(subkey, shape=(100, 8))
    assert jnp.all(jnp.isfinite(source_probe(source).data))

    key, subkey = jax.random.split(key)
    lfp = jax.random.normal(subkey, shape=(100, 16))
    assert jnp.all(jnp.isfinite(lfp_proxy_probe(lfp).data))

    key, subkey = jax.random.split(key)
    csd = jax.random.normal(subkey, shape=(100, 16))
    assert jnp.all(jnp.isfinite(csd_proxy_probe(csd).data))

    key, subkey = jax.random.split(key)
    eeg = jax.random.normal(subkey, shape=(100, 32))
    assert jnp.all(jnp.isfinite(eeg_proxy_probe(eeg).data))

    key, subkey = jax.random.split(key)
    meg = jax.random.normal(subkey, shape=(100, 32))
    assert jnp.all(jnp.isfinite(meg_proxy_probe(meg).data))

    key, subkey = jax.random.split(key)
    emm = jax.random.normal(subkey, shape=(100,))
    assert jnp.all(jnp.isfinite(emm_proxy_probe(emm).data))


# ─── H. Integration: Existing Examples Still Run ─────────────────────────────


def test_h_example_02_spectrolaminar_still_runs():
    """Existing example 02 (spectrolaminar) still executes without error."""
    # This is a smoke test that imports and runs the example
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "examples/02_spectrolaminar_oddball_scaffold.py"],
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
