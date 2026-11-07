"""B1 harness smoke: one tiny cell in-process; every receipt field present/finite."""

import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "b1_harness", Path(__file__).with_name("b1_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _spec(recording: str) -> dict:
    return {"model": "M1", "n": 100, "dtype": "float32", "device": "cpu",
            "batch": 1, "recording": recording, "seeds": [1],
            "duration_ms": 10.0, "dt_ms": 0.1}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_tiny_cell_receipt_fields():
    rec = H.run_cell(_spec("full"))
    assert rec["status"] == "MEASURED", rec.get("error")
    for key in ("cell", "model", "n", "n_steps", "duration_ms", "dt_ms",
                "dtype", "device_requested", "device_used", "backend",
                "batch", "recording", "seeds", "warmup_seed", "build_seed",
                "compile_s", "construct_s", "warmup_s", "run_s",
                "run_s_median", "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "ensemble_s", "ensemble_mode",
                "peak_gpu_mem", "peak_host_rss_mb", "transfer_s",
                "numerical_agreement", "validity", "env", "scaling_rule",
                "lane_refs", "prereg", "notes", "wall_s"):
        assert key in rec, f"missing receipt field {key}"
    assert rec["n_steps"] == 100
    for key in ("compile_s", "construct_s", "warmup_s", "run_s_median",
                "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "peak_host_rss_mb", "wall_s"):
        assert _finite(rec[key]), f"{key}={rec[key]!r} not finite"
    assert len(rec["run_s"]) == 1 and all(_finite(x) for x in rec["run_s"])
    put, get = rec["transfer_s"]["put_s"], rec["transfer_s"]["get_s"]
    assert _finite(put) and _finite(get), (put, get)
    ag = rec["numerical_agreement"]
    assert ag["total_spikes"] >= 0 and _finite(ag["total_spikes"])
    assert set(ag["spike_counts"]) == set(ag["mean_rate_hz"]) >= {"E", "PV"}
    for pop, rate in ag["mean_rate_hz"].items():
        assert _finite(rate), (pop, rate)
    assert rec["validity"]["spikes_gt_zero"] == (ag["total_spikes"] > 0)
    env = rec["env"]
    for key in ("sha", "prereg_sha256", "jax", "jaxlib", "jaxfne_version",
                "device_string", "dtype", "threads", "seed_list", "platform",
                "nvidia_smi_before", "nvidia_smi_after"):
        assert key in env, f"missing env field {key}"
    assert len(env["prereg_sha256"]) == 64
    assert env["seed_list"] == [1]


def test_m3_tiny_cell_receipt_fields():
    spec = {"model": "M3", "n": 1, "dtype": "float32", "device": "cpu",
            "batch": 1, "recording": "full", "seeds": [1],
            "duration_ms": 10.0, "dt_ms": 0.1}
    rec = H.run_cell(spec)
    assert rec["status"] == "MEASURED", rec.get("error")
    for key in ("cell", "model", "n", "n_steps", "duration_ms", "dt_ms",
                "dtype", "device_requested", "device_used", "backend",
                "batch", "recording", "seeds", "warmup_seed", "build_seed",
                "compile_s", "construct_s", "warmup_s", "run_s",
                "run_s_median", "run_s_q1", "run_s_q3", "run_s_iqr",
                "throughput_neuron_steps_s", "ensemble_s", "ensemble_mode",
                "peak_gpu_mem", "peak_host_rss_mb", "transfer_s",
                "numerical_agreement", "validity", "env", "scaling_rule",
                "lane_refs", "prereg", "notes", "wall_s"):
        assert key in rec, f"missing receipt field {key}"
    assert rec["n_steps"] == 100
    ag = rec["numerical_agreement"]
    assert ag["total_spikes"] > 0, "I_ext=10 must drive HH spikes"
    assert rec["validity"]["spikes_gt_zero"] is True
    assert rec["validity"]["mean_E_rate_hz"] > 0


def test_recording_off_rss_reported():
    full = H.run_cell(_spec("full"))
    off = H.run_cell(_spec("off"))
    assert full["status"] == "MEASURED" and off["status"] == "MEASURED"
    # Presence (not a magnitude): both modes report peak host RSS.
    assert _finite(full["peak_host_rss_mb"])
    assert _finite(off["peak_host_rss_mb"])
