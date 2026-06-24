"""ED9 HDP-controller evidence harness: structure + stabilization + truth gates.

Mirrors test_ed9_evidence.py for scripts/ed9_hdp_evidence.py.
"""
import importlib.util
import pathlib


def _load():
    p = pathlib.Path(__file__).parent.parent / "scripts" / "ed9_hdp_evidence.py"
    spec = importlib.util.spec_from_file_location("ed9_hdp", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ed9_hdp_bundle_structure_and_stabilization(tmp_path):
    ed9 = _load()
    bundle, digest, out = ed9.run(n=120, seeds=2, duration_ms=400.0, dt_ms=0.5,
                                  alpha=0.05, gamma=0.5, k_ctrl=0.15, k_hdp=0.01,
                                  out_dir=str(tmp_path / "ed9_hdp"))
    # 3-way ablation grid present (not 4-way -- see script docstring)
    assert set(bundle["conditions"]) == {"null", "h_dynamics", "both"}
    # null + ablation + repeated seeds recorded with stats
    for c in bundle["conditions"].values():
        assert c["rate_spread_hz"]["n"] == 2 and "std" in c["rate_spread_hz"]
        assert c["all_vm_finite"] is True
    # null control: H stays pinned at 1.0
    assert bundle["conditions"]["null"]["H_mean"]["mean"] == 1.0
    assert bundle["conditions"]["null"]["H_std"]["mean"] == 0.0
    # h_dynamics: H moves away from 1.0 once alpha/gamma/K_ctrl are nonzero
    assert bundle["conditions"]["h_dynamics"]["H_std"]["mean"] > 0.0
    # truth gates: method evidence, NOT mechanism
    assert bundle["claim_status"] == "computational_control_proxy_not_biological_mechanism"
    assert bundle["biological_learning_claim"] is False
    assert bundle["mechanism_claim_status"] == "not_claimed"
    # receipt written with a real sha256
    assert len(digest) == 64
    assert (pathlib.Path(out) / "ed9_evidence.json").exists()
    assert (pathlib.Path(out) / "ed9_receipt.json").exists()
