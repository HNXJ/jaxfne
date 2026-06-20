"""ED9 homeostasis-controller evidence harness: structure + stabilization + truth gates."""
import importlib.util
import pathlib


def _load():
    p = pathlib.Path(__file__).parent.parent / "scripts" / "ed9_homeostasis_evidence.py"
    spec = importlib.util.spec_from_file_location("ed9", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ed9_bundle_structure_and_stabilization(tmp_path):
    ed9 = _load()
    bundle, digest, out = ed9.run(n=120, seeds=2, duration_ms=400.0, dt_ms=0.5,
                                  k_gain=8.0, eta=0.05, out_dir=str(tmp_path / "ed9"))
    # ablation grid present
    assert set(bundle["conditions"]) == {"null", "intrinsic", "synaptic", "both"}
    # null + ablation + repeated seeds recorded with stats
    for c in bundle["conditions"].values():
        assert c["rate_spread_hz"]["n"] == 2 and "std" in c["rate_spread_hz"]
        assert c["all_vm_finite"] is True
    # stabilization: controller (both) reduces the hyper-vs-quiet spread vs null
    assert (bundle["conditions"]["both"]["rate_spread_hz"]["mean"]
            < bundle["conditions"]["null"]["rate_spread_hz"]["mean"])
    # truth gates: method evidence, NOT mechanism
    assert bundle["claim_status"] == "computational_control_proxy_not_biological_mechanism"
    assert bundle["biological_learning_claim"] is False
    assert bundle["mechanism_claim_status"] == "not_claimed"
    # receipt written with a real sha256
    assert len(digest) == 64
    assert (pathlib.Path(out) / "ed9_evidence.json").exists()
    assert (pathlib.Path(out) / "ed9_receipt.json").exists()
