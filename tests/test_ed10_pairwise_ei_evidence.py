"""ED10 pairwise-E/I-HDP evidence harness: structure + truth gates + the two
mechanistic claims (rescue happens; selective gains redirect adaptation burden
toward the boosted pathway). Mirrors tests/test_ed9_hdp_evidence.py's pattern
for scripts/ed10_hdp_pairwise_ei_selective_rescue.py.
"""
import importlib.util
import pathlib


def _load():
    p = pathlib.Path(__file__).parent.parent / "scripts" / "ed10_hdp_pairwise_ei_selective_rescue.py"
    spec = importlib.util.spec_from_file_location("ed10_pairwise_ei", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ed10_bundle_structure_and_selective_mechanism(tmp_path):
    ed10 = _load()
    bundle, digest, out = ed10.run(
        n=8, seeds=3, duration_ms=1000.0, dt_ms=0.5,
        out_dir=str(tmp_path / "ed10_pairwise_ei"))

    assert set(bundle["conditions"]) == {
        "healthy_reference", "damaged_no_rescue", "damaged_uniform_rescue", "damaged_selective_rescue",
    }
    s = bundle["summary"]

    # The "damaged" condition (weak inhibitory drive) really does suppress E activity
    # relative to the healthy reference.
    assert s["damaged_e_activity"] < s["healthy_e_activity"]

    # Both uniform and selective rescue substantially recover E activity toward healthy.
    assert s["uniform_rescue_fraction"] is not None and s["uniform_rescue_fraction"] > 0.5
    assert s["selective_rescue_fraction"] is not None and s["selective_rescue_fraction"] > 0.5

    # The mechanistic claim this evidence bundle actually supports: selective (k_ie-boosted,
    # others-damped) gains redirect adaptation onto the boosted I->E pathway and away from
    # E-E, relative to uniform gains -- NOT that selective rescue is "better."
    assert s["selective_perturbs_g_ee_less_than_uniform"] is True
    uniform_gie = bundle["conditions"]["damaged_uniform_rescue"]["g_ie_mean_abs_change"]["mean"]
    selective_gie = bundle["conditions"]["damaged_selective_rescue"]["g_ie_mean_abs_change"]["mean"]
    assert selective_gie > uniform_gie

    # Truth gates: method evidence only, no biological/mechanism/disorder claim.
    assert bundle["claim_status"] == "computational_control_proxy_not_biological_mechanism"
    assert bundle["biological_learning_claim"] is False
    assert bundle["mechanism_claim_status"] == "not_claimed"
    assert bundle["disorder_relevance_claim"] is False

    assert len(digest) == 64
    assert (pathlib.Path(out) / "ed10_evidence.json").exists()
    assert (pathlib.Path(out) / "ed10_receipt.json").exists()
