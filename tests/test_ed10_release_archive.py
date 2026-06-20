"""ED10 release-archive receipt: structure + provenance + content hashes + truth gates."""
import importlib.util
import json
import pathlib


def _load():
    p = pathlib.Path(__file__).parent.parent / "scripts" / "ed10_release_archive_receipt.py"
    spec = importlib.util.spec_from_file_location("ed10", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ed10_release_archive_structure_and_gates(tmp_path):
    ed10 = _load()
    archive, digest, out = ed10.run(out_dir=str(tmp_path / "ed10"))

    # release identity bound in
    assert archive["package_version"]  # non-empty
    prov = archive["provenance"]
    assert "sha" in prov and "branch" in prov and "dirty" in prov

    # evidence chain: hashes only existing artifacts, never fabricates
    chain = archive["evidence_chain"]
    for name, h in chain["artifact_sha256"].items():
        assert len(h) == 64
        assert name in chain["archived_artifacts"]
    # an archived artifact must actually exist on disk
    for rel in chain["archived_artifacts"].values():
        assert (pathlib.Path(__file__).parent.parent / rel).is_file()
    assert chain["complete"] is (len(chain["missing_artifacts"]) == 0)

    # conservative truth gates carried, never escalated
    g = archive["truth_gates"]
    assert g["claim_level"] == "computational_scaffold"
    assert g["field_solver_status"] == "linear_solver"
    assert g["physical_amplitude_calibrated"] is False
    assert g["biological_learning_claim"] is False
    assert g["mechanism_claim_status"] == "not_claimed"

    # self-hashed receipt written
    assert len(digest) == 64
    assert (pathlib.Path(out) / "ed10_release_archive.json").exists()
    receipt_path = pathlib.Path(out) / "ed10_receipt.json"
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["sha256"] == digest
    assert receipt["truth_status"].startswith("provenance")
