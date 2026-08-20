#!/usr/bin/env python3
"""Adversarial validation of harness invariants and Gate 0 failure modes."""
import json
import pytest
from pathlib import Path
from scripts.harness.gate0_git_reality import check_gate0

ROOT = Path(__file__).resolve().parents[1]


def test_gate0_pass_on_clean_state():
    """Verify Gate 0 executes and passes on the current clean synchronized state."""
    res = check_gate0(fetch=False)
    assert res == 0, "Gate 0 must pass on clean synchronized checkout"


def test_frozen_paths_immutable_hash_integrity():
    """Verify that none of the 28 publication evidence artifacts have drifted."""
    frozen_manifest = json.loads((ROOT / ".opencode" / "frozen_paths.json").read_text())
    assert len(frozen_manifest["files"]) == 28
    import hashlib
    for rel_path, expected_sha in frozen_manifest["files"].items():
        p = ROOT / rel_path
        assert p.exists(), f"Frozen file missing: {rel_path}"
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Frozen hash drift detected in {rel_path}"


def test_current_task_ephemeral_and_typed():
    """Verify CURRENT_TASK.md is tiny, ephemeral, and contains explicit typed identities."""
    task_text = (ROOT / "scratch" / "CURRENT_TASK.md").read_text()
    assert len(task_text.splitlines()) < 40, "CURRENT_TASK.md must remain concise"
    assert "declared_core" in task_text
    assert "release_candidate" in task_text
    assert "receipt_head" in task_text
    assert "delta_C_core: 0" in task_text


def test_skills_have_required_sections():
    """Verify all canonical skills follow the WHEN/AUTHORITIES/RULES/STEPS/STOP/VERIFY/DONE structure."""
    required_sections = ["## WHEN", "## AUTHORITIES", "## RULES", "## STEPS", "## STOP", "## VERIFY", "## DONE"]
    for skill_path in (ROOT / "skills").glob("*/SKILL.md"):
        content = skill_path.read_text()
        for sec in required_sections:
            assert sec in content, f"Skill {skill_path.parent.name} missing section {sec}"


def test_release_receipt_typed_identities():
    """Verify release receipt v4 records distinct C_core, C_release, and hashes."""
    receipt = json.loads((ROOT / "artifacts" / "release" / "v0_4_17_release_receipt.json").read_text())
    assert receipt["schema"] == "jaxfne.release_receipt.v4"
    assert receipt["core_candidate"] == "0ff37e40375e1c76d07f354803dccffbabb9d3a6"
    assert receipt["release_candidate"] == "867398e9d5c1d8812736369a6599604a42a296ce"
    assert "02f8a4bb152a811ce2a57099a1439876025214de3032a1540bda69714f96e9b0" in receipt["distribution"]["wheel"]
    assert "87146f94191c853428a80be2606ae70fc6278d9ecb982750ac04190aaeafcb2c" in receipt["distribution"]["sdist"]
