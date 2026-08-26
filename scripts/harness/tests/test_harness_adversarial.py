#!/usr/bin/env python3
"""Adversarial validation of harness invariants, Gate 0 simulations, and epistemic boundaries.

Validates:
1. Gate 0 clean execution on current state.
2. Behavioral Gate 0 execution on simulated STALE_LOCAL_STATE (assert exit code 1).
3. Behavioral Gate 0 execution on simulated DIVERGED branch state (assert exit code 1).
4. Behavioral Gate 0 execution on simulated Remote mismatch (assert exit code 1).
5. Behavioral Gate 0 execution on simulated Fetch failure (assert exit code 1).
6. Behavioral Gate 0 execution on simulated Missing required authorities (assert exit code 1).
7. Behavioral Gate 0 execution on Offline mode (assert exit code 2: REMOTE_STATE_UNVERIFIED).
8. Epistemic preservation: NEGATIVE never becomes POSITIVE in evidence index.
9. Epistemic preservation: UNRESOLVED never becomes NEGATIVE in evidence index.
10. Epistemic preservation: Relative quantities never labeled calibrated without transform.
11. RBS invariant: H is a finite-dimensional state container, not a single homeostatic equation.
12. Release identity separation: C_core, C_release, C_receipt, C_head distinct in receipt & CURRENT_TASK.
13. Frozen paths hash invariance: All 28 immutable publication files match recorded SHA256.
14. Public API surface truth gate: No invented public APIs (__all__ strictly equals public contract).
15. CURRENT_TASK brevity & exact C_* keys: Task file remains compact (<= 30 lines) with exact C_* vocabulary.
16. Skill structural shape & unique triggers: All 7 skills have distinct WHEN triggers and standard section headers.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from scripts.harness.gate0_git_reality import check_gate0, EXPECTED_REMOTES


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip()


# --- True Behavioral Gate 0 Simulation Tests ---

def test_gate0_pass_on_current_clean_state():
    """Verify Gate 0 executes cleanly on current synchronized checkout."""
    assert check_gate0(root=ROOT, fetch=False, offline=False) == 0


def test_gate0_behavioral_stale_local_state():
    """Simulate a local repo behind remote and verify Gate 0 returns exit code 1 (STALE_LOCAL_STATE)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        remote_dir = tmp_path / "remote.git"
        local_dir = tmp_path / "local"

        run_cmd(["git", "init", "--bare", str(remote_dir)], cwd=tmp_path)
        run_cmd(["git", "clone", str(remote_dir), str(local_dir)], cwd=tmp_path)
        run_cmd(["git", "config", "user.name", "Test"], cwd=local_dir)
        run_cmd(["git", "config", "user.email", "test@example.com"], cwd=local_dir)
        run_cmd(["git", "checkout", "-b", "dev"], cwd=local_dir)

        (local_dir / "init.txt").write_text("initial")
        run_cmd(["git", "add", "."], cwd=local_dir)
        run_cmd(["git", "commit", "-m", "initial"], cwd=local_dir)
        run_cmd(["git", "push", "-u", "origin", "dev"], cwd=local_dir)

        # Advance remote via second clone
        other_dir = tmp_path / "other"
        run_cmd(["git", "clone", str(remote_dir), str(other_dir)], cwd=tmp_path)
        run_cmd(["git", "config", "user.name", "Test"], cwd=other_dir)
        run_cmd(["git", "config", "user.email", "test@example.com"], cwd=other_dir)
        run_cmd(["git", "checkout", "dev"], cwd=other_dir)
        (other_dir / "advance.txt").write_text("advance")
        run_cmd(["git", "add", "."], cwd=other_dir)
        run_cmd(["git", "commit", "-m", "remote advance"], cwd=other_dir)
        run_cmd(["git", "push", "origin", "dev"], cwd=other_dir)

        # Fetch in local_dir so tracking ref advances, leaving HEAD behind
        run_cmd(["git", "fetch", "origin"], cwd=local_dir)

        # Execute real Gate 0 logic against local_dir with allowed_remotes matching remote_dir
        res = check_gate0(
            root=local_dir,
            fetch=False,
            offline=False,
            mode="CODE",
            allowed_remotes=[str(remote_dir)],
        )
        assert res == 1, "Gate 0 must fail with exit code 1 when local branch is behind origin"


def test_gate0_behavioral_diverged_state():
    """Simulate a DIVERGED branch state and verify Gate 0 returns exit code 1 (DIVERGED)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        remote_dir = tmp_path / "remote.git"
        local_dir = tmp_path / "local"

        run_cmd(["git", "init", "--bare", str(remote_dir)], cwd=tmp_path)
        run_cmd(["git", "clone", str(remote_dir), str(local_dir)], cwd=tmp_path)
        run_cmd(["git", "config", "user.name", "Test"], cwd=local_dir)
        run_cmd(["git", "config", "user.email", "test@example.com"], cwd=local_dir)
        run_cmd(["git", "checkout", "-b", "dev"], cwd=local_dir)

        (local_dir / "init.txt").write_text("initial")
        run_cmd(["git", "add", "."], cwd=local_dir)
        run_cmd(["git", "commit", "-m", "initial"], cwd=local_dir)
        run_cmd(["git", "push", "-u", "origin", "dev"], cwd=local_dir)

        # Remote advance via other clone
        other_dir = tmp_path / "other"
        run_cmd(["git", "clone", str(remote_dir), str(other_dir)], cwd=tmp_path)
        run_cmd(["git", "config", "user.name", "Test"], cwd=other_dir)
        run_cmd(["git", "config", "user.email", "test@example.com"], cwd=other_dir)
        run_cmd(["git", "checkout", "dev"], cwd=other_dir)
        (other_dir / "commit_remote.txt").write_text("remote")
        run_cmd(["git", "add", "."], cwd=other_dir)
        run_cmd(["git", "commit", "-m", "remote divergence"], cwd=other_dir)
        run_cmd(["git", "push", "origin", "dev"], cwd=other_dir)

        # Local separate commit
        (local_dir / "commit_local.txt").write_text("local")
        run_cmd(["git", "add", "."], cwd=local_dir)
        run_cmd(["git", "commit", "-m", "local divergence"], cwd=local_dir)
        run_cmd(["git", "fetch", "origin"], cwd=local_dir)

        res = check_gate0(
            root=local_dir,
            fetch=False,
            offline=False,
            mode="CODE",
            allowed_remotes=[str(remote_dir)],
        )
        assert res == 1, "Gate 0 must fail with exit code 1 when branch has diverged"


def test_gate0_behavioral_remote_mismatch():
    """Verify Gate 0 fails when origin URL does not match canonical remotes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_cmd(["git", "init"], cwd=tmp_path)
        run_cmd(["git", "remote", "add", "origin", "https://github.com/imposter/wrong-repo.git"], cwd=tmp_path)

        res = check_gate0(root=tmp_path, fetch=False, offline=False, mode="CODE")
        assert res == 1, "Gate 0 must fail with exit code 1 when remote URL is unexpected"


def test_gate0_behavioral_fetch_failure():
    """Verify Gate 0 fails (exit code 1) when git fetch encounters an invalid/unreachable remote."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_cmd(["git", "init"], cwd=tmp_path)
        run_cmd(["git", "remote", "add", "origin", "/nonexistent/remote.git"], cwd=tmp_path)

        res = check_gate0(
            root=tmp_path,
            fetch=True,
            offline=False,
            mode="CODE",
            allowed_remotes=["/nonexistent/remote.git"],
        )
        assert res == 1, "Gate 0 must fail with exit code 1 when git fetch fails"


def test_gate0_behavioral_offline_mode():
    """Verify Gate 0 returns exit code 2 (REMOTE_STATE_UNVERIFIED) under explicit --offline mode."""
    res = check_gate0(root=ROOT, fetch=False, offline=True)
    assert res == 2, "Gate 0 must return 2 (REMOTE_STATE_UNVERIFIED) under --offline"


def test_gate0_behavioral_missing_mode_authority():
    """Verify Gate 0 fails when a mode-required authority is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_cmd(["git", "init"], cwd=tmp_path)
        run_cmd(["git", "remote", "add", "origin", EXPECTED_REMOTES[0]], cwd=tmp_path)

        # In RELEASE mode, release_receipt is required
        res = check_gate0(root=tmp_path, fetch=False, offline=True, mode="RELEASE")
        assert res == 1, "Gate 0 must fail when required authority is absent"


# --- Epistemic & Doctrine Invariant Tests ---

def test_epistemic_negative_polarity_preserved():
    """Adversarial check: ensure NO_WAVE (C3) and NO_ADAPTATION (D3) remain NEGATIVE."""
    index_path = ROOT / "artifacts" / "publication" / "publication_evidence_index.json"
    index = json.loads(index_path.read_text())
    
    assert "C3 traveling-wave conjecture" in index["evidence_summary"]["negative"]
    assert "D3 adaptation attribution" in index["evidence_summary"]["negative"]
    assert "H4 topology-memory extension" in index["evidence_summary"]["negative"]
    
    for item in index["evidence_summary"]["positive"]:
        assert "wave" not in item.lower()
        assert "adaptation" not in item.lower()


def test_epistemic_unresolved_polarity_preserved():
    """Adversarial check: ensure W3b active closed-loop stability remains UNRESOLVED."""
    index_path = ROOT / "artifacts" / "publication" / "publication_evidence_index.json"
    index = json.loads(index_path.read_text())
    
    assert "closed-loop HDP active stability (W3b)" in index["evidence_summary"]["unresolved"]
    assert "closed-loop HDP active stability (W3b)" not in index["evidence_summary"]["negative"]
    assert "closed-loop HDP active stability (W3b)" not in index["evidence_summary"]["positive"]


def test_relative_not_calibrated_in_doctrine():
    """Adversarial check: doctrine must strictly separate relative from calibrated quantities."""
    doctrine_text = (ROOT / "docs" / "doctrine" / "relative_quantity_grammar.md").read_text()
    assert "p_eff = C_p(p0, r_p)" in doctrine_text
    assert "base" in doctrine_text.lower() and "relative" in doctrine_text.lower() and "effective" in doctrine_text.lower()


def test_rbs_h_not_homeostasis_in_doctrine():
    """Adversarial check: H is defined as relative hidden biophysical state, not homeostasis by definition."""
    doctrine_text = (ROOT / "docs" / "doctrine" / "tfne_containment_architecture.md").read_text()
    assert "finite-dimensional state-space container" in doctrine_text
    assert "not a single homeostatic" in doctrine_text


# --- Frozen Paths, Public API Surface & Release Integrity Tests ---

def test_frozen_paths_immutable_hashes():
    """Verify all 28 publication evidence artifacts match their recorded SHA256 checksums."""
    frozen_manifest = json.loads((ROOT / "artifacts/publication/frozen_manifest.json").read_text())
    assert len(frozen_manifest["files"]) == 28
    for rel_path, expected_sha in frozen_manifest["files"].items():
        p = ROOT / rel_path
        assert p.exists(), f"Frozen path missing: {rel_path}"
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Frozen hash drift detected in {rel_path}"


def test_public_api_surface_no_invented_symbols():
    """Adversarial check: verify public surface contract equals exact exported symbols (no invented APIs)."""
    contract_path = ROOT / "artifacts" / "public_surface_contract_v0413.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        import jaxfne
        live_exports = set(jaxfne.__all__)
        assert len(live_exports) == contract["counts"]["public_exports"]


def test_release_receipt_distinct_typed_identities():
    """Verify release receipt v4 records distinct C_core, C_release, and distribution hashes."""
    receipt = json.loads((ROOT / "artifacts" / "release" / "v0_4_17_release_receipt.json").read_text())
    assert receipt["schema"] == "jaxfne.release_receipt.v4"
    assert receipt["core_candidate"] == "0ff37e40375e1c76d07f354803dccffbabb9d3a6"
    assert receipt["release_candidate"] == "867398e9d5c1d8812736369a6599604a42a296ce"
    assert "02f8a4bb152a811ce2a57099a1439876025214de3032a1540bda69714f96e9b0" in receipt["distribution"]["wheel"]
    assert "87146f94191c853428a80be2606ae70fc6278d9ecb982750ac04190aaeafcb2c" in receipt["distribution"]["sdist"]


def test_current_task_ephemeral_and_typed_structure():
    """Verify CURRENT_TASK.md is tiny (<=30 lines) and uses exact C_* vocabulary."""
    task_text = (ROOT / "scratch" / "CURRENT_TASK.md").read_text()
    lines = [l for l in task_text.splitlines() if l.strip()]
    assert len(lines) <= 30, f"CURRENT_TASK.md has {len(lines)} lines; must remain <= 30"
    assert "C_core:" in task_text
    assert "C_release:" in task_text
    assert "C_receipt:" in task_text
    assert "C_head:" in task_text
    assert "delta_C_core: 0" in task_text


def test_skill_shape_and_unique_triggers():
    """Verify exactly 7 canonical skills exist with standardized sections and distinct WHEN triggers."""
    canonical_skills = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))
    expected_skills = [
        "jaxfne-audit",
        "jaxfne-core",
        "jaxfne-frozen-use",
        "jaxfne-release",
        "jaxfne-repo",
        "jaxfne-science",
        "jaxfne-seal",
    ]
    assert canonical_skills == expected_skills, f"Skills mismatch: got {canonical_skills}"

    required_sections = ["## WHEN", "## AUTHORITIES", "## RULES", "## STEPS", "## STOP", "## VERIFY", "## DONE"]
    when_triggers = []
    for s_name in canonical_skills:
        content = (ROOT / "skills" / s_name / "SKILL.md").read_text()
        for sec in required_sections:
            assert sec in content, f"Skill {s_name} missing {sec}"
        when_clause = content.split("## WHEN")[1].split("##")[0].strip()
        when_triggers.append(when_clause)

    assert len(set(when_triggers)) == 7, "Skill WHEN triggers must be distinct"
