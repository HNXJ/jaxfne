"""Tests for scripts/release/ — Phase 0 release-control gates.

These tests verify that release scripts correctly enforce their safety
invariants without making any remote mutations.
"""

import json
import subprocess
import sys
import os
import tempfile
import tarfile
import zipfile
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "release"
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(script_name: str, args: list[str], extra_env: dict | None = None):
    """Run a release script and return (exit_code, stdout, stderr)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def run_bash_script(script_name: str, args: list[str]):
    """Run a shell script and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        ["bash", str(SCRIPTS_DIR / script_name), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


class TestReconcileReleaseTarget:
    """Tests for reconcile_release_target.py."""

    def test_requires_version_arg(self):
        """Fails without --version."""
        code, _, _ = run_script("reconcile_release_target.py", [])
        assert code != 0

    def test_requires_target_sha_arg(self):
        """Fails without --target-sha."""
        code, _, _ = run_script("reconcile_release_target.py", ["--version", "0.3.14"])
        assert code != 0

    def test_produces_json_with_required_fields(self):
        """Output JSON contains all required fields."""
        code, stdout, _ = run_script(
            "reconcile_release_target.py",
            ["--version", "0.3.14", "--target-sha", "a" * 40],
        )
        data = json.loads(stdout)
        required_fields = [
            "version",
            "intended_release_sha",
            "origin_main_sha",
            "local_head_sha",
            "ci_head_sha",
            "ci_conclusion",
            "tag_object_sha",
            "tag_peeled_sha",
            "working_tree_clean",
            "release_target_reconciled",
            "safe_to_repair_tag",
            "safe_to_upload",
            "failure_reasons",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_false_on_sha_mismatch(self):
        """release_target_reconciled=false when intended SHA differs from origin/main."""
        # Provide a nonsense SHA — it will never match origin/main
        fake_sha = "deadbeef" * 5
        code, stdout, _ = run_script(
            "reconcile_release_target.py",
            ["--version", "0.3.14", "--target-sha", fake_sha],
        )
        data = json.loads(stdout)
        assert data["release_target_reconciled"] is False
        assert len(data["failure_reasons"]) > 0
        assert code == 1

    def test_safe_to_upload_always_false(self):
        """safe_to_upload is always False from this script (no upload authority)."""
        code, stdout, _ = run_script(
            "reconcile_release_target.py",
            ["--version", "0.3.14", "--target-sha", "a" * 40],
        )
        data = json.loads(stdout)
        assert data["safe_to_upload"] is False

    def test_safe_to_repair_tag_always_false(self):
        """safe_to_repair_tag is always False from this script."""
        code, stdout, _ = run_script(
            "reconcile_release_target.py",
            ["--version", "0.3.14", "--target-sha", "a" * 40],
        )
        data = json.loads(stdout)
        assert data["safe_to_repair_tag"] is False

    def test_no_ci_run_id_reports_failure_reason(self):
        """Without --ci-run-id, ci_head_sha unknown and failure_reasons includes note."""
        code, stdout, _ = run_script(
            "reconcile_release_target.py",
            ["--version", "0.3.14", "--target-sha", "a" * 40],
        )
        data = json.loads(stdout)
        reasons = data["failure_reasons"]
        assert any("ci-run-id" in r.lower() or "ci_head_sha" in r.lower() for r in reasons)


class TestAssertReleaseFreeze:
    """Tests for assert_release_freeze.py."""

    def test_passes_when_no_freeze_declared(self):
        """Exits 0 and prints pass message when no freeze declaration found."""
        # Run from a temp dir with no AGENTS.md/lockfile
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text('[project]\nversion = "0.3.14"\n')
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "assert_release_freeze.py")],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 0
            assert "not active" in result.stdout.lower() or "pass" in result.stdout.lower()

    def test_reads_agents_md_freeze_true(self):
        """Exits 1 when AGENTS.md RELEASE_STATE block declares freeze=true and SHA mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=tmpdir,
                capture_output=True,
                env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
                     "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
            )
            # State must be inside the marked block; bare lines outside are ignored
            agents_content = (
                "# Agent Coordination\n\n"
                "<!-- RELEASE_STATE_BEGIN -->\n"
                "release_freeze: true\n"
                "intended_release_sha: " + "a" * 40 + "\n"
                "<!-- RELEASE_STATE_END -->\n"
            )
            Path(tmpdir, "AGENTS.md").write_text(agents_content)
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "assert_release_freeze.py")],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com"},
            )
            # Freeze active + SHA mismatch => exit 1
            assert result.returncode == 1

    def test_reads_lockfile_freeze_state(self):
        """Reads .release_target.json for freeze state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lockfile = {
                "release_freeze": True,
                "intended_release_sha": "b" * 40,
            }
            Path(tmpdir, ".release_target.json").write_text(json.dumps(lockfile))
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=tmpdir,
                capture_output=True,
                env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
                     "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "assert_release_freeze.py")],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                env={**os.environ},
            )
            # Freeze active + SHA mismatch => exit 1
            assert result.returncode == 1


class TestValidateReleaseArtifacts:
    """Tests for validate_release_artifacts.py."""

    def test_fails_missing_dist_normal_mode(self):
        """Exit 1 when dist/ absent in normal mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "validate_release_artifacts.py"),
                 "--version", "0.3.14"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 1
            assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_passes_missing_dist_allow_missing(self):
        """Exit 0 when dist/ absent and --allow-missing-dist set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "validate_release_artifacts.py"),
                 "--version", "0.3.14", "--allow-missing-dist"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 0
            assert "skipping" in result.stdout.lower() or "allow-missing" in result.stdout.lower()

    def test_fails_empty_dist_normal_mode(self):
        """Exit 1 when dist/ empty in normal mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "dist").mkdir()
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "validate_release_artifacts.py"),
                 "--version", "0.3.14"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 1

    def test_detects_wrong_wheel_version(self):
        """Exit 1 when wheel METADATA version does not match expected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir, "dist")
            dist_dir.mkdir()
            wheel_path = dist_dir / "jaxfne-9.9.9-py3-none-any.whl"
            with zipfile.ZipFile(wheel_path, "w") as zf:
                zf.writestr(
                    "jaxfne-9.9.9.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: jaxfne\nVersion: 9.9.9\n",
                )
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "validate_release_artifacts.py"),
                 "--version", "0.3.14"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 1
            assert "9.9.9" in result.stdout

    def test_passes_valid_wheel_and_sdist(self):
        """Exit 0 when both wheel and sdist have correct version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir, "dist")
            dist_dir.mkdir()

            # Minimal valid wheel
            wheel_path = dist_dir / "jaxfne-0.3.14-py3-none-any.whl"
            with zipfile.ZipFile(wheel_path, "w") as zf:
                zf.writestr(
                    "jaxfne-0.3.14.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: jaxfne\nVersion: 0.3.14\n",
                )

            # Minimal valid sdist
            sdist_path = dist_dir / "jaxfne-0.3.14.tar.gz"
            with tarfile.open(sdist_path, "w:gz") as tar:
                import io
                content = b'[project]\nversion = "0.3.14"\n'
                info = tarfile.TarInfo(name="jaxfne-0.3.14/pyproject.toml")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "validate_release_artifacts.py"),
                 "--version", "0.3.14"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            assert result.returncode == 0
            assert "valid" in result.stdout.lower()


class TestPrintTagReceipt:
    """Tests for print_tag_receipt.sh."""

    def test_exits_nonzero_missing_tag(self):
        """Exits nonzero when tag does not exist on origin and --allow-missing not set."""
        code, stdout, stderr = run_bash_script(
            "print_tag_receipt.sh",
            ["v99.99.99"],
        )
        assert code != 0

    def test_allow_missing_exits_zero(self):
        """Exits 0 when tag missing and --allow-missing is set."""
        code, stdout, stderr = run_bash_script(
            "print_tag_receipt.sh",
            ["v99.99.99", "--allow-missing"],
        )
        assert code == 0
        assert "allow-missing" in stdout.lower() or "skipping" in stdout.lower()

    def test_real_tag_prints_receipt(self):
        """Prints object and peeled SHAs for the existing v0.3.14 tag."""
        code, stdout, stderr = run_bash_script(
            "print_tag_receipt.sh",
            ["v0.3.14"],
        )
        assert code == 0
        assert "tag object SHA" in stdout or "peeled commit" in stdout
