"""
Hygiene test for Claude agent context.

Validates that durable agent context is properly maintained and protected.
"""

import json
from pathlib import Path


class TestAgentContextHygiene:
    """Validate Claude agent context structure and discipline."""

    def test_durable_context_exists(self):
        """Assert durable context file exists at canonical location."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        assert context_path.exists(), f"Durable context missing: {context_path}"

    def test_durable_context_not_empty(self):
        """Assert context file has substantive content."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')
        assert len(content) > 500, "Context file too short (< 500 chars)"

    def test_durable_context_has_key_sections(self):
        """Assert context includes key guidance sections."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8').lower()

        required_sections = [
            "project identity",
            "repository",
            "api contract",
            "public vs private",
            "public wording",
            "tutorial deliverable",
            "generated output",
            "validation",
            "failure mode",
            "always do",
            "never do"
        ]

        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_durable_context_mentions_canonical_api(self):
        """Assert context documents canonical Configuration API."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        api_terms = [
            "Configuration()",
            ".runtime(",
            ".column(",
            ".cell_types(",
            ".probes(",
            "jtfne.construct(",
            "jtfne.simulate(",
            "signals.field"
        ]

        for term in api_terms:
            assert term in content, f"Missing API reference: {term}"

    def test_durable_context_mentions_public_private_separation(self):
        """Assert context explains public/private surface boundaries."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        # Should mention docs, tutorials as public
        assert "docs" in content.lower(), "Missing public surface: docs"
        assert "tutorial" in content.lower(), "Missing public surface: tutorial"

        # Should mention internal_docs as private
        assert "internal_docs" in content.lower(), "Missing private surface: internal_docs"

    def test_durable_context_mentions_generated_output_policy(self):
        """Assert context documents generated output handling."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        assert "tutorial_outputs" in content, "Missing generated output location"
        assert "JAXFNE_VALIDATE_TUTORIAL_OUTPUTS" in content, "Missing artifact gate env var"

    def test_durable_context_mentions_report_contract(self):
        """Assert context specifies validation report requirements."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        report_items = ["SHA", "branch", "test", "report", "receipt"]
        for item in report_items:
            assert item.lower() in content.lower(), f"Report contract missing: {item}"

    def test_claude_local_is_not_tracked(self):
        """Assert .claude/ directory is not tracked by git."""
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", ".claude"],
            capture_output=True,
            text=True
        )

        # Should return empty (no tracked .claude files)
        assert result.stdout.strip() == "", ".claude/ files are tracked (should be ignored)"

    def test_claude_is_ignored(self):
        """Assert .claude/ is in .gitignore."""
        gitignore_path = Path(".gitignore")
        assert gitignore_path.exists(), ".gitignore not found"

        content = gitignore_path.read_text(encoding='utf-8')
        assert ".claude" in content, ".claude/ not in .gitignore"

    def test_durable_context_warns_against_common_mistakes(self):
        """Assert context documents known failure modes."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        mistake_keywords = [
            "jbiophysic",           # wrong repo
            "stale artifact",       # stale outputs
            "API contract",         # assumption risk
            "low-level kernel",     # wrong API layer
            "tutorial milestone",   # version confusion
            "public wording",       # language discipline
            "generated output",     # tracking discipline
        ]

        for keyword in mistake_keywords:
            assert keyword.lower() in content.lower(), f"Missing failure mode guidance: {keyword}"

    def test_durable_context_jaxfne_version_stable(self):
        """Assert context confirms package version remains 0.3.5."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        # Should mention version staying at 0.3.5
        assert "0.3.5" in content, "Missing version stability note"

    def test_no_secrets_in_context(self):
        """Assert context contains no API keys or credentials."""
        context_path = Path("internal_docs/agent_context/claude/CLAUDE.md")
        content = context_path.read_text(encoding='utf-8')

        secret_patterns = [
            "api_key",
            "password",
            "token",
            "secret",
            "credential",
            "private_key",
        ]

        for pattern in secret_patterns:
            assert pattern.lower() not in content.lower(), f"Potential secret in context: {pattern}"


class TestRepositoryStructure:
    """Validate core repository structure."""

    def test_canonical_repo_path(self):
        """Assert we're in the jaxfne repository."""
        repo_path = Path.cwd()
        assert (repo_path / "jaxfne").exists(), "jaxfne/ package directory not found"
        assert (repo_path / "tests").exists(), "tests/ directory not found"
        assert (repo_path / "docs").exists(), "docs/ directory not found"

    def test_internal_docs_exists(self):
        """Assert internal_docs directory exists."""
        assert Path("internal_docs").exists(), "internal_docs/ directory not found"

    def test_agent_context_directory_structure(self):
        """Assert agent context has expected structure."""
        context_dir = Path("internal_docs/agent_context/claude")
        assert context_dir.exists(), "agent_context/claude directory not found"
        assert (context_dir / "CLAUDE.md").exists(), "CLAUDE.md not found in context dir"
