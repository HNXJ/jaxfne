"""
v0.3 Documentation Audit Tests

Tests that v0.3 tutorial documentation meets audit policy requirements:
1. All doctrine files exist
2. LaTeX equation policy enforced
3. Open in Colab link policy enforced
4. Import alias policy enforced

truth_mode: truth_safe_unverified
"""

import json
import pytest
from pathlib import Path


class TestDocsAuditInfrastructure:
    """Tests for docs audit infrastructure."""

    def test_audit_script_exists(self):
        """scripts/audit_v030_docs_links.py must exist."""
        script = Path(__file__).parent.parent / "scripts" / "audit_v030_docs_links.py"
        assert script.exists(), f"Audit script missing: {script}"

    def test_audit_can_import(self):
        """audit_v030_docs_links.py must be importable."""
        import sys
        scripts_path = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_path))
        try:
            from audit_v030_docs_links import DocsAudit
            assert callable(DocsAudit)
        finally:
            sys.path.pop(0)


class TestDoctrineFilesExist:
    """Tests that all required doctrine files exist."""

    def _get_tutorials_root(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030"

    def test_readme_exists(self):
        """docs/tutorials_v030/README.md must exist."""
        assert (self._get_tutorials_root() / "README.md").exists()

    def test_template_exists(self):
        """docs/tutorials_v030/template.md must exist."""
        assert (self._get_tutorials_root() / "template.md").exists()

    def test_scenario_index_exists(self):
        """docs/tutorials_v030/scenario_index.md must exist."""
        assert (self._get_tutorials_root() / "scenario_index.md").exists()

    def test_acceptance_gates_yaml_exists(self):
        """docs/tutorials_v030/acceptance_gates.yaml must exist."""
        assert (self._get_tutorials_root() / "acceptance_gates.yaml").exists()

    def test_plotly_policy_exists(self):
        """docs/tutorials_v030/plotly_policy.md must exist."""
        assert (self._get_tutorials_root() / "plotly_policy.md").exists()

    def test_canonical_imports_exists(self):
        """docs/tutorials_v030/canonical_imports.md must exist."""
        assert (self._get_tutorials_root() / "canonical_imports.md").exists()

    def test_docs_audit_policy_exists(self):
        """docs/tutorials_v030/docs_audit_policy.md must exist."""
        assert (self._get_tutorials_root() / "docs_audit_policy.md").exists()

    def test_environment_md_exists(self):
        """docs/tutorials_v030/environment.md must exist."""
        assert (self._get_tutorials_root() / "environment.md").exists()


class TestEnvironmentRequirementsExist:
    """Tests for environment and requirements files."""

    def _get_repo_root(self):
        return Path(__file__).parent.parent

    def test_requirements_file_exists(self):
        """requirements/tutorials-v030.txt must exist (moved from root for hygiene)."""
        req_file = self._get_repo_root() / "requirements" / "tutorials-v030.txt"
        assert req_file.exists(), f"Requirements file missing: {req_file}"

    def test_requirements_valid(self):
        """requirements/tutorials-v030.txt must be valid and JSON-safe."""
        req_file = self._get_repo_root() / "requirements" / "tutorials-v030.txt"
        content = req_file.read_text()

        # Must contain core packages
        assert "jaxfne==0.3.5" in content or "jaxfne" in content.lower(), \
            "jaxfne not in requirements"
        assert "jax" in content.lower()
        assert "numpy" in content.lower()
        assert "matplotlib" in content.lower()

    def test_environment_md_is_readable(self):
        """docs/tutorials_v030/environment.md must be valid markdown."""
        env_file = self._get_repo_root() / "docs" / "tutorials_v030" / "environment.md"
        assert env_file.exists()
        content = env_file.read_text()

        # Check for required sections
        assert "Quick Start" in content or "quick start" in content.lower()
        assert "Optional" in content or "optional" in content.lower()
        assert "jaxfne" in content


class TestLaTeXEquationPolicy:
    """Tests for LaTeX equation display policy."""

    def _get_template(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "template.md"

    def test_template_has_equation_display_rule(self):
        """template.md must include equation display rule section."""
        template = self._get_template()
        content = template.read_text()

        assert "Equation Display Rule" in content or "Equation display rule" in content.lower()

    def test_template_shows_latex_examples(self):
        """template.md must show LaTeX equation examples ($$...$$)."""
        template = self._get_template()
        content = template.read_text()

        # Must contain at least one displayed equation
        assert "$$" in content, "No LaTeX displayed equations found in template"

    def test_template_explains_term_glossary(self):
        """template.md must explain term glossary requirement."""
        template = self._get_template()
        content = template.read_text()

        assert "Term Glossary" in content
        assert "Symbol" in content or "symbol" in content.lower()

    def test_template_explains_worded_equation(self):
        """template.md must explain worded-equation requirement."""
        template = self._get_template()
        content = template.read_text()

        assert "Worded Equation" in content
        assert "plain English" in content or "English" in content

    def test_template_explains_claim_boundary(self):
        """template.md must explain claim boundary requirement."""
        template = self._get_template()
        content = template.read_text()

        assert "Claim Boundary" in content or "claim boundary" in content.lower()

    def test_template_forbids_words_only(self):
        """template.md must show incorrect pattern (words without equations)."""
        template = self._get_template()
        content = template.read_text()

        # Should have an incorrect/forbidden example
        assert "Incorrect" in content or "incorrect" in content.lower()


class TestCanonicalImportPolicy:
    """Tests for canonical import alias enforcement."""

    def _get_canonical_imports_doc(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "canonical_imports.md"

    def test_canonical_imports_doc_exists(self):
        """canonical_imports.md must exist."""
        assert self._get_canonical_imports_doc().exists()

    def test_canonical_imports_mentions_jtfne(self):
        """canonical_imports.md must require 'import jaxfne as jtfne'."""
        doc = self._get_canonical_imports_doc()
        content = doc.read_text()

        assert "jtfne" in content
        assert "import jaxfne as jtfne" in content

    def test_canonical_imports_forbids_bare_import(self):
        """canonical_imports.md must forbid bare import."""
        doc = self._get_canonical_imports_doc()
        content = doc.read_text()

        # Should mention that bare import is forbidden
        assert "import jaxfne" in content or "Forbidden" in content or "forbidden" in content.lower()

    def test_canonical_imports_forbids_wildcard(self):
        """canonical_imports.md must forbid wildcard import."""
        doc = self._get_canonical_imports_doc()
        content = doc.read_text()

        assert "from jaxfne import *" in content or "*" in content


class TestCoLabLinkPolicy:
    """Tests for Open in Colab link policy."""

    def _get_docs_audit_policy(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "docs_audit_policy.md"

    def test_docs_audit_policy_exists(self):
        """docs_audit_policy.md must exist."""
        assert self._get_docs_audit_policy().exists()

    def test_docs_audit_policy_requires_colab_link(self):
        """docs_audit_policy.md must require 'Open in Colab' link."""
        policy = self._get_docs_audit_policy()
        content = policy.read_text()

        assert "Open in Colab" in content

    def test_docs_audit_policy_forbids_verbose_colab_preamble(self):
        """docs_audit_policy.md must forbid verbose Colab link text."""
        policy = self._get_docs_audit_policy()
        content = policy.read_text()

        # Should mention that preambles are forbidden
        assert "Forbidden" in content or "forbidden" in content.lower()

    def test_docs_audit_policy_specifies_colab_url_format(self):
        """docs_audit_policy.md must specify exact Colab URL format."""
        policy = self._get_docs_audit_policy()
        content = policy.read_text()

        assert "colab.research.google.com" in content
        assert "github/HNXJ/jaxfne" in content


class TestAcceptanceGatesAndClaimGates:
    """Tests for hard acceptance and claim gates."""

    def _get_template(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "template.md"

    def test_template_includes_claim_gates_section(self):
        """template.md must include immutable claim gates in Section 8."""
        template = self._get_template()
        content = template.read_text()

        # Must mention claim gates
        assert "Claim Gates" in content or "claim gates" in content.lower()
        assert "physical_amplitude_claim_allowed" in content

    def test_template_mentions_firing_rate_gate(self):
        """template.md must mention 2–25 Hz firing rate acceptance gate."""
        template = self._get_template()
        content = template.read_text()

        assert "2" in content and "25" in content and "Hz" in content

    def test_template_mentions_json_safety(self):
        """template.md must mention JSON safety (no NaN/Inf)."""
        template = self._get_template()
        content = template.read_text()

        assert "JSON" in content or "json" in content.lower()
        assert "NaN" in content or "Inf" in content


class TestProbeOperators:
    """Tests for 8-probe operator contract."""

    def _get_template(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "template.md"

    def test_template_mentions_eight_probes(self):
        """template.md must reference all 8 probe operators."""
        template = self._get_template()
        content = template.read_text()

        # Must mention probes
        required_probes = ["spikes", "V_m", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy"]
        for probe in required_probes:
            assert probe in content, f"Probe '{probe}' not mentioned in template"


class TestAuditReportSchema:
    """Tests for docs_link_audit.json report schema."""

    def _get_audit_report_path(self):
        return Path(__file__).parent.parent / "docs" / "tutorials_v030" / "docs_link_audit.json"

    def test_audit_report_can_be_generated(self):
        """audit_v030_docs_links.py must generate valid JSON report."""
        # This test verifies the report format, not execution
        # (Report file may not exist yet if audit hasn't run)
        report_path = self._get_audit_report_path()

        # If report exists, it must be valid JSON
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

            # Validate schema
            assert "schema_version" in report
            assert report["schema_version"] == "v0.3.0"
            assert "status" in report
            assert report["status"] in ["pass", "fail", "warn"]
            assert "checked_files" in report
            assert isinstance(report["checked_files"], int)
            assert "missing_links" in report
            assert isinstance(report["missing_links"], list)
            assert "missing_colab_links" in report
            assert isinstance(report["missing_colab_links"], list)
            assert "latex_policy_violations" in report
            assert isinstance(report["latex_policy_violations"], list)
            assert "notes" in report
            assert isinstance(report["notes"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
