"""P1 build contract: --out X.pdf must always produce PDF, scientific captions, DOI validator rename."""

from __future__ import annotations

import subprocess
from pathlib import Path

import scripts.build_manuscript as bm


def test_backend_choices_deterministic():
    assert bm.build.__code__.co_varnames[1] == "backend"
    # Direct check: build rejects unknown backend
    try:
        bm.build(Path("/tmp/should_fail.pdf"), backend="bogus")
        assert False, "unknown backend should raise SystemExit"
    except SystemExit as e:
        assert "unknown backend" in str(e)


def test_validate_doi_syntax_renamed_and_alias():
    assert hasattr(bm, "validate_doi_syntax"), "validate_doi_syntax missing (rename required)"
    assert hasattr(bm, "verify_dois"), "verify_dois alias missing for backward compat"
    # Both should be callable and equivalent (regex-only syntax check)
    bm.validate_doi_syntax()
    bm.verify_dois()


def test_inject_scientific_captions_no_audit_language(tmp_path):
    md = "Intro Fig. 1 text.\n\nMore Fig. 5 text.\n\nAlso Fig. 2 Fig. 6 Fig. 7."
    out = bm.inject(md)
    assert "PASSED" not in out, "audit PASSED leaked into caption"
    assert "SHA" not in out, "audit SHA leaked into caption"
    # Scientific captions expected
    assert "TFNE grammar" in out, "Figure 1 scientific caption missing"
    assert "traveling" in out.lower() or "Traveling" in out, "Figure 5 scientific caption missing"
    assert "RBS" in out or "HDP" in out, "Figure 6 scientific caption missing"
    assert "E-integration" in out or "integration" in out.lower(), "Figure 7 scientific caption missing"
    # Figures 2-4 shared caption
    assert "Canonical source" in out or "Figures 2" in out


def test_build_reportlab_always_pdf(tmp_path):
    out = tmp_path / "manuscript.pdf"
    res = bm.build(out, backend="reportlab")
    assert res.suffix == ".pdf", f"reportlab backend returned {res.suffix}, expected .pdf"
    assert res.exists() and res.stat().st_size > 0
    # Content is PDF (starts with %PDF)
    assert res.read_bytes()[:4] == b"%PDF"


def test_build_auto_always_pdf_without_pandoc(monkeypatch, tmp_path):
    # Simulate pandoc absent
    monkeypatch.setattr(bm.shutil, "which", lambda x: None)
    out = tmp_path / "auto_no_pandoc.pdf"
    res = bm.build(out, backend="auto")
    assert res.suffix == ".pdf"
    assert res.exists()
    assert res.read_bytes()[:4] == b"%PDF"


def test_build_auto_always_pdf_with_mock_pandoc_no_pdf_engine(monkeypatch, tmp_path):
    """Reproduced defect: pandoc present but no pdf engine must not return .tex when .pdf requested."""
    monkeypatch.setattr(bm.shutil, "which", lambda cmd: "/usr/bin/pandoc" if cmd == "pandoc" else None)

    orig_run = bm.subprocess.run

    def fake_run(cmd, check=False, capture_output=False, timeout=None, **kw):
        if isinstance(cmd, list) and cmd and cmd[0] == "pandoc":
            out_path = Path(cmd[3])
            if out_path.suffix == ".pdf":
                # Simulate missing pdf engine: do NOT create file
                return subprocess.CompletedProcess(cmd, 0)
            if out_path.suffix in (".tex",) or out_path.name.endswith(".tmp.tex"):
                out_path.write_text("fake tex")
                return subprocess.CompletedProcess(cmd, 0)
        return orig_run(cmd, check=check, capture_output=capture_output, timeout=timeout, **kw)

    monkeypatch.setattr(bm.subprocess, "run", fake_run)
    out = tmp_path / "audit-manuscript.pdf"
    res = bm.build(out, backend="auto")
    assert res.suffix == ".pdf", f"auto with mock pandoc returned {res.suffix}, expected .pdf (defect: returned .tex)"
    assert res.exists()
    assert res.read_bytes()[:4] == b"%PDF"
    # Ensure we did NOT return the intermediate tex as final output
    assert res.name == "audit-manuscript.pdf"


def test_build_pandoc_backend_fallback_to_reportlab_when_no_pdf_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(bm.shutil, "which", lambda cmd: "/usr/bin/pandoc" if cmd == "pandoc" else None)
    orig_run = bm.subprocess.run

    def fake_run(cmd, check=False, capture_output=False, timeout=None, **kw):
        if isinstance(cmd, list) and cmd and cmd[0] == "pandoc":
            out_path = Path(cmd[3])
            if out_path.suffix == ".pdf":
                return subprocess.CompletedProcess(cmd, 0)
            out_path.write_text("fake tex")
            return subprocess.CompletedProcess(cmd, 0)
        return orig_run(cmd, check=check, capture_output=capture_output, timeout=timeout, **kw)

    monkeypatch.setattr(bm.subprocess, "run", fake_run)
    out = tmp_path / "pandoc_fallback.pdf"
    res = bm.build(out, backend="pandoc")
    assert res.suffix == ".pdf"
    assert res.read_bytes()[:4] == b"%PDF"


def test_build_tex_explicit_honored_with_pandoc(monkeypatch, tmp_path):
    monkeypatch.setattr(bm.shutil, "which", lambda cmd: "/usr/bin/pandoc" if cmd == "pandoc" else None)
    orig_run = bm.subprocess.run

    def fake_run(cmd, check=False, capture_output=False, timeout=None, **kw):
        if isinstance(cmd, list) and cmd and cmd[0] == "pandoc":
            out_path = Path(cmd[3])
            out_path.write_text("fake tex")
            return subprocess.CompletedProcess(cmd, 0)
        return orig_run(cmd, check=check, capture_output=capture_output, timeout=timeout, **kw)

    monkeypatch.setattr(bm.subprocess, "run", fake_run)
    out = tmp_path / "explicit.tex"
    res = bm.build(out, backend="pandoc")
    assert res.suffix == ".tex"
    assert res.exists()
