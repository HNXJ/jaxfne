"""Regression: harness boundary audit must read text files as UTF-8.

Seen on Windows under a cp1252 locale: ``audit_public_docs()`` crashed with
``UnicodeDecodeError`` (byte 0x9d, the trailing byte of U+201D RIGHT DOUBLE
QUOTATION MARK in UTF-8) because it called ``Path.read_text()`` without an
explicit encoding.
"""

import locale

import scripts.harness.audit_public_private_boundary as boundary


def test_audit_public_docs_reads_utf8_under_non_utf8_locale(tmp_path, monkeypatch):
    """A UTF-8 docs page must audit cleanly even when the locale is not UTF-8."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "page.md").write_bytes("curly quote: \u201d\n".encode("utf-8"))
    (tmp_path / "mkdocs.yml").write_text("nav:\n  - Home: page.md\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "ROOT", tmp_path)
    monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "cp1252")
    if hasattr(locale, "getencoding"):
        monkeypatch.setattr("locale.getencoding", lambda *a, **k: "cp1252")
    assert boundary.audit_public_docs() == []
