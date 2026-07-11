"""Packaging markers for downstream type checkers."""

from pathlib import Path


def test_py_typed_marker_present() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "jaxfne" / "py.typed").is_file()
