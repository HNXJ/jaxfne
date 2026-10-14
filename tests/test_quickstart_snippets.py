"""Executable check for every Python block in docs/quickstart.md.

Each ```python block is extracted in document order and exec'd in one shared,
fresh namespace (reader order, so later blocks reuse earlier names like
``model``/``cfg``/``signals`` exactly as a reader running the page top to
bottom would). A block that needs a long simulation, a browser, or files is
shrunk only by string substitution from _OVERRIDES (per-block, with reasons).
The gate runs the shrunk sizes only; block 6 was run once at its documented
n = 2000 (construct 8.6 s, 2026-10-13). A block whose optional dependency is not
installed is reported with a warning; CI installs every extra these blocks use.
"""

import importlib.util
import re
import warnings
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"

# Maps python-block index -> [(old, new)] substitutions. Each `old` must occur
# at least once in its block (asserted), so a doc edit that removes the anchor
# fails loudly instead of silently unshrinking.
# Reasons: full-size runs (n=1000/2000, 1000 ms, 50 AGSDR steps) take minutes;
# the shrunk values exercise the same construct -> simulate -> readout path.
_OVERRIDES: dict[int, list[tuple[str, str, str]]] = {
    2: [("duration_ms=1000.0", "duration_ms=20.0",
         "1000 ms reference run too slow for a gate; 20 ms covers the same path")],
    3: [("n_neurons=100", "n_neurons=20", "smaller inline circuit, same API"),
         ("duration_ms=1000.0", "duration_ms=20.0",
          "1000 ms run too slow for a gate; 20 ms covers the same path")],
    4: [("duration_ms=1000.0", "duration_ms=20.0",
          "1000 ms HDP run too slow for a gate; 20 ms covers the same path")],
    5: [("n=1000", "n=60", "smaller column, same builder path"),
         ("duration_ms=1000.0", "duration_ms=20.0",
          "1000 ms run too slow for a gate; 20 ms covers the same path")],
    6: [("n=2000", "n=80", "smaller column, same connections/homeostasis path"),
         ("duration_ms=1000.0", "duration_ms=20.0",
          "1000 ms run too slow for a gate; 20 ms covers the same path")],
    7: [("n=1000", "n=60", "smaller column, same 3D-view path"),
         ('output_html="network3d.html"', 'output_html={HTML_OUT}',
          "write the panel under tmp_path instead of the repo tree")],
    8: [("steps=50", "steps=2",
          "50 AGSDR candidates too slow for a gate; 2 exercise the tune loop")],
}

_EXPECTED_BLOCKS = 9

# Block index -> optional module the block imports (the `jaxley` extra).
_OPTIONAL = {1: "jaxley"}


def _python_blocks() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    return re.findall(r"^```python[^\n]*\r?\n(.*?)\r?\n```", text, re.DOTALL | re.MULTILINE)


def test_quickstart_block_count():
    assert len(_python_blocks()) == _EXPECTED_BLOCKS, (
        f"docs/quickstart.md holds {len(_python_blocks())} python blocks; "
        "update _OVERRIDES/_EXPECTED_BLOCKS alongside the doc edit"
    )


def test_quickstart_snippets(tmp_path):
    blocks = _python_blocks()
    assert len(blocks) == _EXPECTED_BLOCKS
    html_out = str(tmp_path / "network3d.html").replace("\\", "/")
    ns: dict = {}
    for idx, block in enumerate(blocks):
        module = _OPTIONAL.get(idx)
        if module and importlib.util.find_spec(module) is None:
            warnings.warn(f"quickstart block {idx} not run: optional module {module!r} missing")
            continue
        src = block
        for old, new, _reason in _OVERRIDES.get(idx, []):
            new = new.format(HTML_OUT=f'"{html_out}"')
            assert old in src, (
                f"block {idx}: anchor {old!r} absent; the doc moved, update _OVERRIDES"
            )
            src = src.replace(old, new)
        exec(compile(src, f"<quickstart block {idx}>", "exec"), ns)
