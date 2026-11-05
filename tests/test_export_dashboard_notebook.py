"""scripts/export_dashboard_notebook.py: figures re-encoded as WebP, code hidden, error outputs refused."""

import base64
import io
import json

import nbformat
import pytest
from PIL import Image

from scripts import export_dashboard_notebook as exp


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _notebook(tmp_path, error=False, figure=True):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# Suite title\nmore"))
    cell = nbformat.v4.new_code_cell("secret_code_line = 1")
    if figure:
        cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": _png_b64()}))
    if error:
        cell.outputs.append(nbformat.v4.new_output("error", ename="NameError", evalue="plt", traceback=[]))
    nb.cells.append(cell)
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, path)
    return path


def test_export_shows_code_and_reencodes_figures(tmp_path):
    html, n = exp.export(_notebook(tmp_path))
    assert n == 1, "fixture: one figure"
    assert "data:image/webp;base64," in html
    assert "data:image/png" not in html
    assert "secret_code_line" in html
    assert "<title>Suite title</title>" in html


def test_export_hide_code_keeps_outputs_only(tmp_path):
    html, n = exp.export(_notebook(tmp_path), show_code=False)
    assert n == 1, "fixture: one figure"
    assert "secret_code_line" not in html


@pytest.mark.parametrize("kw", [{"error": True}, {"figure": False}])
def test_export_refuses_error_or_empty_notebook(tmp_path, kw):
    with pytest.raises(SystemExit):
        exp.export(_notebook(tmp_path, **kw))


def test_main_upserts_one_sorted_row(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUT", tmp_path / "out")
    nb = _notebook(tmp_path)
    common = ["--title", "T", "--paper", "P", "--data", "D"]
    assert exp.main([str(nb), "--slug", "zeta", *common]) == 0
    assert exp.main([str(nb), "--slug", "alpha", *common]) == 0
    assert exp.main([str(nb), "--slug", "zeta", *common]) == 0
    rows = json.loads((tmp_path / "out" / "index.json").read_text(encoding="utf-8"))["notebooks"]
    assert [r["slug"] for r in rows] == ["alpha", "zeta"]
    assert rows[0]["panels"] == 1 and rows[0]["html"] == "alpha.html"
    assert exp.main([str(_notebook(tmp_path, error=True)), "--slug", "bad", *common]) == 1
    assert not (tmp_path / "out" / "bad.html").exists()
