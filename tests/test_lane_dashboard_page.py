"""Lane dashboard page: standalone_page.py packs the template, the runtime shim and the commit stamp, the
datasets and app.js land beside it; app.js parses (node --check) and fills data only through textContent;
every KPI and table carries its dataset marks."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "scripts" / "lane_dashboard"
TOOL = DIR / "standalone_page.py"
SETS = ("lanes", "analyses")


def test_page_has_template_shim_and_stamp(tmp_path):
    dash, out = tmp_path / "dash", tmp_path / "out"
    dash.mkdir()
    (dash / "doc_ds.json").write_text(json.dumps({
        "description": "One row per lane at commit abc1234.", "updated": {"at": "2026-01-02T03:04:05Z"}}))
    for n in SETS:
        (dash / f"{n}.json").write_text(json.dumps({"rows": [{"id": n}]}))
    rc = subprocess.run([sys.executable, str(TOOL), "--dash", str(dash), "--out", str(out)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    page = (out / "index.html").read_text(encoding="utf-8")
    tpl = (DIR / "template" / "index.html").read_text(encoding="utf-8")
    assert tpl in page
    assert "window.dash" in page and page.index("window.dash") < page.index('src="app.js"')
    assert "d3.min.js" in page and page.index("d3.min.js") < page.index('src="app.js"')
    assert "commit abc1234, 2026-01-02T03:04:05Z" in page
    assert page.startswith('<meta charset="utf-8">'), "the page declares its encoding"
    assert "onData(fn) { subs.push(fn); notify(); }" in page, "marks fill when data arrived before app.js"
    assert (out / "app.js").read_bytes() == (DIR / "template" / "app.js").read_bytes()
    for n in SETS:
        assert json.loads((out / f"{n}.json").read_text())["rows"] == [{"id": n}]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    rc = subprocess.run(["node", "--check", str(DIR / "template" / "app.js")], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_data_text_never_goes_through_innerhtml():
    js = (DIR / "template" / "app.js").read_text(encoding="utf-8")
    assert "textContent" in js, "fixture: the page must write text"
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", js)


def test_every_number_is_a_dataset_mark():
    html = (DIR / "template" / "index.html").read_text(encoding="utf-8")
    kpis = re.findall(r'<b class="n"[^>]*>', html)
    assert kpis, "fixture: the template must hold KPIs"
    for k in kpis:
        assert all(f"{a}=" in k for a in ("data-source", "data-field", "data-where")), k
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
    assert tables, "fixture: the template must hold tables"
    for t in tables:
        head = re.match(r"<table[^>]*>", t).group(0)
        assert "data-source=" in head and "data-row-key=" in head, head
        ths = re.findall(r"<th(?:\s[^>]*)?>", t)
        assert ths and all("data-field=" in th for th in ths), t[:200]
    assert not re.search(r">\s*\d[\d,.]*\s*<", re.sub(r"<script.*?</script>", "", html, flags=re.S)), \
        "a number typed into the template"
