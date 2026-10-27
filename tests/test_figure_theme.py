"""A figure-site panel follows the page that frames it: light or dark, in Chromium.

``themed_html`` injects one script that starts from the OS scheme and switches on the
``jaxfne-scheme`` message from its parent. The browser test frames the themed file in a parent
page, posts each scheme from the parent and reads the layout the figure rendered with. Plotly's CDN
is answered from the copy the installed plotly package ships, so the test runs offline. Each test
skips when Playwright or its Chromium is missing; any other failure fails the test.
"""
from __future__ import annotations

import http.server
import json
import re
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import plotly
import plotly.graph_objects as go
import pytest
from plotly.io.json import to_json_plotly
from plotly.subplots import make_subplots

from jaxfne.vis.figure_theme import DARK, DIV_ID, LIGHT, themed_html

pytestmark = pytest.mark.optional_oracle

PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
CDN = "https://cdn.plot.ly/**"
MESSAGE = "jaxfne-scheme"
SCRIPT = re.compile(r"<script>\n\(function \(\) \{\n  var DIV.*?\}\)\(\);\n</script>\n", re.DOTALL)
SCHEMES = re.compile(r"  var SCHEMES = (\{.*\});\n")
PARENT = """<!doctype html><html><body>
<iframe name="figure" src="figure.html" width="960" height="640"></iframe>
</body></html>
"""
# What the test reads from the rendered figure: the page and plot backgrounds, the ink, a 2D axis line
# and a 3D scene's background and axis line.
READ = """() => {
  const l = document.getElementById("jx-figure")._fullLayout;
  return {paper: l.paper_bgcolor, plot: l.plot_bgcolor, ink: l.font.color,
          line: l.xaxis.linecolor, scene: l.scene.bgcolor, sceneline: l.scene.xaxis.linecolor};
}"""


def _expected(palette: dict) -> dict:
    return {"paper": palette["paper_bgcolor"], "plot": palette["plot_bgcolor"],
            "ink": palette["font"]["color"], "line": palette["xaxis"]["linecolor"],
            "scene": palette["scene"]["bgcolor"], "sceneline": palette["scene"]["xaxis"]["linecolor"]}


def _builder_figure() -> go.Figure:
    """A 2D trace and a 3D trace, with the page colours and template the atlas builders set."""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]])
    fig.add_trace(go.Scatter(x=[0, 1, 2], y=[0, 1, 4], mode="lines+markers", name="2d"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="markers", name="3d"), row=1, col=2)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), scene=dict(bgcolor="#000000"))
    assert [t.type for t in fig.data] == ["scatter", "scatter3d"], "fixture: one 2D and one 3D trace"
    return fig


def test_themed_html_holds_the_switch_both_schemes_and_the_unchanged_data():
    fig = _builder_figure()
    html = themed_html(fig)
    assert html.count("</body>") == 1
    assert f'event.data.type !== "{MESSAGE}"' in html, "the switch must answer the message type"
    assert "event.source !== window.parent" in html, "the switch must answer its parent only"
    assert "matchMedia(\"(prefers-color-scheme: dark)\")" in html, "a page opened directly follows the OS"
    match = SCHEMES.search(html)
    assert match, "the script must carry both schemes"
    schemes = json.loads(match.group(1))
    assert set(schemes) == {"light", "dark"}
    assert schemes["light"]["paper_bgcolor"] == LIGHT["paper_bgcolor"]
    assert schemes["dark"]["paper_bgcolor"] == DARK["paper_bgcolor"]
    assert schemes["dark"]["font.color"] == DARK["font"]["color"]
    # The builder's own layout keys are overridden per axis path, the 3D scene included.
    assert schemes["light"]["xaxis.linecolor"] == LIGHT["xaxis"]["linecolor"]
    assert schemes["light"]["scene.bgcolor"] == LIGHT["scene"]["bgcolor"]
    assert schemes["light"]["scene.xaxis.gridcolor"] == LIGHT["scene"]["xaxis"]["gridcolor"]
    assert schemes["dark"]["scene.zaxis.tickcolor"] == DARK["scene"]["zaxis"]["tickcolor"]
    # Trace data is never changed: the data array plotly writes is still in the page.
    assert to_json_plotly(fig.to_plotly_json()["data"]) in html
    assert DIV_ID in html


def test_themed_html_is_the_plain_page_plus_the_script():
    fig = _builder_figure()
    html = themed_html(fig)
    stripped = SCRIPT.sub("", html)
    assert stripped != html, "the script must be present to be removed"
    assert stripped == fig.to_html(full_html=True, include_plotlyjs="cdn", div_id=DIV_ID)


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        try:
            chromium = pw.chromium.launch(headless=True)
        except sync_api.Error as exc:
            text = str(exc)
            if "Executable doesn't exist" in text or "playwright install" in text:
                pytest.skip("Playwright Chromium is not installed: " + text.splitlines()[0])
            raise
        yield chromium
        chromium.close()


@pytest.fixture
def site(tmp_path):
    (tmp_path / "figure.html").write_text(themed_html(_builder_figure()), encoding="utf-8")
    (tmp_path / "parent.html").write_text(PARENT, encoding="utf-8")
    return tmp_path


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@contextmanager
def _serve(root):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _offline_plotly(route):
    route.fulfill(path=str(PLOTLY_JS), content_type="application/javascript")


def _rendered(frame):
    """Precondition: the figure rendered its two traces and its layout."""
    frame.wait_for_function(
        "() => { const g = document.getElementById('jx-figure');"
        " return !!(g && g._fullLayout && g.querySelectorAll('.main-svg').length > 0); }", timeout=15000)
    assert frame.evaluate("() => document.getElementById('jx-figure').data.length") == 2


def _wait_for(frame, palette):
    want = _expected(palette)
    frame.wait_for_function(
        f"(want) => {{ const r = ({READ})(); return Object.keys(want).every(k => r[k] === want[k]); }}",
        arg=want, timeout=5000)
    assert frame.evaluate(READ) == want


def _post(page, scheme):
    page.evaluate(
        "(scheme) => document.querySelector('iframe[name=figure]').contentWindow.postMessage("
        "{type: 'jaxfne-scheme', scheme: scheme}, '*')", scheme)


def test_the_framed_figure_switches_light_dark_light(browser, site):
    context = browser.new_context(color_scheme="light")
    page = context.new_page()
    try:
        page.route(CDN, _offline_plotly)
        with _serve(site) as base:
            page.goto(f"{base}/parent.html")
            frame = page.frame(name="figure")
            assert frame is not None, "the parent page must frame the figure"
            _rendered(frame)
            _wait_for(frame, LIGHT)
            _post(page, "dark")
            _wait_for(frame, DARK)
            _post(page, "light")
            _wait_for(frame, LIGHT)
    finally:
        context.close()


def test_a_figure_opened_directly_follows_the_os_scheme(browser, site):
    context = browser.new_context(color_scheme="dark")
    page = context.new_page()
    try:
        page.route(CDN, _offline_plotly)
        with _serve(site) as base:
            page.goto(f"{base}/figure.html")
            _rendered(page)
            _wait_for(page, DARK)
    finally:
        context.close()
