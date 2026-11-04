"""Replication dashboard page: dashboard/index.html with app.js, style.css and data/replications.json loads in
headless Chromium and renders from the committed data.

No network: the page is served from a local http.server thread, and every request to another host is answered
locally (JS stubs for the three CDN scripts, an empty stylesheet) or aborted. The test copy drops the CDN
integrity attributes, because a subresource-integrity check rejects any stub body. Skips when Playwright or its
Chromium is not installed (per test, so every leg records the same JUnit nodes; ``optional_oracle`` lets the
comparator classify the skip); any other failure fails the test.
"""
import http.server
import json
import re
import shutil
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.optional_oracle

REPO = Path(__file__).resolve().parents[1]
DASH = REPO / "dashboard"
DATA = DASH / "data" / "replications.json"

# Only what app.js calls: Plotly.newPlot and Plotly.purge, marked.parse, katex.renderToString.
STUBS = {
    "plotly.min.js": "window.Plotly = {newPlot: function () {}, purge: function () {}};",
    "marked.min.js": "window.marked = {parse: function (text) { return text; }};",
    "katex.min.js": "window.katex = {renderToString: function (tex) { return tex; }};",
}
CDN_STYLESHEETS = {"katex.min.css"}

HEADING_IS = "label => { const h = document.querySelector('#view h2'); return h !== null && h.textContent === label; }"


def _answer(route, request):
    parts = urlsplit(request.url)
    if parts.hostname == "127.0.0.1":
        route.continue_()
        return
    name = parts.path.rsplit("/", 1)[-1]
    cors = {"Access-Control-Allow-Origin": "*"}
    if parts.hostname == "cdn.jsdelivr.net" and name in STUBS:
        route.fulfill(status=200, content_type="application/javascript", headers=cors, body=STUBS[name])
    elif parts.hostname == "cdn.jsdelivr.net" and name in CDN_STYLESHEETS:
        route.fulfill(status=200, content_type="text/css", headers=cors, body="")
    else:
        route.abort()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if urlsplit(self.path).path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def _serve(root):
    handler = partial(_Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _build_site(base, broken=False):
    """Copy dashboard/ to base/site/dashboard. The Figures tab fetches ../figures/manifest.json, so the
    copy gets an empty manifest beside it; the repository has none (CI builds it into gh-pages)."""
    site = base / "site"
    shutil.copytree(DASH, site / "dashboard")
    index = site / "dashboard" / "index.html"
    index.write_text(re.sub(r' integrity="[^"]*"', "", index.read_text(encoding="utf-8")), encoding="utf-8")
    manifest = site / "figures" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"build_sha": "0" * 40, "figures": []}), encoding="utf-8")
    if broken:
        app = site / "dashboard" / "app.js"
        app.write_bytes(app.read_bytes() + b"\nthis is not valid javascript (((\n")
    return site


@contextmanager
def _session(browser, site):
    server, base = _serve(site)
    context = browser.new_context()
    try:
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: errors.append(f"console error: {msg.text}") if msg.type == "error" else None)
        page.route("**/*", _answer)
        page.goto(f"{base}/dashboard/index.html")
        yield page, errors
    finally:
        context.close()
        server.shutdown()
        server.server_close()


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


def test_page_renders_from_committed_data_without_errors(browser, tmp_path):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(data["papers"]) >= 2, "fixture: replications.json must hold at least two papers"
    site = _build_site(tmp_path)
    with _session(browser, site) as (page, errors):
        page.wait_for_function(HEADING_IS, arg="Overview", timeout=15000)
        page.wait_for_load_state("networkidle")
        assert errors == [], f"during load: {errors}"

        # Check 2: the header is filled from replications.json, not left at its template text.
        generated = page.text_content("#generated")
        assert generated != "Loading…", "the generated stamp was never filled"
        assert generated.startswith("Generated "), generated
        dev_sha = data["generated"]["dev_sha"]
        assert page.text_content("#devsha") == "build dev@" + dev_sha[:7]
        assert page.get_attribute("#devsha", "href") == "https://github.com/hnxj/jaxfne/commit/" + dev_sha

        # Check 1: every tab, clicked in turn, renders with no page error and no console error.
        tabs = page.eval_on_selector_all("#tabs a", "els => els.map(e => e.dataset.tab)")
        assert tabs == ["overview", "models", "validation", "figures", "roadmap", "method"], tabs
        for tab in tabs:
            page.click(f'#tabs a[data-tab="{tab}"]')
            page.wait_for_function(HEADING_IS, arg=tab.capitalize(), timeout=15000)
            page.wait_for_load_state("networkidle")
            assert errors == [], f"after the {tab} tab: {errors}"

        # Check 3: the models view holds one card per paper in replications.json.
        page.click('#tabs a[data-tab="models"]')
        page.wait_for_function(HEADING_IS, arg="Models", timeout=15000)
        hrefs = page.eval_on_selector_all("#view a.card", "els => els.map(e => e.getAttribute('href'))")
        assert hrefs == ["#model/" + p["id"] for p in data["papers"]], hrefs
        assert errors == [], f"on the models view: {errors}"


def test_collector_reports_a_broken_app_js(browser, tmp_path):
    site = _build_site(tmp_path, broken=True)
    assert b"valid javascript" in (site / "dashboard" / "app.js").read_bytes(), "fixture: the copy must be broken"
    with _session(browser, site) as (page, errors):
        page.wait_for_load_state("networkidle")
        page_errors = [e for e in errors if e.startswith("pageerror")]
        assert page_errors, f"the collector missed the syntax error: {errors}"
