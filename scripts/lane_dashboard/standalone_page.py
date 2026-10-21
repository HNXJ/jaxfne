"""Build a standalone lane dashboard page from scripts/lane_dashboard/template (index.html, app.js) and
the jaxfne_dashboard.py outputs. The template expects a dashboard runtime (dash.calc / dash.data /
dash.onData and theme tokens); this page supplies a small shim that loads the two datasets published
beside it as lanes.json and analyses.json, and d3 from cdnjs.

    python scripts/lane_dashboard/standalone_page.py --dash <dir> --out <dir>

--dash is the --out of jaxfne_dashboard.py. Publish <out>/index.html as an Artifact with app.js and the
two json files as supporting files (skill jaxfne-dashboard). Pattern copied from the omission Draft-S
dashboard.
"""
import argparse
import json
import shutil
from pathlib import Path

SOURCES = ("lanes", "analyses")

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dash", type=Path, required=True, help="output folder of jaxfne_dashboard.py")
ap.add_argument("--out", type=Path, required=True, help="folder for index.html, app.js and the datasets")
a = ap.parse_args()
here, dash, out = Path(__file__).resolve().parent, a.dash, a.out
out.mkdir(parents=True, exist_ok=True)
tpl = (here / "template" / "index.html").read_text(encoding="utf-8")
ds = json.loads((dash / "doc_ds.json").read_text(encoding="utf-8"))
stamp = ds["updated"]["at"]
commit = ds["description"].rsplit("commit ", 1)[-1].rstrip(".")

TOKENS = """
:root{--font-anthropic-sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
--font-anthropic-serif:ui-serif,Georgia,"Times New Roman",serif;
--color-bg:#f7f8f6;--color-panel:#eceee9;--color-fg:#1c1f1d;--color-fg-muted:#59605b;
--color-border-line:#d6dad3;--color-good:#2e7d32;--color-warn:#a15c00;--color-bad:#b3261e;
--cds-radius:8px;--cds-gap-md:12px;--cds-pad-md:10px;--cds-pad-lg:14px;
--cds-font-size-caption:.8125rem;--cds-font-size-heading:1.125rem;--cds-font-weight-medium:500;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--color-bg:#1b1e1c;--color-panel:#262a27;
--color-fg:#eceee9;--color-fg-muted:#aab1ab;--color-border-line:#3a3f3b;--color-good:#81c784;
--color-warn:#ffb74d;--color-bad:#ef9a9a;color-scheme:dark}}
:root[data-theme="dark"]{--color-bg:#1b1e1c;--color-panel:#262a27;--color-fg:#eceee9;--color-fg-muted:#aab1ab;
--color-border-line:#3a3f3b;--color-good:#81c784;--color-warn:#ffb74d;--color-bad:#ef9a9a;color-scheme:dark}
html,body{margin:0;background:var(--color-bg);color:var(--color-fg)}
body{padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.dash-skeleton{min-height:8rem;opacity:.5}
.stamp{max-width:1280px;margin:0 auto 1rem;color:var(--color-fg-muted);font:var(--cds-font-size-caption)/1.4 var(--font-anthropic-sans)}
"""

SHIM = """
window.dash = (function () {
  const SOURCES = %s;
  const state = {}, calcs = {}, subs = [];
  SOURCES.forEach((n) => { state[n] = { status: 'loading' }; });
  function data(name) {
    const c = calcs[name];
    if (!c) return state[name] || { status: 'error', error: 'unknown source ' + name };
    const ins = c.inputs.map((i) => data(i));
    if (ins.some((x) => x.status === 'error')) return { status: 'error' };
    if (ins.some((x) => x.status !== 'ok')) return { status: 'loading' };
    try { return { status: 'ok', data: c.fn(...ins.map((x) => x.data)) }; }
    catch (e) { return { status: 'error', error: String(e) }; }
  }
  function fillMarks() {
    document.querySelectorAll('[data-source][data-field][data-where]').forEach((el) => {
      const res = data(el.dataset.source);
      if (res.status !== 'ok') return;
      const [k, v] = el.dataset.where.split('=');
      const row = res.data.find((r) => String(r[k]) === v);
      el.textContent = row && row[el.dataset.field] != null ? String(row[el.dataset.field]) : '—';
    });
  }
  function notify() { fillMarks(); subs.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } }); }
  SOURCES.forEach((n) => {
    fetch(n + '.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((j) => { state[n] = { status: 'ok', data: j.rows || [] }; })
      .catch((e) => { state[n] = { status: 'error', error: String(e) }; })
      .finally(notify);
  });
  return {
    calc(name, spec) { calcs[name] = spec; },
    data,
    onData(fn) { subs.push(fn); notify(); },
  };
})();
""" % json.dumps(list(SOURCES))

page = (
    '<meta charset="utf-8">\n<title>jaxfne Lane Dashboard</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
    + "<style>" + TOKENS + "</style>\n"
    + f'<p class="stamp">Read from the jaxfne repository at commit {commit}, {stamp}.</p>\n'
    + "<script>" + SHIM + "</script>\n"
    + tpl
    + '\n<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>'
    + '\n<script src="app.js"></script>\n'
)
(out / "index.html").write_text(page, encoding="utf-8", newline="\n")
shutil.copyfile(here / "template" / "app.js", out / "app.js")
for n in SOURCES:
    shutil.copyfile(dash / f"{n}.json", out / f"{n}.json")
print(f"wrote {out} (commit {commit}, {stamp})")
