"""Export one executed replication notebook to the dashboard's Notebooks tab.

Writes ``dashboard/notebooks/<slug>.html`` (outputs only, code cells hidden) and
upserts the notebook's row in ``dashboard/notebooks/index.json``. Embedded PNG
figures are re-encoded as WebP (quality 82) to keep the page small. Rows are
sorted by slug and carry no timestamp, so the same inputs give the same bytes.

    python scripts/export_dashboard_notebook.py EXECUTED.ipynb --slug caminos-plos \
        --title "..." --paper "..." --data "F:/warehouse/caminos (figP_out, plosr_7f7a660e)"

Exit codes: 0 written, 1 the notebook holds an error output or no figure, 2 usage.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import re
import subprocess
import sys

import nbformat
from nbconvert import HTMLExporter
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "notebooks"
PNG_URI = re.compile(r'src="data:image/png;base64,([A-Za-z0-9+/=\s]+)"')
WEBP_QUALITY = 82
# The basic template carries no page shell or CSS; this shell keeps the page light and readable.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ margin: 0 auto; max-width: 1100px; padding: 16px; background: #fff; color: #1b1f24;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
img {{ max-width: 100%; height: auto; }}
pre {{ white-space: pre-wrap; font-size: 13px; color: #5b6573; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #dde1e6; padding: 2px 6px; }}
</style></head>
<body>
{body}
</body></html>
"""


def _webp(match: re.Match) -> str:
    img = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=WEBP_QUALITY, method=6)
    return 'src="data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode("ascii") + '"'


def export(nb_path: pathlib.Path) -> tuple[str, int]:
    """Return (html, figure count); refuse a notebook with an error output or no figure."""
    nb = nbformat.read(nb_path, as_version=4)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    if errors:
        raise SystemExit(f"error output in {nb_path}: {errors[0].get('ename')}: {errors[0].get('evalue', '')[:200]}")
    nb.metadata.pop("widgets", None)
    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    body, n = PNG_URI.subn(_webp, body)
    title = next((c.source.splitlines()[0].lstrip("# ") for c in nb.cells if c.cell_type == "markdown"), nb_path.stem)
    html = PAGE.format(title=title, body=body)
    if n == 0:
        raise SystemExit(f"no figure in {nb_path}")
    return html, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook", type=pathlib.Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--data", required=True, help="where the runs the notebook reads live")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9-]+", args.slug):
        ap.error("slug must be lowercase kebab-case")
    try:
        html, n = export(args.notebook)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.slug}.html").write_bytes(html.encode("utf-8"))
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"notebooks": []}
    rows = [r for r in index["notebooks"] if r["slug"] != args.slug]
    rows.append({"slug": args.slug, "title": args.title, "paper": args.paper, "html": f"{args.slug}.html",
                 "panels": n, "build_sha": sha, "data": args.data})
    index["notebooks"] = sorted(rows, key=lambda r: r["slug"])
    index_path.write_bytes((json.dumps(index, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {args.slug}.html ({len(html.encode('utf-8'))} bytes, {n} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
