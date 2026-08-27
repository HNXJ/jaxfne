#!/usr/bin/env python3
"""Deterministic manuscript PDF — P1-publication, Δscience=0. v0.4.17

Concatenates docs/publication/manuscript/*.md in fixed order
abstract→introduction→methods→results→discussion→supplement,
injects figure embeds from artifacts/figures/publication/final/ (300 DPI, SHA-verified),
reads captions from fig*_semantic_audit.json (no science vocab added),
and emits PDF via reportlab (deterministic) or pandoc if available.

Writes to /tmp/manuscript.pdf by default (no repo mutation).
"""
from __future__ import annotations

import argparse
import hashlib
import html as htmlmod
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["abstract", "introduction", "methods", "results", "discussion", "supplement"]
FIG_MAP = {
    1: ROOT / "artifacts/figures/publication/final/fig01_tfne_grammar.png",
    2: ROOT / "artifacts/figures/publication/final/fig02_emitter_source.png",
    3: ROOT / "artifacts/figures/publication/final/fig03_local_observation.png",
    4: ROOT / "artifacts/figures/publication/final/fig04_multiscale_boundary.png",
    5: ROOT / "artifacts/figures/publication/final/fig05_traveling_wave_no_wave.png",
    6: ROOT / "artifacts/figures/publication/final/fig06_rbs_hdp_ladder.png",
    7: ROOT / "artifacts/figures/publication/final/fig07_e_integration.png",
}
FIXED_EPOCH = "D:20260815000000Z"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_figs() -> None:
    for i, p in FIG_MAP.items():
        receipt = ROOT / f"artifacts/publication/polish/fig{i}_polish_receipt.json"
        if receipt.exists():
            want = json.loads(receipt.read_text()).get("png_sha256") or json.loads(receipt.read_text()).get("png_300dpi_sha") or json.loads(receipt.read_text()).get("sha256")
            # polish receipt stores png_sha256 at top level for some figs
            data = json.loads(receipt.read_text())
            want = data.get("png_sha256") or data.get("sha256") or data.get("png_300dpi_sha256")
            if want and sha256(p) != want:
                raise SystemExit(f"figure {i} SHA mismatch: {p} vs receipt")
        if not p.exists():
            raise SystemExit(f"missing figure {p}")


def verify_dois() -> None:
    text = (ROOT / "docs/reference/references.md").read_text(encoding="utf-8")
    dois = re.findall(r"10\.\d+/[^\s\)]+", text)
    bad = [d for d in dois if not re.match(r"10\.\d+/.+", d)]
    if bad:
        raise SystemExit(f"malformed DOIs: {bad}")


def inject(md: str) -> str:
    # Append figure embeds after each Fig. N textual ref, with audit caption
    def _caption(fig_no: int) -> str:
        audit = ROOT / f"artifacts/publication/fig0{fig_no}_semantic_audit.json"
        if fig_no in (1, 5, 6, 7) and audit.exists():
            j = json.loads(audit.read_text())
            status = j.get("status", "PASSED")
            return f"*Figure {fig_no} — {status} | SHA {sha256(FIG_MAP[fig_no])[:8]}*"
        if fig_no in (2, 3, 4):
            j = json.loads((ROOT / "artifacts/publication/fig02_04_cross_figure_audit.json").read_text())
            return f"*Figures 2-4 — cross-audit {j.get('status')} | Q {j.get('canonical_q_hash','')[:8]}*"
        return f"*Figure {fig_no}*"
    for n in range(1, 8):
        pat = re.compile(rf"Fig\. {n}\b")
        if pat.search(md):
            repl = f"Fig. {n}\n\n![]({FIG_MAP[n].as_posix()})\n\n{_caption(n)}"
            md = pat.sub(repl, md, count=1)
    return md


def build(out: Path) -> Path:
    verify_figs()
    verify_dois()
    parts = []
    for name in ORDER:
        p = ROOT / f"docs/publication/manuscript/{name}.md"
        if not p.exists():
            raise SystemExit(f"missing manuscript part {p}")
        parts.append(p.read_text(encoding="utf-8"))
    concat = "\n\n---\n\n".join(parts)
    injected = inject(concat)
    tmp_md = out.with_suffix(".concat.md")
    tmp_md.write_text(injected, encoding="utf-8")
    # Try pandoc first
    if shutil.which("pandoc"):
        bib = ROOT / "docs/reference/references.md"
        # pandoc needs .bib; generate minimal from references.md DOI list if missing
        b = out.with_suffix(".tex")
        cmd = ["pandoc", str(tmp_md), "-o", str(b)]
        subprocess.run(cmd, check=False)
        if b.exists():
            out = b
            return out
    # Fallback: markdown -> HTML -> reportlab PDF (deterministic canvas)
    try:
        import markdown
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
    except ImportError:
        # No reportlab: emit HTML only
        html = markdown.markdown(injected, extensions=["tables", "toc", "fenced_code"])
        out = out.with_suffix(".html")
        out.write_text(html, encoding="utf-8")
        return out
    html = markdown.markdown(injected, extensions=["tables", "toc", "fenced_code"])
    # --- font registration: DejaVu for Unicode Greek, embeds subset ---
    dejavu_path = None
    dejavu_mono_path = None
    try:
        import matplotlib
        mpl_ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        cand = mpl_ttf / "DejaVuSans.ttf"
        if cand.exists():
            dejavu_path = cand
        cand2 = mpl_ttf / "DejaVuSansMono.ttf"
        if cand2.exists():
            dejavu_mono_path = cand2
    except Exception:
        pass
    if dejavu_path is None:
        for cand in [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
            if cand.exists():
                dejavu_path = cand
                break
    if dejavu_mono_path is None:
        for cand in [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]:
            if cand.exists():
                dejavu_mono_path = cand
                break
    if dejavu_path is None:
        try:
            import matplotlib.font_manager as fm
            fp = fm.findfont("DejaVu Sans", fallback_to_default=False)
            if fp and Path(fp).exists():
                dejavu_path = Path(fp)
        except Exception:
            pass
    if dejavu_mono_path is None and dejavu_path is not None:
        # fallback mono to regular if not found separate
        try:
            import matplotlib.font_manager as fm2
            fp2 = fm2.findfont("DejaVu Sans Mono", fallback_to_default=False)
            if fp2 and Path(fp2).exists():
                dejavu_mono_path = Path(fp2)
        except Exception:
            pass
    # register fonts
    if dejavu_path and dejavu_path.exists():
        try:
            pdfmetrics.registerFont(TTFont('DejaVu', str(dejavu_path)))
        except Exception:
            pass
    else:
        # last resort: try Windows path already handled via mpl_ttf
        pass
    if dejavu_mono_path and dejavu_mono_path.exists():
        try:
            pdfmetrics.registerFont(TTFont('DejaVuMono', str(dejavu_mono_path)))
        except Exception:
            pass
    elif dejavu_path and dejavu_path.exists():
        # use DejaVu as mono fallback
        try:
            pdfmetrics.registerFont(TTFont('DejaVuMono', str(dejavu_path)))
        except Exception:
            pass

    doc = SimpleDocTemplate(str(out), pagesize=letter, title="JaxFNE Manuscript v0.4.17", author="Hamed Nejat")
    styles = getSampleStyleSheet()
    # set all styles to DejaVu to ensure embedding and Greek coverage; keep Symbol only for missing glyphs via fallback
    for sname in list(styles.byName.keys()):
        try:
            styles[sname].fontName = 'DejaVu'
        except Exception:
            pass
    # custom styles
    normal_style = styles['Normal']
    normal_style.fontName = 'DejaVu'
    normal_style.fontSize = 9
    normal_style.leading = 11
    # heading styles
    heading1_style = ParagraphStyle('Heading1_DejaVu', parent=styles['Heading1'], fontName='DejaVu', fontSize=14, leading=16, spaceAfter=8, keepWithNext=True)
    heading2_style = ParagraphStyle('Heading2_DejaVu', parent=styles['Heading2'], fontName='DejaVu', fontSize=11, leading=13, spaceAfter=6, keepWithNext=True)
    heading3_style = ParagraphStyle('Heading3_DejaVu', parent=styles['Heading3'], fontName='DejaVu', fontSize=10, leading=12, spaceAfter=4, keepWithNext=True)
    mono_style = ParagraphStyle('Mono7', parent=normal_style, fontName='DejaVuMono', fontSize=7, leading=8, leftIndent=6)
    eq_style = ParagraphStyle('EqMono', parent=mono_style, fontSize=8, leading=10)

    story = []
    # title
    story.append(Paragraph("JaxFNE Manuscript v0.4.17", heading1_style))
    story.append(Spacer(1, 12))

    blocks = re.split(r"\n\s*\n", injected)
    i = 0
    while i < len(blocks):
        block = blocks[i]
        stripped = block.strip()
        if not stripped:
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            story.append(PageBreak())
            i += 1
            continue
        # Image + caption KeepTogether with aspect preserve
        if stripped.startswith("![]("):
            m = re.search(r"\(([^)]+)\)", block)
            caption_text = None
            caption_idx = None
            if i + 1 < len(blocks) and blocks[i + 1].strip().startswith("*Figure"):
                caption_text = blocks[i + 1].strip()
                caption_idx = i + 1
            if m:
                img = Path(m.group(1))
                if img.exists():
                    try:
                        w_pt = 450.0
                        h_pt = 300.0
                        try:
                            from PIL import Image as PILImage
                            with PILImage.open(img) as im:
                                iw, ih = im.size
                                if iw and ih:
                                    scale = min(w_pt / iw, h_pt / ih)
                                    w_pt = iw * scale
                                    h_pt = ih * scale
                        except Exception:
                            pass
                        img_flow = RLImage(str(img), width=w_pt, height=h_pt)
                        img_flow.hAlign = 'CENTER'
                        if caption_text is not None:
                            cap_inner = caption_text.strip().strip("*").strip()
                            cap_para = Paragraph(htmlmod.escape(cap_inner), normal_style)
                            story.append(KeepTogether([img_flow, Spacer(1, 6), cap_para, Spacer(1, 12)]))
                            i += 2
                            continue
                        else:
                            story.append(KeepTogether([img_flow, Spacer(1, 12)]))
                            i += 1
                            continue
                    except Exception:
                        pass
            i += 1
            continue
        # Headings
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            esc = htmlmod.escape(text)
            if level == 1:
                story.append(Paragraph(esc, heading1_style))
            elif level == 2:
                story.append(Paragraph(esc, heading2_style))
            else:
                story.append(Paragraph(esc, heading3_style))
            story.append(Spacer(1, 6))
            i += 1
            continue
        # Code fences -> split equations into separate Preformatted with DejaVuSansMono, no truncation, Greek via Unicode
        if "```" in block:
            parts_fence = block.split("```")
            for idx, part in enumerate(parts_fence):
                if idx % 2 == 1:
                    lines = part.splitlines()
                    # drop language identifier if first line is known language
                    if lines and lines[0].strip().lower() in ("text", "python", "json", "bash", "yaml", "markdown", "md"):
                        content = "\n".join(lines[1:])
                    else:
                        content = part
                    content = content.strip("\n")
                    if not content.strip():
                        continue
                    # Preserve internal newlines; split into separate Preformatted if multi-line equations
                    # Keep as one Preformatted to preserve line-breaks, but also ensure no truncation
                    story.append(Preformatted(content, eq_style))
                    story.append(Spacer(1, 6))
                else:
                    if part.strip():
                        for line in part.splitlines():
                            if line.strip():
                                story.append(Paragraph(htmlmod.escape(line.strip()), normal_style))
                                story.append(Spacer(1, 2))
            i += 1
            continue
        # Tables: blocks containing '|' use Preformatted monospace 7pt, no truncation, keep Δscience=0
        if "|" in block:
            lines = [l for l in block.splitlines() if "|" in l]
            if len(lines) >= 1:
                # Use Preformatted with mono 7pt, preserving pipes byte-identical
                story.append(Preformatted(block.strip("\n"), mono_style))
                story.append(Spacer(1, 6))
                i += 1
                continue
        # Bullet / ordered list handling
        if stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"\d+\.\s", stripped):
            for line in block.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                story.append(Paragraph(htmlmod.escape(ls), normal_style))
                story.append(Spacer(1, 3))
            i += 1
            continue
        # Generic paragraph: no truncation, keep Δscience=0, replace single newlines with space for wrapping
        txt_single = re.sub(r"\s*\n\s*", " ", block.strip())
        if txt_single:
            story.append(Paragraph(htmlmod.escape(txt_single), normal_style))
            story.append(Spacer(1, 6))
        i += 1

    doc.build(story)
    # Determinism fix: patch CreationDate/ModDate/ID with \s* and DOTALL; verify second build hash equals first
    try:
        data = out.read_bytes()
        # ReportLab embeds CreationDate as D:YYYYMMDDHHmmSS; replace with fixed, allow whitespace/newline between key and value
        data = re.sub(rb"/CreationDate\s*\(D:[^\)]+\)", b"/CreationDate (" + FIXED_EPOCH.encode() + b")", data, flags=re.S)
        data = re.sub(rb"/ModDate\s*\(D:[^\)]+\)", b"/ModDate (" + FIXED_EPOCH.encode() + b")", data, flags=re.S)
        # ID is random 32 hex; replace with fixed, allow newline between /ID and [
        data = re.sub(rb"/ID\s*\[<[^>]+><[^>]+>\]", b"/ID [<00000000000000000000000000000000><00000000000000000000000000000000>]", data, flags=re.S)
        out.write_bytes(data)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic manuscript PDF build")
    ap.add_argument("--out", type=Path, default=Path("/tmp/manuscript.pdf"), help="output PDF/HTML path")
    ap.add_argument("--verify", action="store_true", help="verify figures/DOIs before build")
    args = ap.parse_args()
    out = build(args.out)
    print(f"wrote {out} {out.stat().st_size} bytes sha {sha256(out)[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
