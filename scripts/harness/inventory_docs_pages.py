#!/usr/bin/env python3
"""Inventory all 81 public pages in mkdocs.yml."""
import re
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
mk = yaml.safe_load((ROOT / "mkdocs.yml").read_text())


def extract_nav(nav_item):
    items = []
    if isinstance(nav_item, dict):
        for k, v in nav_item.items():
            if isinstance(v, str):
                items.append((k, v))
            else:
                items.extend(extract_nav(v))
    elif isinstance(nav_item, list):
        for x in nav_item:
            items.extend(extract_nav(x))
    return items


nav_pairs = extract_nav(mk.get("nav", []))

print(f"Found {len(nav_pairs)} navigation entries.")
print("| Path | Title | Words | Equations | Code Blocks | Figures | Status | Action Reason |")
print("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|")

total_words = 0
for title, rel in nav_pairs:
    p = ROOT / "docs" / rel
    if not p.exists():
        print(f"| `{rel}` | {title} | 0 | 0 | 0 | 0 | REMOVE | Missing on disk |")
        continue
    content = p.read_text()
    words = len(content.split())
    total_words += words
    eq_blocks = len(re.findall(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])", content))
    code_blocks = len(re.findall(r"```", content)) // 2
    figures = len(re.findall(r"!\[.*?\]\(.*?\)", content))

    status = "KEEP"
    reason = "Compact & accurate"
    if words > 2500:
        status = "UPDATE"
        reason = "Candidate for dry equation/code compression"
    elif "legacy" in content.lower() or "deprecated" in content.lower():
        status = "UPDATE"
        reason = "Ensure secondary/compact status"

    print(f"| `{rel}` | {title} | {words} | {eq_blocks} | {code_blocks} | {figures} | {status} | {reason} |")

print(f"\nTotal Word Count: {total_words}")
