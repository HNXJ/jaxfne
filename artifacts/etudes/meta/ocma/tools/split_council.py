"""Split a council workflow result ({"first": [rater...], "second": [rater...]}) into one file per rater and round.

Usage: python artifacts/etudes/meta/ocma/split_council.py <workflow output> <out dir>
Writes <out dir>/r<k>.json (round 1) and r<k>.rev.json (feedback round), k = A, B, C, ...
Exit 0 written, 2 unreadable input or no "first" list.
"""
import json
import sys
from pathlib import Path


def main(src, out):
    text = Path(src).read_text(encoding="utf-8")
    start = text.find("{")
    try:
        data = json.loads(text[start:])
    except ValueError as e:
        print(f"unreadable {src}: {e}")
        return 2
    data = data.get("result", data)
    if isinstance(data.get("batches"), list):
        # full council: merge each rater's papers across batches into one record list per round
        merged = {"first": [], "second": []}
        for b in data["batches"]:
            if not b:
                print("a batch is null")
                continue
            for key in merged:
                for i, rater in enumerate(b.get(key) or []):
                    while len(merged[key]) <= i:
                        merged[key].append({"papers": []})
                    if rater is None:
                        print(f"batch {b.get('lines')} {key}[{i}] is null")
                        continue
                    merged[key][i]["papers"].extend(rater.get("papers", []))
        data = merged
    if not isinstance(data.get("first"), list):
        print("no 'first' list")
        return 2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for tag, key in (("", "first"), (".rev", "second")):
        for i, rater in enumerate(data.get(key) or []):
            if rater is None:
                print(f"{key}[{i}] is null")
                continue
            (out / f"r{chr(65 + i)}{tag}.json").write_text(
                json.dumps(rater, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            n += 1
    print(f"wrote {n} files to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
