#!/usr/bin/env python3
"""Balance metrics for a Labyrinth `.lab/` graph -- Phase 1 of the graph-rebalance
plan (artifacts/developer/plans.json :: lab-graph-rebalance-ten-phase).

The graph is meant to be an ensemble, not a tree: roughly log2(N) degree per node and
roughly log2(log2(N)) depth. Nothing else in the toolchain measures whether that holds,
so every claim about "the graph is more balanced now" was unfalsifiable self-report.
This makes it a number.

    python3 scripts/lab_balance.py                  # human-readable report
    python3 scripts/lab_balance.py --json           # machine-readable
    python3 scripts/lab_balance.py --gate           # exit 1 if targets are missed

Degree is UNDIRECTED: a link contributes to both endpoints. The graph's relations
(refines/derives_from today) are directed, but balance is a question about how many
things a node is connected to at all, not about which way the arrow points.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

DEFAULT_LAB = Path(__file__).resolve().parent.parent / "artifacts" / ".lab"

# Targets, all derived from N rather than hardcoded, so they scale with the graph.
#   degree band   : log2(N), within a tolerance factor
#   max degree    : 2*log2(N) -- above this a node is a star hub, not a member
#   gini          : 0.25, a conventional "reasonably even" threshold
GINI_TARGET = 0.25
DEGREE_BAND_TOLERANCE = 0.7  # band is [log2N*(1-t), log2N*(1+t)]


def load_nodes(lab_dir: Path) -> tuple[dict, list[str]]:
    """Every *.json in .lab that actually carries an `id` is a node. Anything else
    (suggestions.json, and whatever else the toolchain drops there) is skipped and
    reported rather than silently ignored -- a malformed node would otherwise vanish
    from the metrics and make the graph look better than it is."""
    nodes, skipped = {}, []
    for path in sorted(lab_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append(f"{path.name} (unreadable: {type(exc).__name__})")
            continue
        if isinstance(data, dict) and "id" in data:
            nodes[data["id"]] = data
        else:
            skipped.append(path.name)
    return nodes, skipped


def build_edges(nodes: dict) -> tuple[dict, dict, collections.Counter, set]:
    """Returns (out_adj, in_adj, relation_census, dangling_targets)."""
    out_adj = collections.defaultdict(list)
    in_adj = collections.defaultdict(list)
    relations = collections.Counter()
    for node_id, node in nodes.items():
        for link in node.get("generated", {}).get("links", []):
            target = link.get("to")
            if not target:
                continue
            out_adj[node_id].append(target)
            in_adj[target].append(node_id)
            relations[link.get("relation", "<none>")] += 1
    dangling = {t for targets in out_adj.values() for t in targets} - set(nodes)
    return out_adj, in_adj, relations, dangling


def gini(values: list[int]) -> float:
    """Gini coefficient of the degree distribution: 0 = every node equally connected,
    1 = one node holds every edge. A star graph sits near 1; the target is < 0.25."""
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    cumulative = sum((2 * i - n + 1) * v for i, v in enumerate(ordered))
    return cumulative / (n * sum(ordered))


def depth_of(node_id: str, out_adj: dict, nodes: dict, seen: frozenset = frozenset()) -> int:
    """Distance to the nearest root (a node with no outgoing links), following the
    SHORTEST parent path. Cycles are cut by `seen` rather than raising -- the graph is
    explicitly not a tree, so a cycle is a legitimate shape here, not a corruption."""
    if node_id in seen:
        return 0
    parents = [t for t in out_adj.get(node_id, []) if t in nodes]
    if not parents:
        return 0
    return 1 + min(depth_of(p, out_adj, nodes, seen | {node_id}) for p in parents)


def repeated_segment_ids(nodes: dict) -> list[str]:
    """Ids like `jaxfne-jaxfne`, where a segment repeats ADJACENTLY. These are
    grammar-VALID (objective_grammar.py has no dedup rule) but semantically degenerate:
    they happen when a child's own key equals the parent's inherited last segment.

    Adjacency matters. A non-adjacent repeat is usually legitimate English --
    `plan-file-by-file-review` repeats "file" on purpose -- so flagging every repeat
    would put a permanent false positive in front of the Phase 10 gate."""
    result = []
    for node_id in nodes:
        segments = node_id.split("-")
        if any(a == b for a, b in zip(segments, segments[1:])):
            result.append(node_id)
    return sorted(result)


def analyze(lab_dir: Path) -> dict:
    nodes, skipped = load_nodes(lab_dir)
    n = len(nodes)
    if n == 0:
        return {"error": f"no nodes found in {lab_dir}"}

    out_adj, in_adj, relations, dangling = build_edges(nodes)
    undirected_edges = {
        frozenset((src, dst))
        for src, targets in out_adj.items()
        for dst in targets
    }
    degrees = {i: len(out_adj.get(i, [])) + len(in_adj.get(i, [])) for i in nodes}
    values = sorted(degrees.values())
    mean_degree = sum(values) / n
    log2n = math.log2(n)
    max_degree_target = 2 * log2n
    band_low = log2n * (1 - DEGREE_BAND_TOLERANCE)
    band_high = log2n * (1 + DEGREE_BAND_TOLERANCE)
    depths = collections.Counter(depth_of(i, out_adj, nodes) for i in nodes)
    hubs = sorted(
        ((i, d) for i, d in degrees.items() if d > max_degree_target),
        key=lambda kv: -kv[1],
    )
    top_share = (
        sum(d for _, d in sorted(degrees.items(), key=lambda kv: -kv[1])[:2])
        / (2 * len(undirected_edges)) if undirected_edges else 0.0
    )

    return {
        "lab_dir": str(lab_dir),
        "nodes": n,
        "undirected_edges": len(undirected_edges),
        "non_node_files": skipped,
        "degree": {
            "mean": round(mean_degree, 2),
            "median": values[n // 2],
            "max": values[-1],
            "min": values[0],
            "variance": round(sum((v - mean_degree) ** 2 for v in values) / n, 1),
            "gini": round(gini(values), 3),
            "histogram": dict(sorted(collections.Counter(values).items())),
        },
        "targets": {
            "log2_n": round(log2n, 2),
            "edges_for_log2n_degree": round(n * log2n / 2),
            "edges_delta": round(n * log2n / 2) - len(undirected_edges),
            "depth_log2_log2_n": round(math.log2(log2n), 2) if log2n > 1 else 0.0,
            "max_degree_cap": round(max_degree_target, 1),
            "degree_band": [round(band_low, 1), round(band_high, 1)],
            "gini": GINI_TARGET,
        },
        "structure": {
            "depth_histogram": dict(sorted(depths.items())),
            "roots": sorted(i for i in nodes if not out_adj.get(i)),
            "in_degree_zero": sum(1 for i in nodes if not in_adj.get(i)),
            "pendants_degree_le_1": sum(1 for v in values if v <= 1),
            "in_band": sum(1 for v in values if band_low <= v <= band_high),
            "top2_edge_share": round(top_share, 3),
            "hubs_over_cap": [{"id": i, "degree": d} for i, d in hubs],
        },
        "defects": {
            "dangling_targets": sorted(dangling),
            "repeated_segment_ids": repeated_segment_ids(nodes),
        },
        "relations": dict(relations.most_common()),
        "kinds": dict(collections.Counter(x.get("kind") for x in nodes.values()).most_common()),
    }


def gate_failures(report: dict) -> list[str]:
    """The conditions Phase 10 must drive to zero. Kept separate from analyze() so the
    report stays a pure measurement and the gate stays a pure policy."""
    failures = []
    if report["degree"]["gini"] > GINI_TARGET:
        failures.append(
            f"gini {report['degree']['gini']} > target {GINI_TARGET}")
    if report["degree"]["max"] > report["targets"]["max_degree_cap"]:
        failures.append(
            f"max degree {report['degree']['max']} > cap {report['targets']['max_degree_cap']}")
    if report["defects"]["dangling_targets"]:
        failures.append(
            f"dangling edges: {', '.join(report['defects']['dangling_targets'])}")
    if report["defects"]["repeated_segment_ids"]:
        failures.append(
            f"repeated-segment ids: {', '.join(report['defects']['repeated_segment_ids'])}")
    if report["structure"]["pendants_degree_le_1"]:
        failures.append(
            f"{report['structure']['pendants_degree_le_1']} pendant node(s) with degree <= 1")
    return failures


def render(report: dict) -> str:
    d, t, s = report["degree"], report["targets"], report["structure"]
    lines = [
        f"Labyrinth balance report -- {report['lab_dir']}",
        "",
        f"  nodes {report['nodes']}   undirected edges {report['undirected_edges']}"
        f"   (target {t['edges_for_log2n_degree']}, delta {t['edges_delta']:+d})",
        "",
        "  DEGREE",
        f"    mean     {d['mean']:>7}   target ~{t['log2_n']} = log2(N)",
        f"    median   {d['median']:>7}",
        f"    max      {d['max']:>7}   cap {t['max_degree_cap']} = 2*log2(N)",
        f"    gini     {d['gini']:>7}   target < {t['gini']}",
        f"    variance {d['variance']:>7}",
        f"    in band {t['degree_band']}: {s['in_band']}/{report['nodes']} nodes",
        "",
        "  STRUCTURE",
        f"    depth histogram      {s['depth_histogram']}   target peak {t['depth_log2_log2_n']}",
        f"    roots                {s['roots']}",
        f"    never referenced     {s['in_degree_zero']}/{report['nodes']} (in-degree 0)",
        f"    pendants (deg <= 1)  {s['pendants_degree_le_1']}",
        f"    top-2 edge share     {s['top2_edge_share']:.1%}",
    ]
    if s["hubs_over_cap"]:
        lines.append("    hubs over cap:")
        lines += [f"        {h['degree']:>4}  {h['id']}" for h in s["hubs_over_cap"]]
    lines += [
        "",
        f"  RELATIONS  {report['relations']}",
        f"  KINDS      {report['kinds']}",
    ]
    defects = report["defects"]
    if defects["dangling_targets"] or defects["repeated_segment_ids"]:
        lines += ["", "  DEFECTS"]
        for target in defects["dangling_targets"]:
            lines.append(f"    dangling edge -> {target} (no such node)")
        for node_id in defects["repeated_segment_ids"]:
            lines.append(f"    repeated segment in id: {node_id}")
    if report["non_node_files"]:
        lines.append(f"\n  non-node files skipped: {', '.join(report['non_node_files'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lab-dir", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--gate", action="store_true", help="exit 1 if targets are missed")
    args = parser.parse_args()

    if not args.lab_dir.is_dir():
        print(f"no .lab directory at {args.lab_dir}", file=sys.stderr)
        return 2

    report = analyze(args.lab_dir)
    if "error" in report:
        print(report["error"], file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2) if args.json else render(report))

    if args.gate:
        failures = gate_failures(report)
        if failures:
            print("\nGATE FAILED:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
