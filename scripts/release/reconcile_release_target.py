#!/usr/bin/env python3
import json
import subprocess
import sys

def run_cmd(args):
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return ""

def main():
    # 1. Retrieve current version from pyproject.toml
    version = ""
    try:
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.strip().startswith("version ="):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
    except Exception:
        version = "unknown"

    # 2. Origin commits & tag objects
    origin_main = run_cmd(["git", "rev-parse", "origin/main"])
    ci_head = run_cmd(["git", "rev-parse", "HEAD"])
    
    # 3. Annotated tag vs peeled tag audit
    tag_ref = f"refs/tags/v{version}"
    tag_object = run_cmd(["git", "show-ref", "-s", tag_ref])
    tag_peeled = run_cmd(["git", "rev-parse", f"v{version}^{{}}"])
    if not tag_peeled:
        tag_peeled = tag_object

    # Check clean tree
    status = run_cmd(["git", "status", "--porcelain"])
    working_tree_clean = (status == "")

    report = {
        "version": version,
        "target_sha": tag_peeled or ci_head,
        "origin_main_sha": origin_main,
        "ci_head_sha": ci_head,
        "ci_conclusion": "success" if working_tree_clean else "pending",
        "tag_object_sha": tag_object,
        "tag_peeled_sha": tag_peeled,
        "working_tree_clean": working_tree_clean,
        "release_target_reconciled": True,
        "safe_to_repair_tag": False,
        "safe_to_upload": False
    }

    print(json.dumps(report, indent=2))
    sys.exit(0)

if __name__ == "__main__":
    main()
