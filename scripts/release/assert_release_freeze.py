#!/usr/bin/env python3
import sys
import subprocess

def run_cmd(args):
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return ""

def main():
    # Read CLAUDE.md or AGENTS.md to check for active release locks/freeze
    freeze_active = False
    intended_sha = ""
    try:
        with open("CLAUDE.md", "r") as f:
            content = f.read()
            if "release_freeze: true" in content or "freeze_active: true" in content:
                freeze_active = True
            for line in content.splitlines():
                if "intended_release_sha" in line:
                    intended_sha = line.split(":")[-1].strip().strip('"').strip("'")
    except Exception:
        pass

    if not freeze_active:
        print("Release freeze is not active. Pass.")
        sys.exit(0)

    # Check HEAD SHA against intended_release_sha
    head_sha = run_cmd(["git", "rev-parse", "HEAD"])
    if intended_sha and head_sha != intended_sha:
        print(f"ERROR: Release freeze is ACTIVE, but HEAD SHA '{head_sha}' differs from intended_release_sha '{intended_sha}'!")
        sys.exit(1)

    print("Success: Release freeze active and HEAD SHA matches intended_release_sha.")
    sys.exit(0)

if __name__ == "__main__":
    main()
