#!/usr/bin/env python3
import os
import sys
import tarfile
import zipfile

def main():
    if not os.path.exists("dist"):
        print("No 'dist' directory found. Nothing to validate. Pass.")
        sys.exit(0)

    files = os.listdir("dist")
    if not files:
        print("Empty 'dist' directory. Pass.")
        sys.exit(0)

    print(f"Inspecting {len(files)} built artifacts in 'dist'...")
    for f in files:
        fpath = os.path.join("dist", f)
        if f.endswith(".tar.gz"):
            try:
                with tarfile.open(fpath, "r:gz") as tar:
                    members = tar.getnames()
                    print(f"✓ Tarball {f} contains {len(members)} entries.")
                    # Verify basic structure
                    if not any("pyproject.toml" in m for m in members):
                        print("✗ ERROR: Missing pyproject.toml inside source distribution!")
                        sys.exit(1)
            except Exception as e:
                print(f"✗ ERROR parsing sdist {f}: {e}")
                sys.exit(1)
        elif f.endswith(".whl"):
            try:
                with zipfile.ZipFile(fpath, "r") as zip_ref:
                    members = zip_ref.namelist()
                    print(f"✓ Wheel {f} contains {len(members)} entries.")
                    if not any("METADATA" in m for m in members):
                        print("✗ ERROR: Missing METADATA file inside wheel distribution!")
                        sys.exit(1)
            except Exception as e:
                print(f"✗ ERROR parsing wheel {f}: {e}")
                sys.exit(1)

    print("Success: Built artifacts validated successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
