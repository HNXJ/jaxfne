import sys
import json

# Record sys.modules keys before importing jaxfne
before_import = set(sys.modules.keys())

# Perform root import
import jaxfne as jtfne

# Record sys.modules keys after importing jaxfne
after_import = set(sys.modules.keys())

imported = sorted(list(after_import - before_import))

# Detect heavy third-party packages
heavy_packages = ["matplotlib", "plotly", "pandas", "optax", "jaxley"]
loaded_heavy = []
for pkg in heavy_packages:
    # check if any loaded module starts with pkg name
    if any(m == pkg or m.startswith(pkg + ".") for m in imported):
        loaded_heavy.append(pkg)

report = {
    "imported_modules_count": len(imported),
    "loaded_heavy_packages": loaded_heavy,
    "is_lightweight": len(loaded_heavy) == 0,
    "imported_list": imported,
}

print(json.dumps(report, indent=2))
