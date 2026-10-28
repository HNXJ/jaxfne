"""Build the grid cache once through fig3_analysis._load_grid (read-only on the run tree).

Run: C:/workspace/venv_py314_jax011/Scripts/python.exe -I C:/workspace/pcl_diag/build_cache.py
"""
import os
import sys

DIAG_DIR = os.environ.get("PCL_DIAG_DIR", "")
if not DIAG_DIR:
    sys.exit("PCL_DIAG_DIR is unset: set it to the PCL diagnostic workspace folder")
sys.path.insert(0, DIAG_DIR)

import diag_common as dc  # noqa: E402

data = dc.load_grid()
print("cache", dc.CACHE)
print("counts_s shape", data["counts_s"].shape, "counts_c shape", data["counts_c"].shape)
print("cells_s", int(data["cells_s"][0]), "to", int(data["cells_s"][-1]), "n", len(data["cells_s"]))
print("cells_c", int(data["cells_c"][0]), "to", int(data["cells_c"][-1]), "n", len(data["cells_c"]))
print("time keys simple", len(data["times_s"]), "complex", len(data["times_c"]))
print("total counts simple", int(data["counts_s"].sum()), "complex", int(data["counts_c"].sum()))
print("build_cache DONE")
