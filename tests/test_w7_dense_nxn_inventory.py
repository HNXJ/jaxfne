"""W7 inventory closure tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "artifacts" / "audit" / "w7_dense_nxn_inventory.json"


def test_w7_inventory_exists_and_closed():
    assert INVENTORY.is_file(), "run scripts/audit_w7_dense_nxn_sites.py to generate inventory"
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert data["w7_verdict"] == "CLOSED"
    assert data["unexplained_count"] == 0
    assert data["static_site_count"] >= 16
    assert len(data["measurements"]) >= 4


def test_w7_bounded_degree_dual_storage_W_documented():
    """W10/W11: edge_list backend can still retain a dense emitter.W (COMPATIBILITY_ONLY)."""
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    cell = next(c for c in data["measurements"] if c["cell"] == "bounded_degree_N1000_K100")
    assert cell["realized"]["recurrent_backend"] == "edge_list"
    assert cell["realized"]["k_max_in_realized"] == 100
    w = next(a for a in cell["nxn_arrays_materialized"] if ".W" in a["ARRAY"])
    assert tuple(w["SHAPE"]) == (1000, 1000)
    assert w["R_k"] > 1000  # class-shared / unused dense W (see W10 U_k=0)


def test_w7_sparse_direct_has_no_nxn_persistent_arrays():
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    cell = next(c for c in data["measurements"] if c["cell"] == "sparse_direct_N6000_p01")
    assert cell["realized"]["recurrent_backend"] == "edge_list"
    assert cell["nxn_count"] == 0
    assert cell["nxn_bytes_total"] == 0
