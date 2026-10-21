"""0.5.7-A5 (T-19): the per-type default drives read one table.

``jaxfne.emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS`` owns each cell type's
``"drive"``. Each former duplicate site must follow a monkeypatched entry.
"""

import jaxfne as jtfne
import jaxfne.emitters as emitters
from jaxfne._construct_presets import suite2_celltype_presets
from jaxfne.tutorial_utils import make_cell_type_catalog

PATCHED_DRIVE = 6.25


def _canonical_e(monkeypatch):
    canonical = emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS
    assert canonical["E"]["drive"] != PATCHED_DRIVE, "fixture must change the value"
    monkeypatch.setitem(canonical["E"], "drive", PATCHED_DRIVE)


def test_config_drive_default_reads_canonical_table(monkeypatch):
    _canonical_e(monkeypatch)
    cfg = jtfne.Configuration().drive()
    assert cfg.metadata["drive"]["baseline_drive_by_cell_type"]["E"] == PATCHED_DRIVE


def test_tutorial_catalog_drive_reads_canonical_table(monkeypatch):
    _canonical_e(monkeypatch)
    assert make_cell_type_catalog()["E"].drive == PATCHED_DRIVE


def test_suite2_presets_drive_reads_canonical_table(monkeypatch):
    _canonical_e(monkeypatch)
    assert suite2_celltype_presets()["E"]["drive"] == PATCHED_DRIVE
