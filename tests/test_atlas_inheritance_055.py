"""AT-00-R3: mechanical source-vs-registry inheritance check (import, not subprocess)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import artifacts.atlas.at_manifest as M
import scripts.check_atlas_inheritance as C

SOURCE = ROOT / "artifacts" / "project_sources" / "8_atlas.md"


def test_source_block_parses_to_registry_stage_sets():
    sets, notes = C.parse_source_sets(SOURCE)
    assert list(sets) == ["S1", "S2-4", "S5-7", "S8-9", "S10"]
    for stage in C.STAGE_ORDER:
        assert sets[stage] == set(M.STAGE_COMPONENTS[stage]), stage
    assert any("8_atlas.md" in n for n in notes)


def test_inheritance_valid_on_current_tree():
    errors, sets, _notes = C.check(SOURCE, ROOT)
    assert errors == []
    assert C.diff_against_registry(sets) == []
    assert M.inheritance_check() == []
