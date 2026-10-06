"""Atlas manuscript progression (AT-00-R6).

The five stages and the simulations each covers, taken from the section
grouping of ``artifacts/project_sources/8_atlas.md`` (S1 anchor; S2-S4 pair
experiments; S5-S7 emergence; S8-S9 two-area systems; S10 synthesis).
``check_progression`` verifies a manuscript outline against that order.
"""

from __future__ import annotations

from typing import Any

__all__ = ["STAGES", "SIM_ORDER", "check_progression"]

STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("one physical neuron", ("S1",)),
    ("interacting neurons", ("S2", "S3", "S4")),
    ("emergent population field", ("S5", "S6", "S7")),
    ("adaptive interacting areas", ("S8", "S9")),
    ("genome-defined multiarea model", ("S10",)),
)

SIM_ORDER: tuple[str, ...] = tuple(s for _, sims in STAGES for s in sims)


def check_progression(outline: list[dict[str, Any]]) -> list[str]:
    """Defects of a manuscript outline ``[{"stage": name, "sims": [...]}]``.

    Stages must follow ``STAGES`` order, each stage must carry exactly its
    simulations in order, and no simulation may be missing or repeated.
    """
    out: list[str] = []
    names = [o.get("stage") for o in outline]
    if names != [n for n, _ in STAGES]:
        out.append(f"stage order {names} != {[n for n, _ in STAGES]}")
    for o, (name, sims) in zip(outline, STAGES):
        if list(o.get("sims", [])) != list(sims):
            out.append(f"{name}: sims {list(o.get('sims', []))} != {list(sims)}")
    return out
