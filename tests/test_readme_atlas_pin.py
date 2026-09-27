"""The published canonical atlas stays pinned to the realized canonical configuration.

A correctness repair that changes the realized canonical column (0.5.2 item 2 did)
must fail here, so the atlas is regenerated and re-pinned on purpose instead of
going stale behind a generator nobody reruns.
"""

import importlib.util
from pathlib import Path

import jaxfne as jtfne

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_readme_atlas.py"


def test_canonical_config_hash_matches_the_pin():
    spec = importlib.util.spec_from_file_location("generate_readme_atlas", SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    model = jtfne.construct(
        jtfne.load_canonical_neuronal_tensor(gen.CANONICAL_TENSOR),
        jtfne.RuntimeConfiguration(seed=gen.CANONICAL_SEED, duration_ms=gen.CANONICAL_DURATION_MS,
                                   dt_ms=gen.CANONICAL_DT_MS),
    )
    assert model.summary()["config_hash"] == gen.EXPECTED_CONFIG_HASH, (
        "canonical column changed: regenerate the atlas "
        "(python scripts/generate_readme_atlas.py) and re-pin EXPECTED_CONFIG_HASH"
    )
