"""0.5.5 ENGINE items 3+4: Atlas spec registry, manifests, inheritance check."""

import importlib
import json
import pathlib

import pytest

import artifacts.atlas.at01_at06_052 as A052
import artifacts.atlas.at07_at04_053 as A053
import artifacts.atlas.at_manifest as M

# Cost keys written by the AT-02 runner: top-level "wall_s"
# (at01_at06_052.py:569), per-arm "wall_s" (:539) and "mem_peak_b" (via
# _retained, :497-502). Wall time and peak bytes vary run to run; everything
# else must reproduce exactly at fixed seeds.
_COST_KEYS = ("wall_s", "mem_peak_b")


def _without_cost(node):  # wall-time/memory keys stripped at every dict level
    if isinstance(node, dict):
        return {k: _without_cost(v) for k, v in node.items() if k not in _COST_KEYS}
    if isinstance(node, list):
        return [_without_cost(v) for v in node]
    return node


def test_registry_runners_resolve_to_callables():
    for at_id, entry in M.REGISTRY.items():
        mod_name, func_name = entry["runner"].split(":")
        mod = importlib.import_module(f"artifacts.atlas.{mod_name}")
        assert callable(getattr(mod, func_name, None)), at_id


def test_manifest_json_roundtrip_and_digest_stable():
    for at_id in M.REGISTRY:
        first = M.manifest(at_id)
        assert json.loads(json.dumps(first)) == first, at_id
        second = M.manifest(at_id)
        assert second["spec_digest"] == first["spec_digest"], at_id
        assert second["spec_digest"] == M.spec_digest(first["spec"]), at_id


def test_inheritance_check_empty_and_adversarial(monkeypatch):
    assert M.inheritance_check() == []
    orig = {stage: list(comps) for stage, comps in M.STAGE_COMPONENTS.items()}
    dropped = dict(orig)
    dropped["S5-7"] = [c for c in dropped["S5-7"] if c != "W"]
    monkeypatch.setattr(M, "STAGE_COMPONENTS", dropped)
    assert any("S5-7 drops parent components ['W']" in v for v in M.inheritance_check())
    adds_nothing = dict(orig)
    adds_nothing["S8-9"] = list(adds_nothing["S5-7"])
    monkeypatch.setattr(M, "STAGE_COMPONENTS", adds_nothing)
    assert any("S8-9 adds no component over S5-7" in v for v in M.inheritance_check())
    monkeypatch.setattr(M, "STAGE_COMPONENTS", orig)
    monkeypatch.setitem(M.REGISTRY, "AT-05", {**M.REGISTRY["AT-05"], "stage": "S1"})
    monkeypatch.setattr(M, "at_spec", lambda at_id: {"components": sorted(orig["S5-7"])})
    assert any(v.startswith("AT-05: components") for v in M.inheritance_check())


@pytest.mark.slow
@pytest.mark.parametrize("at_id", list(M.REGISTRY))
def test_regeneration_reproducible_at_manifest_seeds(at_id):
    assert M.at_spec("AT-02")["seeds"] == {"build": A052.SEED, "run": A052.SEED}
    run = M.resolve_runner(at_id)
    assert _without_cost(run()) == _without_cost(run()), at_id


def test_specs_carry_no_transcribed_literals_and_citations_resolve():
    root = pathlib.Path(M.__file__).resolve().parents[2]
    for at_id in M.REGISTRY:
        spec = M.at_spec(at_id)
        assert "transcribed" not in spec, at_id
        for c in spec["cited"]:
            text = (root / c["file"]).read_text(encoding="utf-8")
            assert c["anchor"] in text, (at_id, c["field"], c["anchor"])


def test_spec_constant_is_consumed_by_the_run(monkeypatch):
    """A spec input must drive the run, not only the spec (AT-10: 3 areas x 2)."""
    import artifacts.atlas.at01_at10_toy as TOY

    def shape():
        return list(M.resolve_runner("AT-10")()["spikes"]["shape"])

    base = shape()
    monkeypatch.setattr(TOY, "AT10_N_PER_AREA", 3)
    assert M.at_spec("AT-10")["inputs"]["n_per_area"] == 3
    assert shape() == [base[0], 9] != base


def test_spec_reads_runner_constants_not_retyped(monkeypatch):
    before = M.at_spec("AT-07")
    assert before["seeds"]["run"] == A053.RUN_SEED
    monkeypatch.setattr(A053, "RUN_SEED", 999)
    after = M.at_spec("AT-07")
    assert after["seeds"]["run"] == 999
    assert after != before
