"""0.5.5 ENGINE items 3+4: Atlas spec registry, manifests, inheritance check."""

import importlib
import json

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
def test_at02_regeneration_reproducible_at_manifest_seeds():
    spec = M.at_spec("AT-02")
    assert spec["seeds"] == {"build": A052.SEED, "run": A052.SEED}
    first = A052.run_at02()
    second = A052.run_at02()
    assert _without_cost(first) == _without_cost(second)


def test_spec_reads_runner_constants_not_retyped(monkeypatch):
    before = M.at_spec("AT-07")
    assert before["seeds"]["run"] == A053.RUN_SEED
    monkeypatch.setattr(A053, "RUN_SEED", 999)
    after = M.at_spec("AT-07")
    assert after["seeds"]["run"] == 999
    assert after != before
