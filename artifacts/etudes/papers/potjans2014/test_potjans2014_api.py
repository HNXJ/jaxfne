"""Smoke test for the potjans2014 lane API (one test: build the spec)."""

from artifacts.etudes.papers.potjans2014 import potjans2014_api as L


def test_build_spec_smoke():
    """The scaffold spec compiles edges without simulating (fixture must build)."""
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"
