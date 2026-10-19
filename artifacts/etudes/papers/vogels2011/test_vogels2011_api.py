"""Smoke test for the vogels2011 lane API (one test: build the spec)."""

from artifacts.etudes.papers.vogels2011 import vogels2011_api as L


def test_build_spec_smoke():
    """The scaffold spec compiles edges without simulating (fixture must build)."""
    spec = L.build_spec(n=20, seed=0)
    assert spec is not None, "fixture must build the lane spec"
