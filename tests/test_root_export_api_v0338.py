"""v0.3.38: the v0.3.37 root export grammar is now formal ``jaxfne.__all__`` API.

The six export/figure helpers were root-importable since v0.3.37 but absent from
``__all__``. This test pins them as first-class public symbols and guards against
regression of the API-grammar contract.
"""

EXPECTED = {
    "save_figure",
    "save_figures",
    "export_report",
    "export_tutorial_artifacts",
    "plot_raster",
    "plot_spectrolaminar_suite",
}


def test_v0337_export_helpers_are_formal_root_api():
    import jaxfne as jtfne

    missing_attrs = {name for name in EXPECTED if not hasattr(jtfne, name)}
    missing_all = EXPECTED - set(jtfne.__all__)

    assert not missing_attrs, f"root-accessible attrs missing: {sorted(missing_attrs)}"
    assert not missing_all, f"not in __all__: {sorted(missing_all)}"


def test_export_helpers_are_callable():
    import jaxfne as jtfne

    for name in EXPECTED:
        assert callable(getattr(jtfne, name)), f"{name} is not callable"


def test_no_duplicate_public_names():
    import jaxfne as jtfne

    dupes = sorted({n for n in jtfne.__all__ if jtfne.__all__.count(n) > 1})
    assert not dupes, f"duplicate names in __all__: {dupes}"
