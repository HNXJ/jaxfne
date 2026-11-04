"""Session-wide pytest fixtures.

Only one concern lives here: resetting jaxfne's module-level
`compilation_registry` singleton between tests. Confirmed via source
(`jaxfne/validation.py::CompilationRegistry`) that it tracks per-function
shape-signature baselines in `self.baselines`, persisted for the lifetime of
the Python process -- with no per-test isolation, a test earlier in the run
order can set a baseline shape for e.g. `simulate`, and an unrelated later
test calling `simulate` with a different (but individually-correct) shape
trips a false "Re-compilation guard alert" that has nothing to do with any
real recompilation inside that later test. Only
`tests/test_v0320_recompilation_guards.py`'s own tests called
`compilation_registry.reset()` themselves; nothing reset it for the rest of
the suite, so a full-suite run's alert count depended on test order rather
than actual behavior.
"""
import jax
import pytest

from jaxfne import compilation_registry


@pytest.fixture(autouse=True)
def _reset_jax_x64_between_tests(request):
    """Protocol analysis modules may enable x64 at import; reset before each test."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    if "test_protocol_w_" in request.node.fspath.basename:
        jax.config.update("jax_enable_x64", True)
    elif request.node.get_closest_marker("x64") is not None:
        jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(autouse=True)
def _reset_compilation_registry():
    compilation_registry.reset()
    yield
    compilation_registry.reset()


JUNIT_MARKER_PROPERTY = "jaxfne_marker"
# Markers the JUnit comparator (scripts/compare_pytest_junit.py) classifies.
JUNIT_MARKERS = ("data_gated", "optional_oracle", "bitwise_diagnostic")


def pytest_collection_modifyitems(config, items):
    """Record the JUnit-classified markers as user properties.

    Pytest writes them into the testcase's <properties> block under the default xunit2 family.
    Appending to item.user_properties avoids record_property, which warns under xunit2.
    """
    for item in items:
        for name in JUNIT_MARKERS:
            if item.get_closest_marker(name) is not None:
                item.user_properties.append((JUNIT_MARKER_PROPERTY, name))

