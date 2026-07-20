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
import pytest

from jaxfne import compilation_registry


@pytest.fixture(autouse=True)
def _reset_compilation_registry():
    compilation_registry.reset()
    yield
    compilation_registry.reset()
