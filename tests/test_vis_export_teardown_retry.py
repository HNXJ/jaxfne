"""P-011: a Kaleido browser-teardown error is retried once; other errors propagate."""

import pytest

from jaxfne.vis import exporters

TEARDOWN = RuntimeError("Couldn't close or kill browser subprocess")


class _Fig:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def write_image(self, out):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


def test_teardown_error_is_retried_once():
    fig = _Fig([TEARDOWN])
    exporters._write_image(fig, "x.png")
    assert fig.calls == 2


@pytest.mark.parametrize("errors", [[TEARDOWN, TEARDOWN], [RuntimeError("other")]])
def test_repeated_or_other_errors_propagate(errors):
    with pytest.raises(RuntimeError):
        exporters._write_image(_Fig(errors), "x.png")
