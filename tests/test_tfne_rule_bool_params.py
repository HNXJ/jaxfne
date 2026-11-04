"""TFNE rule property allow_self: the bare words true and false are read
exactly, the way `associative` is read. bool("false") is True, so
`allow_self = false` used to read as true; any other value is refused.
"""

import pytest

from jaxfne.tfne import TFNEError, parse, realize, resolve

PROG = """
O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5; allow_self = %s];
A := [C = {E}; N = 3];
B := [C = {E}; N = 3];
V := A O[k] B;
x : V : y
"""


def _realize(word):
    program = parse(PROG % word)
    return realize(resolve(program), program, seed=0)


@pytest.mark.parametrize(
    "word, expected",
    [("true", True), ("false", False), ("True", True), ("FALSE", False)],
)
def test_allow_self_words_are_read_exactly(word, expected):
    assert parse(PROG % word).rules["k"].params["allow_self"] == word
    realized = _realize(word)
    (origin,) = realized.I["rule_origins"].values()
    assert origin["params"]["allow_self"] is expected


@pytest.mark.parametrize("bad", ["maybe", "1", "0"])
def test_allow_self_other_values_are_refused(bad):
    with pytest.raises(TFNEError, match="allow_self"):
        _realize(bad)
