"""0.5.5 item 5c: adversarial semantic substitutions against the agent surface.

Each case substitutes one meaning (units, type, parameter, mechanism, time
grid, API name) and requires a refusal or a FAIL, never a silent PASS.
"""

import dataclasses
import importlib
import re
from pathlib import Path

import pytest

import jaxfne as J
from jaxfne import agent as A

AB = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"
TASK_SKILLS = ("model", "network", "state", "plasticity", "fields", "simulate", "verify", "inspect")


def _spec(mech="AMPA", delay=4.0):
    return (f"O[k] := [direction = >; mechanism = {mech}; probability = 1.0; "
            f"weight = 0.5; delay = {delay}];\n{AB}\nx : A O[k] B : y\n")


def _run(mech="AMPA", dt_ms=1.0):
    return A.simulate(A.realize(_spec(mech), seed=11), duration_ms=30.0, dt_ms=dt_ms, seed=0)


def test_time_units_and_types_are_refused_not_rounded():
    r = A.realize(_spec(), seed=11)
    for dt in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            A.simulate(r, duration_ms=30.0, dt_ms=dt, seed=0)
    with pytest.raises(TypeError, match="real number"):
        A.simulate(r, duration_ms=30.0, dt_ms="1.0", seed=0)
    # 30.5 ms at dt 1 ms would run 30 steps and still report time EQUAL.
    with pytest.raises(ValueError, match="whole number"):
        A.simulate(r, duration_ms=30.5, dt_ms=1.0, seed=0)
    # Seconds passed as ms: 0.03 is not a whole number of 1 ms steps.
    with pytest.raises(ValueError, match="whole number"):
        A.simulate(r, duration_ms=0.03, dt_ms=1.0, seed=0)
    with pytest.raises(TypeError, match="seed"):
        A.simulate(r, duration_ms=30.0, dt_ms=1.0)


def test_parameter_substitutions_fail_their_class():
    declared = _run("AMPA")
    swapped = dataclasses.replace(_run("NMDA"), realization=declared.realization)
    verdicts = {p: A.verify(swapped, p)["verdict"]
                for p in ("weight_identity", "mechanism_identity", "delay_identity")}
    assert verdicts == {"weight_identity": "PASS", "mechanism_identity": "FAIL",
                        "delay_identity": "PASS"}
    # Executed at 1 ms, recorded as 0.5 ms: both the grid and the delay steps disagree.
    relabeled = dataclasses.replace(declared, dt_ms=0.5)
    assert A.verify(relabeled, "time_identity")["verdict"] == "FAIL"
    assert A.verify(relabeled, "delay_identity")["verdict"] == "FAIL"
    with pytest.raises(J.tfne.TFNEError, match="E_MECHANISM_UNRESOLVED"):
        _run("GABA")  # ambiguous between GABA_A and GABA_B


_QUALIFIED = re.compile(r"`((?:jaxfne|Model|Configuration)(?:\.\w+)+)")


def _stale_symbols(text):
    """Qualified names in a skill that do not resolve against live code."""
    stale = []
    for name in _QUALIFIED.findall(text):
        head, *attrs = name.split(".")
        if head == "jaxfne":
            obj, i = importlib.import_module("jaxfne"), 0
            while i < len(attrs) and hasattr(obj, attrs[i]):
                obj, i = getattr(obj, attrs[i]), i + 1
            while i < len(attrs):
                try:
                    obj = importlib.import_module(f"{obj.__name__}.{attrs[i]}")
                    i += 1
                except (ImportError, AttributeError):
                    break
            ok = i == len(attrs)
        else:
            cls = getattr(J, head)
            fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
            ok = len(attrs) == 1 and (hasattr(cls, attrs[0]) or attrs[0] in fields)
        if not ok:
            stale.append(name)
    return stale


def test_task_skills_name_only_live_symbols():
    texts = {n: Path(f"artifacts/skills/jaxfne-{n}/SKILL.md").read_text(encoding="utf-8")
             for n in TASK_SKILLS}
    assert sum(len(_QUALIFIED.findall(t)) for t in texts.values()) >= 20
    assert {n: _stale_symbols(t) for n, t in texts.items() if _stale_symbols(t)} == {}
    # A renamed API is caught, not skipped.
    renamed = texts["verify"].replace("jaxfne.agent.verify", "jaxfne.agent.check")
    assert set(_stale_symbols(renamed)) == {"jaxfne.agent.check"}
    assert _stale_symbols("`Model.simulate_many`") == ["Model.simulate_many"]
