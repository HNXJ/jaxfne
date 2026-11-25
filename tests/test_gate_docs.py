"""The docs gate and the dev gate run the prose hygiene tests.

2026-10-27: a docs edit passed the docs gate and CI Fast with three "X, not Y" hits that only
the broad gate's ``tests/test_public_docs_hygiene.py`` caught.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

import scripts.run_test_gate as rtg

HYGIENE = "tests/test_public_docs_hygiene.py"
PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"


def test_dev_gate_collects_the_docs_hygiene_tests():
    assert HYGIENE in rtg.DEV_PYTEST_TARGETS


def test_docs_gate_runs_the_docs_hygiene_tests(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(rtg, "_run", lambda cmd, family=None, **kw: calls.append((family, cmd)))
    rtg.gate_docs()
    assert calls, "the docs gate ran nothing"
    pytest_calls = [cmd for family, cmd in calls if family == "pytest_docs"]
    assert len(pytest_calls) == 1
    assert HYGIENE in pytest_calls[0]
    assert "pytest_docs" in rtg.CHECK_FAMILIES
    assert "pytest_docs" in rtg.GATE_CHECK_FAMILIES["docs"]


def test_docs_gate_runs_the_tree_size_check(monkeypatch):
    """2026-11-24: a docs lane added one test file and merged at files=2826 > budget_files=2825."""
    calls: list[str] = []
    monkeypatch.setattr(rtg, "_run", lambda cmd, family=None, **kw: calls.append(family))
    rtg.gate_docs()
    assert "tree_size_audit" in calls
    assert "tree_size_audit" in rtg.GATE_CHECK_FAMILIES["docs"]


def test_gate_env_defaults_without_gpu_preallocation_and_keeps_caller_values(monkeypatch):
    """2026-11-25: xdist workers each preallocated GPU memory (9.5 h hang); CPU is the default."""
    try:
        monkeypatch.delenv(PREALLOCATE, raising=False)
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
        importlib.reload(rtg)
        assert rtg.PYTEST_ENV[PREALLOCATE] == "false"
        assert rtg.PYTEST_ENV["JAX_PLATFORMS"] == "cpu"
        monkeypatch.setenv(PREALLOCATE, "true")
        monkeypatch.setenv("JAX_PLATFORMS", "gpu")
        importlib.reload(rtg)
        assert rtg.PYTEST_ENV[PREALLOCATE] == "true"
        assert rtg.PYTEST_ENV["JAX_PLATFORMS"] == "gpu"
    finally:
        monkeypatch.undo()
        importlib.reload(rtg)


def test_gate_step_hang_becomes_a_failure(monkeypatch):
    """A step past the cap fails with 124 (the coreutils timeout code) instead of hanging."""
    monkeypatch.setattr(rtg, "GATE_STEP_TIMEOUT_S", 2.0)
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as err:
        rtg._run([sys.executable, "-c", "import time; time.sleep(30)"], family=None)
    assert err.value.returncode == 124
    assert time.monotonic() - started < 15


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX")
def test_gate_step_timeout_kills_grandchildren(monkeypatch, tmp_path):
    """pytest-xdist workers are grandchildren: the cap must take the whole process group."""
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setattr(rtg, "GATE_STEP_TIMEOUT_S", 3.0)
    with pytest.raises(subprocess.CalledProcessError) as err:
        rtg._run([sys.executable, str(parent), str(pid_file)], family=None)
    assert err.value.returncode == 124
    assert pid_file.exists(), "fixture must start the grandchild before the cap"
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    # SIGKILL is asynchronous and the orphan is reaped by init, so poll for its disappearance.
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "grandchild still alive 5 s after the cap"
        time.sleep(0.05)


def _is_dead(pid: int) -> bool:
    """True once the pid is gone or a zombie (killed, not yet reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


DRIVER_SRC = """\
import sys
sys.path.insert(0, sys.argv[2])
import scripts.run_test_gate as rtg
rtg._run([sys.executable, sys.argv[3], sys.argv[1]], family=None)
"""

STEP_SRC = """\
import os, pathlib, subprocess, sys, time
grand = subprocess.Popen(["sleep", "300"])
pathlib.Path(sys.argv[1]).write_text(f"{os.getpid()} {grand.pid}", encoding="utf-8")
time.sleep(300)
"""


@pytest.mark.skipif(os.name != "posix", reason="process-group signals are POSIX")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_signal_to_gate_stops_the_step_tree(tmp_path, sig):
    """A SIGTERM or SIGHUP to the gate stops the step's group; nothing is orphaned."""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER_SRC, encoding="utf-8")
    step = tmp_path / "step.py"
    step.write_text(STEP_SRC, encoding="utf-8")
    pid_file = tmp_path / "pids.txt"
    gate = subprocess.Popen(
        [sys.executable, str(driver), str(pid_file), str(rtg.ROOT), str(step)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 20
        while True:
            text = pid_file.read_text(encoding="utf-8") if pid_file.exists() else ""
            if len(text.split()) == 2:
                break
            assert time.monotonic() < deadline, "step never started"
            time.sleep(0.05)
        pids = [int(p) for p in text.split()]
        gate.send_signal(sig)
        returncode = gate.wait(timeout=30)
        deadline = time.monotonic() + 10
        for pid in pids:
            while not _is_dead(pid):
                assert time.monotonic() < deadline, f"pid {pid} outlived the gate's {sig.name}"
                time.sleep(0.1)
        assert returncode == 128 + sig
    finally:
        if gate.poll() is None:
            gate.kill()
        for pid in pids[:1]:
            try:
                os.killpg(pid, signal.SIGKILL)  # the step leads its own group
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_slow_collection_runs_on_cpu_with_a_cap(monkeypatch):
    """2026-11-25: the slow-node collection inherited the GPU env and had no cap."""
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="tests/test_x.py::test_a\n", stderr="")

    monkeypatch.setattr(rtg.subprocess, "run", fake_run)
    monkeypatch.setattr(rtg, "GATE_STEP_TIMEOUT_S", 7.0)
    assert rtg._slow_node_ids() == frozenset({"tests/test_x.py::test_a"})
    assert seen["env"]["JAX_PLATFORMS"] == "cpu"
    assert seen["timeout"] == 7.0


def test_slow_collection_timeout_fails_loudly(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(rtg.subprocess, "run", fake_run)
    monkeypatch.setattr(rtg, "GATE_STEP_TIMEOUT_S", 7.0)
    with pytest.raises(RuntimeError, match="rc 124"):
        rtg._slow_node_ids()


def test_isolated_wheel_smoke_env_defaults_to_cpu(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(rtg, "_run", lambda cmd, **kwargs: seen.update(kwargs))
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv(PREALLOCATE, raising=False)
    rtg.run_isolated_wheel_smoke(sys.executable, "0.0.0", cwd=tmp_path)
    assert seen["env"]["JAX_PLATFORMS"] == "cpu"
    assert seen["env"][PREALLOCATE] == "false"


@pytest.mark.skipif(os.name != "posix", reason="signal handlers are POSIX here")
def test_capped_run_skips_restoring_a_handler_python_cannot_reinstall(monkeypatch):
    """signal.signal(sig, None) raises TypeError, so a previous handler of None is not restored."""
    calls: list[tuple[int, object]] = []

    def fake_signal(sig, handler):
        calls.append((sig, handler))
        return None

    monkeypatch.setattr(rtg.signal, "signal", fake_signal)
    rtg._run_capped([sys.executable, "-c", "pass"], cwd=rtg.ROOT, env=rtg.PYTEST_ENV)
    assert calls, "fixture must install the stop handlers"
    assert all(handler is not None for _, handler in calls)
