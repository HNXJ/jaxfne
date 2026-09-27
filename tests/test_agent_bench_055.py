"""0.5.5 item 5d: frozen Atlas task set and its scorer (agent-native steps 7 and 9)."""

import dataclasses
import json
from pathlib import Path

import numpy as np

from artifacts.atlas.at_bundle import bundle
from artifacts.atlas.at_manifest import REGISTRY, at_spec, spec_digest
import pytest

from artifacts.benchmark.agent_bench import (
    ARMS, CLASSES, TASK_SET, freeze_task, packet, prepare_worktree, score)

TASKS = {t["at_id"]: t for t in json.loads(TASK_SET.read_text(encoding="utf-8"))["tasks"]}


# Specs corrected after the freeze without changing any output (H1, 2026-09-27): these
# recorded n_contacts=4 while the executed field had 16 contacts. The frozen tasks and
# their scored results stay byte-for-byte; the correction is the only allowed difference.
CORRECTED_AFTER_FREEZE = {at: {("recording", "n_contacts"): (4, 16)} for at in ("AT-02", "AT-03", "AT-04", "AT-05")}


def _leaf_diff(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        return {d: v for k in set(a) | set(b) for d, v in _leaf_diff(a.get(k), b.get(k), (*path, k)).items()}
    return {} if a == b else {path: (a, b)}


def test_task_set_covers_registry_with_current_specs():
    added_after_freeze = ["AT-10-N20"]  # registered after the 0.5.5 task set was frozen
    assert list(TASKS) == [k for k in REGISTRY if k not in added_after_freeze]
    for at_id, task in TASKS.items():
        live = json.loads(json.dumps(at_spec(at_id)))
        if at_id in CORRECTED_AFTER_FREEZE:
            assert _leaf_diff(task["spec"], live) == CORRECTED_AFTER_FREEZE[at_id], at_id
        else:
            assert task["spec_digest"] == spec_digest(at_spec(at_id)), at_id
        assert task["intent"] and task["arms"], at_id


def test_reference_reproduces_frozen_task_and_scores_full():
    fresh, frozen = freeze_task("AT-10"), TASKS["AT-10"]
    assert [fresh[k] for k in ("intent", "spec", "spec_digest")] == [
        frozen[k] for k in ("intent", "spec", "spec_digest")]
    ref = bundle("AT-10")
    full = score(TASKS["AT-10"], ref)
    assert not full["missing_arms"] and not full["extra_arms"]
    assert {c: v["score"] for c, v in full["classes"].items()} == {c: 1.0 for c in CLASSES}


def test_substituted_candidates_lose_the_right_class():
    ref = bundle("AT-10")["main"]
    sig = ref["signals"]
    short = dataclasses.replace(sig, time_ms=sig.time_ms[:10], V_m=sig.V_m[:10])
    s = score(TASKS["AT-10"], {"main": {**ref, "signals": short}})["classes"]
    assert s["execution"]["failed"] == ["main.n_steps"] and s["model"]["score"] == 1.0
    w0 = np.asarray([r["weight"] for r in ref["model"].edge_table()])
    s = score(TASKS["AT-10"], {"main": {**ref, "hdp": {"w_final": 2 * w0}}})["classes"]
    assert s["plasticity"]["failed"] == ["main.plastic"] and s["execution"]["score"] == 1.0
    # Same structure, different dynamics (e.g. another drive): only dynamics fails.
    shifted = dataclasses.replace(sig, V_m=sig.V_m + 1.0)
    s = score(TASKS["AT-10"], {"main": {**ref, "signals": shifted}})["classes"]
    assert s["dynamics"]["failed"] == ["main.mean_vm"]
    assert all(s[c]["score"] == 1.0 for c in CLASSES if c != "dynamics")
    renamed = score(TASKS["AT-10"], {"other": ref})
    assert renamed["missing_arms"] == ["main"] and renamed["extra_arms"] == ["other"]
    assert all(v["score"] == 0.0 for v in renamed["classes"].values())


def test_packets_leak_no_answers_and_prepare_refuses_a_plain_directory(tmp_path):
    for task in TASKS.values():
        texts = {arm: packet(task, arm) for arm in ARMS}
        for text in texts.values():
            assert "artifacts/atlas" not in text and "run_at" not in text, task["at_id"]
        assert "jaxfne-core" in texts["skills"] and "jaxfne-core" not in texts["direct"]
    for root in (tmp_path, Path.cwd()):  # no .git; this repository itself
        with pytest.raises(ValueError, match="not a disposable git checkout"):
            prepare_worktree(root, "direct")
