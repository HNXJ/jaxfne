"""PCL R2 comparison harness: run the event oracle on a C++-layout network folder.

The authors' C++ side runs separately in WSL (``neuvisys-exe <net> <events.h5> <nbPass>``);
this module only reads its output folder. It runs the pure-path event oracle
(``pcl_event_ref.PCLEventRef``, learning on) on a network folder in the C++ layout, saves
the result back into that folder in the same layout the C++ ``save()`` writes, and compares
two network folders after a run.

C++ facts (the authors' source in WSL ``~/pcl_cpp``):
- Init weights are unseeded (``WeightMatrix.cpp:7``), so both arms start from one copied
  network folder (``init_network``, ``copy_network``); the ``seed`` of ``run_oracle`` only
  seeds the constructor's init weights, which loading overwrites.
- ``Util::winnerTakeAll`` uses a time-seeded ``std::sample``, so R2 runs C++ against C++
  (a null) before reading oracle-vs-C++ differences.
- ``save()`` calls ``resetAllNeurons()`` (``NetworkHandle.cpp:200``) before
  ``saveNetwork()`` (233), and ``Neuron::resetNeuron`` zeroes ``m_totalSpike`` and clears
  ``m_trackingSpikeTrain`` (``Neuron.cpp:616, 626``), so every saved ``<idx>.json``
  carries ``count_spike`` 0 and an empty ``spike_train``; threshold, rate and decay are
  the reset values while weights survive.
- Layer-0 shared weights are written ``weights/0/<k>.npy`` with shape ``(2, 1, 1, W, H)``
  (polarity, camera, synapse, x, y), float64; the oracle only supports one camera and
  one synapse, so the two inserted axes are size 1. ``<k>lli.npy`` and the layer-1
  files already match the C++ shapes.
- R2 ``compare`` decides on weights (rtol 1e-12); the saved ``count_spike``
  fields are still read (a missing field is refused) and reported per layer
  (``cells_equal_count``, ``counts_reset``: true when every count on both
  sides is 0) but do not gate ``exact``. Spike evidence is the run result
  (``n_spikes``, ``last_spike_t``), which the saved folder no longer carries.

``save_network`` mirrors that order: reset the neurons, then write. ``networkState.json``
is not written: its schema is unknown here and neither the loader nor ``compare`` reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


# P-025 repo-root guard in the detector-recognized form (five parents up from
# artifacts/etudes/papers/pcl/port/).
_ROOT = str(Path(__file__).resolve().parents[5])
_HERE = Path(__file__).resolve().parent
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import pcl_event_ref as ref  # noqa: E402

RTOL = 1e-12  # weight agreement for `exact`, as the port's replay pins use


def copy_network(src, dst) -> Path:
    """Byte-copy a network folder; refuse when ``dst`` already exists."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise ValueError(f"copy_network: source folder is missing: {src}")
    if dst.exists():
        raise FileExistsError(f"copy_network: refusing an existing dst: {dst}")
    shutil.copytree(src, dst)
    return dst


def _code_sha(root=None) -> str:
    """The git HEAD SHA, read from the ``.git`` files (no subprocess)."""
    root = Path(root) if root is not None else Path(_ROOT)
    git = root / ".git"
    if git.is_file():  # a worktree pointer: "gitdir: <path>"
        first = git.read_text(encoding="utf-8").strip()
        if first.startswith("gitdir:"):
            target = first[len("gitdir:"):].strip()
            git = Path(target) if Path(target).is_absolute() else root / target
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        p = git / head[len("ref:"):].strip()
        if p.is_file():
            head = p.read_text(encoding="utf-8").strip()
        else:  # packed ref
            want = head[len("ref:"):].strip()
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or line.startswith("^"):
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == want:
                    head = sha.strip()
                    break
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError(f"unresolvable git HEAD in {git}")
    return head


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(net_dir) -> tuple[str, int]:
    """Combined sha256 over a folder: sorted ``relpath + NUL + bytes`` of every file."""
    root = Path(net_dir)
    h = hashlib.sha256()
    n = 0
    for p in sorted((p for p in root.rglob("*") if p.is_file()),
                    key=lambda p: p.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + p.read_bytes())
        n += 1
    return h.hexdigest(), n


def _reset_neurons(net: "ref.PCLEventRef") -> None:
    """``Neuron::resetNeuron`` as ``save()`` applies it: state to reset, weights kept.

    Potential, adaptation, timers and counters go to 0, the threshold to ``VTHRESH``, the
    rate average to 0 and the learning decay to 1.0; the STDP and inhibition buffers
    empty. The spike list survives so the written ``spike_train`` names every spike.
    """
    for cells, cfg in ((net.s_cells, net.scfg), (net.c_cells, net.ccfg)):
        for cell in cells:
            cell.V = 0.0
            cell.A = 0.0
            cell.tlast = 0
            cell.t_spk = 0
            cell.t_last_spk = 0
            cell.thr = float(cfg.VTHRESH)
            cell.count = 0
            cell.rate = 0.0
            cell.decay = 1.0
            cell.ev.clear()
            cell.loc.clear()
            cell.lat.clear()
            cell.td.clear()
            cell.vspikes.clear()


def _save_inhibition_maps(d: Path, suffix: str, mask: np.ndarray, target: np.ndarray) -> None:
    """One ``<n><suffix>.npy`` per cell, values in the loader's iteration order."""
    import libstdcxx_umap as um

    orders: dict[bytes, np.ndarray] = {}
    for n in range(target.shape[0]):
        keys = np.flatnonzero(mask[n])
        sig = keys.tobytes()
        order = orders.get(sig)
        if order is None:
            order = np.asarray(um.iteration_order(keys.tolist()), dtype=np.int64)
            orders[sig] = order
        np.save(d / f"{n}{suffix}.npy", np.asarray(target[n, order], dtype=np.float64))


def save_network(net: "ref.PCLEventRef", net_dir) -> Path:
    """Save the network into a folder in the C++ layout, after the C++ reset.

    Per layer ``weights/<layer>/``: the shared matrices (``<k>.npy``) and local rows
    (``<k>lli.npy``) for layer 0, one ``<c>.npy`` / ``<c>lli.npy`` per complex cell for
    layer 1, per-cell ``<n>li.npy`` / ``<n>tdi.npy`` maps where the pathway is on, and
    one ``<idx>.json`` per neuron with ``count_spike``, ``threshold``, ``spiking_rate``,
    ``learning_decay`` and ``spike_train``. The folder stays loadable by ``PCLEventRef``.
    """
    import libstdcxx_umap  # noqa: F401  # the loader needs the same iteration order

    root = Path(net_dir)
    d0, d1 = root / "weights" / "0", root / "weights" / "1"
    d0.mkdir(parents=True, exist_ok=True)
    d1.mkdir(parents=True, exist_ok=True)
    _reset_neurons(net)
    for k in range(net.ND):
        w = np.asarray(net.w_s[k], dtype=np.float64)[:, None, None, :, :]
        np.save(d0 / f"{k}.npy", w)
        np.save(d0 / f"{k}lli.npy", np.asarray(net.wi_s[k], dtype=np.float64))
    if net.lat_on and net.LAT is not None:
        _save_inhibition_maps(d0, "li", net.lat_mask, net.LAT)
    if net.td_on and net.TD is not None:
        _save_inhibition_maps(d0, "tdi", net.td_mask, net.TD)
    for c in range(net.NC):
        cz = c % net.CD
        np.save(d1 / f"{c}.npy", np.asarray(net.w_c[cz], dtype=np.float64))
        np.save(d1 / f"{c}lli.npy", np.asarray(net.wi_c[cz], dtype=np.float64))
    for d, cells in ((d0, net.s_cells), (d1, net.c_cells)):
        for idx, cell in enumerate(cells):
            (d / f"{idx}.json").write_text(json.dumps({
                "count_spike": 0,  # reset on save; the run result carries the spikes
                "threshold": float(cell.thr),
                "spiking_rate": float(cell.rate),
                "learning_decay": float(cell.decay),
                "spike_train": [],  # reset on save; empty whatever the run did
            }), encoding="utf-8")
    return root


def init_network(configs_dir, net_dir, *, seed: int) -> Path:
    """A shared starting folder: the three config JSONs plus the oracle's seeded init weights.

    The C++ init is unseeded, so R2 starts both arms from this folder; the C++ loads its
    weights because ``weights/<layer>/0.npy`` exists (``SpikingNetwork.cpp:641-697``).
    """
    src, root = Path(configs_dir), Path(net_dir)
    if root.exists():
        raise FileExistsError(f"init_network: refusing an existing folder: {root}")
    (root / "configs").mkdir(parents=True)
    for name in ("network_config.json", "simple_cell_config.json", "complex_cell_config.json"):
        shutil.copyfile(src / name, root / "configs" / name)
    for sub in ("figures", "gabors", "images", "statistics"):
        (root / sub).mkdir()
    return save_network(ref.PCLEventRef(root, seed=int(seed), learning=True), root)


def run_oracle(net_dir, events_path, nb_pass: int, *, seed: int) -> dict:
    """Run the oracle on a C++-layout folder the way ``neuvisys-exe <net> <events> <nbPass>`` feeds it.

    ``nb_pass`` is the C++ ``nbPass``: the file is fed that many times, pass ``k`` with its
    timestamps shifted by ``k * (last - first)`` of the raw file (``NetworkHandle.cpp:479-516``).
    Events load through ``load_cpp_events`` (camera 0 only, rebased to the first event, as
    ``NetworkHandle.cpp:511-512``) and feed with learning on. The C++ reads in packets of
    10000 events and flags each packet's last event; that flag only flushes the synaptic
    delay queue, which ``neuron1Synapses = 1`` leaves empty (the oracle refuses more), so the
    oracle feeds one stream. The seed seeds the constructor's init weights, which loading
    overwrites. The result saves back into ``net_dir`` in the C++ layout.
    """
    import h5py

    nb_pass = int(nb_pass)
    if nb_pass < 1:
        raise ValueError(f"run_oracle: nb_pass must be >= 1, got {nb_pass}")
    net = ref.PCLEventRef(Path(net_dir), seed=int(seed), learning=True)
    ev = ref.load_cpp_events(events_path)  # camera 0, rebased, as the C++ feeds
    with h5py.File(events_path, "r") as f:
        span = int(f["events"]["t"][-1]) - int(f["events"]["t"][0])
    n = int(ev["t"].shape[0])
    for k in range(nb_pass):
        net.run(ev["t"] + k * span, ev["x"], ev["y"], ev["p"])
    last_spike_t = -1
    for cells in (net.s_cells, net.c_cells):
        for cell in cells:
            for t in cell.spikes:
                if int(t) > last_spike_t:
                    last_spike_t = int(t)
    save_network(net, net_dir)
    return {"n_events": n * nb_pass, "n_spikes": int(net.n_spikes()),
            "last_spike_t": last_spike_t, "net_dir": str(Path(net_dir))}


def _layer_sizes(net_dir: Path) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    cfg = json.loads((net_dir / "configs" / "network_config.json").read_text(encoding="utf-8"))
    return (tuple(int(v) for v in cfg["layerSizes"][0]),
            tuple(int(v) for v in cfg["layerSizes"][1]))


def _read_state(path: Path) -> tuple[int, list]:
    """(``count_spike``, ``spike_train``) of one neuron file; the train is empty unless tracked."""
    state = json.loads(path.read_text(encoding="utf-8"))
    if "count_spike" not in state:
        raise ValueError(f"{path}: refusing a neuron file without 'count_spike'")
    return int(state["count_spike"]), list(state.get("spike_train", []))


def _read_weight(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.float64:
        raise ValueError(f"{path}: need float64 weights, got {arr.dtype}")
    return arr


def compare(dir_a, dir_b) -> dict:
    """Compare two network folders after a run, per layer.

    Per layer: ``cells``, ``cells_equal_count`` (cells with equal ``count_spike``),
    ``total_spikes_a`` / ``total_spikes_b``, ``counts_reset`` (true when
    every count on both sides is 0, as the C++ reset leaves them),
    ``max_abs_weight_diff`` and
    ``max_rel_weight_diff`` over every weight file (relative: ``|a-b|/max(|a|,|b|)``,
    0 where both are 0). ``exact`` holds when every weight file agrees
    within rtol 1e-12 (atol 0); the counts are reported but do not gate it.
    The top-level ``exact`` (and ``counts_reset``) hold when both
    layers agree. A missing neuron file, a missing ``count_spike`` field, a weight
    file present on one side only, or a shape or dtype mismatch is refused.
    """
    a, b = Path(dir_a), Path(dir_b)
    sa, sb = _layer_sizes(a), _layer_sizes(b)
    if sa != sb:
        raise ValueError(f"compare: layer sizes differ: {sa} vs {sb}")
    out: dict = {}
    for layer, shape in ((0, sa[0]), (1, sa[1])):
        n = int(shape[0]) * int(shape[1]) * int(shape[2])
        da, db = a / "weights" / str(layer), b / "weights" / str(layer)
        if not da.is_dir() or not db.is_dir():
            raise ValueError(f"compare: weights/{layer} is missing on one side")
        fa = sorted(p.name for p in da.glob("*.npy"))
        fb = sorted(p.name for p in db.glob("*.npy"))
        if fa != fb:
            raise ValueError(f"compare: weights/{layer} file sets differ")
        equal = 0
        ta = tb = 0
        for idx in range(n):
            pa, pb = da / f"{idx}.json", db / f"{idx}.json"
            if not pa.is_file() or not pb.is_file():
                raise ValueError(f"compare: neuron file {idx}.json "
                                 f"is missing in weights/{layer} on one side")
            ca, _ = _read_state(pa)
            cb, _ = _read_state(pb)
            ta += ca
            tb += cb
            equal += ca == cb
        max_abs = 0.0
        max_rel = 0.0
        weights_ok = True
        for name in fa:
            A, B = _read_weight(da / name), _read_weight(db / name)
            if A.shape != B.shape:
                raise ValueError(f"compare: {layer}/{name} shapes differ: "
                                 f"{A.shape} vs {B.shape}")
            if A.size:
                d = np.abs(A - B)
                max_abs = max(max_abs, float(d.max()))
                denom = np.maximum(np.abs(A), np.abs(B))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(denom > 0, d / denom, 0.0)
                max_rel = max(max_rel, float(rel.max()))
            if not np.allclose(A, B, rtol=RTOL, atol=0.0):
                weights_ok = False
        exact = weights_ok
        counts_reset = (ta == 0 and tb == 0)
        out[str(layer)] = {"cells": n, "cells_equal_count": equal,
                           "total_spikes_a": ta, "total_spikes_b": tb,
                           "counts_reset": bool(counts_reset),
                           "max_abs_weight_diff": max_abs,
                           "max_rel_weight_diff": max_rel, "exact": bool(exact)}
    out["exact"] = bool(out["0"]["exact"] and out["1"]["exact"])
    out["counts_reset"] = bool(out["0"]["counts_reset"] and out["1"]["counts_reset"])
    return out


def _write_receipt(path: Path, record: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    import os

    os.replace(tmp, path)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="write a starting folder: configs plus seeded init weights")
    i.add_argument("--configs", required=True, help="folder holding the three config JSONs")
    i.add_argument("--out", required=True)
    i.add_argument("--seed", default=0, type=int)
    o = sub.add_parser("oracle", help="run the oracle on a network folder with learning on")
    o.add_argument("--net", required=True)
    o.add_argument("--events", required=True)
    o.add_argument("--nb-pass", required=True, type=int, help="passes over the file (C++ nbPass)")
    o.add_argument("--seed", default=0, type=int)
    o.add_argument("--out", required=True, help="receipt json")
    c = sub.add_parser("compare", help="compare two network folders after a run")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out", required=True, help="receipt json")
    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "init":
        print(f"initialized {init_network(args.configs, args.out, seed=args.seed)}", flush=True)
        return 0
    if args.cmd == "oracle":
        digest_before, n_before = _dir_digest(args.net)
        res = run_oracle(args.net, args.events, args.nb_pass, seed=args.seed)
        _write_receipt(Path(args.out), {
            "tool": "run_r2_reference",
            "subcommand": "oracle",
            "argv": sys.argv[1:] if argv is None else list(argv),
            "code_sha": _code_sha(),
            "inputs": {"net": {"path": str(Path(args.net)),
                               "sha256": digest_before,
                               "n_files": n_before},
                       "events": {"path": str(Path(args.events)),
                                  "sha256": _sha256_file(Path(args.events))}},
            "result": res,
        })
        print(f"fed {res['n_events']} events in {args.nb_pass} pass(es); "
              f"spikes {res['n_spikes']}; saved {res['net_dir']} out={args.out}", flush=True)
        return 0
    cmp = compare(args.a, args.b)
    da, na = _dir_digest(args.a)
    db, nb = _dir_digest(args.b)
    _write_receipt(Path(args.out), {
        "tool": "run_r2_reference",
        "subcommand": "compare",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "code_sha": _code_sha(),
        "inputs": {"a": {"path": str(Path(args.a)), "sha256": da, "n_files": na},
                   "b": {"path": str(Path(args.b)), "sha256": db, "n_files": nb}},
        "comparison": cmp,
    })
    print(f"exact={cmp['exact']} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
