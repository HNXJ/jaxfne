"""Event-exact reference simulator for the PCL two-layer network (port stage A).

Oracle for the time-binned port (stage B). It follows the authors' C++, read-only, at
F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_C++/libs/network/src:

- event loop: NetworkHandle::feedEvents (NetworkHandle.cpp:109), loadHDF5Events (:479)
- addEvent, addNeuronEvent, local/lateral/top-down inhibition: SpikingNetwork.cpp:41-179
- connections and weight sharing: SpikingNetwork.cpp:188-466
- membrane, spike, inhibition: SimpleNeuron.cpp:47-159, ComplexNeuron.cpp:30-90, Neuron.cpp
- STDP: SimpleNeuron.cpp:186-296, ComplexNeuron.cpp:95-170
- periodic rate and threshold update: NetworkHandle.cpp:137-148, SpikingNetwork.cpp:474-489

Times are integer microseconds (C++ uint64). Config taus are in ms and are scaled by E3 (Config.cpp).

Deviations from the C++ (also in the port report):
- Initial weights come from a seeded numpy RNG. The C++ draws them from std::random_device
  (WeightMatrix.cpp:7), so no C++ run can be reproduced bit for bit.
- Weights load only in the .npy layout that SpikingNetwork::loadWeights reads (.npz is not
  reached by that path). Lateral (<i>li.npy) and top-down (<i>tdi.npy) maps are WeightMap
  (std::unordered_map): the file holds values in the map's iteration order (WeightMap.cpp:131-145),
  which the C++ does not fix and the files do not record. They are refused when a file exists.
- "static" in layerInhibitions is read as "local" (declared alias). The trained configs use
  "static"; the C++ checks only "local" (SpikingNetwork.cpp:351), so it runs them without local
  inhibition.
- Loaded neuron state is what Neuron::readJson reads (Neuron.cpp:495-500): spiking_rate and
  learning_decay. The C++ writes threshold but does not read it (Neuron.cpp:497 is commented out),
  and count_spike feeds m_totalSpike, which no dynamics reads.
- Complex neurons of one depth hold one shared matrix (ComplexNeuron.hpp:14-15), so their files
  must agree. A disagreement is refused.
- Sums use numpy (pairwise), so weights agree with the C++ to about 1e-15 relative, not bitwise.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import deque
from pathlib import Path


def _repo_root(start: Path) -> Path:
    """Nearest ancestor that holds pyproject.toml."""
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_ROOT = str(_repo_root(Path(__file__).resolve().parent))
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

E3 = 1000  # Config.hpp:25, microseconds per millisecond
E6 = 1000000  # Config.hpp:26
NBPOLARITY = 2  # Util.hpp:17
NEG_LIMIT = -20.0  # Neuron.cpp:26, m_negativeLimits
CAP = 1000  # boost::circular_buffer capacity: Neuron.cpp:22, SimpleNeuron.cpp:21-25
ALPHA = 0.6  # Neuron.cpp:360, rate smoothing

# DefaultConfig.cpp:9-24 (network layout), :26-52 (simple cell), :53-71 (complex cell)
DEFAULT_NETWORK = {
    "nbCameras": 1,
    "neuron1Synapses": 1,
    "sharingType": "patch",
    "neuronType": ["SimpleCell", "ComplexCell"],
    "layerInhibitions": [["local"], ["local"]],
    "interLayerConnections": [[-1], [0]],
    "layerPatches": [[[150], [100], [0]], [[0], [0], [0]]],
    "layerSizes": [[9, 9, 64], [6, 6, 32]],
    "neuronSizes": [[[10, 10, 1]], [[4, 4, 64]]],
    "neuronOverlap": [[3, 3, 1], [3, 3, 64]],
    "neuronInhibitionRange": [4, 4],
    "vfWidth": 346,
    "vfHeight": 260,
    "measurementInterval": 100,
}
DEFAULT_SIMPLE = {
    "VTHRESH": 30,
    "VRESET": -10,
    "TRACKING": "none",
    "TAU_SRA": 100,
    "TAU_RP": 5,
    "TAU_M": 18,
    "TAU_LTP": 7,
    "TAU_LTD": 7,
    "TARGET_SPIKE_RATE": 0.75,
    "SYNAPSE_DELAY": 0,
    "STDP_LEARNING": "all",
    "NORM_FACTOR": 50,
    "LATERAL_NORM_FACTOR": 6500,
    "TOPDOWN_NORM_FACTOR": 2000,
    "DECAY_RATE": 0,
    "MIN_THRESH": 4,
    "ETA_LTP": 0.00077,
    "ETA_LTD": -0.00021,
    "ETA_ILTP": 0.00046,
    "ETA_ILTD": -0.00046,
    "ETA_SRA": 0,
    "ETA_TA": 0,
    "ETA_RP": 10,
    "ETA_INH": 1500,
}
DEFAULT_COMPLEX = {
    "VTHRESH": 3,
    "VRESET": -10,
    "TRACKING": "none",
    "TAU_M": 50,
    "TAU_LTP": 40,
    "TAU_LTD": 40,
    "TAU_RP": 5,
    "STDP_LEARNING": "all",
    "NORM_FACTOR": 1000,
    "DECAY_RATE": 0,
    "ETA_LTP": 0.002,
    "ETA_LTD": 0.002,
    "ETA_ILTP": 0.008,
    "ETA_ILTD": -0.008,
    "ETA_INH": 600,
    "ETA_RP": 10,
}


class CellConfig:
    """Scalars of one cell type, as NeuronConfig loads them (Config.cpp:62-136), in microseconds."""

    def __init__(self, kind: str, raw: dict) -> None:
        if kind not in ("simple", "complex"):
            raise ValueError(kind)
        self.kind = kind
        self.VTHRESH = float(raw["VTHRESH"])
        self.VRESET = float(raw["VRESET"])
        self.TAU_M = E3 * float(raw["TAU_M"])
        self.TAU_LTP = E3 * float(raw["TAU_LTP"])
        self.TAU_LTD = E3 * float(raw["TAU_LTD"])
        self.TAU_RP = E3 * float(raw["TAU_RP"])
        self.ETA_LTP = float(raw["ETA_LTP"])
        self.ETA_LTD = float(raw["ETA_LTD"])
        self.ETA_ILTP = float(raw["ETA_ILTP"])
        self.ETA_ILTD = float(raw["ETA_ILTD"])
        self.ETA_INH = float(raw["ETA_INH"])
        self.NORM_FACTOR = float(raw["NORM_FACTOR"])
        self.STDP_LEARNING = str(raw["STDP_LEARNING"])
        self.DELTA_RP = float(raw["ETA_RP"])  # Config.cpp:74 and :126
        if kind == "simple":
            self.TAU_SRA = E3 * float(raw["TAU_SRA"])
            self.ETA_SR = float(raw["ETA_TA"])  # Config.cpp:73
            self.DELTA_SRA = float(raw["ETA_SRA"])  # Config.cpp:75
            self.TARGET_SPIKE_RATE = float(raw["TARGET_SPIKE_RATE"])
            self.MIN_THRESH = float(raw["MIN_THRESH"])
            self.LATERAL_NORM_FACTOR = float(raw["LATERAL_NORM_FACTOR"])
            self.TOPDOWN_NORM_FACTOR = float(raw["TOPDOWN_NORM_FACTOR"])
        else:
            # Config.cpp:104-136 does not read these keys; NeuronConfig leaves them at 0.
            self.TAU_SRA = 0.0
            self.ETA_SR = 0.0
            self.DELTA_SRA = 0.0
            self.TARGET_SPIKE_RATE = 0.0
            self.MIN_THRESH = 0.0
            self.LATERAL_NORM_FACTOR = 0.0
            self.TOPDOWN_NORM_FACTOR = 0.0
        # Neuron constructor (Neuron.cpp:31-34): rates are scaled by the norm and the inhibition eta.
        self.eLTP = self.ETA_LTP * self.NORM_FACTOR
        self.eLTD = self.ETA_LTD * self.NORM_FACTOR
        self.iLTP = self.ETA_ILTP * self.ETA_INH
        self.iLTD = self.ETA_ILTD * self.ETA_INH
        # SimpleNeuron constructor (SimpleNeuron.cpp:35-38).
        self.pliLTP = self.ETA_ILTP * self.LATERAL_NORM_FACTOR
        self.pliLTD = self.ETA_ILTD * self.LATERAL_NORM_FACTOR
        self.ptiLTP = self.ETA_ILTP * self.TOPDOWN_NORM_FACTOR
        self.ptiLTD = self.ETA_ILTD * self.TOPDOWN_NORM_FACTOR


class Cell:
    """Scalar state of one neuron, named after the C++ members (Neuron.hpp:44-74)."""

    __slots__ = ("V", "A", "tlast", "t_spk", "t_last_spk", "thr", "count", "rate", "decay",
                 "ev", "loc", "lat", "td", "spikes", "vspikes")

    def __init__(self, thr: float) -> None:
        self.V = 0.0  # m_potential
        self.A = 0.0  # m_adaptationPotential
        self.tlast = 0  # m_timestampLastEvent
        self.t_spk = 0  # m_spikingTime; 0 is the C++ "never spiked" sentinel
        self.t_last_spk = 0  # m_lastSpikingTime
        self.thr = thr  # m_threshold
        self.count = 0  # m_spikeRateCounter
        self.rate = 0.0  # m_spikingRateAverage
        self.decay = 1.0  # m_decay; learningDecay is never called in the C++
        self.ev = deque(maxlen=CAP)  # m_events: (t, x, y, p) simple, (t, x, y, z) complex
        self.loc = deque(maxlen=CAP)  # local inhibition: (t, source depth)
        self.lat = deque(maxlen=CAP)  # lateral inhibition: (t, source index)
        self.td = deque(maxlen=CAP)  # top-down inhibition: (t, source index)
        self.spikes: list[int] = []  # recorded spike times, microseconds
        self.vspikes: list[int] = []  # m_vectorSpikingTime (complex only)


def _refractory(cell: Cell, cfg: CellConfig, t: int) -> float:
    """Neuron::refractoryPotential (Neuron.cpp:51-58)."""
    if cell.t_spk == 0:
        return 0.0
    return cfg.DELTA_RP * math.exp(-(t - cell.t_spk) / cfg.TAU_RP)


def _spike(cell: Cell, cfg: CellConfig, t: int, adaptive: bool) -> None:
    """SimpleNeuron::spike (SimpleNeuron.cpp:166) and ComplexNeuron::spike (:77)."""
    cell.t_last_spk = cell.t_spk
    cell.t_spk = t
    cell.count += 1
    cell.spikes.append(t)
    cell.V = cfg.VRESET
    if adaptive:
        cell.A += cfg.DELTA_SRA  # Neuron::spikeRateAdaptation (Neuron.cpp:392)


def excite_simple(cell: Cell, cfg: CellConfig, t: int, w: float) -> bool:
    """Synaptic input to a simple cell: SimpleNeuron::membraneUpdate (SimpleNeuron.cpp:141-159)."""
    dt = t - cell.tlast
    cell.V *= math.exp(-dt / cfg.TAU_M)  # potentialDecay (Neuron.cpp:64)
    cell.A *= math.exp(-dt / cfg.TAU_SRA)  # adaptationPotentialDecay (Neuron.cpp:72)
    cell.V += w - _refractory(cell, cfg, t) - cell.A
    cell.tlast = t
    if cell.V < NEG_LIMIT:  # checkNegativeLimits (Neuron.cpp:396)
        cell.V = NEG_LIMIT
    if cell.V > cell.thr:
        _spike(cell, cfg, t, adaptive=True)
        return True
    return False


def excite_complex(cell: Cell, cfg: CellConfig, t: int, w: float) -> bool:
    """Synaptic input to a complex cell: ComplexNeuron::membraneUpdate (ComplexNeuron.cpp:53-71)."""
    dt = t - cell.tlast
    cell.V *= math.exp(-dt / cfg.TAU_M)
    cell.V += w - _refractory(cell, cfg, t)
    cell.tlast = t
    if cell.V < NEG_LIMIT:
        cell.V = NEG_LIMIT
    if cell.V > cell.thr:
        cell.vspikes.append(t)  # ComplexNeuron::spike (:78)
        _spike(cell, cfg, t, adaptive=False)  # no adaptation term for complex cells
        return True
    return False


def inhibit_simple(cell: Cell, cfg: CellConfig, t: int, w: float) -> None:
    """Local, top-down or lateral inhibition of a simple cell: V -= w + R + A.

    SimpleNeuron.cpp:56-64 (local), :70-78 (top-down), :103-113 (lateral). The three are the
    same update; the caller appends the event to the matching buffer.
    """
    dt = t - cell.tlast
    cell.V *= math.exp(-dt / cfg.TAU_M)
    cell.A *= math.exp(-dt / cfg.TAU_SRA)
    cell.V -= w + _refractory(cell, cfg, t) + cell.A
    cell.tlast = t
    if cell.V < NEG_LIMIT:
        cell.V = NEG_LIMIT


def inhibit_complex(cell: Cell, cfg: CellConfig, t: int, w: float) -> None:
    """Local inhibition of a complex cell: V -= w + R (ComplexNeuron.cpp:39-46)."""
    dt = t - cell.tlast
    cell.V *= math.exp(-dt / cfg.TAU_M)
    cell.V -= w + _refractory(cell, cfg, t)
    if cell.V < NEG_LIMIT:
        cell.V = NEG_LIMIT
    cell.tlast = t


def _l1_normalize_inplace(a: np.ndarray, target: float) -> None:
    """WeightMatrix::normalizeL1Norm (WeightMatrix.cpp:151): scale so that sum |a| = target."""
    norm = float(np.abs(a).sum())
    if norm != 0.0:
        a *= target / norm


def _npy(path: Path, size: int) -> np.ndarray:
    """Read a C++ weight file. The C++ reads raw doubles (cnpy), so the dtype must be float64."""
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.float64 or arr.size != size:
        raise ValueError(f"{path}: need float64 with {size} values, got {arr.dtype} {arr.shape}")
    return arr.reshape(-1)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_cpp_events(path, *, n_events: int | None = None, camera0_only: bool = True,
                    rebase: bool = True) -> dict:
    """Read the C++ HDF5 event file (group "events": t, x, y, p, c).

    rebase: subtract the file's first timestamp, as the C++ does (NetworkHandle.cpp:512).
    camera0_only: keep only c == 0, as the C++ does for one-camera networks (NetworkHandle.cpp:511).
    """
    with h5py.File(path, "r") as f:
        g = f["events"]
        total = int(g["t"].shape[0])
        stop = total if n_events is None else min(int(n_events), total)
        first = int(g["t"][0])  # readFirstAndLastTimestamp reads index 0 (NetworkHandle.cpp:530)
        t = g["t"][:stop].astype(np.int64)
        x = g["x"][:stop].astype(np.int64)
        y = g["y"][:stop].astype(np.int64)
        p = (g["p"][:stop] != 0).astype(np.int64)
        c = g["c"][:stop].astype(np.int64)
    if rebase:
        t = t - first
    keep = (c == 0) if camera0_only else np.ones(t.shape[0], dtype=bool)
    return {"t": t[keep], "x": x[keep], "y": y[keep], "p": p[keep], "c": c[keep]}


class PCLEventRef:
    """Event-exact two-layer PCL network with the C++ order of updates.

    directory: a network folder as the C++ reads it (configs/*.json, weights/...). None uses
    the DefaultConfig.cpp values and random initial weights.
    layer_inhibitions: test ablation only. Replaces the config's layerInhibitions and keeps the
    trained weights, e.g. [["static"], ["static"]] drops the lateral and top-down maps.
    """

    def __init__(self, directory=None, *, seed: int = 0, learning: bool = True,
                 activation: bool = True, layer_inhibitions=None) -> None:
        if directory is None:
            net = copy.deepcopy(DEFAULT_NETWORK)
            simple_raw = copy.deepcopy(DEFAULT_SIMPLE)
            complex_raw = copy.deepcopy(DEFAULT_COMPLEX)
            weights_dir = None
        else:
            root = Path(directory)
            net = _read_json(root / "configs" / "network_config.json")
            simple_raw = _read_json(root / "configs" / "simple_cell_config.json")
            complex_raw = _read_json(root / "configs" / "complex_cell_config.json")
            weights_dir = root / "weights"
        if layer_inhibitions is not None:
            net["layerInhibitions"] = copy.deepcopy(layer_inhibitions)
        self.learning = bool(learning)  # False freezes all plasticity (weightUpdate not applied)
        self.activation = bool(activation)  # SpikingNetwork::m_activation
        self.scfg = CellConfig("simple", simple_raw)
        self.ccfg = CellConfig("complex", complex_raw)
        self._layout(net)
        self._geometry(net)
        self._init_weights(seed)
        self.s_cells = [Cell(self.scfg.VTHRESH) for _ in range(self.NS)]
        self.c_cells = [Cell(self.ccfg.VTHRESH) for _ in range(self.NC)]
        self.measurement_interval = E3 * float(net["measurementInterval"])  # Config.cpp:29
        self._last_update = 0  # NetworkHandle m_saveTime.update
        if weights_dir is not None and weights_dir.is_dir():
            self._load_weights(weights_dir)

    # ----- construction -------------------------------------------------------------------

    def _layout(self, net: dict) -> None:
        if int(net["nbCameras"]) != 1:
            raise NotImplementedError("nbCameras != 1 is not ported (NetworkHandle.cpp:511)")
        if int(net["neuron1Synapses"]) != 1:
            raise NotImplementedError("neuron1Synapses > 1 (synaptic queue) is not ported")
        if net["sharingType"] != "patch":
            raise NotImplementedError("only sharingType 'patch' is ported")
        if list(net["neuronType"]) != ["SimpleCell", "ComplexCell"]:
            raise NotImplementedError("only SimpleCell then ComplexCell is ported")
        if int(net["vfWidth"]) < 2 or int(net["vfHeight"]) < 2:
            raise NotImplementedError("a one-pixel field changes m_LatNorm (SpikingNetwork.cpp:55)")
        if [list(map(int, v)) for v in net["interLayerConnections"]] != [[-1], [0]]:
            raise NotImplementedError("only interLayerConnections [[-1],[0]] are ported")
        for layer in net["layerPatches"]:
            if any(len(dim) != 1 for dim in layer):
                raise NotImplementedError("only one patch per layer is ported")
        # Declared alias: the trained configs say "static" for local inhibition (module docstring).
        inh = [["local" if v == "static" else v for v in layer] for layer in net["layerInhibitions"]]
        if not set(inh[0]) <= {"local", "lateral", "topdown"}:
            raise NotImplementedError(f"unknown inhibition in layer 0: {inh[0]}")
        if not set(inh[1]) <= {"local", "lateral", "topdown"}:
            raise NotImplementedError(f"unknown inhibition in layer 1: {inh[1]}")
        # C++ builds complex-layer lateral lists and layer-0 top-down lists, but nothing calls
        # them (SpikingNetwork.cpp:129, :138-143), so those two flags are ignored here.
        self.local_s = "local" in inh[0]
        self.lat_on = "lateral" in inh[0]
        self.local_c = "local" in inh[1]
        self.td_on = "topdown" in inh[1]

    def _geometry(self, net: dict) -> None:
        self.NX, self.NY, self.ND = (int(v) for v in net["layerSizes"][0])
        self.CX, self.CY, self.CD = (int(v) for v in net["layerSizes"][1])
        s_rf = [int(v) for v in net["neuronSizes"][0][0]]
        c_rf = [int(v) for v in net["neuronSizes"][1][0]]
        s_ov = [int(v) for v in net["neuronOverlap"][0]]
        c_ov = [int(v) for v in net["neuronOverlap"][1]]
        if s_rf[2] != 1:
            raise NotImplementedError("simple receptive field must be one depth (SpikingNetwork.cpp:213)")
        if c_rf[2] != self.ND:
            raise NotImplementedError("complex receptive field must span all simple depths")
        self.SX, self.SY = s_rf[0], s_rf[1]
        self.CSX, self.CSY = c_rf[0], c_rf[1]
        p0, p1 = net["layerPatches"][0], net["layerPatches"][1]
        # Offsets: SpikingNetwork.cpp:297-300, patch + i * (neuronSize - overlap).
        self.s_ox = [int(p0[0][0]) + i * (self.SX - s_ov[0]) for i in range(self.NX)]
        self.s_oy = [int(p0[1][0]) + j * (self.SY - s_ov[1]) for j in range(self.NY)]
        self.c_ox = [int(p1[0][0]) + i * (self.CSX - c_ov[0]) for i in range(self.CX)]
        self.c_oy = [int(p1[1][0]) + j * (self.CSY - c_ov[1]) for j in range(self.CY)]
        self.VW = int(net["vfWidth"])
        self.VH = int(net["vfHeight"])
        self.Rx, self.Ry = (int(v) for v in net["neuronInhibitionRange"])
        self.NS = self.NX * self.NY * self.ND
        self.NC = self.CX * self.CY * self.CD
        # Pixel mapping (SpikingNetwork.cpp:382-386), by x and y separately, ascending index.
        self.cover_x = [[i for i in range(self.NX) if self.s_ox[i] <= x < self.s_ox[i] + self.SX]
                        for x in range(self.VW)]
        self.cover_y = [[j for j in range(self.NY) if self.s_oy[j] <= y < self.s_oy[j] + self.SY]
                        for y in range(self.VH)]
        # Complex cells covering simple position (i, j), ascending complex index.
        self.c_cover = {
            (i, j): [(cx, cy) for cx in range(self.CX) for cy in range(self.CY)
                     if self.c_ox[cx] <= i < self.c_ox[cx] + self.CSX
                     and self.c_oy[cy] <= j < self.c_oy[cy] + self.CSY]
            for i in range(self.NX) for j in range(self.NY)
        }
        si = np.repeat(np.arange(self.NX), self.NY * self.ND)
        sj = np.tile(np.repeat(np.arange(self.NY), self.ND), self.NX)
        self.td_mask = None
        if self.td_on:
            ci = np.repeat(np.arange(self.CX), self.CY * self.CD)
            cj = np.tile(np.repeat(np.arange(self.CY), self.CD), self.CX)
            cxo = np.asarray(self.c_ox)[ci]
            cyo = np.asarray(self.c_oy)[cj]
            okx = (si[:, None] >= cxo[None, :]) & (si[:, None] < cxo[None, :] + self.CSX)
            oky = (sj[:, None] >= cyo[None, :]) & (sj[:, None] < cyo[None, :] + self.CSY)
            self.td_mask = okx & oky  # (simple, complex): simple lies in the complex receptive field
        self.lat_mask = None
        if self.lat_on:
            self.lat_mask = np.zeros((self.NS, self.NS), dtype=bool)
            for n in range(self.NS):  # SpikingNetwork.cpp:409-442: window +-R, own position excluded
                dx = np.abs(si - si[n])
                dy = np.abs(sj - sj[n])
                self.lat_mask[n] = (dx <= self.Rx) & (dy <= self.Ry) & ((dx > 0) | (dy > 0))
            self.lat_count = self.lat_mask.sum(axis=1)
            if np.any(self.lat_count == 1):
                raise NotImplementedError("a lateral map of size 1 (SimpleNeuron.cpp:263) is not ported")

    def _init_weights(self, seed: int) -> None:
        """Random U(0,1) initial weights, then L1 normalisation as in addLayer (SpikingNetwork.cpp:339)."""
        rng = np.random.default_rng(seed)
        self.w_s = rng.random((self.ND, NBPOLARITY, self.SX, self.SY))  # shared per depth
        self.wi_s = rng.random((self.ND, self.ND))  # local inhibition, per receiving depth
        self.w_c = rng.random((self.CD, self.CSX, self.CSY, self.ND))  # shared per complex depth
        self.wi_c = rng.random((self.CD, self.CD))
        sc, cc = self.scfg, self.ccfg
        for k in range(self.ND):
            _l1_normalize_inplace(self.w_s[k], sc.NORM_FACTOR)
        for d in range(self.ND):
            _l1_normalize_inplace(self.wi_s[d], sc.ETA_INH)
        for cz in range(self.CD):
            _l1_normalize_inplace(self.w_c[cz], cc.NORM_FACTOR)
            _l1_normalize_inplace(self.wi_c[cz], cc.ETA_INH)
        self.LAT = None
        if self.lat_on:  # WeightMap::addWeight(id, true) sets 1 (WeightMap.cpp:157)
            self.LAT = self.lat_mask.astype(np.float64)
            rs = self.LAT.sum(axis=1)
            scale = np.divide(sc.LATERAL_NORM_FACTOR, rs, out=np.zeros_like(rs), where=rs != 0)
            self.LAT *= scale[:, None]
        self.TD = None
        if self.td_on:
            self.TD = self.td_mask.astype(np.float64)
            rs = self.TD.sum(axis=1)
            scale = np.divide(sc.TOPDOWN_NORM_FACTOR, rs, out=np.zeros_like(rs), where=rs != 0)
            self.TD *= scale[:, None]

    def _load_weights(self, wdir: Path) -> None:
        """SpikingNetwork::loadWeights (SpikingNetwork.cpp:641-702), keyed on <layer>/0.npy existing."""
        d0 = wdir / "0"
        if (d0 / "0.npy").is_file():
            if self.lat_on and (d0 / "0li.npy").is_file():
                raise NotImplementedError("lateral weight files exist: their order is the std::unordered_map "
                                          "iteration order (WeightMap.cpp:131-145), not recorded in the files")
            if self.td_on and (d0 / "0tdi.npy").is_file():
                raise NotImplementedError("top-down weight files exist: their order is the std::unordered_map "
                                          "iteration order (WeightMap.cpp:131-145), not recorded in the files")
            for k in range(self.ND):  # patch sharing: neurons 0..ND-1 at (0,0), SpikingNetwork.cpp:666-678
                shape = self.w_s[k].shape
                self.w_s[k] = _npy(d0 / f"{k}.npy", self.w_s[k].size).reshape(shape)
                self.wi_s[k] = _npy(d0 / f"{k}lli.npy", self.ND)
            self._load_states(d0, self.s_cells)
        d1 = wdir / "1"
        if (d1 / "0.npy").is_file():
            # Complex cells of one depth share one matrix (ComplexNeuron.hpp:14-15), so each depth's
            # files must agree. The C++ loads them in index order, last one wins; a disagreement is refused.
            shared: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for c in range(self.NC):
                cz = c % self.CD
                w = _npy(d1 / f"{c}.npy", self.w_c[cz].size).reshape(self.w_c[cz].shape)
                wi = _npy(d1 / f"{c}lli.npy", self.CD)
                if cz not in shared:
                    shared[cz] = (w, wi)
                elif not (np.array_equal(shared[cz][0], w) and np.array_equal(shared[cz][1], wi)):
                    raise ValueError(f"complex weight files of depth {cz} disagree (cell {c})")
            for cz, (w, wi) in shared.items():
                self.w_c[cz] = w
                self.wi_c[cz] = wi
            self._load_states(d1, self.c_cells)

    @staticmethod
    def _load_states(d: Path, cells: list) -> None:
        """Neuron::loadState reads spiking_rate and learning_decay (Neuron.cpp:431, readJson)."""
        for idx, cell in enumerate(cells):
            p = d / f"{idx}.json"
            if not p.is_file():  # C++ prints and continues (Neuron.cpp:443)
                continue
            state = _read_json(p)
            cell.rate = float(state["spiking_rate"])
            cell.decay = float(state["learning_decay"])

    # ----- plasticity ---------------------------------------------------------------------

    def weight_update_simple(self, n: int) -> None:
        """SimpleNeuron::weightUpdate (SimpleNeuron.cpp:186-296), run at each spike of cell n."""
        cell = self.s_cells[n]
        cfg = self.scfg
        k = n % self.ND
        if self.learning and cfg.STDP_LEARNING in ("excitatory", "all"):
            W = self.w_s[k]
            for t_i, xr, yr, p in cell.ev:  # SimpleNeuron.cpp:196-216
                wj = float(W[p, xr, yr])
                soft_ltp = (3.0 - wj) * 0.33
                d_ltp = soft_ltp * cell.decay * cfg.eLTP * 1.0 * math.exp(-(cell.t_spk - t_i) / cfg.TAU_LTP)
                soft_ltd = wj * 0.33
                W[p, xr, yr] += d_ltp
                if cell.t_last_spk != 0:
                    d_ltd = soft_ltd * cfg.eLTD * 1.0 * math.exp(-(t_i - cell.t_last_spk) / cfg.TAU_LTD)
                    W[p, xr, yr] += d_ltd
                if W[p, xr, yr] < 0:
                    W[p, xr, yr] = 0.0
            _l1_normalize_inplace(W, cfg.NORM_FACTOR)  # :219
            Wi = self.wi_s[k]
            for t_i, src in cell.loc:  # SimpleNeuron.cpp:222-237
                wj = float(Wi[src])
                soft_ltp = (50.0 - wj) * 0.02
                d_ltp = soft_ltp * cfg.iLTP * math.exp(-(cell.t_spk - t_i) / cfg.TAU_LTP)
                soft_ltd = wj * 0.02
                Wi[src] += d_ltp
                if cell.t_last_spk != 0:
                    d_ltd = soft_ltd * cfg.iLTD * math.exp(-(t_i - cell.t_last_spk) / cfg.TAU_LTD)
                    Wi[src] += d_ltd
                if Wi[src] < 0:
                    Wi[src] = 0.0
            _l1_normalize_inplace(Wi, cfg.ETA_INH)  # :238
        if self.learning and cfg.STDP_LEARNING in ("inhibitory", "all"):  # :241, with !m_noLat
            if self.td_on:
                row = self.TD[n]
                for t_i, c in cell.td:  # SimpleNeuron.cpp:244-258
                    wj = float(row[c])
                    soft_ltp = (50.0 - wj) * 0.02
                    d_ltp = soft_ltp * cfg.ptiLTP * math.exp(-(cell.t_spk - t_i) / cfg.TAU_LTP)
                    soft_ltd = wj * 0.02
                    row[c] += d_ltp
                    if cell.t_last_spk != 0:
                        row[c] += soft_ltd * cfg.ptiLTD * math.exp(-(t_i - cell.t_last_spk) / cfg.TAU_LTD)
                    if row[c] < 0:
                        row[c] = 0.0
                _l1_normalize_inplace(row, cfg.TOPDOWN_NORM_FACTOR)  # :259
            if self.lat_on:
                row = self.LAT[n]
                for t_i, src in cell.lat:  # SimpleNeuron.cpp:268-283
                    wj = float(row[src])
                    soft_ltp = (50.0 - wj) * 0.02
                    d_ltp = soft_ltp * cfg.pliLTP * math.exp(-(cell.t_spk - t_i) / cfg.TAU_LTP)
                    soft_ltd = wj * 0.02
                    row[src] += d_ltp
                    if cell.t_last_spk != 0:
                        row[src] += soft_ltd * cfg.pliLTD * math.exp(-(t_i - cell.t_last_spk) / cfg.TAU_LTD)
                    if row[src] < 0:
                        row[src] = 0.0
                if int(self.lat_count[n]) > 1:  # :284-286
                    _l1_normalize_inplace(row, cfg.LATERAL_NORM_FACTOR)
        cell.ev.clear()  # :292-295, always, as weightUpdate always clears
        cell.td.clear()
        cell.loc.clear()
        cell.lat.clear()

    def weight_update_complex(self, c: int) -> None:
        """ComplexNeuron::weightUpdate (ComplexNeuron.cpp:95-170), run at each spike of cell c."""
        cell = self.c_cells[c]
        cfg = self.ccfg
        cz = c % self.CD
        if self.learning and cfg.STDP_LEARNING in ("excitatory", "all"):
            if cell.count == 1:  # :97; the counter is reset to 0 inside this block (:140)
                W = self.w_c[cz]
                for t_i, x, y, z in cell.ev:  # :101-138
                    for t in cell.vspikes:
                        if t >= t_i:  # int(t - event.timestamp()) >= 0 (:103)
                            wj = float(W[x, y, z])
                            soft_ltp = (4.0 - wj) * 0.25
                            W[x, y, z] += soft_ltp * cfg.eLTP * math.exp(-abs(t - t_i) / cfg.TAU_LTP)
                        else:
                            wj = float(W[x, y, z])
                            soft_ltd = wj * 0.25
                            W[x, y, z] += soft_ltd * cfg.eLTD * math.exp(-abs(t - t_i) / cfg.TAU_LTD)
                    if W[x, y, z] < 0:  # clip after all spike pairs of this event (:133-135)
                        W[x, y, z] = 0.0
                _l1_normalize_inplace(W, cfg.NORM_FACTOR)  # :139
                cell.count = 0  # :140
                if cell.vspikes:  # erase(begin, begin + size - 1): keep only the last spike (:141)
                    cell.vspikes = cell.vspikes[-1:]
            Wi = self.wi_c[cz]
            max_i, factor_i = 25.0, 0.04  # ComplexNeuron.cpp:144-145
            for t_i, src in cell.loc:  # :146-165
                wj = float(Wi[src])
                soft_ltp = (max_i - wj) * factor_i
                Wi[src] += soft_ltp * cfg.iLTP * math.exp(-(cell.t_spk - t_i) / cfg.TAU_LTP)
                if cell.t_last_spk != 0:
                    wj2 = float(Wi[src])  # re-read after LTP (:152)
                    soft_ltd = wj2 * factor_i
                    Wi[src] += soft_ltd * cfg.iLTD * math.exp(-(t_i - cell.t_last_spk) / cfg.TAU_LTD)
                elif Wi[src] < 0:  # the clip runs only when there was no previous spike (:159-163)
                    Wi[src] = 0.0
            _l1_normalize_inplace(Wi, cfg.ETA_INH)  # :166
        cell.ev.clear()  # :168-169
        cell.loc.clear()

    # ----- event loop ---------------------------------------------------------------------

    def process_event(self, t: int, x: int, y: int, p: int) -> None:
        """One event, then the periodic update, as NetworkHandle::feedEvents does (:113-122)."""
        if not (0 <= x < self.VW and 0 <= y < self.VH):
            raise ValueError(f"event pixel ({x}, {y}) outside {self.VW}x{self.VH}")
        self._add_event(int(t), int(x), int(y), 1 if p else 0)
        self._update_neurons(int(t))

    def run(self, t, x, y, p) -> None:
        """Feed an event stream in order."""
        for ti, xi, yi, pi in zip(np.asarray(t).tolist(), np.asarray(x).tolist(),
                                  np.asarray(y).tolist(), np.asarray(p).tolist()):
            self.process_event(ti, xi, yi, pi)

    def _add_event(self, t: int, x: int, y: int, p: int) -> None:
        """SpikingNetwork::addEvent, one synapse (SpikingNetwork.cpp:41-67).

        Candidates are visited in ascending neuron index, as the pixel mapping is built
        (SpikingNetwork.cpp:385). A neuron that spikes updates its weights, then sends local
        inhibition, then lateral inhibition (if activation), then forwards to the complex layer.
        """
        sc = self.scfg
        for i in self.cover_x[x]:
            xr = x - self.s_ox[i]
            for j in self.cover_y[y]:
                yr = y - self.s_oy[j]
                base = (i * self.NY + j) * self.ND
                for k in range(self.ND):
                    n = base + k
                    cell = self.s_cells[n]
                    cell.ev.append((t, xr, yr, p))
                    if excite_simple(cell, sc, t, float(self.w_s[k, p, xr, yr])):
                        self._on_simple_spike(n, i, j, k, t)

    def _on_simple_spike(self, n: int, i: int, j: int, k: int, t: int) -> None:
        self.weight_update_simple(n)  # before any inhibition (SpikingNetwork.cpp:54)
        sc = self.scfg
        if self.local_s:  # lateralLocalInhibition (SpikingNetwork.cpp:172-179)
            base = (i * self.NY + j) * self.ND
            for d in range(self.ND):
                if d == k:
                    continue
                m = base + d
                cell = self.s_cells[m]
                cell.loc.append((t, k))
                inhibit_simple(cell, sc, t, float(self.wi_s[d, k]))
        if self.activation and self.lat_on:  # SpikingNetwork.cpp:60-62
            self._lateral(n, t)
        self._forward(i, j, k, t)  # addNeuronEvent (SpikingNetwork.cpp:63-65)

    def _lateral(self, n: int, t: int) -> None:
        """lateralDynamicInhibition (SpikingNetwork.cpp:157-165); targets in ascending index."""
        sc = self.scfg
        for m in np.flatnonzero(self.lat_mask[n]):
            m = int(m)
            cell = self.s_cells[m]
            cell.lat.append((t, n))
            inhibit_simple(cell, sc, t, float(self.LAT[m, n]))

    def _forward(self, i: int, j: int, k: int, t: int) -> None:
        """addNeuronEvent (SpikingNetwork.cpp:87-107): complex cells in ascending index."""
        cc = self.ccfg
        for cx, cy in self.c_cover[(i, j)]:
            ex = i - self.c_ox[cx]
            ey = j - self.c_oy[cy]
            base = (cx * self.CY + cy) * self.CD
            for cz in range(self.CD):
                c = base + cz
                cell = self.c_cells[c]
                cell.ev.append((t, ex, ey, k))
                if excite_complex(cell, cc, t, float(self.w_c[cz, ex, ey, k])):
                    self._on_complex_spike(c, cx, cy, cz, t)

    def _on_complex_spike(self, c: int, cx: int, cy: int, cz: int, t: int) -> None:
        self.weight_update_complex(c)
        cc = self.ccfg
        if self.local_c:  # lateralLocalInhibition for the complex layer (SpikingNetwork.cpp:97, :172)
            base = (cx * self.CY + cy) * self.CD
            for cz2 in range(self.CD):
                if cz2 == cz:
                    continue
                m = base + cz2
                cell = self.c_cells[m]
                cell.loc.append((t, cz))
                inhibit_complex(cell, cc, t, float(self.wi_c[cz2, cz]))
        if self.activation and self.td_on:  # SpikingNetwork.cpp:98-100
            self._topdown(c, t)

    def _topdown(self, c: int, t: int) -> None:
        """topDownDynamicInhibition (SpikingNetwork.cpp:142-151); targets in ascending index."""
        sc = self.scfg
        for s in np.flatnonzero(self.td_mask[:, c]):
            s = int(s)
            cell = self.s_cells[s]
            cell.td.append((t, c))
            inhibit_simple(cell, sc, t, float(self.TD[s, c]))

    def _update_neurons(self, t: int) -> None:
        """NetworkHandle::updateNeurons (NetworkHandle.cpp:137) and updateNeuronsStates (SpikingNetwork.cpp:474)."""
        dt = t - self._last_update
        if not (float(dt) > self.measurement_interval):
            return
        for cells, cfg in ((self.s_cells, self.scfg), (self.c_cells, self.ccfg)):
            for cell in cells:
                sps = cell.count * (E6 / float(dt))  # Neuron::updateState (Neuron.cpp:357-364)
                cell.rate = ALPHA * sps + (1.0 - ALPHA) * cell.rate
                cell.count = 0
                if cell.rate > cfg.TARGET_SPIKE_RATE:  # Neuron::thresholdAdaptation (Neuron.cpp:377-387)
                    cell.thr += cfg.ETA_SR * (1 - math.exp(cfg.TARGET_SPIKE_RATE - cell.rate))
                else:
                    cell.thr -= cfg.ETA_SR * (1 - math.exp(cell.rate - cfg.TARGET_SPIKE_RATE))
                if cell.thr < cfg.MIN_THRESH:
                    cell.thr = cfg.MIN_THRESH
        self._last_update = t

    # ----- output -------------------------------------------------------------------------

    def spike_trains(self) -> tuple[list[list[int]], list[list[int]]]:
        """Spike times in microseconds: (simple per neuron index, complex per neuron index)."""
        return ([list(c.spikes) for c in self.s_cells], [list(c.spikes) for c in self.c_cells])

    def n_spikes(self) -> int:
        return sum(len(c.spikes) for c in self.s_cells) + sum(len(c.spikes) for c in self.c_cells)
