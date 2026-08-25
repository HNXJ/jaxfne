"""E2 executor infrastructure repairs (post-E2b audit). No V1 rerun.

R1: gates generated from executable JSON (no retyped thresholds).
R2: units carried by types, not variable names.
R3: stable declared seed derivation (jax fold_in chain in canonical order), never hash().
"""
import json, re, hashlib, pathlib
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jnp

# ---------- R2: typed units ----------
class Deg(NamedTuple):
    value: float
    unit: str = "deg"

class Ms(NamedTuple):
    value: float
    unit: str = "ms"

class dB(NamedTuple):
    value: float
    unit: str = "dB"

class Hz(NamedTuple):
    value: float
    unit: str = "Hz"

def require_unit(x, unit):
    assert isinstance(x, tuple) and x.unit == unit, f"unit violation: expected {unit}, got {x}"
    return x.value

# ---------- R1: JSON-generated gate predicates ----------
_NUM = r"(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)"

def _split_conjuncts(expr):
    parts = re.split(r"\s*(?:&&|\band\b)\s*", expr)
    return [p.strip() for p in parts if p.strip()]

def _parse_conjunct(c):
    c = re.sub(r"\s+(dB|ms|deg|Hz)\s*$", "", c.strip())  # trailing units are annotations
    m = re.match(rf"^([A-Za-z_|Δ\[\]\(\) ]+?)\s+not\s+in\s+\[{_NUM}\s*,\s*{_NUM}\]$", c)
    if m:
        name, lo, hi = m.group(1).strip(), float(m.group(2)), float(m.group(3))
        return ("notin", name, lo, hi)
    m = re.match(rf"^([A-Za-z_|Δ\[\]\(\) ]+?)\s+in\s+\[{_NUM}\s*,\s*{_NUM}\]$", c)
    if m:
        name, lo, hi = m.group(1).strip(), float(m.group(2)), float(m.group(3))
        return ("in", name, lo, hi)
    m = re.match(rf"^(.*?)\s*(>=|<=|>|<)\s*{_NUM}$", c)
    if m:
        return ("cmp", m.group(1).strip(), m.group(2), float(m.group(3)))
    raise ValueError(f"unparseable conjunct: {c!r}")

def compile_gate(expr, metrics):
    """Return callable(values)->bool. Constants come ONLY from the frozen expr string.
    `metrics` maps canonical metric names -> accessor for alias resolution."""
    conjuncts = [_parse_conjunct(c) for c in _split_conjuncts(expr)]
    def evaluate(vals):
        for kind, name, x, y in conjuncts:
            key = metrics(name)
            v = vals.get(key) if key else None
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v or v in (float('inf'), float('-inf')):
                return False  # missing/non-finite metric -> gate fails closed
            if kind == "cmp":
                op, thr = x, y
                ok = {"<=": v <= thr, ">=": v >= thr, "<": v < thr, ">": v > thr}[op]
            elif kind == "in":
                ok = x <= v <= y
            else:
                ok = not (x <= v <= y)
            if not ok:
                return False
        return True
    return evaluate

class PingGates:
    """All constants parsed from e2_ping_prereg.json classifier strings at load time."""
    def __init__(self, ping_json: dict):
        cl = ping_json["classifiers"]
        self.spec = compile_gate(cl["G_spec"], lambda n: _alias(n))
        self.rate = compile_gate(cl["G_rate"], lambda n: _alias(n))
        self.phase = compile_gate(cl["G_phase"], lambda n: _alias(n))
        self.cycle = compile_gate(cl["G_cycle"], lambda n: _alias(n))
        gz = cl["gray_zones"]
        def band(s):
            m = re.match(rf"{_NUM}\s*-\s*{_NUM}", s)
            return float(m.group(1)), float(m.group(2))
        self.gray_prom = band(gz["prominence"])
        self.gray_plv = band(gz["PLV"])
        self.gray_dphi = band(gz["delta_phi"])
        self.labels = list(cl["labels"])
        self.mutual_exclusion = bool(cl["mutual_exclusion"])

_ALIASES = {
    "prominence": "prom_dB", "bandpower": "band_ratio", "f_peak": "fpk",
    "MD": "md_min", "AC": "ac_min", "|xcorr|": "xcorr_abs",
    "delta_phi": "dphi_deg", "PLV": "plv", "delta_t": "dt_lag_ms",
    "Rayleigh p": "rayleigh_p", "CV_T": "cv_T", "FF": "ff",
    "median p_i": "part_med", "N_cycles": "n_cycles",
}
def _alias(n):
    n = n.strip()
    if n in _ALIASES: return _ALIASES[n]
    return n

# ---------- R3: stable declared seed derivation ----------
def splitmix64(x: int) -> int:
    M = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & M
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)

def domain_keys(master_seed: int, replicate_idx: int, offsets: dict, canonical_order: list):
    """Frozen derivation: K0=PRNGKey(master); fold_in chained through canonical order
    with data = offset_domain + replicate_idx. Returns dict of per-domain jax keys."""
    key = jax.random.PRNGKey(master_seed)
    out = {}
    for dom in canonical_order:
        off = offsets.get(dom, offsets.get(f'K_{dom}'))
        assert off is not None, f'no offset for {dom}'
        key = jax.random.fold_in(key, jnp.uint32(off + replicate_idx))
        out[dom] = key
    return out

def child_seed(parent_key: jax.Array, tag: str) -> int:
    """Stable named child seed (replaces salted hash())."""
    h = hashlib.sha256()
    h.update(np.asarray(parent_key).tobytes())
    h.update(tag.encode())
    return int.from_bytes(h.digest()[:8], "little")

def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def canon_spec_hash(j: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in j.items() if k != "spec_hash"},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
