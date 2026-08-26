"""JDNA Compact Grammar — front-end compiler etude (private).

Implements the BNF/PEG from artifacts/etudes/jdna-compact-grammar-preregistration.md
as a pure desugar to jaxfne.jdna.genome (PseudoGenome) before validate_genome → develop.

Location: artifacts/etudes/jdna-compact-grammar/  (not jaxfne/jdna/)
Provenance: genome_rules_hash  (no new hash)
K_D determinism: develop(G, K_D) PRNG split per area/layer (jaxfne.jdna.genome.develop)
H11: callers generate outputs only in tmp_path; no reliance on gitignored artifacts.

Grammar covered:
  define / inherit / use  (PEG §4.3)
  A-80  (AbsoluteTweak: TargetRef "-" INT)
  A*0.08 (FractionTweak: TargetRef "*" FLOAT)  denominator = N_enclosing (§5.3)
  deep merge per §6.2-§6.4
  compose with -> merge_neuronal_tensors (§7)
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Sequence

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

KEYWORDS = frozenset([
    "define", "inherit", "from", "use", "area", "layer", "connect",
    "compose", "with", "poses", "as", "development", "pose",
])

@dataclass
class Token:
    kind: str
    value: str
    pos: int

_TOKEN_SPEC = [
    ("COMMENT", r"#[^\n]*"),
    ("STRING",  r'"(?:[^"\n\\]|\\.)*"'),
    ("FLOAT",   r"(?:[0-9]+\.[0-9]+|\.[0-9]+)"),
    ("INT",     r"[0-9]+"),
    ("ARROW",   r"->"),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LBRACE",  r"\{"),
    ("RBRACE",  r"\}"),
    ("LBRACK",  r"\["),
    ("RBRACK",  r"\]"),
    ("COLON",   r":"),
    ("COMMA",   r","),
    ("EQUAL",   r"="),
    ("DOT",     r"\."),
    ("MINUS",   r"-"),
    ("STAR",    r"\*"),
    ("SLASH",   r"/"),
]

_MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))

_WS_RE = re.compile(r"[ \t\r\n]+")

def _lex(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(text)
    while i < n:
        m = _WS_RE.match(text, i)
        if m:
            i = m.end()
            continue
        m = _MASTER_RE.match(text, i)
        if not m:
            raise SyntaxError(f"lex error at {i!r}: {text[i:i+20]!r}")
        kind = m.lastgroup  # type: ignore
        val = m.group()
        if kind == "COMMENT":
            i = m.end()
            continue
        tokens.append(Token(kind, val, i))
        i = m.end()
    tokens.append(Token("EOF", "", n))
    return tokens

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass
class Tweak:
    target: str  # e.g. "V1.L4" or "V1.L4.E" or "V1"
    op: str      # "-" or "*"
    value: float  # int for "-", float for "*"
    area: str
    layer: Optional[str] = None
    cell_type: Optional[str] = None

@dataclass
class LayerSpec:
    name: str
    n: Optional[Any] = None          # int or Tweak string for NExpr
    n_tweak: Optional[Tweak] = None
    depth: Optional[Tuple[float,float]] = None
    fractions: Optional[Dict[str,float]] = None
    tolerance: Optional[Dict[str,Tuple[float,float]]] = None
    geometry: Optional[Dict[str,Any]] = None
    sizes: Optional[Dict[str,float]] = None

@dataclass
class AreaSpec:
    name: str
    pose: Optional[Dict[str,Any]] = None
    layers: List[LayerSpec] = field(default_factory=list)
    connects: List[Dict[str,Any]] = field(default_factory=list)

@dataclass
class DevParams:
    params: Dict[str,Any] = field(default_factory=dict)

@dataclass
class DefineStmt:
    name: str
    areas: List[AreaSpec] = field(default_factory=list)
    area_connects: List[Dict[str,Any]] = field(default_factory=list)
    dev: DevParams = field(default_factory=DevParams)

@dataclass
class InheritStmt:
    child: str
    parent: str
    areas: List[AreaSpec] = field(default_factory=list)
    area_connects: List[Dict[str,Any]] = field(default_factory=list)
    dev: DevParams = field(default_factory=DevParams)

@dataclass
class UseStmt:
    base: str
    tweaks: List[Tweak] = field(default_factory=list)
    compose_with: List[str] = field(default_factory=list)
    poses: Optional[List[Dict[str,Any]]] = None
    as_name: Optional[str] = None

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    def peek(self, k=0) -> Token:
        idx = self.pos + k
        if idx < len(self.toks):
            return self.toks[idx]
        return self.toks[-1]

    def cur(self) -> Token:
        return self.toks[self.pos]

    def consume(self) -> Token:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def expect(self, kind: str, value: Optional[str]=None) -> Token:
        t = self.cur()
        if t.kind != kind or (value is not None and t.value != value):
            raise SyntaxError(f"expected {kind} {value!r} at {t.pos}, got {t.kind} {t.value!r}")
        self.pos += 1
        return t

    def match(self, kind: str, value: Optional[str]=None) -> bool:
        t = self.cur()
        if t.kind == kind and (value is None or t.value == value):
            self.pos += 1
            return True
        return False

    def match_ident(self, val: str) -> bool:
        t = self.cur()
        if t.kind == "IDENT" and t.value == val:
            self.pos += 1
            return True
        return False

    def expect_ident(self) -> str:
        t = self.expect("IDENT")
        return t.value

    def parse(self):
        stmts = []
        while self.cur().kind != "EOF":
            if self.cur().kind == "IDENT" and self.cur().value == "define":
                stmts.append(self.parse_define())
            elif self.cur().kind == "IDENT" and self.cur().value == "inherit":
                stmts.append(self.parse_inherit())
            elif self.cur().kind == "IDENT" and self.cur().value == "use":
                stmts.append(self.parse_use())
            else:
                raise SyntaxError(f"unexpected token {self.cur().kind} {self.cur().value!r} at {self.cur().pos}")
        return stmts

    # --- helpers for layer names with slash ---
    def parse_layer_name(self) -> str:
        # IDENT (SLASH IDENT)*
        name = self.expect_ident()
        # allow slash in layer name like L2/3
        while self.cur().kind == "SLASH":
            self.consume()
            # after slash, expect IDENT or INT? e.g., L2/3 where 3 is INT
            t = self.cur()
            if t.kind in ("IDENT", "INT"):
                self.consume()
                name = f"{name}/{t.value}"
            else:
                raise SyntaxError(f"expected IDENT after '/' at {t.pos}")
        return name

    def _parse_hyphenated_name(self) -> str:
        name = self.expect_ident()
        # hyphenated names like base-column, sparse-L4, base-frac08 : IDENT ("-" IDENT)*
        # Only consume "-" when followed by IDENT (not INT), to avoid eating tweak "-" before number
        while self.cur().kind == "MINUS" and self.peek(1).kind == "IDENT":
            self.consume()  # "-"
            name = f"{name}-{self.expect_ident()}"
        return name

    def parse_genome_name(self) -> str:
        return self._parse_hyphenated_name()

    def parse_area_name(self) -> str:
        # area names may also be hyphenated, but typically not; reuse same logic but allow hyphen before ident
        return self._parse_hyphenated_name()

    # --- Define ---
    def parse_define(self) -> DefineStmt:
        self.expect("IDENT", "define")
        name = self.parse_genome_name()
        self.expect("LBRACE")
        areas: List[AreaSpec] = []
        area_connects: List[Dict[str,Any]] = []
        dev = DevParams()
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated define block")
            if self.match_ident("area"):
                areas.append(self.parse_area_block())
            elif self.match_ident("development"):
                dev = self.parse_dev_block()
            elif self.match_ident("connect"):
                area_connects.append(self.parse_area_connect())
            else:
                raise SyntaxError(f"unexpected in define body: {self.cur().kind} {self.cur().value!r} at {self.cur().pos}")
        return DefineStmt(name=name, areas=areas, area_connects=area_connects, dev=dev)

    def parse_inherit(self) -> InheritStmt:
        self.expect("IDENT", "inherit")
        child = self.parse_genome_name()
        self.expect("IDENT", "from")
        parent = self.parse_genome_name()
        self.expect("LBRACE")
        areas: List[AreaSpec] = []
        area_connects: List[Dict[str,Any]] = []
        dev = DevParams()
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated inherit block")
            if self.match_ident("area"):
                areas.append(self.parse_area_override())
            elif self.match_ident("development"):
                dev = self.parse_dev_block()
            elif self.match_ident("connect"):
                area_connects.append(self.parse_area_connect())
            else:
                raise SyntaxError(f"unexpected in inherit body: {self.cur().kind} {self.cur().value!r} at {self.cur().pos}")
        return InheritStmt(child=child, parent=parent, areas=areas, area_connects=area_connects, dev=dev)

    def parse_use(self) -> UseStmt:
        self.expect("IDENT", "use")
        base = self.parse_genome_name()
        tweaks: List[Tweak] = []
        # peek for tweaks: TargetRef "-" INT  or  TargetRef "*" FLOAT
        while True:
            saved = self.pos
            # try_parse_tweak returns None if not a tweak; raises SyntaxError for invalid tweak domain (propagate)
            tw = self.try_parse_tweak()
            if tw is not None:
                tweaks.append(tw)
                continue
            else:
                self.pos = saved
                break
        compose_with: List[str] = []
        poses = None
        as_name = None
        if self.match_ident("compose"):
            self.expect("IDENT", "with")
            compose_with.append(self.parse_genome_name())
            while self.match("COMMA"):
                compose_with.append(self.parse_genome_name())
            if self.match_ident("poses"):
                # poses = <PoseList>  may be "=" then list, or directly list
                if self.match("EQUAL"):
                    poses = self.parse_pose_list()
                else:
                    # if next is "=", consume, else list
                    if self.cur().kind == "EQUAL":
                        self.consume()
                    poses = self.parse_pose_list()
            if self.match_ident("as"):
                as_name = self.parse_genome_name()
        # also allow "as" without compose (e.g., "use G V1.L4-80 as Sparse")
        if as_name is None and self.match_ident("as"):
            as_name = self.parse_genome_name()
        return UseStmt(base=base, tweaks=tweaks, compose_with=compose_with, poses=poses, as_name=as_name)

    def try_parse_tweak(self) -> Optional[Tweak]:
        # TargetRef "-" INT  or TargetRef "*" FLOAT
        # TargetRef = AreaName ["." LayerName ["." CellType]] ?
        # We need to not consume if not a tweak.
        if self.cur().kind != "IDENT":
            return None
        # peek ahead to see if after Ident chain we have MINUS or STAR
        # Save pos, try to parse TargetRef
        start = self.pos
        # parse area
        area = self.expect_ident()
        layer = None
        cell = None
        if self.match("DOT"):
            # layer name may contain slash
            layer = self.parse_layer_name()
            if self.match("DOT"):
                cell = self.expect_ident()
        # now check op
        if self.cur().kind == "MINUS":
            self.consume()
            t = self.expect("INT")
            val = int(t.value)
            if val <= 0:
                raise SyntaxError(f"A-INT must be >0, got {val}")
            target = area if layer is None else (f"{area}.{layer}" if cell is None else f"{area}.{layer}.{cell}")
            return Tweak(target=target, op="-", value=val, area=area, layer=layer, cell_type=cell)
        elif self.cur().kind == "STAR":
            self.consume()
            t = self.cur()
            if t.kind == "FLOAT":
                self.consume()
                val = float(t.value)
            elif t.kind == "INT":
                # treat INT as float for fraction tweak? per spec must be FLOAT, but allow INT
                self.consume()
                val = float(t.value)
            else:
                raise SyntaxError(f"expected FLOAT after '*' at {t.pos}, got {t.kind} {t.value!r}")
            if not (0.0 <= val <= 1.0):
                raise SyntaxError(f"A*FLOAT must be in [0,1], got {val}")
            # allow f=0 or 1 parse-legal, but will be validated later as n_neurons>0 may fail
            target = area if layer is None else (f"{area}.{layer}" if cell is None else f"{area}.{layer}.{cell}")
            return Tweak(target=target, op="*", value=val, area=area, layer=layer, cell_type=cell)
        else:
            # not a tweak
            return None

    def parse_area_block(self) -> AreaSpec:
        area_name = self.parse_area_name()
        pose = None
        if self.cur().kind == "IDENT" and self.cur().value == "pose":
            self.consume()
            pose = self.parse_pose_clause()
        self.expect("LBRACE")
        layers: List[LayerSpec] = []
        connects: List[Dict[str,Any]] = []
        # track duplicate layer names
        seen_layers = set()
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated area block")
            if self.cur().kind == "IDENT" and self.cur().value == "layer":
                self.consume()
                layer_name = self.parse_layer_name()
                if layer_name in seen_layers:
                    raise SyntaxError(f"duplicate layer name {layer_name!r} in area {area_name!r}")
                seen_layers.add(layer_name)
                spec = self.parse_layer_props(layer_name)
                layers.append(spec)
            elif self.cur().kind == "IDENT" and self.cur().value == "connect":
                self.consume()
                connects.append(self.parse_inter_connect())
            else:
                raise SyntaxError(f"unexpected in area block: {self.cur().kind} {self.cur().value!r} at {self.cur().pos}")
        return AreaSpec(name=area_name, pose=pose, layers=layers, connects=connects)

    def parse_area_override(self) -> AreaSpec:
        # same as area block but layers may be partial
        area_name = self.parse_area_name()
        pose = None
        if self.cur().kind == "IDENT" and self.cur().value == "pose":
            self.consume()
            pose = self.parse_pose_clause()
        self.expect("LBRACE")
        layers: List[LayerSpec] = []
        connects: List[Dict[str,Any]] = []
        seen_layers = set()
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated area override block")
            if self.cur().kind == "IDENT" and self.cur().value == "layer":
                self.consume()
                layer_name = self.parse_layer_name()
                if layer_name in seen_layers:
                    raise SyntaxError(f"duplicate layer name {layer_name!r} in area override {area_name!r}")
                seen_layers.add(layer_name)
                # LayerPropsPartial is optional; if next is IDENT with "=" then parse props, else empty spec meaning inherit all
                spec = LayerSpec(name=layer_name)
                # check if next tokens look like a prop key
                while self.cur().kind == "IDENT" and self.cur().value in ("n","depth","fractions","tolerance","geometry","sizes"):
                    self.parse_layer_prop_into(spec)
                layers.append(spec)
            elif self.cur().kind == "IDENT" and self.cur().value == "connect":
                self.consume()
                connects.append(self.parse_inter_connect())
            else:
                raise SyntaxError(f"unexpected in area override: {self.cur().kind} {self.cur().value!r} at {self.cur().pos}")
        return AreaSpec(name=area_name, pose=pose, layers=layers, connects=connects)

    def parse_dev_block(self) -> DevParams:
        self.expect("LBRACE")
        params: Dict[str,Any] = {}
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated development block")
            key = self.expect_ident()
            self.expect("EQUAL")
            t = self.cur()
            if t.kind == "INT":
                self.consume()
                params[key] = int(t.value)
            elif t.kind == "FLOAT":
                self.consume()
                params[key] = float(t.value)
            elif t.kind == "STRING":
                self.consume()
                params[key] = t.value[1:-1]  # strip quotes, handle escapes simplistically
            elif t.kind == "IDENT":
                # allow unquoted stringlike?
                self.consume()
                params[key] = t.value
            else:
                raise SyntaxError(f"expected value for dev param {key} at {t.pos}, got {t.kind} {t.value!r}")
        return DevParams(params=params)

    def parse_pose_clause(self) -> Dict[str,Any]:
        self.expect("LBRACE")
        out: Dict[str,Any] = {}
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated pose clause")
            key = self.expect_ident()
            self.expect("COLON")
            val = self.parse_pose_value()
            out[key] = val
            self.match("COMMA")
        return out

    def parse_pose_value(self) -> Any:
        t = self.cur()
        if t.kind == "STRING":
            self.consume()
            return t.value[1:-1]
        elif t.kind == "FLOAT":
            self.consume()
            return float(t.value)
        elif t.kind == "INT":
            self.consume()
            return int(t.value)
        elif t.kind == "IDENT":
            # e.g., xy plane without quotes? allow
            self.consume()
            return t.value
        elif t.kind == "LBRACK":
            return self.parse_list_value()
        else:
            raise SyntaxError(f"unexpected pose value {t.kind} {t.value!r} at {t.pos}")

    def parse_list_value(self) -> List[Any]:
        self.expect("LBRACK")
        vals: List[Any] = []
        while not self.match("RBRACK"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated list")
            t = self.cur()
            if t.kind == "FLOAT":
                self.consume()
                vals.append(float(t.value))
            elif t.kind == "INT":
                self.consume()
                # keep int
                vals.append(int(t.value))
            elif t.kind == "STRING":
                self.consume()
                vals.append(t.value[1:-1])
            else:
                raise SyntaxError(f"unexpected list element {t.kind} {t.value!r}")
            self.match("COMMA")
        return vals

    def parse_pose_list(self) -> List[Dict[str,Any]]:
        # PoseList is like [{plane:"xy", translation:[0,0,0]}, ...] or single pose?
        # We parse a bracketed list of pose maps: "[" "{" ... "}" ("," "{" ... "}")* "]"
        # But spec shows: poses [{plane:"xy", translation:[0,0,0]}, {plane:"xy", translation:[1,0,0]}]
        # without "=" already consumed. So we check for "[" then.
        if self.cur().kind == "LBRACK":
            self.consume()
            poses: List[Dict[str,Any]] = []
            while not self.match("RBRACK"):
                if self.cur().kind == "LBRACE":
                    # parse pose map
                    self.consume()
                    m: Dict[str,Any] = {}
                    while not self.match("RBRACE"):
                        k = self.expect_ident()
                        self.expect("COLON")
                        v = self.parse_pose_value()
                        m[k] = v
                        self.match("COMMA")
                    poses.append(m)
                    self.match("COMMA")
                else:
                    raise SyntaxError(f"expected pose map at {self.cur().pos}")
            return poses
        else:
            raise SyntaxError(f"expected pose list '[' at {self.cur().pos}, got {self.cur().kind} {self.cur().value!r}")

    def parse_layer_props(self, layer_name: str) -> LayerSpec:
        spec = LayerSpec(name=layer_name)
        # LayerProps ordered: n, depth, fractions, [tolerance], [geometry], [sizes]
        # We'll parse sequentially but allow any order for flexibility while requiring n,depth,fractions somewhere
        # First, loop while next is prop key
        while self.cur().kind == "IDENT" and self.cur().value in ("n","depth","fractions","tolerance","geometry","sizes"):
            self.parse_layer_prop_into(spec)
        # Validate required fields present for define (caller ensures)
        return spec

    def parse_layer_prop_into(self, spec: LayerSpec):
        key = self.expect_ident()
        # "=" is required for n/depth but optional for map-style props (fractions/tolerance/geometry/sizes/pose) per examples
        if self.cur().kind == "EQUAL":
            self.consume()
        else:
            if key in ("n", "depth"):
                raise SyntaxError(f"expected '=' after {key!r} at {self.cur().pos}, got {self.cur().kind} {self.cur().value!r}")
            # for fractions/tolerance/geometry/sizes, allow implicit "="
            pass
        if key == "n":
            # NExpr: INT | TargetRef "-" INT | TargetRef "*" FLOAT
            # check if next is INT directly
            if self.cur().kind == "INT" and self.peek(1).kind not in ("MINUS","STAR","DOT"):
                t = self.consume()
                spec.n = int(t.value)
            else:
                # try tweak
                saved = self.pos
                # need to parse TargetRef then op
                # TargetRef for NExpr: same as tweak target but without op? Actually NExpr is AbsoluteTweak | FractionTweak | INT
                # So we can reuse try_parse_tweak logic but starting from here
                # Parse TargetRef then op
                try:
                    tw = self.try_parse_tweak_from_n()
                    if tw is not None:
                        spec.n_tweak = tw
                        spec.n = tw  # store tweak
                    else:
                        # maybe bare float? not allowed
                        raise SyntaxError(f"expected INT or tweak for n at {self.cur().pos}")
                except SyntaxError:
                    self.pos = saved
                    raise
        elif key == "depth":
            spec.depth = self.parse_depth_band()
        elif key == "fractions":
            spec.fractions = self.parse_fractions_map()
        elif key == "tolerance":
            spec.tolerance = self.parse_tolerance_map()
        elif key == "geometry":
            spec.geometry = self.parse_geometry_map()
        elif key == "sizes":
            spec.sizes = self.parse_sizes_map()
        else:
            raise SyntaxError(f"unknown layer prop {key!r}")

    def try_parse_tweak_from_n(self) -> Optional[Tweak]:
        # same as try_parse_tweak but we are at position where TargetRef starts
        # we reuse try_parse_tweak which expects to parse from current pos
        return self.try_parse_tweak()

    def parse_depth_band(self) -> Tuple[float,float]:
        self.expect("LBRACK")
        lo = self.parse_number()
        self.expect("COMMA")
        hi = self.parse_number()
        self.expect("RBRACK")
        return (float(lo), float(hi))

    def parse_number(self) -> float:
        t = self.cur()
        if t.kind == "FLOAT":
            self.consume()
            return float(t.value)
        elif t.kind == "INT":
            self.consume()
            return float(t.value)
        else:
            raise SyntaxError(f"expected number at {t.pos}, got {t.kind} {t.value!r}")

    def parse_fractions_map(self) -> Dict[str,float]:
        self.expect("LBRACE")
        out: Dict[str,float] = {}
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated fractions map")
            if self.cur().kind == "RBRACE":
                break
            ct = self.expect_ident()
            # allow slash in cell type? no
            self.expect("COLON")
            val = self.parse_number()
            out[ct] = float(val)
            self.match("COMMA")
        return out

    def parse_tolerance_map(self) -> Dict[str,Tuple[float,float]]:
        self.expect("LBRACE")
        out: Dict[str,Tuple[float,float]] = {}
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated tolerance map")
            ct = self.expect_ident()
            self.expect("COLON")
            self.expect("LBRACK")
            lo = self.parse_number()
            self.expect("COMMA")
            hi = self.parse_number()
            self.expect("RBRACK")
            out[ct] = (float(lo), float(hi))
            self.match("COMMA")
        return out

    def parse_geometry_map(self) -> Dict[str,Any]:
        self.expect("LBRACE")
        out: Dict[str,Any] = {}
        while not self.match("RBRACE"):
            if self.cur().kind == "EOF":
                raise SyntaxError("unterminated geometry map")
            key = self.expect_ident()
            self.expect("COLON")
            t = self.cur()
            if t.kind == "STRING":
                self.consume()
                out[key] = t.value[1:-1]
            elif t.kind == "FLOAT":
                self.consume()
                out[key] = float(t.value)
            elif t.kind == "INT":
                self.consume()
                out[key] = int(t.value)
            elif t.kind == "LBRACK":
                out[key] = self.parse_list_value()
            elif t.kind == "IDENT":
                # allow bare ident as value
                self.consume()
                out[key] = t.value
            else:
                raise SyntaxError(f"unexpected geometry value {t.kind} {t.value!r}")
            self.match("COMMA")
        return out

    def parse_sizes_map(self) -> Dict[str,float]:
        return self.parse_fractions_map()

    def parse_inter_connect(self) -> Dict[str,Any]:
        # "connect L4 E -> L2/3 E mechanism:\"AMPA\""  (inside area)
        # parse source_layer, source_type, ->, target_layer, target_type, [mechanism]
        # source_layer may contain slash
        src_layer = self.parse_layer_name()
        # next must be IDENT (source type)
        src_type = self.expect_ident()
        self.expect("ARROW")
        tgt_layer = self.parse_layer_name()
        tgt_type = self.expect_ident()
        mech = None
        # optional mechanism clause: IDENT ":" STRING  or just STRING?
        # allow "mechanism:\"AMPA\"" or "mechanism=\"AMPA\"" or "mechanism: AMPA"
        if self.cur().kind == "IDENT" and self.cur().value == "mechanism":
            self.consume()
            # expect COLON or EQUAL
            if self.cur().kind in ("COLON","EQUAL"):
                self.consume()
            t = self.cur()
            if t.kind == "STRING":
                self.consume()
                mech = t.value[1:-1]
            elif t.kind == "IDENT":
                self.consume()
                mech = t.value
            else:
                raise SyntaxError(f"expected mechanism value at {t.pos}")
        elif self.cur().kind == "COLON" and self.peek(1).kind == "STRING":
            pass  # not expected
        return {"source_layer": src_layer, "source_neuron_type": src_type,
                "target_layer": tgt_layer, "target_neuron_type": tgt_type,
                "mechanism": mech or "AMPA"}

    def parse_area_connect(self) -> Dict[str,Any]:
        # top-level connect for area_connections
        # Could be "connect V1.L2 E -> V1.L4 E mechanism:\"AMPA\""  or simple layer form
        # We try to parse source as TargetRef-like (Area.Layer.CellType) but with space separated cell type?
        # We'll parse flexibly: first part may be Area.Layer or Layer
        # Approach: parse first ident chain: IDENT (DOT IDENT (SLASH ...))? then optionally IDENT as cell type if next not ARROW
        # Simpler: parse as inter_connect but with optional area prefix via DOT
        # Let's look ahead: if after first IDENT we have DOT, then it's area-prefixed
        # source_area, source_layer, source_type
        # For now, parse generically into dict with source_area, source_layer, source_neuron_type, target_area, etc.
        # We'll try to parse source as: [Area "."] Layer CellType
        first = self.expect_ident()
        source_area = None
        source_layer = None
        source_type = None
        # check for DOT
        if self.cur().kind == "DOT":
            self.consume()
            # following is layer name (with slash)
            layer_name = self.parse_layer_name()
            source_area = first
            source_layer = layer_name
            # next should be cell type ident if not ARROW
            if self.cur().kind == "IDENT" and self.peek(1).kind == "ARROW":
                source_type = self.expect_ident()
            elif self.cur().kind == "IDENT":
                # maybe cell type before arrow
                # peek two ahead
                if self.peek(1).kind == "ARROW":
                    source_type = self.expect_ident()
                else:
                    source_type = self.expect_ident()
            else:
                raise SyntaxError(f"expected source neuron type at {self.cur().pos}")
        else:
            # first was layer, second is type, third is arrow?
            # first is actually source_layer, need source_type
            source_layer = first
            # handle slash continuation already? first was plain ident, slash would have been consumed as separate? For layer names with slash, parse_layer_name would have consumed, but we already consumed first as ident only.
            # If next is SLASH, we need to handle: "L2" "/" "3" -> layer "L2/3"
            if self.cur().kind == "SLASH":
                self.consume()
                t = self.cur()
                if t.kind in ("IDENT","INT"):
                    self.consume()
                    source_layer = f"{source_layer}/{t.value}"
                else:
                    raise SyntaxError("expected layer suffix after '/'")
            # now source_type
            source_type = self.expect_ident()
            # No area for inter-like connect at top level? But spec says top-level area_connections have source_area etc.
            # If this is top-level with no area prefix, we can assume source_area remains None and will be filled later as same area? But we treat as within-area style and convert to area_connections with same area? For simplicity, we keep source_area=None and let builder handle.
        self.expect("ARROW")
        # parse target similarly
        tgt_first = self.expect_ident()
        target_area = None
        target_layer = None
        target_type = None
        if self.cur().kind == "DOT":
            self.consume()
            layer_name = self.parse_layer_name()
            target_area = tgt_first
            target_layer = layer_name
            target_type = self.expect_ident()
        else:
            target_layer = tgt_first
            if self.cur().kind == "SLASH":
                self.consume()
                t = self.cur()
                if t.kind in ("IDENT","INT"):
                    self.consume()
                    target_layer = f"{target_layer}/{t.value}"
            target_type = self.expect_ident()
        mech = None
        if self.cur().kind == "IDENT" and self.cur().value == "mechanism":
            self.consume()
            if self.cur().kind in ("COLON","EQUAL"):
                self.consume()
            t = self.cur()
            if t.kind == "STRING":
                self.consume()
                mech = t.value[1:-1]
            elif t.kind == "IDENT":
                self.consume()
                mech = t.value
            else:
                raise SyntaxError(f"expected mechanism value at {t.pos}")
        # Build dict: for genome-level, need source_area/target_area; if missing, use placeholder and let caller decide
        return {"source_area": source_area, "source_layer": source_layer, "source_neuron_type": source_type,
                "target_area": target_area, "target_layer": target_layer, "target_neuron_type": target_type,
                "mechanism": mech or "monotonic_cable_synapse"}

# ---------------------------------------------------------------------------
# Desugar helpers
# ---------------------------------------------------------------------------

def parse_text(text: str):
    toks = _lex(text)
    p = Parser(toks)
    return p.parse()

def parse_file(path: str | Path):
    return parse_text(Path(path).read_text(encoding="utf-8"))

# Helpers to turn parsed specs into PseudoGenome

def _layer_spec_to_kwargs(spec: LayerSpec) -> dict:
    kw: dict[str,Any] = {"name": spec.name}
    # n
    if spec.n is not None and not isinstance(spec.n, Tweak):
        kw["n_neurons"] = int(spec.n)
    elif spec.n_tweak is not None:
        raise ValueError("n tweak inside define not yet desugared: use apply_tweaks for use statements")
    # depth
    if spec.depth is not None:
        kw["depth_band"] = tuple(spec.depth)
    # fractions
    if spec.fractions is not None:
        kw["cell_type_fractions"] = dict(spec.fractions)
    # tolerance
    if spec.tolerance is not None:
        kw["fraction_tolerance"] = {k: tuple(v) for k,v in spec.tolerance.items()}
    # geometry
    if spec.geometry is not None:
        geom = dict(spec.geometry)
        # normalize x_range/y_range if present as list
        for k in ("x_range","y_range","z_range"):
            if k in geom and isinstance(geom[k], list):
                geom[k] = tuple(float(x) for x in geom[k])
        # distribution, value_tag
        kw["geometry"] = geom
    # sizes
    if spec.sizes is not None:
        kw["relative_sizes"] = dict(spec.sizes)
    return kw

def _build_pseudogenome_from_define(stmt: DefineStmt):
    from jaxfne.jdna.genome import PseudoGenome, AreaGenome, LayerGenome, ConnectionRuleGenome
    areas = []
    for a in stmt.areas:
        layers = []
        for ls in a.layers:
            # require n, depth, fractions for define
            if ls.n is None or ls.depth is None or ls.fractions is None:
                raise SyntaxError(f"layer {ls.name!r} in area {a.name!r} missing required n/depth/fractions in define")
            kw = _layer_spec_to_kwargs(ls)
            # defaults for missing optional already handled
            layers.append(LayerGenome(**kw))
        conns = [ConnectionRuleGenome(**c) for c in a.connects]
        pose = dict(a.pose) if a.pose is not None else {"plane":"xy","rotation_deg":0.0,"translation":(0.0,0.0,0.0),"value_tag":"relative"}
        # normalize translation
        if "translation" in pose and isinstance(pose["translation"], list):
            pose["translation"] = tuple(float(x) for x in pose["translation"])
        areas.append(AreaGenome(name=a.name, layers=tuple(layers), inter_connections=tuple(conns), pose=pose))
    area_conns = []
    for c in stmt.area_connects:
        # need to ensure source_area/target_area are set; if None, skip or error
        # If dict has None area, we cannot create area_connections entry; treat as error unless we infer
        if c.get("source_area") is None or c.get("target_area") is None:
            # If both none, it's actually an inter-connection mis-placed at top level; ignore
            continue
        area_conns.append(dict(c))
    return PseudoGenome(name=stmt.name, description="", development_parameters=dict(stmt.dev.params), areas=tuple(areas), area_connections=tuple(area_conns))

def _merge_genomes(parent, child_stmt: InheritStmt):
    from jaxfne.jdna.genome import PseudoGenome, AreaGenome, LayerGenome, ConnectionRuleGenome
    # deep merge per §6.2
    # development_parameters shallow-merged, child overrides
    new_dev = dict(parent.development_parameters)
    new_dev.update(child_stmt.dev.params)
    # Build map area name -> AreaGenome for parent
    parent_areas: Dict[str, AreaGenome] = {a.name: a for a in parent.areas}
    # Track which areas are overridden
    merged_areas: List[AreaGenome] = []
    child_area_names = {a.name for a in child_stmt.areas}
    # First, process all parent areas in order, merging if child overrides
    for pa in parent.areas:
        if pa.name not in child_area_names:
            merged_areas.append(pa)
        else:
            ca = next(a for a in child_stmt.areas if a.name == pa.name)
            # pose deep-merged key-by-key
            new_pose = dict(pa.pose)
            if ca.pose is not None:
                new_pose.update(ca.pose)
                if "translation" in new_pose and isinstance(new_pose["translation"], list):
                    new_pose["translation"] = tuple(float(x) for x in new_pose["translation"])
            # layers merged per layer rules
            child_layers_by_name = {l.name: l for l in ca.layers}
            parent_layers_by_name = {l.name: l for l in pa.layers}
            merged_layers: List[LayerGenome] = []
            # parent layers first, in parent order, with overrides
            for pl in pa.layers:
                if pl.name not in child_layers_by_name:
                    merged_layers.append(pl)
                else:
                    cl = child_layers_by_name[pl.name]
                    # per-field override: each field present in child's LayerPropsPartial replaces parent field
                    # geometry/sizes/pose maps merged key-by-key
                    kw: dict[str,Any] = {}
                    kw["name"] = pl.name
                    # n
                    if cl.n is not None and not isinstance(cl.n, Tweak):
                        kw["n_neurons"] = int(cl.n)
                    elif cl.n_tweak is not None:
                        # n tweak inside inherit: same as absolute? For simplicity, treat as absolute if op="-" else fractional? Use denominator N_area(parent) snapshot
                        # We'll desugar n_tweak via apply step: if op "-", set int; if "*", set fraction of parent area total
                        tw = cl.n_tweak
                        if tw.op == "-":
                            kw["n_neurons"] = int(tw.value)
                        else:
                            # fractional: compute N_enclosing = sum n_neurons for this area in parent
                            N_area = sum(l.n_neurons for l in pa.layers)
                            raw = N_area * float(tw.value)
                            # largest remainder not needed for single layer override; just round
                            kw["n_neurons"] = int(round(raw))
                    else:
                        kw["n_neurons"] = pl.n_neurons
                    kw["depth_band"] = tuple(cl.depth) if cl.depth is not None else pl.depth_band
                    kw["cell_type_fractions"] = dict(cl.fractions) if cl.fractions is not None else dict(pl.cell_type_fractions)
                    # tolerance: field-granular per §6.3: child replaces intervals for listed CTs only
                    if cl.tolerance is not None:
                        merged_tol = dict(pl.fraction_tolerance)
                        merged_tol.update({k: tuple(v) for k,v in cl.tolerance.items()})
                        kw["fraction_tolerance"] = merged_tol
                    else:
                        kw["fraction_tolerance"] = dict(pl.fraction_tolerance)
                    if cl.geometry is not None:
                        mg = dict(pl.geometry)
                        mg.update(cl.geometry)
                        kw["geometry"] = mg
                    else:
                        kw["geometry"] = dict(pl.geometry)
                    if cl.sizes is not None:
                        ms = dict(pl.relative_sizes)
                        ms.update(cl.sizes)
                        kw["relative_sizes"] = ms
                    else:
                        kw["relative_sizes"] = dict(pl.relative_sizes)
                    merged_layers.append(LayerGenome(**kw))
            # add new layers from child that weren't in parent
            for cl in ca.layers:
                if cl.name not in parent_layers_by_name:
                    if cl.n is None or cl.depth is None or cl.fractions is None:
                        raise ValueError(f"new layer {cl.name!r} in inherit must declare n/depth/fractions")
                    kw = _layer_spec_to_kwargs(cl)
                    # need to include name
                    merged_layers.append(LayerGenome(**kw))
            # inter_connections: whole-list replacement if child provides any, else inherited
            if ca.connects:
                new_conns = tuple(ConnectionRuleGenome(**c) for c in ca.connects)
            else:
                new_conns = tuple(pa.inter_connections)
            merged_areas.append(AreaGenome(name=pa.name, layers=tuple(merged_layers), inter_connections=new_conns, pose=new_pose))
    # add new areas from child that weren't in parent
    for ca in child_stmt.areas:
        if ca.name not in parent_areas:
            # must have at least one layer
            if not ca.layers:
                raise ValueError(f"new area {ca.name!r} must have at least one layer")
            layers = []
            for cl in ca.layers:
                if cl.n is None or cl.depth is None or cl.fractions is None:
                    raise ValueError(f"new layer {cl.name!r} in new area {ca.name!r} missing n/depth/fractions")
                layers.append(LayerGenome(**_layer_spec_to_kwargs(cl)))
            conns = tuple(ConnectionRuleGenome(**c) for c in ca.connects)
            pose = dict(ca.pose) if ca.pose is not None else {"plane":"xy","rotation_deg":0.0,"translation":(0.0,0.0,0.0),"value_tag":"relative"}
            if "translation" in pose and isinstance(pose["translation"], list):
                pose["translation"] = tuple(float(x) for x in pose["translation"])
            merged_areas.append(AreaGenome(name=ca.name, layers=tuple(layers), inter_connections=conns, pose=pose))
    # area_connections: whole-list replacement if child declares any, else inherited
    if child_stmt.area_connects:
        # need to map those where source_area may be None? if so, we need area names, assume first area
        new_area_conns = tuple(dict(c) for c in child_stmt.area_connects if c.get("source_area") is not None)
    else:
        new_area_conns = tuple(parent.area_connections)
    return PseudoGenome(name=child_stmt.child, description=parent.description, development_parameters=new_dev, areas=tuple(merged_areas), area_connections=new_area_conns)

def _largest_remainder_allocation(weighted: Dict[str,float], total: int) -> Dict[str,int]:
    """Largest-remainder allocation for arbitrary weighted dict to sum total."""
    floors = {k: int(math.floor(v)) for k,v in weighted.items()}
    remainder = total - sum(floors.values())
    if remainder < 0:
        raise ValueError(f"weighted allocation exceeds total {total}: sum floors {sum(floors.values())} > total")
    if remainder == 0:
        return floors
    # order by fractional remainder descending, then stable key order
    # need to handle remainder > len(types): allocate cyclically in order
    order = sorted(weighted.keys(), key=lambda k: weighted[k] - floors[k], reverse=True)
    # distribute remainder one by one wrapping
    idx = 0
    while remainder > 0:
        k = order[idx % len(order)]
        floors[k] += 1
        remainder -= 1
        idx += 1
        # avoid infinite if weighted all zero? but then floors 0 and remainder total>0, will still allocate equally
    return floors

def _apply_tweaks_to_genome(genome, tweaks: List[Tweak], preserve: bool = True):
    from jaxfne.jdna.genome import PseudoGenome, AreaGenome, LayerGenome
    # group tweaks by area
    # last writer wins per layer
    # Build final per-layer tweak map: dict (area,layer) -> Tweak
    final: Dict[Tuple[str,Optional[str]], Tweak] = {}
    for tw in tweaks:
        # bare area only allowed if that area has exactly one layer
        # validation later
        key = (tw.area, tw.layer)  # layer may be None for bare area
        final[key] = tw
    # Now process each area
    new_areas: List[AreaGenome] = []
    for area in genome.areas:
        N_area0 = sum(l.n_neurons for l in area.layers)
        # collect tweaks targeting this area
        # need to handle bare area tweak: target "V1" with layer None
        area_tweaks: Dict[str, Tweak] = {}  # layer name -> tweak
        bare_tweaks_for_area: List[Tweak] = []
        for (a, lyr), tw in final.items():
            if a != area.name:
                continue
            if lyr is None:
                bare_tweaks_for_area.append(tw)
            else:
                area_tweaks[lyr] = tw
        # handle bare area tweaks: only allowed if exactly one layer
        if bare_tweaks_for_area:
            if len(area.layers) != 1:
                raise SyntaxError(f"bare Area tweak {bare_tweaks_for_area[0].target!r} requires area {area.name!r} to have exactly one layer, has {len(area.layers)}")
            # map bare to the single layer's name
            sole = area.layers[0].name
            # last bare wins if multiple
            last_bare = bare_tweaks_for_area[-1]
            # re-key as that layer
            area_tweaks[sole] = Tweak(target=f"{area.name}.{sole}", op=last_bare.op, value=last_bare.value, area=area.name, layer=sole)
        # separate absolute and fractional
        abs_map: Dict[str,int] = {}
        frac_map: Dict[str,float] = {}
        for lname, tw in area_tweaks.items():
            if tw.op == "-":
                abs_map[lname] = int(tw.value)
            else:
                frac_map[lname] = float(tw.value)
        # if no tweaks for this area, keep as is
        if not abs_map and not frac_map:
            new_areas.append(area)
            continue
        # Validate that all tweak target layers exist
        layer_names = {l.name for l in area.layers}
        for lname in list(abs_map.keys()) + list(frac_map.keys()):
            if lname not in layer_names:
                raise ValueError(f"tweak target {area.name}.{lname!r} references unknown layer")
        # Now compute new n_neurons per layer
        new_n: Dict[str,int] = {}
        if preserve:
            Total = N_area0
            N_fixed = sum(abs_map.values())
            if N_fixed > Total:
                raise ValueError(f"absolute tweaks sum {N_fixed} exceeds area total {Total} for area {area.name!r} in preserving mode")
            Remaining = Total - N_fixed
            # raw fractional targets: N_area0 * f
            raw_frac: Dict[str,float] = {lname: N_area0 * f for lname,f in frac_map.items()}
            sum_raw_frac = sum(raw_frac.values())
            # untreated layers
            untreated = [l for l in area.layers if l.name not in abs_map and l.name not in frac_map]
            sum_orig_untreated = sum(l.n_neurons for l in untreated)
            if frac_map and sum_raw_frac > Remaining + 1e-9:
                # fractional targets exceed remaining: need to scale down fractional to fit Remaining, untreated would be zero?
                # For preserving, we scale fractional proportionally to fit Remaining if untreated empty, else error?
                # If untreated present, excess would push untreated to zero or negative => infeasible
                # We raise if sum_raw_frac > Remaining and untreated not empty
                if untreated:
                    # Need to decide: scale frac down? Spec says largest-remainder over fractional subset, so sum should not exceed Remaining if untreated exists.
                    # We'll raise as validation error
                    raise ValueError(f"fractional tweaks sum {sum_raw_frac} exceeds remaining {Remaining} after absolutes for area {area.name!r}")
                else:
                    # only frac layers, scale them to fit Remaining proportionally
                    # weighted scaled
                    weighted_scaled = {lname: raw * (Remaining / sum_raw_frac) for lname,raw in raw_frac.items()}
                    alloc = _largest_remainder_allocation(weighted_scaled, Remaining)
                    for lname in frac_map:
                        new_n[lname] = alloc[lname]
            else:
                # normal: frac layers get raw, untreated get scaled proportionally
                # frac allocation
                # Use largest remainder for frac layers if needed? raw are already integer-ish but we need integer n
                # We'll allocate frac via rounding with largest remainder to sum to integer round of sum_raw_frac? Actually raw may be float, need integer
                # We'll use largest remainder for the combined non-fixed pool? Simpler: handle frac and untreated together via weighted dict.
                # Build weighted for non-fixed pool to distribute Remaining
                weighted: Dict[str,float] = {}
                for lname, raw in raw_frac.items():
                    weighted[lname] = float(raw)
                if untreated:
                    # remaining for untreated is Remaining - sum_raw_frac
                    rem_for_untreated = Remaining - sum_raw_frac
                    if sum_orig_untreated > 0:
                        for l in untreated:
                            # scale untreated proportionally to original
                            weighted[l.name] = float(l.n_neurons) * (rem_for_untreated / sum_orig_untreated) if sum_orig_untreated else 0.0
                    else:
                        # no untreated sum? shouldn't happen
                        pass
                # Now allocate integer counts summing to Remaining via largest remainder
                if weighted:
                    alloc = _largest_remainder_allocation(weighted, Remaining)
                    for k,v in alloc.items():
                        new_n[k] = v
                # absolute layers
            for lname, v in abs_map.items():
                new_n[lname] = v
        else:
            # mutating mode: total may change: new Total = N_fixed + sum_raw_frac + sum of untreated originals? Actually mutating allows area total to change as old total - old L4 + new
            # For single absolute, new total = old total - old_layer + new
            # For fractional mutating, raw is still N_area0 * f, but area total mutates to sum of new counts (no preserving constraint)
            # We implement mutating as: for each layer, if abs, use abs; if frac, use round(N_area0*f); else keep original
            # This changes area total.
            for l in area.layers:
                if l.name in abs_map:
                    new_n[l.name] = abs_map[l.name]
                elif l.name in frac_map:
                    raw = N_area0 * frac_map[l.name]
                    # largest remainder not needed for single, but we use round
                    # Use largest remainder across frac layers if multiple? We'll just round individually for mutating mode (no sum constraint)
                    new_n[l.name] = int(round(raw))
                else:
                    new_n[l.name] = l.n_neurons
        # Build new layers with updated n_neurons
        new_layers: List[LayerGenome] = []
        for l in area.layers:
            nn = new_n.get(l.name, l.n_neurons)
            if nn <= 0:
                raise ValueError(f"layer {l.name!r}: n_neurons must be positive after tweak, got {nn}")
            new_layers.append(LayerGenome(name=l.name, n_neurons=nn, depth_band=l.depth_band, cell_type_fractions=dict(l.cell_type_fractions), fraction_tolerance=dict(l.fraction_tolerance), geometry=dict(l.geometry), relative_sizes=dict(l.relative_sizes)))
        new_areas.append(AreaGenome(name=area.name, layers=tuple(new_layers), inter_connections=tuple(area.inter_connections), pose=dict(area.pose)))
    # check that after tweaks, no area has zero layers etc. validation will catch
    # Build new genome name: if any tweak, name is original + tweak suffix? For use "as" clause, caller will rename
    return PseudoGenome(name=genome.name, description=genome.description, development_parameters=dict(genome.development_parameters), areas=tuple(new_areas), area_connections=tuple(genome.area_connections))

def desugar_file(text: str, registry: Optional[Dict[str,Any]]=None):
    """Parse text and return dict name -> PseudoGenome after desugaring define/inherit/use."""
    from jaxfne.jdna.genome import validate_genome
    stmts = parse_text(text)
    reg: Dict[str,Any] = dict(registry) if registry else {}
    # also preload canonical if needed? we can load from jaxfne
    for stmt in stmts:
        if isinstance(stmt, DefineStmt):
            if stmt.name in reg:
                raise ValueError(f"duplicate genome name {stmt.name!r}")
            g = _build_pseudogenome_from_define(stmt)
            validate_genome(g)
            reg[stmt.name] = g
        elif isinstance(stmt, InheritStmt):
            if stmt.parent not in reg:
                # try load canonical
                try:
                    from jaxfne.jdna.genome import load_canonical_pseudogenome
                    pg = load_canonical_pseudogenome(stmt.parent)
                    reg[stmt.parent] = pg
                except Exception as e:
                    raise ValueError(f"inherit parent {stmt.parent!r} not found") from e
            parent = reg[stmt.parent]
            child = _merge_genomes(parent, stmt)
            validate_genome(child)
            reg[stmt.child] = child
        elif isinstance(stmt, UseStmt):
            if stmt.base not in reg:
                try:
                    from jaxfne.jdna.genome import load_canonical_pseudogenome
                    pg = load_canonical_pseudogenome(stmt.base)
                    reg[stmt.base] = pg
                except Exception as e:
                    raise ValueError(f"use base {stmt.base!r} not found") from e
            base = reg[stmt.base]
            tweaked = _apply_tweaks_to_genome(base, stmt.tweaks, preserve=True)
            # handle compose? For now, if compose_with present, we just record but not merge at genome level (tensor level)
            # The use "as" rename
            final_name = stmt.as_name or f"{stmt.base}_tweaked"
            tweaked_typed = tweaked  # already PseudoGenome, need to rename
            from jaxfne.jdna.genome import PseudoGenome
            # recreate with new name
            renamed = PseudoGenome(name=final_name, description=tweaked.description, development_parameters=dict(tweaked.development_parameters), areas=tweaked.areas, area_connections=tweaked.area_connections)
            from jaxfne.jdna.genome import validate_genome as vg2
            vg2(renamed)
            reg[final_name] = renamed
            # If compose_with, we could develop and merge tensors; but we leave as separate genomes to be merged by caller via merge_neuronal_tensors after develop
            # For convenience, if compose_with non-empty, also store a note
            if stmt.compose_with:
                # Validate compose targets exist or are canonical
                for cname in stmt.compose_with:
                    if cname not in reg:
                        try:
                            from jaxfne.jdna.genome import load_canonical_pseudogenome
                            reg[cname] = load_canonical_pseudogenome(cname)
                        except Exception:
                            pass
                # pose arity check
                if stmt.poses is not None:
                    total_areas = sum(len(reg[n].areas) for n in [stmt.base] + stmt.compose_with if n in reg)
                    # But reg[final_name] is tweaked version of base, not base itself; use final_name count + others
                    # Count areas in tweaked + compose targets
                    t_areas = len(renamed.areas)
                    other_areas = sum(len(reg[c].areas) for c in stmt.compose_with if c in reg)
                    if len(stmt.poses) != t_areas + other_areas:
                        raise ValueError(f"poses must have one entry per area; got {len(stmt.poses)} for {t_areas+other_areas} areas")
                # Store compose metadata as attribute on renamed? We'll attach via reg key
                reg[f"{final_name}__compose"] = {"with": stmt.compose_with, "poses": stmt.poses, "as": final_name}
        else:
            raise ValueError(f"unknown stmt {stmt}")
    return reg

def develop_with_provenance(genome, seed: int = 0):
    """Wrapper around jaxfne.jdna.genome.develop that returns tensor with provenance."""
    from jaxfne.jdna.genome import develop
    return develop(genome, seed=seed)

def merge_tensors_after_develop(genomes: List[Any], seeds: List[int], poses=None, name="merged"):
    """Develop each genome with its K_D, then merge via merge_neuronal_tensors."""
    from jaxfne.jdna.genome import develop
    from jaxfne.neuronal_tensor import merge_neuronal_tensors, Pose3D
    tensors = [develop(g, seed=s) for g, s in zip(genomes, seeds)]
    # poses may be list of dicts -> convert to Pose3D
    pose_objs = None
    if poses is not None:
        pose_objs = []
        for p in poses:
            # p is dict like {"plane":"xy","translation":[0,0,0], ...}
            translation = p.get("translation", [0.0,0.0,0.0])
            if isinstance(translation, list):
                translation = tuple(float(x) for x in translation)
            pose_objs.append(Pose3D(plane=p.get("plane","xy"), rotation_deg=float(p.get("rotation_deg",0.0)), translation=translation, value_tag=p.get("value_tag","relative")))
    merged = merge_neuronal_tensors(tensors, poses=pose_objs, name=name)
    return merged, tensors
