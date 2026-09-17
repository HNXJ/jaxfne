"""TFNE algebra: compact hierarchical specification language for JaxFNE.

This module compiles the canonical TFNE algebra
(``artifacts/project_sources/7_tfne_algebra.md``, language version ``tfne/2``)
as a **specification-time compiler layer**. The language is authority and this
module is not: conformance is partial, and the measured gap is recorded under
"Compiler conformance" in ``docs/doctrine/tfne_algebra.md``.
It introduces no new simulator. TFNE text resolves to an explicit typed neural
model, which is compiled two ways from the same source::

    TFNE -> resolve -> explicit model -> realize            -> (s, h0, I)
                                      -> to_neuronal_tensor -> construct -> kernels

``realize`` produces ``(s, h0, I)`` for inspection and indexing; execution runs
through ``to_neuronal_tensor`` into the existing JaxFNE kernels. The two agree
on topology and identity (verified edge for edge in
``tests/test_tfne_execution.py``), but the tensor bridge does not carry rule
parameters: a declared connection ``weight`` reaches ``(s, h0, I)`` and not the
executed model, which applies its own scaling and sign convention instead.

Runtime keeps the normal form ``(h_t, x_t; s) -> (h_t+1, y_t)`` with flat,
indexed, vectorized arrays. The hierarchy lives in the specification and in
the realization index map ``I``; the simulation loop never traverses it.

Deterministic minimum rules (the algebra leaves these open; this layer fixes
them so every supported expression has exactly one realization):

1. Bare ``A O B`` / ``A X B`` (no rule, no projection) contribute structure
   only and generate zero edges. Edges come exclusively from explicit
   projections (``>``, ``<``, ``<>``) and rule applications (``O[k]``/``X[k]``).
2. An ordered rule binds its own adjacency (S10): in ``A O[k] B O[j] C``,
   ``k`` reaches ``(A, B)`` and ``j`` reaches ``(B, C)``, never ``(A, C)``.
   Operands compose through frontiers -- an object is its own frontier, and
   a composite ``{A O B}`` exposes ``in(A)`` and ``out(B)`` (S9 derived
   defaults) -- so a rule applied to a composite reaches its frontier rather
   than every member inside it. ``{E}`` boundaries are preserved in TFNE
   paths (``parent.g<i>`` segments, ``g``-ids assigned pre-order by
   creation); ``A O B O C``, ``{A O B} O C`` and ``A O {B O C}`` realize
   identical edge sets with distinct index-map paths.
3. ``A^n`` creates ``n`` indexed instances at ``<scope>.A.1`` ...
   ``<scope>.A.n`` (S6, 1-based); no connectivity is implied. Replication
   applies to a single named reference.
4. Proportion-to-count allocation is largest-remainder with declaration-order
   tiebreak, so ``sum_c N[A.c] == N[A]`` holds exactly and deterministically.
5. Rule applications and bare projections require an explicit ``direction``
   (rule field; ``>``, ``<`` or ``<>``). The compiler raises rather than
   guessing a direction for ``O[k]``/``X[k]``.
6. Exclusions (``A !> B``) subtract from the rule expansion (S14): with
   ``G_0`` the generated projection set, ``G`` is ``G_0`` with the resolved
   exclusion identities removed. An exclusion naming no mechanism removes
   every identity on its route. An exclusion matching no generated
   projection raises ``E_EXCLUSION_UNKNOWN`` rather than passing as a
   no-op, so a stale exclusion cannot survive normalization.
7. ``parse(normalize(p))`` normalizes to ``normalize(p)`` (idempotent replay);
   the realization seed derives from the normalization hash unless overridden.

Name scoping (see ``docs/doctrine/tfne_algebra.md``): bare ``O``/``X`` belong
to the algebra as composition operators and ``H`` names the H-state tensor
exclusively. JaxFNE pipeline stages keep their full names (Emitter, Source,
Field, Probe, Objective, Optimizer, Manifest); runtime state keeps its
meanings (``X`` activity, ``H`` H-state, ``W`` weights, ``B`` delay history,
``K`` kernels, ``Q`` source quantity). Only ``O``, ``X`` and ``H`` are
rejected as structural object names; all other capitals (including the
corpus names ``A``–``D``) are legal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import hashlib
import json

import numpy as np

__all__ = [
    "TFNEError",
    "Program",
    "ExplicitModel",
    "Realization",
    "parse",
    "normalize",
    "resolve",
    "realize",
    "flatten",
    "to_neuronal_tensor",
    "DEFAULT_MODEL",
    "DIRECT_MECHANISM",
]

DEFAULT_MODEL = "izhikevich"
DIRECT_MECHANISM = "tfne_direct"

_PN_TOL = 1e-9


class TFNEError(ValueError):
    """Deterministic TFNE specification/realization failure."""


# --------------------------------------------------------------------------- #
# Reserved names
# --------------------------------------------------------------------------- #

# Single capital letters that cannot name structural objects: O/X are the
# composition operators (a definition under either name could never be
# referenced), and H is the H-state tensor exclusively (no structural H).
# Every other capital — including A, B, C, D used across the algebra and its
# corpus, and JaxFNE runtime/pipeline letters such as E or Q — is a legal
# object name; runtime meanings (X activity, W weights, Q source quantity,
# ...) are preserved by keeping them out of the specification grammar rather
# than by forbidding names. Single-letter names shadowing runtime vocabulary
# are discouraged in documentation but accepted by the compiler.
_RESERVED_CAPITALS = frozenset(["O", "X", "H"])
# Lowercase names owned by the algebra: boundaries and state symbols.
_RESERVED_LOWER = frozenset(["x", "y", "h", "s"])


def _check_object_name(name: str) -> str:
    if not name or not name[0].isupper():
        raise TFNEError(f"object name must be capitalized; got {name!r}")
    if name in _RESERVED_CAPITALS:
        raise TFNEError(
            f"object name {name!r} is reserved (composition operator or "
            "H-state tensor); choose another name"
        )
    return name


def _check_lower_name(name: str, role: str) -> str:
    if not name or not (name[0].islower() or name[0] == "_"):
        raise TFNEError(f"{role} name must be lowercase; got {name!r}")
    return name


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #

_TOKEN_SYMBOLS = (":=", "<>", "!>", "!<", ";", "{", "}", "[", "]", ".", ",",
                  "^", ":", "=", ">", "<")


def _lex(text: str) -> list[tuple[str, str]]:
    """Tokenize TFNE source into (kind, value) pairs.

    Kinds: NAME, INT, FLOAT, SYM, SEP (newline or ';' statement separator).
    """
    toks: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            toks.append(("SEP", "\n"))
            i += 1
            continue
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(("NAME", text[i:j]))
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            dot = False
            while j < n and (text[j].isdigit() or (text[j] == "." and not dot)):
                if text[j] == ".":
                    dot = True
                j += 1
            toks.append(("FLOAT" if dot else "INT", text[i:j]))
            i = j
            continue
        if c == "-" and i + 1 < n and (
            text[i + 1].isdigit()
            or (text[i + 1] == "." and i + 2 < n and text[i + 2].isdigit())
        ):
            j = i + 1
            dot = False
            while j < n and (text[j].isdigit() or (text[j] == "." and not dot)):
                if text[j] == ".":
                    dot = True
                j += 1
            toks.append(("FLOAT" if dot else "INT", text[i:j]))
            i = j
            continue
        matched = False
        for sym in _TOKEN_SYMBOLS:
            if text.startswith(sym, i):
                if sym == ";":
                    toks.append(("SEP", ";"))
                else:
                    toks.append(("SYM", sym))
                i += len(sym)
                matched = True
                break
        if not matched:
            raise TFNEError(f"unexpected character {c!r} at offset {i}")
    return toks


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Ref:
    segments: tuple[str, ...]


@dataclass(frozen=True)
class Select:
    base: tuple[str, ...]
    key: str
    tail: tuple[str, ...] = ()


@dataclass(frozen=True)
class Group:
    body: Any  # Expr


@dataclass(frozen=True)
class Ordered:
    left: Any  # Expr
    right: Any  # Expr
    rule: Optional[str] = None


@dataclass(frozen=True)
class Cross:
    left: Any  # Expr
    right: Any  # Expr
    rule: Optional[str] = None


@dataclass(frozen=True)
class Project:
    direction: str  # '>', '<', '<>'
    left: Any  # Expr (Ref/Select/Group)
    right: Any  # Expr


@dataclass(frozen=True)
class Exclude:
    direction: str  # '!>' or '!<'
    left: Any  # Expr
    right: Any  # Expr


@dataclass(frozen=True)
class Replicate:
    atom: Ref
    n: int


@dataclass(frozen=True)
class ObjDef:
    name: str
    kind: str  # 'expr' | 'props' | 'special'
    body: Any


@dataclass(frozen=True)
class RuleDef:
    name: str
    kind: str  # 'O' | 'X'
    params: Mapping[str, Any]


@dataclass(frozen=True)
class System:
    x: Optional[str]
    x_type: Optional[str]
    body: Any  # Expr
    y: Optional[str]
    y_type: Optional[str]


@dataclass(frozen=True)
class Program:
    defs: Mapping[str, ObjDef]
    rules: Mapping[str, RuleDef]
    system: Optional[System]


# --------------------------------------------------------------------------- #
# Parser (recursive descent)
# --------------------------------------------------------------------------- #

class _Parser:
    def __init__(self, toks: list[tuple[str, str]]):
        self.toks = toks
        self.pos = 0

    def peek(self) -> tuple[str, str]:
        if self.pos < len(self.toks):
            return self.toks[self.pos]
        return ("EOF", "")

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        if tok[0] == "EOF":
            raise TFNEError("unexpected end of input")
        self.pos += 1
        return tok

    def expect(self, kind: str, value: Optional[str] = None) -> str:
        tok = self.next()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise TFNEError(f"expected {kind} {value or ''}; got {tok}")
        return tok[1]

    def at_sym(self, value: str) -> bool:
        tok = self.peek()
        return tok[0] == "SYM" and tok[1] == value

    def at_name(self, value: Optional[str] = None) -> bool:
        tok = self.peek()
        return tok[0] == "NAME" and (value is None or tok[1] == value)

    # -- program -------------------------------------------------------- #
    def parse_program(self) -> Program:
        defs: dict[str, ObjDef] = {}
        rules: dict[str, RuleDef] = {}
        system: Optional[System] = None
        while self.peek()[0] != "EOF":
            while self.peek()[0] == "SEP":
                self.next()
            if self.peek()[0] == "EOF":
                break
            if self._is_ruledef():
                rule = self._parse_ruledef()
                if rule.name in rules:
                    raise TFNEError(f"duplicate rule definition {rule.name!r}")
                rules[rule.name] = rule
            elif self._is_objdef():
                objdef = self._parse_objdef()
                if objdef.name in defs:
                    raise TFNEError(f"duplicate definition {objdef.name!r}")
                defs[objdef.name] = objdef
            else:
                if system is not None:
                    raise TFNEError("only one system expression per program")
                system = self._parse_system()
            while self.peek()[0] == "SEP":
                self.next()
        return Program(defs=defs, rules=rules, system=system)

    def _is_ruledef(self) -> bool:
        # ("O"|"X") "[" NAME "]" ":="
        t = self.toks
        p = self.pos
        return (
            p + 4 < len(t)
            and t[p][0] == "NAME" and t[p][1] in ("O", "X")
            and t[p + 1] == ("SYM", "[")
            and t[p + 2][0] == "NAME"
            and t[p + 3] == ("SYM", "]")
            and t[p + 4] == ("SYM", ":=")
        )

    def _is_objdef(self) -> bool:
        t = self.toks
        p = self.pos
        return (
            p + 1 < len(t)
            and t[p][0] == "NAME"
            and t[p + 1] == ("SYM", ":=")
        )

    def _parse_ruledef(self) -> RuleDef:
        kind = self.expect("NAME")
        if kind not in ("O", "X"):
            raise TFNEError(f"rule kind must be O or X; got {kind!r}")
        self.expect("SYM", "[")
        name = self.expect("NAME")
        _check_lower_name(name, "rule")
        self.expect("SYM", "]")
        self.expect("SYM", ":=")
        self.expect("SYM", "[")
        params = self._parse_propbody()
        self.expect("SYM", "]")
        return RuleDef(name=name, kind=kind, params=params)

    def _parse_objdef(self) -> ObjDef:
        name = self.expect("NAME")
        _check_object_name(name)
        self.expect("SYM", ":=")
        if self.at_sym("["):
            self.next()
            props = self._parse_propbody()
            self.expect("SYM", "]")
            return ObjDef(name=name, kind="props", body=props)
        # special := Base[key]  vs  expr
        if self.peek()[0] == "NAME":
            save = self.pos
            try:
                atom = self._parse_atom()
            except TFNEError:
                atom = None
            if (
                isinstance(atom, Select)
                and not atom.tail
                and self.peek()[0] in ("EOF", "SEP")
            ):
                base = ".".join(atom.base)
                return ObjDef(name=name, kind="special",
                              body={"base": base, "key": atom.key})
            self.pos = save
        body = self._parse_expr()
        return ObjDef(name=name, kind="expr", body=body)

    def _parse_system(self) -> System:
        x = x_type = None
        # left boundary: lowercase NAME [ "[" type "]" ] ":" (not ":=...")
        if self.peek()[0] == "NAME":
            t = self.toks
            p = self.pos
            q = p + 1
            btype: Optional[str] = None
            if (q + 2 < len(t) and t[q] == ("SYM", "[")
                    and t[q + 1][0] == "NAME" and t[q + 2] == ("SYM", "]")):
                btype = t[q + 1][1]
                q += 3
            if q < len(t) and t[q] == ("SYM", ":"):
                x = self.expect("NAME")
                _check_lower_name(x, "boundary")
                if btype is not None:
                    self.expect("SYM", "[")
                    x_type = self.expect("NAME")
                    self.expect("SYM", "]")
                self.expect("SYM", ":")
        body = self._parse_expr()
        y = y_type = None
        if self.at_sym(":"):
            self.next()
            if self.peek()[0] != "NAME":
                raise TFNEError("expected boundary name after ':'")
            y = self.expect("NAME")
            _check_lower_name(y, "boundary")
            if self.at_sym("["):
                self.next()
                y_type = self.expect("NAME")
                self.expect("SYM", "]")
        return System(x=x, x_type=x_type, body=body, y=y, y_type=y_type)

    # -- expressions ---------------------------------------------------- #
    def _parse_expr(self) -> Any:
        left = self._parse_proj()
        while self.at_name("O") or self.at_name("X"):
            op = self.next()[1]
            rule = None
            if self.at_sym("["):
                self.next()
                rule = self.expect("NAME")
                _check_lower_name(rule, "rule reference")
                self.expect("SYM", "]")
            right = self._parse_proj()
            if op == "O":
                left = Ordered(left=left, right=right, rule=rule)
            else:
                left = Cross(left=left, right=right, rule=rule)
        return left

    def _parse_proj(self) -> Any:
        left = self._parse_unary()
        while True:
            tok = self.peek()
            if tok[0] == "SYM" and tok[1] in (">", "<", "<>", "!>", "!<"):
                self.next()
                right = self._parse_unary()
                if tok[1] in ("!>", "!<"):
                    left = Exclude(direction=tok[1], left=left, right=right)
                else:
                    left = Project(direction=tok[1], left=left, right=right)
            else:
                return left

    def _parse_unary(self) -> Any:
        atom = self._parse_atom()
        if self.at_sym("^"):
            self.next()
            tok = self.next()
            if tok[0] != "INT":
                raise TFNEError("replication count must be an integer")
            n = int(tok[1])
            if n < 1:
                raise TFNEError("replication count must be >= 1")
            if not isinstance(atom, Ref) or len(atom.segments) != 1:
                raise TFNEError(
                    "replication (A^n) applies to a single named reference"
                )
            return Replicate(atom=atom, n=n)
        return atom

    def _parse_atom(self) -> Any:
        tok = self.peek()
        if tok == ("SYM", "{"):
            self.next()
            body = self._parse_expr()
            self.expect("SYM", "}")
            return Group(body=body)
        if tok == ("SYM", "("):
            self.next()
            body = self._parse_expr()
            self.expect("SYM", ")")
            return body
        if tok[0] == "NAME":
            if tok[1] in ("O", "X"):
                raise TFNEError(
                    f"unexpected {tok[1]!r} here (reserved composition operator)"
                )
            name = self.next()[1]
            base = [name]
            while self.at_sym("."):
                self.next()
                base.append(self.expect("NAME"))
            # selects: Base[key] with optional tail
            key: Optional[str] = None
            tail: list[str] = []
            while self.at_sym("["):
                if key is not None:
                    raise TFNEError("only one [key] selection per reference")
                self.next()
                key = self.expect("NAME")
                self.expect("SYM", "]")
                while self.at_sym("."):
                    self.next()
                    tail.append(self.expect("NAME"))
            if key is None:
                return Ref(segments=tuple(base))
            return Select(base=tuple(base), key=key, tail=tuple(tail))
        raise TFNEError(f"unexpected token {tok} in expression")

    # -- property bodies ------------------------------------------------ #
    def _parse_propbody(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        while True:
            while self.peek()[0] == "SEP":
                self.next()
            if self.peek()[0] != "NAME":
                break
            # lookahead: NAME "=" (not "=="; '=' sym distinct from ':=') 
            t = self.toks
            p = self.pos
            if not (p + 1 < len(t) and t[p + 1] == ("SYM", "=")):
                break
            key = self.expect("NAME")
            self.expect("SYM", "=")
            props[key] = self._parse_value()
            while self.peek()[0] == "SEP":
                self.next()
            if self.at_sym(","):
                self.next()
                continue
        return props

    def _parse_value(self) -> Any:
        tok = self.peek()
        if tok[0] == "SYM" and tok[1] in (">", "<", "<>"):
            self.next()
            return tok[1]
        if tok[0] in ("INT", "FLOAT"):
            self.next()
            return float(tok[1]) if tok[0] == "FLOAT" else int(tok[1])
        if tok[0] == "NAME":
            # dotted name path value
            parts = [self.next()[1]]
            while self.at_sym("."):
                self.next()
                parts.append(self.expect("NAME"))
            if len(parts) == 1:
                return parts[0]
            return {"path": parts}
        if tok == ("SYM", "{"):
            self.next()
            # set {A, B} or map {A: 1, B: 2}
            first = self.expect("NAME")
            if self.at_sym(":"):
                self.next()
                mapping = {first: self._expect_number()}
                while self.at_sym(","):
                    self.next()
                    k = self.expect("NAME")
                    self.expect("SYM", ":")
                    mapping[k] = self._expect_number()
                self.expect("SYM", "}")
                return {"map": mapping}
            items = [first]
            while self.at_sym(","):
                self.next()
                items.append(self.expect("NAME"))
            self.expect("SYM", "}")
            return {"set": items}
        if tok == ("SYM", "["):
            self.next()
            nested = self._parse_propbody()
            self.expect("SYM", "]")
            return {"dict": nested}
        raise TFNEError(f"unexpected value token {tok}")

    def _expect_number(self) -> Any:
        tok = self.next()
        if tok[0] == "INT":
            return int(tok[1])
        if tok[0] == "FLOAT":
            return float(tok[1])
        raise TFNEError(f"expected number; got {tok}")


def parse(text: str) -> Program:
    """Parse TFNE source text into a :class:`Program`."""
    parser = _Parser(_lex(text))
    program = parser.parse_program()
    if parser.peek()[0] != "EOF":
        raise TFNEError(f"trailing input at {parser.peek()}")
    return program


# --------------------------------------------------------------------------- #
# Normalization (canonical replayable form)
# --------------------------------------------------------------------------- #

def _emit_expr(node: Any) -> str:
    if isinstance(node, Ref):
        return ".".join(node.segments)
    if isinstance(node, Select):
        s = ".".join(node.base) + f"[{node.key}]"
        if node.tail:
            s += "." + ".".join(node.tail)
        return s
    if isinstance(node, Group):
        return "{" + _emit_expr(node.body) + "}"
    if isinstance(node, (Ordered, Cross)):
        op = "O" if isinstance(node, Ordered) else "X"
        if node.rule is not None:
            op += f"[{node.rule}]"
        return f"{_emit_expr(node.left)} {op} {_emit_expr(node.right)}"
    if isinstance(node, (Project, Exclude)):
        return f"{_emit_expr(node.left)} {node.direction} {_emit_expr(node.right)}"
    if isinstance(node, Replicate):
        return f"{_emit_expr(node.atom)}^{node.n}"
    raise TFNEError(f"cannot emit {node!r}")


def _emit_value(value: Any) -> str:
    if isinstance(value, bool):
        raise TFNEError("boolean property values are not supported")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if "set" in value:
            return "{" + ", ".join(value["set"]) + "}"
        if "map" in value:
            inner = ", ".join(f"{k}: {v!r}" if isinstance(v, float)
                              else f"{k}: {v}"
                              for k, v in value["map"].items())
            return "{" + inner + "}"
        if "dict" in value:
            return "[" + _emit_propbody(value["dict"]) + "]"
        if "path" in value:
            return ".".join(value["path"])
    raise TFNEError(f"cannot emit value {value!r}")


def _emit_propbody(props: Mapping[str, Any]) -> str:
    return "; ".join(f"{k} = {_emit_value(v)}" for k, v in props.items())


def normalize(program: Program) -> str:
    """Render the deterministic canonical form of a program.

    Definitions and rules are sorted by name; the system body keeps its
    structure (order is semantic for ``O``/``X`` chains). The result
    re-parses to an identical program (replay property).
    """
    parts: list[str] = []
    for name in sorted(program.rules):
        rule = program.rules[name]
        parts.append(f"{rule.kind}[{name}] := [{_emit_propbody(rule.params)}]")
    for name in sorted(program.defs):
        objdef = program.defs[name]
        if objdef.kind == "props":
            parts.append(f"{name} := [{_emit_propbody(objdef.body)}]")
        elif objdef.kind == "special":
            parts.append(f"{name} := {objdef.body['base']}[{objdef.body['key']}]")
        else:
            parts.append(f"{name} := {_emit_expr(objdef.body)}")
    if program.system is not None:
        sys = program.system
        head = ""
        if sys.x is not None:
            head = sys.x
            if sys.x_type is not None:
                head += f"[{sys.x_type}]"
            head += " : "
        tail = ""
        if sys.y is not None:
            tail = " : " + sys.y
            if sys.y_type is not None:
                tail += f"[{sys.y_type}]"
        parts.append(f"{head}{_emit_expr(sys.body)}{tail}")
    return "; ".join(parts)


def spec_hash(program: Program) -> str:
    """Deterministic sha256 of the canonical normalization."""
    return hashlib.sha256(normalize(program).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Explicit model (resolved hierarchy)
# --------------------------------------------------------------------------- #

@dataclass
class NodeRecord:
    path: str
    name: str
    kind: str  # 'object' | 'composite'
    parent: Optional[str]
    children: list[str] = field(default_factory=list)
    # object payload (kind == 'object')
    cell_types: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    proportions: Mapping[str, float] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    model: str = DEFAULT_MODEL
    special: Optional[str] = None
    declared_default: bool = False


@dataclass
class RelationRecord:
    key: str
    form: str  # 'rule' | 'projection'
    kind: str  # 'O' | 'X' | 'direct'
    rule: Optional[str]
    direction: str  # '>' | '<' ('<>' splits into two records sharing a group)
    pre_label: str
    post_label: str
    pre_scopes: tuple[str, ...]
    post_scopes: tuple[str, ...]
    group: Optional[str] = None  # shared stem for '<>' halves


@dataclass
class ExplicitModel:
    nodes: Mapping[str, NodeRecord]
    order: tuple[str, ...]  # pre-order creation order
    relations: tuple[RelationRecord, ...]
    exclusions: tuple[RelationRecord, ...]
    boundaries: Mapping[str, Any]
    normalization: str
    digest: str
    rule_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _allocate_counts(cell_types: Sequence[str],
                     proportions: Optional[Mapping[str, float]],
                     total: Optional[Any],
                     where: str) -> tuple[dict[str, int], dict[str, float]]:
    """Deterministic largest-remainder allocation with declaration-order tiebreak."""
    ctypes = list(cell_types)
    if proportions is not None:
        for c, p in proportions.items():
            if c not in ctypes:
                raise TFNEError(f"{where}: proportion member {c!r} not in C")
            if not (0.0 <= float(p) <= 1.0):
                raise TFNEError(f"{where}: proportion {c}={p} outside [0,1]")
        if abs(sum(float(p) for p in proportions.values()) - 1.0) > 1e-6:
            raise TFNEError(
                f"{where}: proportions sum to "
                f"{sum(float(p) for p in proportions.values())}, not 1"
            )
    if isinstance(total, Mapping):
        counts = {c: int(total[c]) for c in total}
        for c in counts:
            if c not in ctypes:
                raise TFNEError(f"{where}: count member {c!r} not in C")
        missing = [c for c in ctypes if c not in counts]
        if missing:
            raise TFNEError(f"{where}: N map missing members {missing}")
        n_total = sum(counts.values())
        if proportions is not None:
            for c in ctypes:
                if abs(counts[c] - proportions[c] * n_total) >= 1.0:
                    raise TFNEError(
                        f"{where}: N map inconsistent with P for {c!r}")
        derived = {c: counts[c] / n_total for c in ctypes} if n_total else {
            c: 0.0 for c in ctypes}
        return counts, derived
    n_total = 1 if total is None else int(total)
    if n_total < 1:
        raise TFNEError(f"{where}: N must be >= 1")
    if proportions is None:
        if len(ctypes) == 1:
            return {ctypes[0]: n_total}, {ctypes[0]: 1.0}
        raise TFNEError(f"{where}: P required when C has members {ctypes}")
    exact = [float(proportions[c]) * n_total for c in ctypes]
    base = [int(np.floor(v)) for v in exact]
    remainder = n_total - sum(base)
    # fractional parts, tie-broken by declaration order (stable argsort)
    order = sorted(range(len(ctypes)),
                   key=lambda i: (-(exact[i] - base[i]), i))
    for i in order[:remainder]:
        base[i] += 1
    counts = {c: base[i] for i, c in enumerate(ctypes)}
    assert sum(counts.values()) == n_total
    return counts, {c: float(proportions[c]) for c in ctypes}


@dataclass(frozen=True)
class _Expansion:
    """What expanding one expression contributes to its parent.

    ``members`` are the structural paths handed upward. ``fin``/``fout`` are
    the composition frontiers an ordered rule binds against: for ``{A O B}``
    they are ``in(A)`` and ``out(B)`` (S9 derived defaults), so a rule applied
    to the composite reaches its frontier rather than everything it contains.
    Carrying them separately is what keeps S10 adjacency exact -- the members
    accumulated by a chain are no longer reachable as a rule endpoint.
    """

    members: tuple[str, ...]
    fin: tuple[str, ...]
    fout: tuple[str, ...]


def _leaf_expansion(path: str) -> _Expansion:
    """An object is its own frontier on both sides."""
    return _Expansion(members=(path,), fin=(path,), fout=(path,))


def _flat_expansion(members: Sequence[str]) -> _Expansion:
    """A set of members with no ordering: the whole set is both frontiers."""
    m = tuple(members)
    return _Expansion(members=m, fin=m, fout=m)


class _Resolver:
    def __init__(self, program: Program):
        self.program = program
        self.nodes: dict[str, NodeRecord] = {}
        self.order: list[str] = []
        self.relations: list[RelationRecord] = []
        self.exclusions: list[RelationRecord] = []
        self.group_counter = 0
        self.relation_counter = 0

    # -- node construction -------------------------------------------- #
    def _add_node(self, node: NodeRecord) -> str:
        if node.path in self.nodes:
            raise TFNEError(f"duplicate TFNE path {node.path!r}")
        self.nodes[node.path] = node
        self.order.append(node.path)
        if node.parent is not None and node.parent in self.nodes:
            self.nodes[node.parent].children.append(node.path)
        return node.path

    def _object_props(self, props: Mapping[str, Any], where: str,
                      implicit: bool = False) -> NodeRecord:
        raw_c = props.get("C", {"set": ["cell"]})
        if isinstance(raw_c, Mapping) and "set" in raw_c:
            ctypes = tuple(raw_c["set"])
        else:
            raise TFNEError(f"{where}: C must be a {{type, ...}} set")
        # Cell-type members use the algebra's own capitalized convention
        # (C:={E,PV,SST,VIP}); they live in the selection namespace bound by
        # L[C], not the structural-object namespace, so the reserved-letter
        # rule for object names does not apply to them.
        raw_p = props.get("P")
        proportions = None
        if raw_p is not None:
            if not (isinstance(raw_p, Mapping) and "map" in raw_p):
                raise TFNEError(f"{where}: P must be a {{type: p, ...}} map")
            proportions = {k: float(v) for k, v in raw_p["map"].items()}
        raw_n = props.get("N")
        total: Any = None
        if raw_n is not None:
            if isinstance(raw_n, Mapping) and "map" in raw_n:
                total = {k: int(v) for k, v in raw_n["map"].items()}
            elif isinstance(raw_n, (int, float)):
                total = int(raw_n)
            else:
                raise TFNEError(f"{where}: N must be an integer or map")
        counts, derived = _allocate_counts(ctypes, proportions, total, where)
        raw_g = props.get("G")
        geometry: dict[str, Any] = {}
        if raw_g is not None:
            if not (isinstance(raw_g, Mapping) and "dict" in raw_g):
                raise TFNEError(f"{where}: G must be a [k = v; ...] body")
            geometry = dict(raw_g["dict"])
        model = props.get("model", DEFAULT_MODEL)
        if isinstance(model, Mapping):
            raise TFNEError(f"{where}: model must be a name")
        model = str(model)
        return NodeRecord(path="", name="", kind="object", parent=None,
                          cell_types=ctypes, counts=counts,
                          proportions=derived, geometry=geometry, model=model,
                          declared_default=implicit)

    # -- structure expansion ------------------------------------------- #
    def expand(self, node: Any, scope: str) -> _Expansion:
        """Expand an expression under ``scope``.

        Returns the structural members plus the frontiers that an ordered
        composition binds against (S9 derived defaults, S10 adjacency).
        """
        if isinstance(node, Ref):
            return self._expand_ref(node.segments, scope)
        if isinstance(node, Select):
            return _leaf_expansion(self._resolve_select(node, scope))
        if isinstance(node, Group):
            gid = f"g{self.group_counter}"
            self.group_counter += 1
            path = f"{scope}.{gid}" if scope else gid
            self._add_node(NodeRecord(path=path, name=gid, kind="composite",
                                      parent=scope or None))
            inner = self.expand(node.body, path)
            if not inner.members:
                raise TFNEError(f"empty composite group at {path!r}")
            # S8: the brace is a structural boundary. The composite hands
            # itself upward as one member but composes through its body's
            # frontier, so it is not flattened into the surrounding chain.
            return _Expansion(members=(path,), fin=inner.fin, fout=inner.fout)
        if isinstance(node, Ordered):
            left = self.expand(node.left, scope)
            right = self.expand(node.right, scope)
            if node.rule is not None:
                # S10: a rule binds its own adjacency only -- the left
                # operand's out frontier to the right operand's in frontier.
                # Operands accumulated earlier in the chain are not reachable
                # here, so A O[k] B O[j] C yields k(A,B) and j(B,C), never A>C.
                self._record_rule(node, list(left.fout), list(right.fin))
            return _Expansion(members=left.members + right.members,
                              fin=left.fin, fout=right.fout)
        if isinstance(node, Cross):
            left = self.expand(node.left, scope)
            right = self.expand(node.right, scope)
            if node.rule is not None:
                # X is nonordered: it relates its operands rather than an
                # adjacency, and binds them whole. Unchanged here; per-rule
                # endpoint selection is $L/$R (TFNE2-05).
                self._record_rule(node, list(left.members),
                                  list(right.members))
            return _Expansion(members=left.members + right.members,
                              fin=left.fin + right.fin,
                              fout=left.fout + right.fout)
        if isinstance(node, Project):
            left = self._endpoint_members(node.left, scope, "projection")
            right = self._endpoint_members(node.right, scope, "projection")
            group = None
            if node.direction == "<>":
                group = (f"projection:direct[-]:{_emit_expr(node.left)}<>"
                         f"{_emit_expr(node.right)}")
            for direction in self._split_direction(node.direction):
                self._record_relation("projection",
                                      "direct", None, direction,
                                      _emit_expr(node.left),
                                      _emit_expr(node.right), left, right,
                                      group=group)
            return _Expansion(members=tuple(left) + tuple(right),
                              fin=tuple(left), fout=tuple(right))
        if isinstance(node, Exclude):
            left = self._endpoint_members(node.left, scope, "exclusion")
            right = self._endpoint_members(node.right, scope, "exclusion")
            rec = self._record_relation(
                "exclusion", "direct", None, node.direction,
                _emit_expr(node.left), _emit_expr(node.right), left, right)
            self.exclusions.append(rec)
            return _Expansion(members=tuple(left) + tuple(right),
                              fin=tuple(left), fout=tuple(right))
        if isinstance(node, Replicate):
            return self._expand_replicate(node, scope)
        raise TFNEError(f"cannot expand {node!r}")

    def _expand_ref(self, segments: Sequence[str],
                    scope: str) -> _Expansion:
        if len(segments) != 1:
            raise TFNEError(
                "dotted path {0!r} is addressable in projections only, not as "
                "a structural member".format(".".join(segments))
            )
        name = segments[0]
        if name in ("O", "X"):
            raise TFNEError(f"{name!r} is a reserved operator")
        target = f"{scope}.{name}" if scope else name
        if name in self.program.defs:
            objdef = self.program.defs[name]
            if objdef.kind == "props":
                rec = self._object_props(objdef.body, where=name)
                rec.path = target
                rec.name = name
                rec.parent = scope or None
                self._add_node(rec)
                return _leaf_expansion(target)
            if objdef.kind == "special":
                base = objdef.body["base"]
                if base not in self.program.defs:
                    raise TFNEError(
                        f"{name}: specialization base {base!r} undefined")
                basedef = self.program.defs[base]
                if basedef.kind != "props":
                    raise TFNEError(
                        f"{name}: specialization base {base!r} must declare "
                        "properties")
                rec = self._object_props(basedef.body, where=name)
                rec.path = target
                rec.name = name
                rec.parent = scope or None
                rec.special = f"{base}[{objdef.body['key']}]"
                self._add_node(rec)
                return _leaf_expansion(target)
            # transparent expression definition: expand under own namespace
            composite = NodeRecord(path=target, name=name, kind="composite",
                                   parent=scope or None)
            self._add_node(composite)
            inner = self.expand(objdef.body, target)
            if not inner.members:
                raise TFNEError(f"definition {name!r} expands to nothing")
            # A named definition is a structural boundary like a brace: it
            # composes through its body's frontier, not every member.
            return _Expansion(members=(target,), fin=inner.fin,
                              fout=inner.fout)
        # implicit degenerate leaf (valid at cardinality one)
        _check_object_name(name)
        rec = self._object_props({}, where=name, implicit=True)
        rec.path = target
        rec.name = name
        rec.parent = scope or None
        self._add_node(rec)
        return _leaf_expansion(target)

    def _expand_replicate(self, node: Replicate,
                          scope: str) -> _Expansion:
        name = node.atom.segments[0]
        paths: list[str] = []
        # S6: A^n = {A.1 ... A.n}. Instance indices are 1-based; the path
        # A.1 names the first instance, not the second.
        for i in range(1, node.n + 1):
            stem = f"{scope}.{name}" if scope else name
            inst = f"{stem}.{i}"
            if name in self.program.defs:
                objdef = self.program.defs[name]
                if objdef.kind == "props":
                    rec = self._object_props(objdef.body, where=name)
                    rec.path = inst
                    rec.name = name
                    rec.parent = scope or None
                    self._add_node(rec)
                elif objdef.kind == "special":
                    base = objdef.body["base"]
                    basedef = self.program.defs[base]
                    rec = self._object_props(basedef.body, where=name)
                    rec.path = inst
                    rec.name = name
                    rec.parent = scope or None
                    rec.special = f"{base}[{objdef.body['key']}]"
                    self._add_node(rec)
                else:
                    comp = NodeRecord(path=inst, name=name, kind="composite",
                                      parent=scope or None)
                    self._add_node(comp)
                    self.expand(objdef.body, inst)
            else:
                _check_object_name(name)
                rec = self._object_props({}, where=name, implicit=True)
                rec.path = inst
                rec.name = name
                rec.parent = scope or None
                self._add_node(rec)
            paths.append(inst)
        # S6: instances carry no connectivity and no order, so the whole set
        # is both frontiers. Replication behaviour is otherwise unchanged.
        return _flat_expansion(paths)

    def _endpoint_members(self, node: Any, scope: str,
                          role: str) -> list[str]:
        members = self.expand_endpoint(node, scope)
        if not members:
            raise TFNEError(f"{role} endpoint expands to nothing")
        return members

    def expand_endpoint(self, node: Any, scope: str) -> list[str]:
        if isinstance(node, (Ref, Select)):
            label = (_emit_expr(node))
            found = self._find_paths(node, scope)
            if found:
                return found
            # A defined head that was never expanded as structure (e.g. a
            # projection-only program `x : V1.L4[E] > ... : y`): expand the
            # definition at the use site, then resolve again.
            head = (node.segments[0] if isinstance(node, Ref)
                    else node.base[0])
            multiseg = (isinstance(node, Ref) and len(node.segments) > 1) \
                or isinstance(node, Select)
            if multiseg and head in self.program.defs:
                target = f"{scope}.{head}" if scope else head
                if target not in self.nodes:
                    self._expand_ref((head,), scope)
                    found = self._find_paths(node, scope)
                    if found:
                        return found
            # single undefined name: create implicit member at use site
            # (single defined name: expand the definition via _expand_ref)
            if isinstance(node, Ref) and len(node.segments) == 1:
                return list(self._expand_ref(node.segments, scope).members)
            raise TFNEError(
                f"projection endpoint {label!r} matches no realized object")
        if isinstance(node, Group):
            return list(self.expand(node, scope).members)
        raise TFNEError(
            "projection endpoints must be references, selections or groups")

    def _find_paths(self, node: Any, scope: str) -> list[str]:
        """Resolve an endpoint to existing node paths or subtree scopes."""
        if isinstance(node, Select):
            base_path = self._resolve_base(node.base, scope)
            if base_path is None:
                return []
            key = node.key
            rec = self.nodes.get(base_path)
            if rec is None or rec.kind != "object":
                return []
            if key not in rec.counts:
                raise TFNEError(
                    f"selection {key!r} not a member of {base_path!r}")
            path = base_path + "." + key
            if node.tail:
                path += "." + ".".join(node.tail)
                # tails address virtual sub-paths; accept if head resolves
            return [path]
        assert isinstance(node, Ref)
        absolute = ".".join(node.segments)
        if absolute in self.nodes or self._is_member_path(absolute):
            return [absolute]
        if scope:
            qualified = scope + "." + absolute
            if qualified in self.nodes or self._is_member_path(qualified):
                return [qualified]
        # prefix scope: all nodes under it
        hits = [p for p in self.order
                if p == absolute or p.startswith(absolute + ".")]
        if hits:
            return hits
        if scope:
            qualified = scope + "." + absolute
            hits = [p for p in self.order
                    if p == qualified or p.startswith(qualified + ".")]
            if hits:
                return hits
        return []

    def _resolve_base(self, base: tuple[str, ...], scope: str) -> Optional[str]:
        absolute = ".".join(base)
        if absolute in self.nodes:
            return absolute
        if scope and (scope + "." + absolute) in self.nodes:
            return scope + "." + absolute
        return None

    def _resolve_select(self, node: Select, scope: str) -> str:
        found = self._find_paths(node, scope)
        if not found:
            raise TFNEError(
                f"selection {_emit_expr(node)!r} matches no realized object")
        return found[0]

    def _is_member_path(self, path: str) -> bool:
        if "." not in path:
            return False
        head, _, member = path.rpartition(".")
        rec = self.nodes.get(head)
        return rec is not None and member in rec.counts

    # -- relations ------------------------------------------------------ #
    def _record_rule(self, node: Any, left: list[str],
                     right: list[str]) -> RelationRecord:
        assert isinstance(node, (Ordered, Cross))
        assert node.rule is not None
        if node.rule not in self.program.rules:
            raise TFNEError(f"connection rule {node.rule!r} undefined")
        ruledef = self.program.rules[node.rule]
        direction = ruledef.params.get("direction")
        if direction not in (">", "<", "<>"):
            raise TFNEError(
                f"rule {node.rule!r} must declare direction ('>', '<', '<>'); "
                "the algebra fixes no default direction for O[k]/X[k]")
        kind = "O" if isinstance(node, Ordered) else "X"
        group = None
        if direction == "<>":
            group = (f"rule:{kind}[{node.rule}]:{_emit_expr(node.left)}<>"
                     f"{_emit_expr(node.right)}")
        for split in self._split_direction(direction):
            self._record_relation("rule", kind, node.rule, split,
                                  _emit_expr(node.left),
                                  _emit_expr(node.right), left, right,
                                  group=group)
        return self.relations[-1]

    @staticmethod
    def _split_direction(direction: str) -> tuple[str, ...]:
        # A<>B is bidirectional: two directed records sharing one group stem.
        if direction == "<>":
            return (">", "<")
        return (direction,)

    def _record_relation(self, form: str, kind: str, rule: Optional[str],
                         direction: str, pre_label: str, post_label: str,
                         left: list[str], right: list[str],
                         group: Optional[str] = None) -> RelationRecord:
        i = self.relation_counter
        self.relation_counter += 1
        rname = rule if rule is not None else "-"
        key = (f"{form[0]}{i}:{kind}[{rname}]:"
               f"{pre_label}{direction}{post_label}")
        rec = RelationRecord(key=key, form=form, kind=kind, rule=rule,
                             direction=direction, pre_label=pre_label,
                             post_label=post_label,
                             pre_scopes=tuple(left),
                             post_scopes=tuple(right),
                             group=group)
        if form != "exclusion":
            self.relations.append(rec)
        return rec


def resolve(program: Program) -> ExplicitModel:
    """Expand definitions, replication, groups and rules to an explicit model."""
    resolver = _Resolver(program)
    if program.system is not None:
        resolver.expand(program.system.body, "")
    boundaries: dict[str, Any] = {}
    if program.system is not None:
        sys = program.system
        boundaries = {"x": sys.x, "x_type": sys.x_type,
                      "y": sys.y, "y_type": sys.y_type}
    normalization = normalize(program)
    return ExplicitModel(nodes=dict(resolver.nodes),
                         order=tuple(resolver.order),
                         relations=tuple(resolver.relations),
                         exclusions=tuple(resolver.exclusions),
                         boundaries=boundaries,
                         normalization=normalization,
                         digest=spec_hash(program),
                         rule_params={k: dict(v.params)
                                      for k, v in program.rules.items()})


# --------------------------------------------------------------------------- #
# Realization (s, h0, I)
# --------------------------------------------------------------------------- #

@dataclass
class Realization:
    """Flattened execution structures plus the realization index map.

    ``s`` holds everything fixed after realization (topology, initial
    parameters, geometry, declarations). ``h0`` holds the initial mutable
    state with execution shapes. ``I`` is the index map preserving TFNE
    identity through flattening.
    """
    s: Mapping[str, Any]
    h0: Mapping[str, Any]
    I: Mapping[str, Any]  # noqa: E741 - algebra symbol for the index map
    explicit: ExplicitModel

    # -- inspection --------------------------------------------------- #
    def path_to_slice(self, path: str) -> tuple[int, int]:
        table = self.I["object_slices"]
        if path in table:
            return table[path]
        member_table = self.I["member_slices"]
        if path in member_table:
            return member_table[path]
        raise TFNEError(f"TFNE path {path!r} has no realized slice")

    def slice_to_path(self, neuron_id: int) -> str:
        paths = self.I["neuron_paths"]
        if not 0 <= int(neuron_id) < len(paths):
            raise TFNEError(f"neuron id {neuron_id} out of range")
        return paths[int(neuron_id)]

    def neuron_ids_in_scope(self, prefix: str) -> list[int]:
        return [i for i, p in enumerate(self.I["neuron_paths"])
                if p == prefix or p.startswith(prefix + ".")]

    def rule_to_edges(self, key: str) -> tuple[int, int]:
        table = self.I["rule_slices"]
        if key not in table:
            raise TFNEError(f"relation {key!r} realized no edges")
        return table[key]

    def edge_to_rule(self, edge_idx: int) -> str:
        origin = self.I["edge_rule"]
        if not 0 <= int(edge_idx) < len(origin):
            raise TFNEError(f"edge index {edge_idx} out of range")
        return origin[int(edge_idx)]

    def relation_origin(self, key: str) -> Mapping[str, Any]:
        origins = self.I["rule_origins"]
        if key not in origins:
            raise TFNEError(f"relation {key!r} unknown")
        return origins[key]

    def relation_group(self, key: str) -> list[str]:
        """All directed relation keys sharing one '<>' group stem (or [key])."""
        group = self.relation_origin(key).get("group")
        if not group:
            return [key]
        return sorted(k for k, o in self.I["rule_origins"].items()
                      if o.get("group") == group)


def _rule_connection_params(params_in: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = dict(params_in) if params_in is not None else {}
    direction = params.get("direction", ">")
    if direction not in (">", "<", "<>"):
        raise TFNEError(f"invalid direction {direction!r}")
    mechanism = params.get("mechanism", DIRECT_MECHANISM)
    if isinstance(mechanism, Mapping):
        raise TFNEError("mechanism must be a name")
    probability = float(params.get("probability", 1.0))
    if not 0.0 <= probability <= 1.0:
        raise TFNEError(f"probability {probability} outside [0,1]")
    weight = params.get("weight", 1.0)
    if isinstance(weight, Mapping):
        raise TFNEError("weight maps are not supported; use a scalar")
    weight = float(weight)
    if not np.isfinite(weight):
        raise TFNEError("rule weight must be finite")
    allow_self = bool(params.get("allow_self", False))
    return {"direction": direction, "mechanism": str(mechanism),
            "probability": probability, "weight": weight,
            "allow_self": allow_self,
            "delay": params.get("delay"),
            "plasticity": params.get("plasticity")}


def realize(explicit: ExplicitModel, program: Optional[Program] = None,
            seed: Optional[int] = None) -> Realization:
    """Realize an explicit model to ``(s, h0, I)`` flat structures.

    Edges compile through :func:`jaxfne.connectivity.compile_connection_rules`
    (host-side, deterministic). ``seed`` defaults to the leading 31 bits of
    the normalization digest.
    """
    from .connectivity import compile_connection_rules

    # -- neuron assignment (pre-order, declaration order within leaves) -- #
    neuron_paths: list[str] = []
    object_slices: dict[str, tuple[int, int]] = {}
    member_slices: dict[str, tuple[int, int]] = {}
    for path in explicit.order:
        rec = explicit.nodes[path]
        if rec.kind != "object":
            continue
        start = len(neuron_paths)
        for ctype in rec.cell_types:
            mstart = len(neuron_paths)
            for _ in range(int(rec.counts.get(ctype, 0))):
                neuron_paths.append(path)
            member_slices[f"{path}.{ctype}"] = (mstart, len(neuron_paths))
        object_slices[path] = (start, len(neuron_paths))
    n_neurons = len(neuron_paths)

    def scope_ids(scopes: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for scope in scopes:
            if scope in object_slices:
                s0, s1 = object_slices[scope]
                ids.extend(range(s0, s1))
                continue
            if scope in member_slices:
                s0, s1 = member_slices[scope]
                ids.extend(range(s0, s1))
                continue
            # subtree scope (composites, groups, replicas stems)
            hits = [i for i, p in enumerate(neuron_paths)
                    if p == scope or p.startswith(scope + ".")]
            if not hits:
                raise TFNEError(
                    f"relation scope {scope!r} matches no realized neurons")
            ids.extend(hits)
        # deterministic order, deduplicated
        return sorted(set(ids))

    # -- connection rules ---------------------------------------------- #
    if seed is None:
        seed = int(explicit.digest[:8], 16) % (2 ** 31)
    rows = []
    for nid, path in enumerate(neuron_paths):
        segments = path.split(".")
        rows.append({
            "neuron_id": nid,
            "area": segments[0] if segments else path,
            "layer": segments[1] if len(segments) > 1 else segments[0],
            "cell_type": _member_of(explicit, path, nid, object_slices),
            "tfne_path": path,
            "object": path,
            "x": 0.0, "y": 0.0, "z": 0.0,
        })
    connections: list[dict[str, Any]] = []
    mechanisms: list[dict[str, Any]] = []
    mech_index: dict[str, int] = {}
    rule_lookup: dict[str, Any] = dict(explicit.rule_params)
    if program is not None:
        for k, v in program.rules.items():
            rule_lookup.setdefault(k, dict(v.params))

    def mech_id(name: str) -> int:
        if name not in mech_index:
            mech_index[name] = len(mechanisms)
            mechanisms.append({"name": name, "kind": "tfne_rule"})
        return mech_index[name]

    # -- S14: G = G_0 \ E_-, subtracted over projection identities ------- #
    # Rule expansion produces G_0; exclusions resolve to identities E_- and
    # are removed from it. Exclusions name no mechanism, so an exclusion
    # resolves to every identity in G_0 sharing its (src, dst) route. An
    # exclusion that resolves to nothing is invalid (E_EXCLUSION_UNKNOWN)
    # rather than a silent no-op: a stale exclusion must not survive
    # normalization.
    excluded_routes: set[tuple[str, str]] = set()
    for exc in explicit.exclusions:
        excluded_routes.update(_scope_pairs(exc))
    matched_routes: set[tuple[str, str]] = set()

    for rel in explicit.relations:
        rule_params = rule_lookup.get(rel.rule) if rel.rule else None
        params = _rule_connection_params(rule_params)
        if rel.direction not in (">", "<"):
            raise TFNEError(
                f"relation {rel.key!r} has unresolved direction "
                f"{rel.direction!r}")
        routes = _scope_pairs(rel)
        removed = [r for r in routes if r in excluded_routes]
        matched_routes.update(removed)
        surviving = [r for r in routes if r not in excluded_routes]
        if not surviving:
            continue
        mech_id(params["mechanism"])

        def emit(name: str, src: Sequence[int], dst: Sequence[int]) -> None:
            if not src or not dst:
                return
            connections.append({
                "name": name,
                "source": {"ids": list(src)},
                "target": {"ids": list(dst)},
                "mechanism": params["mechanism"],
                "probability": params["probability"],
                "weight": params["weight"],
                "allow_self_connections": params["allow_self"],
            })

        if not removed:
            # Whole relation survives: emit it as one connection, keeping the
            # relation key and edge ordering that I indexes against.
            src_scopes = (rel.pre_scopes if rel.direction == ">"
                          else rel.post_scopes)
            dst_scopes = (rel.post_scopes if rel.direction == ">"
                          else rel.pre_scopes)
            emit(rel.key, scope_ids(src_scopes), scope_ids(dst_scopes))
        else:
            # Partially excluded: the relation no longer has a single
            # (sources x targets) identity, so each surviving route becomes
            # its own connection under a route-qualified key.
            for src_scope, dst_scope in surviving:
                emit(f"{rel.key}|{src_scope}>{dst_scope}",
                     scope_ids([src_scope]), scope_ids([dst_scope]))

    for exc in explicit.exclusions:
        if not any(r in matched_routes for r in _scope_pairs(exc)):
            raise TFNEError(
                f"E_EXCLUSION_UNKNOWN: exclusion {exc.key!r} matches no "
                f"projection generated by rule expansion")
    compiled = compile_connection_rules(
        rows, connections, mechanisms, seed=int(seed))
    edge_pre = np.asarray(compiled.edge_pre, dtype=np.int64)
    edge_post = np.asarray(compiled.edge_post, dtype=np.int64)
    edge_weight = np.asarray(compiled.edge_weight, dtype=np.float64)
    edge_mech = np.asarray(compiled.edge_mechanism, dtype=np.int64)
    n_edges = int(edge_pre.shape[0])

    # -- rule -> edge ranges (compiler preserves per-rule contiguity) -- #
    rule_slices: dict[str, tuple[int, int]] = {}
    edge_rule: list[str] = []
    cursor = 0
    for conn_id, conn in enumerate(compiled.connection_table):
        key = conn["name"]
        count = int(conn["n_edges"])
        rule_slices[key] = (cursor, cursor + count)
        edge_rule.extend([key] * count)
        cursor += count
    assert cursor == n_edges

    # -- s / h0 ---------------------------------------------------------- #
    geometry = {p: dict(rec.geometry) for p, rec in explicit.nodes.items()
                if rec.kind == "object" and rec.geometry}
    models = {p: rec.model for p, rec in explicit.nodes.items()
              if rec.kind == "object"}
    proportions = {p: dict(rec.proportions) for p, rec in explicit.nodes.items()
                   if rec.kind == "object"}
    counts = {p: dict(rec.counts) for p, rec in explicit.nodes.items()
              if rec.kind == "object"}
    rule_params = {k: dict(v) for k, v in rule_lookup.items()}
    origins = {rel.key: {"form": rel.form, "kind": rel.kind,
                         "rule": rel.rule,
                         "direction": rel.direction,
                         "group": rel.group,
                         "pre_label": rel.pre_label,
                         "post_label": rel.post_label,
                         "pre_scopes": list(rel.pre_scopes),
                         "post_scopes": list(rel.post_scopes),
                         "params": (_rule_connection_params(
                             rule_lookup.get(rel.rule))
                             if rel.rule else _rule_connection_params(None))}
               for rel in explicit.relations}
    s: dict[str, Any] = {
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "neuron_paths": list(neuron_paths),
        "edge_pre": edge_pre,
        "edge_post": edge_post,
        "edge_weight": edge_weight,
        "edge_mechanism": edge_mech,
        "mechanism_table": [dict(m) for m in compiled.mechanism_table],
        "connection_table": [dict(c) for c in compiled.connection_table],
        "geometry": geometry,
        "models": models,
        "proportions": proportions,
        "counts": counts,
        "rule_params": rule_params,
        "boundaries": dict(explicit.boundaries),
        "seed": int(seed),
        "normalization": explicit.normalization,
        "digest": explicit.digest,
        "value_tag": "relative",
    }
    h0: dict[str, Any] = {
        "v": np.zeros((n_neurons,), dtype=np.float32),
        "u": np.zeros((n_neurons,), dtype=np.float32),
        "spikes": np.zeros((n_neurons,), dtype=np.float32),
        "syn": np.zeros((n_edges,), dtype=np.float32),
        "H": np.zeros((n_neurons, 1), dtype=np.float32),
        "w": np.asarray(edge_weight, dtype=np.float32).copy(),
    }
    index_map: dict[str, Any] = {
        "object_slices": dict(object_slices),
        "member_slices": dict(member_slices),
        "neuron_paths": list(neuron_paths),
        "rule_slices": dict(rule_slices),
        "edge_rule": list(edge_rule),
        "rule_origins": dict(origins),
        "exclusions": [e.key for e in explicit.exclusions],
    }
    return Realization(s=s, h0=h0, I=index_map, explicit=explicit)


def _member_of(explicit: ExplicitModel, path: str, nid: int,
               object_slices: Mapping[str, tuple[int, int]]) -> str:
    rec = explicit.nodes[path]
    s0, _ = object_slices[path]
    offset = nid - s0
    for ctype in rec.cell_types:
        offset -= int(rec.counts.get(ctype, 0))
        if offset < 0:
            return ctype
    raise TFNEError(f"neuron {nid} outside member blocks of {path!r}")


def flatten(text: str, seed: Optional[int] = None) -> Realization:
    """Parse, resolve and realize TFNE source in one call."""
    program = parse(text)
    return realize(resolve(program), program, seed=seed)


# --------------------------------------------------------------------------- #
# NeuronalTensor bridge (explicit typed neural model)
# --------------------------------------------------------------------------- #

def to_neuronal_tensor(explicit: ExplicitModel):
    """Map an explicit model onto a JaxFNE ``NeuronalTensor``.

    Top-level members become areas; child composites (or leaves directly
    under an area) become layers; leaf cell types become neuron types with
    realized fractions. Relations inside one area become
    ``InterConnection`` entries; relations across areas become
    ``AreaConnection`` entries. Mechanism names come from the referenced
    rules (bare projections use :data:`DIRECT_MECHANISM`).
    """
    from .neuronal_tensor import (Area, AreaConnection, Geometry3D,
                                  InterConnection, Layer, NeuronalTensor,
                                  NeuronType)

    leaves = [p for p in explicit.order
              if explicit.nodes[p].kind == "object"]

    def top(path: str) -> str:
        return path.split(".")[0]

    def area_of(path: str) -> str:
        return top(path)

    def layer_of(path: str) -> str:
        # Nearest named scope strictly below the area root: for V1.L1 this is
        # the leaf itself (laminar layer L1); for V1.g0.L1 the group g0.
        segments = path.split(".")
        if len(segments) >= 2:
            return segments[1]
        return segments[0]

    areas: dict[str, list[str]] = {}
    for leaf in leaves:
        areas.setdefault(area_of(leaf), []).append(leaf)

    built_areas = []
    for area_name, members in areas.items():
        layers: dict[str, list[str]] = {}
        for leaf in members:
            layers.setdefault(layer_of(leaf), []).append(leaf)
        built_layers = []
        for layer_name, lmembers in layers.items():
            total = sum(sum(explicit.nodes[m].counts.values())
                        for m in lmembers)
            types = []
            seen: dict[str, int] = {}
            for leaf in lmembers:
                for ctype in explicit.nodes[leaf].cell_types:
                    seen[ctype] = seen.get(ctype, 0) + int(
                        explicit.nodes[leaf].counts.get(ctype, 0))
            for ctype, count in seen.items():
                types.append(NeuronType.make(
                    ctype, fraction=(count / total if total else 0.0),
                    value_tag="relative"))
            geo = Geometry3D(value_tag="relative")
            first_geo = None
            for leaf in lmembers:
                if explicit.nodes[leaf].geometry:
                    first_geo = explicit.nodes[leaf].geometry
                    break
            if first_geo:
                kwargs: dict[str, Any] = {"value_tag": "relative"}
                try:
                    if "distribution" in first_geo:
                        kwargs["distribution"] = str(
                            first_geo["distribution"])
                    for ax in ("x", "y", "z"):
                        lo = first_geo.get(f"{ax}0", first_geo.get(
                            f"{ax}_range"))
                        hi = first_geo.get(f"{ax}1")
                        if lo is not None and hi is not None:
                            kwargs[f"{ax}_range"] = (float(lo), float(hi))
                    geo = Geometry3D(**kwargs)
                except (TypeError, ValueError):
                    geo = Geometry3D(value_tag="relative")
            built_layers.append(Layer(name=layer_name,
                                      neuron_types=tuple(types),
                                      geometry=geo, n_neurons=int(total)))
        inter: list[InterConnection] = []
        for rel in explicit.relations:
            for pre_scope, post_scope in _scope_pairs(rel):
                for pre_leaf in _leaves_under(explicit, pre_scope, leaves):
                    for post_leaf in _leaves_under(explicit, post_scope,
                                                   leaves):
                        if area_of(pre_leaf) != area_name or \
                                area_of(post_leaf) != area_name:
                            continue
                        for pre_t in explicit.nodes[pre_leaf].cell_types:
                            for post_t in explicit.nodes[
                                    post_leaf].cell_types:
                                inter.append(InterConnection(
                                    source_layer=layer_of(pre_leaf),
                                    source_neuron_type=pre_t,
                                    target_layer=layer_of(post_leaf),
                                    target_neuron_type=post_t,
                                    mechanism=_relation_mechanism(
                                        explicit, rel)))
        built_areas.append(Area(name=area_name, layers=tuple(built_layers),
                                inter_connections=tuple(inter)))
    area_conns: list[AreaConnection] = []
    for rel in explicit.relations:
        for pre_scope, post_scope in _scope_pairs(rel):
            for pre_leaf in _leaves_under(explicit, pre_scope, leaves):
                for post_leaf in _leaves_under(explicit, post_scope, leaves):
                    if area_of(pre_leaf) == area_of(post_leaf):
                        continue
                    for pre_t in explicit.nodes[pre_leaf].cell_types:
                        for post_t in explicit.nodes[post_leaf].cell_types:
                            area_conns.append(AreaConnection(
                                source_area=area_of(pre_leaf),
                                source_layer=layer_of(pre_leaf),
                                source_neuron_type=pre_t,
                                target_area=area_of(post_leaf),
                                target_layer=layer_of(post_leaf),
                                target_neuron_type=post_t,
                                mechanism=_relation_mechanism(explicit, rel)))
    tensor = NeuronalTensor(areas=tuple(built_areas),
                            area_connections=tuple(area_conns),
                            name="tfne",
                            provenance={"tfne_digest": explicit.digest,
                                        "tfne_normalization":
                                            explicit.normalization})
    return tensor


def _scope_pairs(rel: RelationRecord) -> list[tuple[str, str]]:
    pairs = []
    if rel.direction in (">", "!>"):
        pairs = [(a, b) for a in rel.pre_scopes for b in rel.post_scopes]
    elif rel.direction in ("<", "!<"):
        pairs = [(b, a) for a in rel.pre_scopes for b in rel.post_scopes]
    else:
        pairs = [(a, b) for a in rel.pre_scopes for b in rel.post_scopes]
        pairs += [(b, a) for a in rel.pre_scopes for b in rel.post_scopes]
    return pairs


def _leaves_under(explicit: ExplicitModel, scope: str,
                  leaves: Sequence[str]) -> list[str]:
    if scope in explicit.nodes and explicit.nodes[scope].kind == "object":
        return [scope]
    if "." in scope and scope.rpartition(".")[0] in explicit.nodes:
        head = scope.rpartition(".")[0]
        if scope.rpartition(".")[2] in explicit.nodes[head].counts:
            # member selection: owning leaf only
            return [head]
    return [lf for lf in leaves
            if lf == scope or lf.startswith(scope + ".")]


def _relation_mechanism(explicit: ExplicitModel, rel: RelationRecord) -> str:
    if rel.rule is None:
        return DIRECT_MECHANISM
    params = explicit.rule_params.get(rel.rule, {})
    mech = params.get("mechanism", f"tfne_{rel.rule}")
    if isinstance(mech, Mapping):
        raise TFNEError("mechanism must be a name")
    return str(mech)


def realization_summary(realization: Realization) -> dict[str, Any]:
    """JSON-safe summary of a realization (hashes, shapes, digests)."""
    s = realization.s
    blob = {"digest": s["digest"], "seed": s["seed"],
            "n_neurons": s["n_neurons"], "n_edges": s["n_edges"],
            "edge_pre": [int(v) for v in np.asarray(s["edge_pre"]).tolist()],
            "edge_post": [int(v) for v in np.asarray(s["edge_post"]).tolist()],
            "edge_weight": [float(v)
                            for v in np.asarray(s["edge_weight"]).tolist()]}
    payload = json.dumps(blob, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {"digest": s["digest"], "seed": s["seed"],
            "n_neurons": s["n_neurons"], "n_edges": s["n_edges"],
            "normalization": s["normalization"],
            "realization_sha256": hashlib.sha256(payload).hexdigest(),
            "value_tag": s["value_tag"]}
