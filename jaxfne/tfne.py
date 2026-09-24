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
   applies to a single named reference. ``A^{nX}`` / ``A^{nO}`` (S6.1)
   develop the instances under ``X`` / sequential ``O``; the braced form
   keeps ``SEG^10 X Q`` meaning replication followed by composition.
4. ``in[A]`` / ``out[A]`` declare composition frontiers (S9): named
   properties, like ``order[A]``, naming immediate members of ``A``. A
   declared side overrides the derived default for that side only.
5. Rules with bodies (S12) expand each statement against the syntactic
   operands: ``$L`` / ``$R`` denote whole operands, ``.out`` / ``.in``
   narrow to resolved interfaces, and plain names address immediate
   members within their own side. Direction lives in the statements;
   a per-statement ``[mech=...]`` overrides the rule default.
6. Proportion-to-count allocation is largest-remainder with declaration-order
   tiebreak, so ``sum_c N[A.c] == N[A]`` holds exactly and deterministically.
7. Flat rules require an explicit ``direction`` field (``>``, ``<`` or
   ``<>``); rules with bodies carry direction in each statement instead.
   The compiler raises rather than    guessing a direction for ``O[k]``/``X[k]``.
8. Exclusions (``A !> B``) subtract from the rule expansion (S14): with
   ``G_0`` the generated projection set, ``G`` is ``G_0`` with the resolved
   exclusion identities removed. An exclusion naming no mechanism removes
   every identity on its route. An exclusion matching no generated
   projection raises ``E_EXCLUSION_UNKNOWN`` rather than passing as a
   no-op, so a stale exclusion cannot survive normalization.
9. ``parse(normalize(p))`` normalizes to ``normalize(p)`` (idempotent replay);
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
import re

import numpy as np

__all__ = [
    "TFNEError",
    "TFNEAddressUnknown",
    "TFNEAmbiguousExpansion",
    "TFNEExclusionUnknown",
    "TFNEFrontierUnresolved",
    "TFNEInvalidProportion",
    "TFNEMechanismNotPermitted",
    "TFNEMechanismUnresolved",
    "TFNEMissingPolicy",
    "TFNEOrderViolation",
    "TFNEProjectionRedundant",
    "Program",
    "ExplicitModel",
    "Realization",
    "parse",
    "normalize",
    "resolve",
    "realize",
    "flatten",
    "to_neuronal_tensor",
    "to_configuration",
    "DEFAULT_MODEL",
    "DIRECT_MECHANISM",
]

DEFAULT_MODEL = "izhikevich"
DIRECT_MECHANISM = "tfne_direct"

#: Reserved compiler-internal mechanism namespace. Names here are created
#: by the compiler, never declared by authors: `tfne_direct` is the
#: direct-coupling default for projections/rules without a mechanism.
MECHANISM_INTERNAL_PREFIX = "tfne_"

# S25 mechanism vocabulary classes. CANONICAL names come from
# `standard_receptor_specs()`; EXPLICIT_ALIAS is reserved for aliases
# declared by an authoritative definition (none exist: e.g. GABA is
# ambiguous between GABA_A and GABA_B, so no alias can be correct);
# CUSTOM_DEFINED needs a sufficient executable definition (a tau);
# the rest fail closed.
MECHANISM_CANONICAL = "CANONICAL"
MECHANISM_EXPLICIT_ALIAS = "EXPLICIT_ALIAS"
MECHANISM_CUSTOM_DEFINED = "CUSTOM_DEFINED"
MECHANISM_UNRESOLVED = "UNRESOLVED"
MECHANISM_NOT_PERMITTED = "NOT_PERMITTED"

#: Placeholder kinetics for the compiler-internal direct coupling. Not a
#: receptor value: bare projections declare no filtering, and changing this
#: would retune every existing direct-coupled trajectory, so it stays put
#: as documented placeholder rather than a biological claim.
DIRECT_MECHANISM_TAU_MS = 0.1

_PN_TOL = 1e-9


class TFNEError(ValueError):
    """Deterministic TFNE specification/realization failure."""


# S25 failure taxonomy: semantic failure classes, each a TFNEError so
# existing handlers keep working. Parser/lexer structural errors stay
# plain TFNEError — the vocabulary is semantic, not parser-specific.
class TFNEAddressUnknown(TFNEError):
    """A path or member reference resolving to nothing (S7/S25)."""


class TFNEAmbiguousExpansion(TFNEError):
    """An expansion with no unique reading: ungrouped same-rule X chains
    (S11) and other ambiguous/nonunique expansions (S25)."""


class TFNEExclusionUnknown(TFNEError):
    """An exclusion matching no generated projection (S14/S25)."""


class TFNEFrontierUnresolved(TFNEError):
    """A declared or required interface that cannot be derived uniquely
    (S9/S25)."""


class TFNEInvalidProportion(TFNEError):
    """An invalid proportion, count, or type specification in a
    declaration body (S5/S25). Also covers TFNE `G` geometry bounds:
    relative coordinates are fractions of the area's extent in [0,1]
    (0.5.2 decision 0a), so the same class refuses out-of-range,
    half-declared, degenerate or non-finite geometry bounds."""


class TFNEMechanismUnresolved(TFNEError):
    """A required mechanism with no resolvable identity (S13/S25)."""


class TFNEMechanismNotPermitted(TFNEError):
    """A mechanism known but inadmissible here: reserved namespace,
    contradictory canonical kinetics, or inadmissible definition
    (S13/S25)."""


class TFNEMissingPolicy(TFNEError):
    """A required realization/allocation policy the specification does
    not supply (S5/S25)."""


class TFNEOrderViolation(TFNEError):
    """An order declaration that cannot be honoured exactly (S20.1)."""


class TFNEProjectionRedundant(TFNEError):
    """An explicit projection identical to generated output (S13/S25)."""


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
                  "^", ":", "=", ">", "<", "(", ")")


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
        # S12 rule metavariables: `$L` / `$R` lex as one META token. A lone
        # `$` (or `$` before a non-name) stays a hard lexer error, so `$`
        # cannot silently appear anywhere bodies are not parsed.
        if c == "$" and i + 1 < n and (
            text[i + 1].isalpha() or text[i + 1] == "_"
        ):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(("META", text[i + 1:j]))
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
class GroupStmts:
    """Brace composite with `;`-separated statements (S14/S25 atomicity).

    Each statement expands independently; a statement whose resolved
    projection is invalid contributes nothing instead of aborting the
    composite. The group still hands itself upward as one member (S8).
    """
    body: tuple[Any, ...]


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
    # S6.1: relation among instances. None = bare `A^n` (instances only);
    # "X" / "O" = `A^{nX}` / `A^{nO}`, instances joined by that operator.
    rel: Optional[str] = None


@dataclass(frozen=True)
class PrefixApply:
    """S6 prefix rule application: `O[k](SEG^n)` chains rule `k` over the
    replicated instances as ordered adjacencies. Only `O[k]` is defined;
    only plain `A^n` replication is a valid target."""
    rule: str
    target: Any  # Replicate


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
    # S12 rule body: projection statements over `$L` / `$R`. Empty for
    # flat (params-only) rules, which keep their exact legacy behavior.
    body: tuple[Any, ...] = ()


@dataclass(frozen=True)
class RuleEndpoint:
    """One side of a rule-body projection statement (`tfne/2` S12).

    `form` is "meta" (`$L` / `$R` with an optional `out` / `in` tail),
    "ref" (a member reference resolved within the operand), or "set"
    (a collection of member references).
    """
    form: str
    head: str = ""           # "L" | "R" for meta; dotted ref for ref
    tail: tuple[str, ...] = ()  # interface selector for meta
    members: tuple[str, ...] = ()  # refs for set


@dataclass(frozen=True)
class RuleStmt:
    """One projection statement inside a rule body: endpoints, direction,
    and an optional per-statement mechanism overriding the rule default."""
    left: RuleEndpoint
    direction: str  # '>' | '<' | '<>'
    right: RuleEndpoint
    mechanism: Optional[str] = None


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
    # `tfne/2` S20: scope name -> the declared order of its immediate members.
    # Only an `order[A] := [...]` statement populates this. Enumeration,
    # structural listing and composition order never do.
    orders: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # `tfne/2` S9: scope name -> {"in": members, "out": members} for the
    # sides a definition explicitly declares. Only an `in[A] := [...]` /
    # `out[A] := [...]` statement populates this; a declared side overrides
    # the derived default for that side only, never the other side.
    frontiers: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict)


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
        orders: dict[str, tuple[str, ...]] = {}
        frontiers: dict[str, dict[str, tuple[str, ...]]] = {}
        system: Optional[System] = None
        while self.peek()[0] != "EOF":
            while self.peek()[0] == "SEP":
                self.next()
            if self.peek()[0] == "EOF":
                break
            if self._is_orderdef():
                scope, members = self._parse_orderdef()
                if scope in orders:
                    raise TFNEOrderViolation(
                        f"E_ORDER_DUPLICATE: duplicate order declaration for "
                        f"{scope!r}")
                orders[scope] = members
            elif self._is_frontierdef():
                side, scope, members = self._parse_frontierdef()
                if side in frontiers.get(scope, {}):
                    raise TFNEFrontierUnresolved(
                        f"E_FRONTIER_UNRESOLVED: duplicate {side} frontier "
                        f"declaration for {scope!r}; an interface declared "
                        f"twice cannot be derived uniquely")
                frontiers.setdefault(scope, {})[side] = members
            elif self._is_ruledef():
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
        return Program(defs=defs, rules=rules, system=system,
                       orders=orders, frontiers=frontiers)

    def _is_frontierdef(self) -> bool:
        # ("in"|"out") "[" NAME "]" ":="
        t = self.toks
        p = self.pos
        return (
            p + 2 < len(t)
            and t[p][0] == "NAME" and t[p][1] in ("in", "out")
            and t[p + 1] == ("SYM", "[")
            and t[p + 2][0] == "NAME"
        )

    def _parse_frontierdef(self) -> tuple[str, str, tuple[str, ...]]:
        """Parse ``in[A] := [m1, ...]`` / ``out[A] := [m1, ...]`` (S9).

        Frontiers are interface metadata, so — like ``order[A]`` — they take
        a named property rather than a new operator. Members name immediate
        members of ``A`` (subset allowed: an interface need not expose every
        member); validation against the resolved model happens at resolve().
        """
        side = self.expect("NAME")                    # "in" / "out"
        self.expect("SYM", "[")
        scope = self._parse_order_member()            # bare name or dotted path
        self.expect("SYM", "]")
        self.expect("SYM", ":=")
        self.expect("SYM", "[")
        members: list[str] = []
        while not self.at_sym("]"):
            members.append(self._parse_order_member())
            if self.at_sym(","):
                self.next()
                continue
            break
        self.expect("SYM", "]")
        if not members:
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: {side}[{scope}] declares no "
                f"members; an empty interface resolves nothing")
        return side, scope, tuple(members)

    def _is_orderdef(self) -> bool:
        # "order" "[" NAME "]" ":="
        t = self.toks
        p = self.pos
        return (
            p + 2 < len(t)
            and t[p] == ("NAME", "order")
            and t[p + 1] == ("SYM", "[")
            and t[p + 2][0] == "NAME"
        )

    def _parse_orderdef(self) -> tuple[str, tuple[str, ...]]:
        """Parse ``order[A] := [m1, m2, ...]`` (`tfne/2` S20).

        The members are immediate member names of ``A``, not paths. Ordering
        is metadata rather than structure, so this deliberately reuses the
        existing bracket/comma tokens instead of taking a new operator.
        """
        self.expect("NAME")                       # "order"
        self.expect("SYM", "[")
        scope = self._parse_order_member()        # bare name or dotted path
        self.expect("SYM", "]")
        self.expect("SYM", ":=")
        self.expect("SYM", "[")
        members: list[str] = []
        while not self.at_sym("]"):
            members.append(self._parse_order_member())
            if self.at_sym(","):
                self.next()
                continue
            break
        self.expect("SYM", "]")
        if not members:
            raise TFNEOrderViolation(
                f"E_ORDER_EMPTY: order[{scope}] declares no members")
        return scope, tuple(members)

    def _parse_order_member(self) -> str:
        """One member reference: ``L4``, or a replica such as ``SEG.2``.

        ``SEG.2`` lexes as NAME plus FLOAT ``.2`` because the lexer takes a
        leading dot greedily, so the numeric tail is reattached here rather
        than by loosening the lexer for every other construct.
        """
        name = self.expect("NAME")
        parts = [name]
        while True:
            tok = self.peek()
            if tok[0] == "FLOAT" and tok[1].startswith("."):
                self.next()
                tail = tok[1][1:]
                if not tail.isdigit():
                    raise TFNEOrderViolation(
                        f"E_ORDER_MEMBER_INVALID: {name!r} has a malformed "
                        f"replica index {tok[1]!r}")
                parts.append(tail)
                continue
            if self.at_sym("."):
                self.next()
                parts.append(self.expect("NAME"))
                continue
            break
        return ".".join(parts)

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
        body: list[RuleStmt] = []
        while not self.at_sym("]"):
            while self.peek()[0] == "SEP":
                self.next()
            if self.at_sym("]"):
                break
            body.append(self._parse_rulestmt(kind, name))
        self.expect("SYM", "]")
        return RuleDef(name=name, kind=kind, params=params,
                       body=tuple(body))

    def _parse_rulestmt(self, kind: str, rule: str) -> RuleStmt:
        """One S12 body statement: endpoint, direction, endpoint, mechanism.

        Direction lives in the statement (bodies routinely mix `>` and `<`),
        never implicitly. A trailing `[mech=NAME]` overrides the rule-level
        mechanism for this statement's projections only.
        """
        left = self._parse_rule_endpoint()
        tok = self.peek()
        if tok[0] != "SYM" or tok[1] not in (">", "<", "<>"):
            raise TFNEError(
                f"rule {kind}[{rule}] body statements need an explicit "
                f"direction ('>', '<', '<>'); got {tok}")
        direction = self.next()[1]
        right = self._parse_rule_endpoint()
        mechanism: Optional[str] = None
        if self.at_sym("["):
            self.next()
            if self.peek() != ("NAME", "mech"):
                raise TFNEError(
                    f"rule {kind}[{rule}] statement options support only "
                    f"[mech=NAME]; got {self.peek()}")
            self.next()
            self.expect("SYM", "=")
            mechanism = self.expect("NAME")
            self.expect("SYM", "]")
        return RuleStmt(left=left, direction=direction, right=right,
                        mechanism=mechanism)

    def _parse_rule_endpoint(self) -> RuleEndpoint:
        """One S12 endpoint: `$L[.out|.in]`, a member ref, or `{a, b}`."""
        tok = self.peek()
        if tok[0] == "META":
            self.next()
            if tok[1] not in ("L", "R"):
                raise TFNEError(
                    f"rule metavariables are $L and $R; got ${tok[1]}")
            tail: list[str] = []
            while self.at_sym("."):
                self.next()
                part = self.expect("NAME")
                if part not in ("out", "in"):
                    raise TFNEError(
                        f"interface selection on ${tok[1]} supports only "
                        f".out / .in; got .{part}")
                tail.append(part)
            return RuleEndpoint(form="meta", head=tok[1],
                                tail=tuple(tail))
        if tok == ("SYM", "{"):
            self.next()
            members = [self._parse_order_member()]
            while self.at_sym(","):
                self.next()
                members.append(self._parse_order_member())
            self.expect("SYM", "}")
            return RuleEndpoint(form="set", members=tuple(members))
        if tok[0] == "NAME":
            return RuleEndpoint(form="ref", head=self._parse_order_member())
        raise TFNEError(f"expected a rule endpoint; got {tok}")

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
        # S6 prefix rule application `O[k](...)` lives in operand position:
        # after an infix `O[k]` the same tokens are a grouped operand, so
        # only a leading `O[k](` is the prefix form.
        if (self.at_name("O") or self.at_name("X")):
            t = self.toks
            p = self.pos
            if (p + 5 < len(t)
                    and t[p + 1] == ("SYM", "[")
                    and t[p + 2][0] == "NAME"
                    and t[p + 3] == ("SYM", "]")
                    and t[p + 4] == ("SYM", "(")):
                kind = self.expect("NAME")
                self.expect("SYM", "[")
                rule = self.expect("NAME")
                self.expect("SYM", "]")
                self.expect("SYM", "(")
                if kind != "O":
                    raise TFNEError(
                        "prefix rule application is O[k](...) (S6); "
                        "X[k](...) is not defined — X-joined replication "
                        "is A^{nX} (S6.1)")
                target = self._parse_expr()
                self.expect("SYM", ")")
                if not isinstance(target, Replicate) or target.rel is not None:
                    raise TFNEError(
                        f"O[k](...) prefix applies to plain A^n "
                        f"replication (S6); got {_emit_expr(target)!r}")
                return PrefixApply(rule=rule, target=target)
        atom = self._parse_atom()
        if self.at_sym("^"):
            self.next()
            # S6.1: braced `A^{nX}` / `A^{nO}` develop instances under a
            # relation. Braces are required so `SEG^10 X Q` (replication
            # followed by cross composition) keeps its existing meaning.
            if self.at_sym("{"):
                self.next()
                tok = self.next()
                if tok[0] != "INT":
                    raise TFNEError("replication count must be an integer")
                n = int(tok[1])
                rel: Optional[str] = None
                if self.at_sym("}"):
                    self.next()
                else:
                    rtok = self.next()
                    if rtok[0] != "NAME" or rtok[1] not in ("O", "X"):
                        raise TFNEError(
                            "replication relation must be O or X, "
                            f"as in A^{{nO}} or A^{{nX}}; got {rtok}")
                    rel = rtok[1]
                    self.expect("SYM", "}")
            else:
                tok = self.next()
                if tok[0] != "INT":
                    raise TFNEError("replication count must be an integer")
                n = int(tok[1])
                rel = None
            if n < 1:
                raise TFNEError("replication count must be >= 1")
            if not isinstance(atom, Ref) or len(atom.segments) != 1:
                raise TFNEError(
                    "replication (A^n) applies to a single named reference"
                )
            return Replicate(atom=atom, n=n, rel=rel)
        return atom

    def _parse_atom(self) -> Any:
        tok = self.peek()
        if tok == ("SYM", "{"):
            self.next()
            # S14/S25: `;`-separated statements, each atomic. SEP is the
            # separator (so newlines behave exactly like `;`, as at program
            # level); a trailing separator before `}` is allowed.
            while self.peek()[0] == "SEP":
                self.next()
            first = self._parse_expr()
            stmts = [first]
            while True:
                had_sep = False
                while self.peek()[0] == "SEP":
                    self.next()
                    had_sep = True
                if self.at_sym("}"):
                    break
                if not had_sep:
                    # `_parse_expr` never consumes a leading SEP, so reaching
                    # here means juxtaposed expressions, not statements.
                    raise TFNEError(
                        f"expected ';' or '}}' in composite; got {self.peek()}")
                stmts.append(self._parse_expr())
            self.expect("SYM", "}")
            if len(stmts) == 1:
                return Group(body=first)
            return GroupStmts(body=tuple(stmts))
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

def _emit_rule_endpoint(ep: RuleEndpoint) -> str:
    if ep.form == "meta":
        return "$" + ep.head + "".join(f".{t}" for t in ep.tail)
    if ep.form == "ref":
        return ep.head
    return "{" + ", ".join(ep.members) + "}"


def _emit_rulestmt(stmt: RuleStmt) -> str:
    text = (f"{_emit_rule_endpoint(stmt.left)} {stmt.direction} "
            f"{_emit_rule_endpoint(stmt.right)}")
    if stmt.mechanism is not None:
        text += f" [mech={stmt.mechanism}]"
    return text


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
    if isinstance(node, GroupStmts):
        return "{" + "; ".join(_emit_expr(s) for s in node.body) + "}"
    if isinstance(node, PrefixApply):
        return f"O[{node.rule}]({_emit_expr(node.target)})"
    if isinstance(node, (Ordered, Cross)):
        op = "O" if isinstance(node, Ordered) else "X"
        if node.rule is not None:
            op += f"[{node.rule}]"
        return f"{_emit_expr(node.left)} {op} {_emit_expr(node.right)}"
    if isinstance(node, (Project, Exclude)):
        return f"{_emit_expr(node.left)} {node.direction} {_emit_expr(node.right)}"
    if isinstance(node, Replicate):
        base = f"{_emit_expr(node.atom)}^{node.n}"
        if node.rel is None:
            return base
        return f"{_emit_expr(node.atom)}^{{{node.n}{node.rel}}}"
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
        body = f"{rule.kind}[{name}] := [{_emit_propbody(rule.params)}"
        for stmt in getattr(rule, "body", ()):
            sep = "" if body.endswith("[") else "; "
            body += sep + _emit_rulestmt(stmt)
        parts.append(body + "]")
    for scope in sorted(getattr(program, "orders", {})):
        members = ", ".join(getattr(program, "orders")[scope])
        parts.append(f"order[{scope}] := [{members}]")
    for scope in sorted(getattr(program, "frontiers", {})):
        for side in ("in", "out"):
            if side in getattr(program, "frontiers")[scope]:
                members = ", ".join(
                    getattr(program, "frontiers")[scope][side])
                parts.append(f"{side}[{scope}] := [{members}]")
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
    # S12: per-statement mechanism from a rule-body `[mech=...]`, overriding
    # the rule-level mechanism for this relation's projections only.
    mechanism: Optional[str] = None


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
                raise TFNEInvalidProportion(
                    f"{where}: proportion member {c!r} not in C")
            if not (0.0 <= float(p) <= 1.0):
                raise TFNEInvalidProportion(
                    f"{where}: proportion {c}={p} outside [0,1]")
        if abs(sum(float(p) for p in proportions.values()) - 1.0) > 1e-6:
            raise TFNEInvalidProportion(
                f"{where}: proportions sum to "
                f"{sum(float(p) for p in proportions.values())}, not 1"
            )
    if isinstance(total, Mapping):
        counts = {c: int(total[c]) for c in total}
        for c in counts:
            if c not in ctypes:
                raise TFNEInvalidProportion(
                    f"{where}: count member {c!r} not in C")
        missing = [c for c in ctypes if c not in counts]
        if missing:
            raise TFNEMissingPolicy(
                f"{where}: N map missing members {missing}")
        n_total = sum(counts.values())
        if proportions is not None:
            for c in ctypes:
                if abs(counts[c] - proportions[c] * n_total) >= 1.0:
                    raise TFNEInvalidProportion(
                        f"{where}: N map inconsistent with P for {c!r}")
        derived = {c: counts[c] / n_total for c in ctypes} if n_total else {
            c: 0.0 for c in ctypes}
        return counts, derived
    n_total = 1 if total is None else int(total)
    if n_total < 1:
        raise TFNEInvalidProportion(f"{where}: N must be >= 1")
    if proportions is None:
        if len(ctypes) == 1:
            return {ctypes[0]: n_total}, {ctypes[0]: 1.0}
        raise TFNEMissingPolicy(
            f"{where}: P required when C has members {ctypes}")
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


def _validate_geometry_body(where: str, geometry: Mapping[str, Any]) -> None:
    """Refuse a TFNE `G` body that is not relative [0,1] coordinates.

    0.5.2 decision 0a: a declared range is a pair of fractions of the
    area's extent, within [0,1]; a range outside [0,1] is refused. A
    half-declared axis (one bound only), a degenerate range (`hi <= lo`)
    and non-finite bounds are refused alongside, mirroring JDNA's
    `_axis_range` refusal rather than inventing a second policy. An
    undeclared axis defaults to the full extent downstream; `distribution`
    and unknown keys pass through untouched (distribution support is a
    downstream concern, not a language one).
    """
    for axis in ("x", "y", "z"):
        lo_key, hi_key, range_key = f"{axis}0", f"{axis}1", f"{axis}_range"
        if range_key in geometry:
            pair = geometry[range_key]
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                raise TFNEInvalidProportion(
                    f"E_GEOMETRY_OUT_OF_RANGE: {where}: G {range_key} must be "
                    f"a [lo; hi] pair of numbers in [0,1]; got {pair!r}")
            bounds = (lo, hi)
        else:
            lo = geometry.get(lo_key)
            hi = geometry.get(hi_key)
            if lo is None and hi is None:
                continue
            if lo is None or hi is None:
                raise TFNEInvalidProportion(
                    f"E_GEOMETRY_OUT_OF_RANGE: {where}: partial {axis} "
                    f"domain declares only one bound ({lo_key}={lo!r}, "
                    f"{hi_key}={hi!r}); a half-domain cannot be honoured "
                    f"exactly, so it is refused rather than half-defaulted")
            try:
                bounds = (float(lo), float(hi))
            except (TypeError, ValueError):
                raise TFNEInvalidProportion(
                    f"E_GEOMETRY_OUT_OF_RANGE: {where}: G {axis} bounds "
                    f"must be numbers in [0,1]; got ({lo!r}, {hi!r})")
        lo, hi = bounds
        if not (bool(np.isfinite(lo)) and bool(np.isfinite(hi))):
            raise TFNEInvalidProportion(
                f"E_GEOMETRY_OUT_OF_RANGE: {where}: G {axis} bounds must "
                f"be finite numbers in [0,1]; got ({lo!r}, {hi!r})")
        if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
            raise TFNEInvalidProportion(
                f"E_GEOMETRY_OUT_OF_RANGE: {where}: G {axis} range "
                f"({lo}, {hi}) lies outside [0,1]; declared geometry is "
                f"relative (fractions of the area's extent), so an "
                f"outside range is refused rather than rescaled")
        if not hi > lo:
            raise TFNEInvalidProportion(
                f"E_GEOMETRY_OUT_OF_RANGE: {where}: degenerate G {axis} "
                f"domain [{lo}, {hi}]; bounds must order")


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


def _natural_component_key(component: str) -> tuple:
    """Typed natural key for one path component (`tfne/2` S20).

    Digit runs compare as integers and text runs as text, so `L1 < L2 < L10`
    rather than the lexical `L1 < L10 < L2`. Each run carries a type tag so
    tuples of mixed runs stay comparable.
    """
    return tuple(
        (0, int(run), "") if run.isdigit() else (1, 0, run)
        for run in re.findall(r"\d+|\D+", component)
    )


def _natural_path_key(path: str) -> tuple:
    """Typed natural key for a full canonical path.

    Compared component by component, so `SEG.2 < SEG.10`, and a parent sorts
    before its own children because its key is a proper prefix.
    """
    return tuple(_natural_component_key(c) for c in path.split("."))


def _scope_children(nodes: Mapping[str, Any],
                    scope: str) -> list[tuple[str, str]]:
    """Immediate members of `scope` as `(full path, remainder)` pairs.

    Membership follows resolved parent links, so replicated instances such
    as ``SEG.1`` are members of their scope even though the remainder
    contains a dot. Shared by order validation (S20.1) and frontier
    validation (S9): both name immediate members, never descendants.
    """
    out: list[tuple[str, str]] = []
    for path, rec in nodes.items():
        parent = getattr(rec, "parent", None)
        if parent is None:
            parent = path.rpartition(".")[0] if "." in path else ""
        if parent != scope:
            continue
        if scope and path.startswith(scope + "."):
            out.append((path, path[len(scope) + 1:]))
        else:
            out.append((path, path.rpartition(".")[2]))
    return out


def _declared_ranks(nodes: Mapping[str, Any],
                    orders: Mapping[str, Sequence[str]]
                    ) -> dict[tuple[str, str], int]:
    """Validate `order[A] := [...]` declarations into `(parent, remainder)` ranks.

    `tfne/2` S20: only an explicit declaration overrides natural ordering, so
    this rejects anything it cannot honour exactly rather than ordering a
    partial answer. A declaration must name every immediate member of its
    scope exactly once. Immediate membership follows the resolved parent
    links, so replicated instances such as ``SEG.1`` are members of their
    scope even though their remainder contains a dot.
    """
    ranks: dict[tuple[str, str], int] = {}
    for scope, members in orders.items():
        if scope in nodes:
            resolved = scope
        else:
            # A bare name may address a nested object, but only if it does so
            # unambiguously. Guessing between two candidates would reorder a
            # scope the author did not name.
            candidates = sorted(path for path in nodes
                                if path.endswith("." + scope))
            if not candidates:
                raise TFNEOrderViolation(
                    f"E_ORDER_SCOPE_UNKNOWN: order[{scope}] names no object "
                    f"in the resolved model")
            if len(candidates) > 1:
                raise TFNEOrderViolation(
                    f"E_ORDER_SCOPE_AMBIGUOUS: order[{scope}] matches "
                    f"{candidates!r}; name the full path")
            resolved = candidates[0]
        scope = resolved
        child_components = [rem for _, rem in _scope_children(nodes, scope)]

        normalized: list[str] = []
        for member in members:
            local = (member[len(scope) + 1:]
                     if member.startswith(scope + ".") else member)
            if "." in local and local not in child_components:
                raise TFNEOrderViolation(
                    f"E_ORDER_NOT_IMMEDIATE: order[{scope}] names "
                    f"{member!r}, which is not an immediate member of "
                    f"{scope!r}; a declaration orders its own members only")
            normalized.append(local)

        seen: set[str] = set()
        for local in normalized:
            if local in seen:
                raise TFNEOrderViolation(
                    f"E_ORDER_DUPLICATE_MEMBER: order[{scope}] names "
                    f"{local!r} more than once")
            seen.add(local)

        unknown = [m for m in normalized if m not in child_components]
        if unknown:
            raise TFNEOrderViolation(
                f"E_ORDER_MEMBER_UNKNOWN: order[{scope}] names {unknown!r}, "
                f"which are not members of {scope!r}; members are "
                f"{sorted(child_components)!r}")

        missing = [c for c in child_components if c not in seen]
        if missing:
            raise TFNEOrderViolation(
                f"E_ORDER_INCOMPLETE: order[{scope}] omits {sorted(missing)!r}; "
                f"an explicit order must name every immediate member exactly "
                f"once, otherwise the omitted members would silently fall back "
                f"to a different rule")

        for index, local in enumerate(normalized):
            ranks[(scope, local)] = index
    return ranks


def _validate_frontier_members(nodes: Mapping[str, Any], target: str,
                               side: str, scope_key: str,
                               members: Sequence[str]) -> tuple[str, ...]:
    """Validate one declared side into full member paths (`tfne/2` S9).

    A declared frontier names a subset of the scope's immediate members —
    an interface need not expose everything — but anything it cannot honour
    exactly is refused with `E_FRONTIER_UNRESOLVED` rather than partially
    applied: unknown members, descendants rather than members, duplicates.
    """
    pairs = _scope_children(nodes, target)
    by_remainder = {rem: full for full, rem in pairs}
    seen: set[str] = set()
    full_paths: list[str] = []
    for member in members:
        local = (member[len(target) + 1:]
                 if member.startswith(target + ".") else member)
        if local not in by_remainder:
            if "." in local:
                raise TFNEFrontierUnresolved(
                    f"E_FRONTIER_UNRESOLVED: {side}[{scope_key}] names "
                    f"{member!r}, which is not an immediate member of "
                    f"{target!r}; a frontier exposes its own members only")
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: {side}[{scope_key}] names "
                f"{member!r}, which is not a member of {target!r}; "
                f"members are {sorted(by_remainder)!r}")
        if local in seen:
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: {side}[{scope_key}] names "
                f"{local!r} more than once")
        seen.add(local)
        full_paths.append(by_remainder[local])
    return tuple(full_paths)


def _verify_frontier_declarations(nodes: Mapping[str, Any],
                                  frontiers: Mapping[str, Mapping[str, Any]],
                                  applied: set[str]) -> None:
    """End-of-resolve check: every declared frontier resolved and applied.

    Bare scope names resolve only when unambiguous against the final model —
    applying to one of several same-named scopes would complete an interface
    the author did not single out. Leaf scopes cannot declare: a leaf object
    exposes only itself.
    """
    for scope_key in frontiers:
        if scope_key in nodes:
            resolved = scope_key
        else:
            candidates = sorted(path for path in nodes
                                if path.endswith("." + scope_key))
            if not candidates:
                raise TFNEFrontierUnresolved(
                    f"E_FRONTIER_UNRESOLVED: frontier on {scope_key!r} "
                    f"names no object in the resolved model")
            if len(candidates) > 1:
                raise TFNEFrontierUnresolved(
                    f"E_FRONTIER_UNRESOLVED: frontier on {scope_key!r} "
                    f"matches {candidates!r}; name the full path")
            resolved = candidates[0]
        rec = nodes[resolved]
        if getattr(rec, "kind", None) != "composite":
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: frontier on {scope_key!r} "
                f"addresses {resolved!r}, which is a leaf object; a leaf "
                f"exposes only itself")
        if resolved.rpartition(".")[2].isdigit():
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: frontier on {scope_key!r} "
                f"addresses {resolved!r}, which is a replica instance; "
                f"declare on the scope holding the replicas instead")
        if resolved not in applied:
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: frontier on {scope_key!r} "
                f"was never applied during expansion")


def _ordering_key(path: str,
                  ranks: Mapping[tuple[str, str], int],
                  nodes: Mapping[str, Any] | None = None) -> tuple:
    """Realization ordering key: declared rank where given, else natural.

    Ranked edge by edge along the resolved parent chain, so a declaration
    applies only among the siblings it names and nested scopes order
    independently. Both branches are 3-tuples of the same shape, and their
    leading tag differs, so the natural key is never compared against a
    declared rank. Remainders keep their dots (``SEG.1``), so replicated
    instances order as immediate members of their scope.
    """
    if nodes is None:
        key: list[tuple] = []
        components = path.split(".")
        for depth, component in enumerate(components):
            parent = ".".join(components[:depth])
            rank = ranks.get((parent, component))
            if rank is None:
                key.append((1, 0, _natural_component_key(component)))
            else:
                key.append((0, rank, ()))
        return tuple(key)
    chain: list[tuple[str, str]] = []
    current: str | None = path
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        rec = nodes.get(current) if nodes is not None else None
        parent = getattr(rec, "parent", None) if rec is not None else None
        if parent is None:
            if rec is not None:
                parent = ""
            elif current and "." in current:
                parent = current.rpartition(".")[0]
            else:
                parent = ""
        if parent in ("", None):
            chain.append(("", current))
            break
        if isinstance(parent, str) and current.startswith(parent + "."):
            remainder = current[len(parent) + 1:]
        else:
            remainder = current.rpartition(".")[2]
        chain.append((parent, remainder))
        current = parent if parent in nodes else (None if parent == "" else parent)
        if parent not in nodes and parent != "":
            # Ancestor outside the resolved model (should not happen for
            # explicit.order paths); terminate with a natural edge.
            chain.append(("", parent))
            break
    key = []
    for parent, remainder in reversed(chain):
        rank = ranks.get((parent, remainder))
        if rank is None:
            natural = tuple(_natural_component_key(c)
                            for c in remainder.split("."))
            key.append((1, 0, natural))
        else:
            key.append((0, rank, ()))
    return tuple(key)


class _Resolver:
    def __init__(self, program: Program):
        self.program = program
        self.nodes: dict[str, NodeRecord] = {}
        self.order: list[str] = []
        self.relations: list[RelationRecord] = []
        self.exclusions: list[RelationRecord] = []
        self.group_counter = 0
        self.relation_counter = 0
        # Scopes whose declared in/out frontiers were substituted during
        # expansion. Checked at end of resolve(): a declaration that never
        # applied names something unresolvable.
        self.applied_frontiers: set[str] = set()
        # S14/S25 atomicity depth: while expanding brace-group statements,
        # a statement whose resolved projection is invalid contributes
        # nothing instead of aborting the composite. All other errors
        # propagate (resolve() either returns whole or raises).
        self._brace_atomic = 0

    def _apply_declared_frontiers(self, target: str,
                                  inner: _Expansion) -> _Expansion:
        """Substitute declared `in`/`out` for `target` (`tfne/2` S9).

        Declarations match by exact scope path or by bare name against the
        nodes expanded so far; final uniqueness is enforced at end of
        resolve(), so an ambiguous bare name always fails closed. A declared
        side overrides the derived default for that side only. With no
        matching declaration the derived expansion passes through unchanged.
        """
        hits = [(key, sides) for key, sides in
                self.program.frontiers.items()
                if key == target or target.endswith("." + key)]
        if len(hits) > 1:
            names = sorted(key for key, _ in hits)
            raise TFNEFrontierUnresolved(
                f"E_FRONTIER_UNRESOLVED: {names!r} all match {target!r}; "
                f"a scope with competing frontier declarations cannot be "
                f"derived uniquely")
        if not hits:
            return inner
        scope_key, sides = hits[0]
        fin, fout = inner.fin, inner.fout
        if "in" in sides:
            fin = _validate_frontier_members(
                self.nodes, target, "in", scope_key, sides["in"])
        if "out" in sides:
            fout = _validate_frontier_members(
                self.nodes, target, "out", scope_key, sides["out"])
        self.applied_frontiers.add(target)
        return _Expansion(members=inner.members, fin=fin, fout=fout)

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
            raise TFNEInvalidProportion(f"{where}: C must be a {{type, ...}} set")
        # Cell-type members use the algebra's own capitalized convention
        # (C:={E,PV,SST,VIP}); they live in the selection namespace bound by
        # L[C], not the structural-object namespace, so the reserved-letter
        # rule for object names does not apply to them.
        raw_p = props.get("P")
        proportions = None
        if raw_p is not None:
            if not (isinstance(raw_p, Mapping) and "map" in raw_p):
                raise TFNEInvalidProportion(
                    f"{where}: P must be a {{type: p, ...}} map")
            proportions = {k: float(v) for k, v in raw_p["map"].items()}
        raw_n = props.get("N")
        total: Any = None
        if raw_n is not None:
            if isinstance(raw_n, Mapping) and "map" in raw_n:
                total = {k: int(v) for k, v in raw_n["map"].items()}
            elif isinstance(raw_n, (int, float)):
                total = int(raw_n)
            else:
                raise TFNEInvalidProportion(
                    f"{where}: N must be an integer or map")
            counts, derived = _allocate_counts(ctypes, proportions, total, where)
        raw_g = props.get("G")
        geometry: dict[str, Any] = {}
        if raw_g is not None:
            if not (isinstance(raw_g, Mapping) and "dict" in raw_g):
                raise TFNEInvalidProportion(
                    f"{where}: G must be a [k = v; ...] body")
            geometry = dict(raw_g["dict"])
            _validate_geometry_body(where, geometry)
        model = props.get("model", DEFAULT_MODEL)
        if isinstance(model, Mapping):
            raise TFNEInvalidProportion(f"{where}: model must be a name")
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
            return self._apply_declared_frontiers(
                path, _Expansion(members=(path,), fin=inner.fin,
                                 fout=inner.fout))
        if isinstance(node, GroupStmts):
            gid = f"g{self.group_counter}"
            self.group_counter += 1
            path = f"{scope}.{gid}" if scope else gid
            self._add_node(NodeRecord(path=path, name=gid, kind="composite",
                                      parent=scope or None))
            members: list[str] = []
            fin: list[str] = []
            fout: list[str] = []
            self._brace_atomic += 1
            try:
                for stmt in node.body:
                    part = self.expand(stmt, path)
                    members.extend(part.members)
                    fin.extend(part.fin)
                    fout.extend(part.fout)
            finally:
                self._brace_atomic -= 1
            if not members:
                raise TFNEError(f"empty composite group at {path!r}")
            return self._apply_declared_frontiers(
                path, _Expansion(members=(path,), fin=tuple(fin),
                                 fout=tuple(fout)))
        if isinstance(node, Ordered):
            left = self.expand(node.left, scope)
            right = self.expand(node.right, scope)
            if node.rule is not None and left.fout and right.fin:
                # S10: a rule binds its own adjacency only -- the left
                # operand's out frontier to the right operand's in frontier.
                # Operands accumulated earlier in the chain are not reachable
                # here, so A O[k] B O[j] C yields k(A,B) and j(B,C), never A>C.
                # An atomicly-dropped side leaves an empty frontier: binding
                # nothing is recorded (fin/fout are never empty otherwise).
                self._record_rule(node, left, right)
            return _Expansion(members=left.members + right.members,
                              fin=left.fin, fout=right.fout)
        if isinstance(node, Cross):
            left = self.expand(node.left, scope)
            right = self.expand(node.right, scope)
            if node.rule is not None:
                # S11: X is not globally associative. An ungrouped chain
                # under one rule (`A X[k] B X[k] C`) requires grouping
                # unless the rule declares an associative policy
                # (`associative = true`). Braces, parens-free groups, and
                # named-definition boundaries all count as grouping: only
                # direct syntactic nesting of bare Cross nodes refuses.
                for nested in (node.left, node.right):
                    if (isinstance(nested, Cross)
                            and nested.rule == node.rule):
                        ruledef = self.program.rules.get(node.rule)
                        policy = ""
                        if ruledef is not None:
                            policy = str(ruledef.params.get(
                                "associative", "")).lower()
                        if policy != "true":
                            raise TFNEAmbiguousExpansion(
                                f"E_AMBIGUOUS_EXPANSION: ungrouped "
                                f"{_emit_expr(nested)} X[{node.rule}] ... "
                                f"has no unique association; group it "
                                f"({{...}}) or declare "
                                f"`associative = true` on X[{node.rule}]")
                # X is nonordered: it relates its operands rather than an
                # adjacency, and binds them whole. Per-operand endpoint
                # selection inside a body is $L/$R (S12).
                self._record_rule(node, left, right)
            return _Expansion(members=left.members + right.members,
                              fin=left.fin + right.fin,
                              fout=left.fout + right.fout)
        if isinstance(node, Project):
            left = self._endpoint_members(node.left, scope, "projection")
            right = self._endpoint_members(node.right, scope, "projection")
            if not left or not right:
                # Atomicly dropped: an invalid resolved projection
                # contributes nothing (S14) — no relation, no members.
                return _Expansion(members=(), fin=(), fout=())
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
            if not left or not right:
                # Atomicly dropped exclusion records nothing: with no routes
                # it cannot match, and must not trip E_EXCLUSION_UNKNOWN.
                return _Expansion(members=(), fin=(), fout=())
            rec = self._record_relation(
                "exclusion", "direct", None, node.direction,
                _emit_expr(node.left), _emit_expr(node.right), left, right)
            self.exclusions.append(rec)
            return _Expansion(members=tuple(left) + tuple(right),
                              fin=tuple(left), fout=tuple(right))
        if isinstance(node, Replicate):
            if node.rel is None:
                return self._expand_replicate(node, scope)
            return self._expand_replicate_joined(node, scope)
        if isinstance(node, PrefixApply):
            # S6: `O[k](SEG^n)` chains rule k over the instances as ordered
            # adjacencies: SEG.1 O[k] SEG.2 ... O[k] SEG.n. Degenerate n=1
            # yields the instance with no adjacencies. Each pair expands
            # through the ordinary rule path, so body rules apply per pair
            # with the instances as operands.
            rep = self._expand_replicate(node.target, scope)
            paths = list(rep.members)
            for pre, post in zip(paths, paths[1:]):
                self._record_rule(
                    Ordered(left=Ref(segments=(pre,)),
                            right=Ref(segments=(post,)),
                            rule=node.rule),
                    _leaf_expansion(pre), _leaf_expansion(post))
            return _Expansion(members=tuple(paths),
                              fin=(paths[0],), fout=(paths[-1],))
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
            return self._apply_declared_frontiers(
                target, _Expansion(members=(target,), fin=inner.fin,
                                   fout=inner.fout))
        # implicit degenerate leaf (valid at cardinality one)
        _check_object_name(name)
        rec = self._object_props({}, where=name, implicit=True)
        rec.path = target
        rec.name = name
        rec.parent = scope or None
        self._add_node(rec)
        return _leaf_expansion(target)

    def _replicate_instance(self, name: str, i: int, scope: str) -> str:
        """Create one indexed instance ``<scope>.<name>.<i>`` (1-based)."""
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
        return inst

    def _expand_replicate(self, node: Replicate,
                          scope: str) -> _Expansion:
        name = node.atom.segments[0]
        paths: list[str] = []
        # S6: A^n = {A.1 ... A.n}. Instance indices are 1-based; the path
        # A.1 names the first instance, not the second.
        for i in range(1, node.n + 1):
            paths.append(self._replicate_instance(name, i, scope))
        # S6: instances carry no connectivity and no order, so the whole set
        # is both frontiers. Replication behaviour is otherwise unchanged.
        return _flat_expansion(paths)

    def _expand_replicate_joined(self, node: Replicate,
                                 scope: str) -> _Expansion:
        """S6.1: `A^{nX}` / `A^{nO}` join instances under that operator.

        Each instance is one member with itself as frontier (as in
        `_expand_replicate`); the join then follows bare-operator frontier
        semantics. X exposes every instance on both frontiers; O chains them,
        exposing the first as `fin` and the last as `fout`, so a rule applied
        to the composite reaches exactly the chain ends.
        """
        name = node.atom.segments[0]
        paths = [self._replicate_instance(name, i, scope)
                 for i in range(1, node.n + 1)]
        if node.rel == "X":
            return _Expansion(members=tuple(paths),
                              fin=tuple(paths), fout=tuple(paths))
        assert node.rel == "O"
        return _Expansion(members=tuple(paths),
                          fin=(paths[0],), fout=(paths[-1],))

    def _endpoint_members(self, node: Any, scope: str,
                          role: str) -> list[str]:
        members = self.expand_endpoint(node, scope)
        if not members:
            # S14/S25 atomicity: inside a brace-group statement, a resolved
            # projection with no members contributes nothing instead of
            # aborting the composite. Outside, it stays a hard error —
            # an empty endpoint is never silently valid at top level.
            if self._brace_atomic:
                return []
            raise TFNEAddressUnknown(f"{role} endpoint expands to nothing")
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
            if self._brace_atomic:
                # S14/S25 atomicity: an endpoint resolving to nothing drops
                # its statement. A contradictory selection against an
                # existing object still raises (see _find_paths): absence
                # is lenient, contradiction is not.
                return []
            raise TFNEAddressUnknown(
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
    def _remainder_of(self, path: str) -> str:
        """Immediate-member remainder of `path` under its resolved parent."""
        rec = self.nodes.get(path)
        parent = getattr(rec, "parent", None) if rec is not None else None
        if parent:
            assert path.startswith(parent + ".")
            return path[len(parent) + 1:]
        return path.rpartition(".")[2]

    def _eval_rule_endpoint(self, ep: RuleEndpoint,
                            left: _Expansion, right: _Expansion,
                            rule: str, position: str) -> list[str]:
        """Resolve one S12 endpoint to member scopes (`tfne/2` S12).

        `$L` / `$R` address the syntactic operands explicitly: bare they
        denote the whole operand, `.out` / `.in` narrow to the resolved
        interface (declared-or-derived, never bypassed). Plain names and
        `{...}` collections belong to the endpoint's own statement position
        (left position -> left operand) and address immediate members
        within it; anything absent or outside is `E_ADDRESS_UNKNOWN`.
        """
        if ep.form == "meta":
            exp = left if ep.head == "L" else right
            if not ep.tail:
                return list(exp.members)
            if ep.tail == ("out",):
                return list(exp.fout)
            return list(exp.fin)
        side = left if position == "L" else right
        refs = (ep.head,) if ep.form == "ref" else ep.members
        return self._resolve_operand_refs(refs, side, rule)

    def _resolve_operand_refs(self, refs: Sequence[str],
                              side: _Expansion, rule: str) -> list[str]:
        """Resolve member refs within one operand's member scopes.

        A ref matches an immediate member (child remainder, replica-aware,
        or the member scope itself) or an exact node path at/under a member
        scope. Names absent from the operand, or paths outside it, are
        refused: rule expansion operates on resolved interfaces and never
        reconnects arbitrary descendants.
        """
        members = list(side.members)
        out: list[str] = []
        for ref in refs:
            hits: list[str] = []
            if ref in self.nodes and self._within_members(ref, members):
                hits.append(ref)
            else:
                for scope in members:
                    for full, rem in _scope_children(self.nodes, scope):
                        if rem == ref and full not in hits:
                            hits.append(full)
                    if scope in self.nodes and self._remainder_of(scope) == ref \
                            and scope not in hits:
                        hits.append(scope)
            if not hits:
                raise TFNEAddressUnknown(
                    f"E_ADDRESS_UNKNOWN: rule {rule!r} names {ref!r}, "
                    f"which is absent from its operand "
                    f"{[str(m) for m in members]!r}")
            out.extend(hits)
        seen: set[str] = set()
        ordered = [h for h in out if not (h in seen or seen.add(h))]
        return ordered

    @staticmethod
    def _within_members(path: str, members: Sequence[str]) -> bool:
        return any(path == m or path.startswith(m + ".") for m in members)

    def _record_rule(self, node: Any, left: _Expansion,
                     right: _Expansion) -> RelationRecord:
        assert isinstance(node, (Ordered, Cross))
        assert node.rule is not None
        if node.rule not in self.program.rules:
            raise TFNEError(f"connection rule {node.rule!r} undefined")
        ruledef = self.program.rules[node.rule]
        kind = "O" if isinstance(node, Ordered) else "X"
        if getattr(ruledef, "body", ()):
            return self._record_rule_body(node, ruledef, kind, left, right)
        # Flat (params-only) rules keep their exact legacy behavior: O binds
        # fout->fin, X binds whole members.
        pre = list(left.fout) if kind == "O" else list(left.members)
        post = list(right.fin) if kind == "O" else list(right.members)
        direction = ruledef.params.get("direction")
        if direction not in (">", "<", "<>"):
            raise TFNEError(
                f"rule {node.rule!r} must declare direction ('>', '<', '<>'); "
                "the algebra fixes no default direction for O[k]/X[k]")
        group = None
        if direction == "<>":
            group = (f"rule:{kind}[{node.rule}]:{_emit_expr(node.left)}<>"
                     f"{_emit_expr(node.right)}")
        for split in self._split_direction(direction):
            self._record_relation("rule", kind, node.rule, split,
                                  _emit_expr(node.left),
                                  _emit_expr(node.right), pre, post,
                                  group=group)
        return self.relations[-1]

    def _record_rule_body(self, node: Any, ruledef: RuleDef, kind: str,
                          left: _Expansion,
                          right: _Expansion) -> RelationRecord:
        """Expand S12 body statements against the syntactic operands.

        Each statement's endpoints evaluate with `$L`/`$R` bound to the
        operand expansions and plain names resolved within their own side.
        Direction lives in the statements (bodies routinely mix `>` and
        `<`), so a rule-level `direction` alongside a body is refused
        rather than silently ignored.
        """
        if "direction" in ruledef.params:
            raise TFNEError(
                f"rule {ruledef.name!r} declares both a body and a "
                f"top-level direction; direction lives in the body "
                f"statements")
        for stmt in ruledef.body:
            pre = self._eval_rule_endpoint(stmt.left, left, right,
                                           ruledef.name, "L")
            post = self._eval_rule_endpoint(stmt.right, left, right,
                                            ruledef.name, "R")
            label = _emit_rulestmt(stmt)
            group = None
            if stmt.direction == "<>":
                group = (f"rule:{kind}[{ruledef.name}]:{label}")
            for split in self._split_direction(stmt.direction):
                self._record_relation("rule", kind, ruledef.name, split,
                                      _emit_rule_endpoint(stmt.left),
                                      _emit_rule_endpoint(stmt.right),
                                      pre, post,
                                      group=group,
                                      mechanism=stmt.mechanism)
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
                         group: Optional[str] = None,
                         mechanism: Optional[str] = None) -> RelationRecord:
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
                             group=group, mechanism=mechanism)
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
    declared_ranks = _declared_ranks(resolver.nodes, program.orders)
    _verify_frontier_declarations(
        resolver.nodes, program.frontiers, resolver.applied_frontiers)
    _check_projection_redundancy(resolver)
    _nodes_for_key = dict(resolver.nodes)
    return ExplicitModel(nodes=dict(resolver.nodes),
                         # `tfne/2` S20: source declaration order does not
                         # determine realization indexing. Sorting here rather
                         # than at each consumer keeps realization and
                         # `to_neuronal_tensor` on one order; if they diverged,
                         # the specs would address the wrong neurons while
                         # every count still matched.
                         order=tuple(sorted(
                             resolver.order,
                             key=lambda p: _ordering_key(
                                 p, declared_ranks, _nodes_for_key))),
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
        if rel.mechanism is not None:
            # S12 per-statement mechanism overrides the rule default for
            # this relation's projections only.
            params["mechanism"] = rel.mechanism
        if rel.direction not in (">", "<"):
            raise TFNEError(
                f"relation {rel.key!r} has unresolved direction "
                f"{rel.direction!r}")
        # No silent parameter substitution. `delay` would change the dynamics
        # and nothing consumes it -- neither `compile_connection_rules` nor
        # the edge compiler carries a delay -- so accepting it would be a
        # claim the runtime cannot keep. Refuse instead of dropping.
        #
        # `plasticity` is deliberately not refused: it names a rule identity
        # for the separate registrable HDP surface rather than a connection
        # parameter, and is preserved as inspectable provenance in
        # `rule_params` / `relation_origin`. Declared and recorded is not the
        # same as declared and discarded.
        if params.get("delay") is not None:
            raise TFNEError(
                f"E_PARAM_UNSUPPORTED: rule {rel.rule!r} declares a delay, "
                f"which no JaxFNE execution path consumes. Remove it or "
                f"extend the compiler; it will not be silently ignored."
            )
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
            raise TFNEExclusionUnknown(
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
                         "params": dict(
                             _rule_connection_params(
                                 rule_lookup.get(rel.rule))
                             if rel.rule else _rule_connection_params(None),
                             **({"mechanism": rel.mechanism}
                                if rel.mechanism is not None else {}))}
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
        # The authoritative resolved connectivity: exactly the specs these
        # edges were compiled from, declared parameters included. Execution
        # reads these rather than re-deriving connectivity, so there is one
        # resolved representation instead of two that can disagree
        # (:func:`to_configuration`).
        "connection_specs": [dict(c) for c in connections],
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

def _tfne_area_of(path: str) -> str:
    """Top-level scope: the area a realized leaf belongs to."""
    return path.split(".")[0]


def _tfne_layer_of(path: str) -> str:
    """Nearest named scope below the area root: for V1.L1 the leaf itself
    (laminar layer L1); for V1.g0.L1 the group g0."""
    segments = path.split(".")
    if len(segments) >= 2:
        return segments[1]
    return segments[0]


def _tfne_geometry_domains(
    explicit: ExplicitModel,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Per-(area, layer) declared geometry domains for construction.

    Reads each realized leaf's `G` body (already validated relative [0,1]
    at resolve time) and groups by the same (area, layer) mapping
    :func:`to_neuronal_tensor` samples positions under, so the construction
    stage can honour the declaration at the block it actually samples. An
    undeclared axis is unconstrained: the block keeps its historical extent
    for it. Blocks with no non-full declaration are omitted, so
    construction without a sub-range declaration takes the historical code
    path unchanged (bit-identical).

    One sampled block cannot honour two different declared domains on one
    axis, so that is refused (`E_GEOMETRY_AMBIGUOUS`) rather than averaged;
    a single declaration wins for the whole block (undeclared leaves
    inherit the block domain).
    """
    declared: dict[tuple[str, str], dict[str, set[tuple[float, float]]]] = {}
    for path in explicit.order:
        rec = explicit.nodes[path]
        if rec.kind != "object":
            continue
        key = (_tfne_area_of(path), _tfne_layer_of(path))
        geo = rec.geometry or {}
        for axis in ("x", "y", "z"):
            if f"{axis}_range" in geo:
                lo, hi = geo[f"{axis}_range"]
                val: tuple[float, float] | None = (float(lo), float(hi))
            else:
                lo, hi = geo.get(f"{axis}0"), geo.get(f"{axis}1")
                val = None if lo is None or hi is None else (float(lo), float(hi))
            if val is not None:
                declared.setdefault(key, {}).setdefault(axis, set()).add(val)
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for (area, layer), axes in declared.items():
        block: dict[str, list[float]] = {}
        for axis, vals in axes.items():
            if len(vals) > 1:
                raise TFNEError(
                    f"E_GEOMETRY_AMBIGUOUS: leaves in layer "
                    f"{area!r}/{layer!r} declare different {axis} domains "
                    f"({sorted(vals)}); one sampled block cannot honour "
                    f"both exactly")
            (lo, hi), = vals
            if (lo, hi) != (0.0, 1.0):
                block[axis] = [lo, hi]
        if block:
            out.setdefault(area, {})[layer] = block
    return out


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
                                   NeuronType, StaticParams)

    def _connection_static(rel: RelationRecord) -> Any:
        """Kinetics for one relation's connections (TFNE2-07/PARAM-03).

        The declared mechanism resolves to canonical or custom kinetics;
        unresolvable names fail closed here — where invention would occur —
        rather than inheriting the 0.1 placeholder. `realize()` is
        unaffected: identity transfers without kinetics.
        """
        resolved = _resolve_relation_mechanism(explicit, rel)
        reversal = ({resolved.identity: resolved.reversal_mV}
                    if resolved.reversal_mV is not None else {})
        return StaticParams(dT_ms=resolved.tau_ms,
                            reversal_potentials_mV=reversal)

    leaves = [p for p in explicit.order
              if explicit.nodes[p].kind == "object"]

    def top(path: str) -> str:
        return _tfne_area_of(path)

    def area_of(path: str) -> str:
        return top(path)

    def layer_of(path: str) -> str:
        # Nearest named scope strictly below the area root: for V1.L1 this is
        # the leaf itself (laminar layer L1); for V1.g0.L1 the group g0.
        return _tfne_layer_of(path)

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
                                        explicit, rel),
                                    static=_connection_static(rel)))
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
                                mechanism=_relation_mechanism(explicit, rel),
                                static=_connection_static(rel)))
    tensor = NeuronalTensor(areas=tuple(built_areas),
                            area_connections=tuple(area_conns),
                            name="tfne",
                            provenance={"tfne_digest": explicit.digest,
                                        "tfne_normalization":
                                            explicit.normalization})
    return tensor


def to_configuration(realization: Realization, *,
                     duration_ms: float = 1000.0,
                     dt_ms: float = 0.1,
                     emitter: str = DEFAULT_MODEL,
                     dtype: str = "float32"):
    """Build an executable ``Configuration`` from a realization.

    Execution consumes the realization's own resolved connectivity
    (``I["connection_specs"]``) rather than re-deriving it, so the executed
    model carries the parameters TFNE declared::

        TFNE -> resolve -> explicit model -> realize -> (s, h0, I)
                                                     -> to_configuration -> construct

    ``to_neuronal_tensor`` supplies the population structure only. Its
    ``InterConnection``/``AreaConnection`` entries cannot express a declared
    weight, probability or delay, so the connectivity they imply is replaced
    here by the realized specs. Going through the tensor for structure is what
    keeps neuron identity aligned: the constructed neuron table is in the same
    order as ``I["neuron_paths"]``, which lets the specs address neurons by
    realized id. That alignment is an invariant, not a coincidence, and is
    gated by ``tests/test_tfne_execution.py``.

    A rule's declared weight carries its own polarity. It is passed as a
    magnitude plus an explicit ``sign``, because an unsigned rule would
    otherwise inherit the presynaptic neuron's intrinsic sign and silently
    overrule what the specification declared.

    Declared geometry reaches execution as fractions of each sampled
    block's extent: per-(area, layer) sub-range domains are resolved by
    :func:`_tfne_geometry_domains` and recorded in
    ``metadata["tfne_geometry"]`` (relative, ``value_tag="relative"``),
    which the construction stage samples instead of the full block.
    Layers without a sub-range declaration take the historical path
    unchanged.
    """
    from dataclasses import replace as _replace

    from .neuronal_tensor import (
        StaticParams as _StaticParams,
        neuronal_tensor_to_configuration,
    )

    explicit = realization.explicit
    if explicit is None:
        raise TFNEError("realization carries no explicit model to construct from")
    tensor = to_neuronal_tensor(explicit)
    cfg = neuronal_tensor_to_configuration(
        tensor, seed=int(realization.s["seed"]), duration_ms=duration_ms,
        dt_ms=dt_ms, emitter=emitter, dtype=dtype)

    # Synaptic kinetics are not a TFNE-declared parameter: the realized
    # mechanism table carries `tau_ms: None` and `declared_not_simulated`.
    # Inherit whatever the structural bridge declared for each mechanism kind
    # (it derives tau from the connection's own `static.dT_ms`) rather than
    # inventing a value here. In particular it must NOT come from `dt_ms`:
    # tying synaptic decay to the integration timestep would make a refined
    # timestep silently change the synapse model, so the model would not
    # converge under dt-refinement.
    bridge_tau: dict[str, float] = {}
    for m in cfg.metadata.get("circuit", {}).get("mechanisms", []):
        kind = m.get("kind")
        tau = m.get("params", {}).get("tau_ms")
        if kind is not None and tau is not None:
            bridge_tau.setdefault(str(kind), float(tau))
    default_tau = float(_StaticParams().dT_ms)

    mechanisms: list[dict[str, Any]] = []
    declared: dict[str, str] = {}
    rules: list[dict[str, Any]] = []
    for spec in realization.I["connection_specs"]:
        mech = str(spec["mechanism"])
        name = declared.get(mech)
        if name is None:
            name = f"{mech}__tfne__{len(declared)}"
            declared[mech] = name
            mechanisms.append({
                "name": name, "kind": mech,
                "params": {"tau_ms": bridge_tau.get(mech, default_tau)},
            })
        weight = float(spec["weight"])
        rules.append({
            "name": spec["name"],
            "source": {"ids": list(spec["source"]["ids"])},
            "target": {"ids": list(spec["target"]["ids"])},
            "probability": float(spec["probability"]),
            "weight": abs(weight),
            "sign": "excitatory" if weight >= 0.0 else "inhibitory",
            "mechanism": name,
            "status": "declared_not_compiled",
        })

    metadata = {k: v for k, v in cfg.metadata.items()}
    circuit = {k: v for k, v in metadata.get("circuit", {}).items()}
    circuit["connections"] = rules
    circuit["mechanisms"] = mechanisms
    metadata["circuit"] = circuit
    metadata["tfne_digest"] = explicit.digest
    geo_domains = _tfne_geometry_domains(explicit)
    if geo_domains:
        metadata["tfne_geometry"] = {
            "value_tag": "relative",
            "declared": {
                p: dict(rec.geometry) for p, rec in explicit.nodes.items()
                if rec.kind == "object" and rec.geometry},
            "domains": geo_domains,
        }
    return _replace(cfg, metadata=metadata)


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


def _check_projection_redundancy(resolver: Any) -> None:
    """Reject explicit projections identical to rule output (S13/S25).

    Projection identity is `(src, dst, mechanism)` over realized leaves
    with direction normalized — leaf-level, because a coarser explicit
    scope over an already-generated leaf route would double-count edges
    while looking structurally distinct. Any overlap between an explicit
    (bare) projection and canonically generated rule output is refused:
    S14 additionally requires explicit additions disjoint from `G_0`, so
    partial overlap cannot be a valid addition either. Mechanism
    comparison is textual (declared names): resolving here would
    front-load kinetics refusal into `resolve()`.
    """
    leaves = [p for p in resolver.order
              if resolver.nodes[p].kind == "object"]

    def leaf_routes(rel: Any) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for pre, post in _scope_pairs(rel):
            pres = [lf for lf in leaves
                    if lf == pre or lf.startswith(pre + ".")]
            posts = [lf for lf in leaves
                     if lf == post or lf.startswith(post + ".")]
            out.extend((a, b) for a in pres for b in posts)
        return out

    def effective_mech(rel: Any) -> str:
        if rel.mechanism is not None:
            return str(rel.mechanism)
        if rel.rule is None:
            return DIRECT_MECHANISM
        ruledef = resolver.program.rules.get(rel.rule)
        if ruledef is None:
            return DIRECT_MECHANISM
        raw = ruledef.params.get("mechanism", DIRECT_MECHANISM)
        return raw if isinstance(raw, str) else DIRECT_MECHANISM

    generated: set[tuple[str, str, str]] = set()
    for rel in resolver.relations:
        if rel.form != "rule":
            continue
        mech = effective_mech(rel)
        for route in leaf_routes(rel):
            generated.add((route[0], route[1], mech))
    for rel in resolver.relations:
        if rel.form != "projection":
            continue
        for route in leaf_routes(rel):
            identity = (route[0], route[1], DIRECT_MECHANISM)
            if identity in generated:
                raise TFNEProjectionRedundant(
                    f"E_PROJECTION_REDUNDANT: explicit projection "
                    f"{rel.key!r} reproduces the canonically generated "
                    f"({identity[0]!r}, {identity[1]!r}, "
                    f"{identity[2]!r}); identical output is invalid")


def _relation_mechanism(explicit: ExplicitModel, rel: RelationRecord) -> str:
    if rel.mechanism is not None:
        return str(rel.mechanism)
    if rel.rule is None:
        return DIRECT_MECHANISM
    params = explicit.rule_params.get(rel.rule, {})
    mech = params.get("mechanism", DIRECT_MECHANISM)
    if isinstance(mech, Mapping):
        raise TFNEError("mechanism must be a name")
    return str(mech)


@dataclass(frozen=True)
class ResolvedMechanism:
    """One declared mechanism traced to executable kinetics (S25/TFNE2-07).

    `declared` is the name as written; `identity` the canonical name it
    resolves to (itself for canonical/custom); `status` one of the
    MECHANISM_* classes; `tau_ms`/`reversal_mV` the kernel-consumed
    kinetics (`reversal_mV` may be None: metadata only); `sign` the
    canonical sign (+1/-1) or None when the name carries none (custom and
    internal mechanisms inherit E/I from the source cell type downstream).
    """
    declared: str
    identity: str
    status: str
    tau_ms: float
    reversal_mV: Optional[float]
    sign: Optional[int]


def _canonical_receptor_specs() -> Mapping[str, Any]:
    from .emitters import standard_receptor_specs
    return standard_receptor_specs()


def resolve_mechanism(name: Optional[str],
                      rule_params: Optional[Mapping[str, Any]] = None
                      ) -> ResolvedMechanism:
    """Resolve a declared mechanism name to executable kinetics.

    `tfne/2` S13/S25: an executable projection resolves its mechanism from
    its generating rule (or statement) or its explicit specification.
    Canonical names resolve to the canonical table; a non-canonical name
    needs a sufficient executable definition (a finite positive `tau_ms`
    in the rule params); anything else fails closed. Absent mechanism
    means direct coupling (`tfne_direct`), matching the realization path's
    long-standing default — silence is not read as a receptor claim.
    """
    params = dict(rule_params) if rule_params is not None else {}
    if name is None:
        return ResolvedMechanism(
            declared=DIRECT_MECHANISM, identity=DIRECT_MECHANISM,
            status=MECHANISM_CUSTOM_DEFINED, tau_ms=DIRECT_MECHANISM_TAU_MS,
            reversal_mV=None, sign=None)
    if name == DIRECT_MECHANISM:
        return resolve_mechanism(None, rule_params)
    if name.startswith(MECHANISM_INTERNAL_PREFIX):
        raise TFNEMechanismNotPermitted(
            f"E_MECHANISM_NOT_PERMITTED: mechanism {name!r} lives in the "
            f"compiler-internal {MECHANISM_INTERNAL_PREFIX!r} namespace; "
            f"declare a vocabulary name or a sufficient definition instead")
    canonical = _canonical_receptor_specs()
    if name in canonical:
        spec = canonical[name]
        tau = params.get("tau_ms")
        if tau is not None:
            try:
                tau_f = float(tau)
            except (TypeError, ValueError):
                raise TFNEMechanismNotPermitted(
                    f"E_MECHANISM_NOT_PERMITTED: mechanism {name!r} "
                    f"declares a non-numeric tau_ms={tau!r}; an executable "
                    f"definition needs a finite positive number")
            if tau_f != float(spec.tau_ms):
                raise TFNEMechanismNotPermitted(
                    f"E_MECHANISM_NOT_PERMITTED: mechanism {name!r} is "
                    f"canonical with tau_ms={spec.tau_ms}, but the rule "
                    f"declares tau_ms={tau!r}; a canonical identity with "
                    f"non-canonical kinetics is not permitted")
        return ResolvedMechanism(
            declared=name, identity=name, status=MECHANISM_CANONICAL,
            tau_ms=float(spec.tau_ms), reversal_mV=spec.reversal_mV,
            sign=int(spec.sign))
    tau = params.get("tau_ms")
    if tau is None or isinstance(tau, bool):
        raise TFNEMechanismUnresolved(
            f"E_MECHANISM_UNRESOLVED: mechanism {name!r} is not a canonical "
            f"receptor {sorted(canonical)} and declares no sufficient "
            f"executable definition (a finite positive `tau_ms`); refusing "
            f"rather than inventing kinetics")
    try:
        tau_f = float(tau)
    except (TypeError, ValueError):
        raise TFNEMechanismUnresolved(
            f"E_MECHANISM_UNRESOLVED: mechanism {name!r} declares a "
            f"non-numeric tau_ms={tau!r}; refusing rather than inventing "
            f"kinetics")
    if not bool(np.isfinite(tau_f)) or not tau_f > 0:
        raise TFNEMechanismNotPermitted(
            f"E_MECHANISM_NOT_PERMITTED: mechanism {name!r} declares an "
            f"inadmissible tau_ms={tau!r}; an executable definition needs "
            f"a finite positive number")
    return ResolvedMechanism(
        declared=name, identity=name, status=MECHANISM_CUSTOM_DEFINED,
        tau_ms=tau_f, reversal_mV=None, sign=None)


def _resolve_relation_mechanism(explicit: ExplicitModel,
                               rel: RelationRecord) -> ResolvedMechanism:
    """Resolve one relation's mechanism: statement `[mech=]` wins over the
    rule default; the rule params supply any custom definition."""
    rule_params = explicit.rule_params.get(rel.rule) if rel.rule else None
    if rel.mechanism is not None:
        return resolve_mechanism(rel.mechanism, rule_params)
    if rel.rule is None:
        return resolve_mechanism(None, None)
    mech = (rule_params or {}).get("mechanism")
    if isinstance(mech, Mapping):
        raise TFNEError("mechanism must be a name")
    return resolve_mechanism(mech, rule_params)


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
