"""The references page and the docstrings cite each other.

`docs/reference/references.md` lists, per published source, the result jaxfne implements and the
functions that implement it. The page and the docstrings are two copies of one claim, so this
module holds them to each other:

* every table has one of three headers (method, replication, background);
* every DOI link is well formed and its text and target name the same DOI;
* every method and replication row carries exactly one DOI, and every row names a result or a
  relation;
* every function a method row lists exists and cites that row's DOI in its docstring;
* every DOI cited anywhere in `jaxfne/` is on the page;
* every DOI a public docstring cites is on a method row that lists that function.

Whether a DOI is registered needs the network, which the suite does not use; the page states
the date its DOIs were resolved.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reference" / "references.md"
PACKAGE = ROOT / "jaxfne"

DOI_SHAPE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
#: The target may hold one level of balanced parentheses, as a SICI-style DOI needs.
PAGE_LINK = re.compile(r"\[doi:([^\]]+)\]\(https://doi\.org/((?:[^()\s]|\([^()\s]*\))+)\)")
#: A DOI in source text; trailing punctuation and an unbalanced ")" are trimmed by `_norm`.
SOURCE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
CODE_SPAN = re.compile(r"`(jaxfne\.[A-Za-z0-9_.]+)`")

METHOD = ("Reference", "Result implemented", "Functions")
REPLICATION = ("Reference", "Result compared", "Where")
BACKGROUND = ("Reference", "Relation to jaxfne")


def _norm(doi: str) -> str:
    doi = doi.rstrip(".,;")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi.lower()


def _rows():
    """``(header, cells)`` for every body row of every table on the page."""
    rows, header = [], None
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = tuple(c.strip() for c in line.strip("|").split("|"))
        if cells[0] == "Reference":
            assert cells in (METHOD, REPLICATION, BACKGROUND), f"unknown table header: {line}"
            header = cells
            continue
        assert header is not None, f"a row sits under no header: {line[:80]}"
        assert len(cells) == len(header), f"row width differs from its header: {line[:80]}"
        rows.append((header, cells))
    return rows


def _row_dois(cell: str) -> list[str]:
    return [_norm(text) for text, _ in PAGE_LINK.findall(cell)]


def _resolve(dotted: str):
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def _method_rows():
    return [cells for header, cells in _rows() if header == METHOD]


def test_page_has_rows_under_each_header():
    headers = [header for header, _ in _rows()]
    for want in (METHOD, REPLICATION, BACKGROUND):
        assert headers.count(want) >= 1, want
    assert len(_method_rows()) >= 5, "the method tables must not empty out"


def test_doi_links_are_well_formed_and_self_consistent():
    links = PAGE_LINK.findall(PAGE.read_text(encoding="utf-8"))
    assert len(links) >= 50, "the page must carry its DOIs as links"
    for text, target in links:
        assert DOI_SHAPE.match(text), text
        assert text == target, (text, target)


@pytest.mark.parametrize("header, cells", _rows(), ids=lambda x: str(x)[:40])
def test_every_row_names_a_result_and_one_doi(header, cells):
    assert cells[1].strip(), f"a row names no result or relation: {cells[0][:60]}"
    n = len(_row_dois(cells[0]))
    if header in (METHOD, REPLICATION):
        assert n == 1, f"a method or replication row needs one DOI: {cells[0][:60]}"
    else:
        assert n <= 1, cells[0][:60]


def test_every_listed_function_exists_and_cites_its_row():
    checked = 0
    for reference, _, functions in _method_rows():
        (doi,) = _row_dois(reference)
        names = CODE_SPAN.findall(functions)
        assert names, f"a method row lists no function: {reference[:60]}"
        for name in names:
            doc = inspect.getdoc(_resolve(name)) or ""
            cited = {_norm(d) for d in SOURCE_DOI.findall(doc)}
            assert doi in cited, f"{name} does not cite doi:{doi}"
            checked += 1
    assert checked >= 8


def _package_dois() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for doi in SOURCE_DOI.findall(path.read_text(encoding="utf-8")):
            found.setdefault(_norm(doi), []).append(path.relative_to(ROOT).as_posix())
    return found


def test_every_doi_in_the_package_is_on_the_page():
    on_page = {_norm(text) for text, _ in PAGE_LINK.findall(PAGE.read_text(encoding="utf-8"))}
    found = _package_dois()
    assert found, "the package must cite at least one DOI"
    missing = {doi: files for doi, files in found.items() if doi not in on_page}
    assert not missing, f"DOIs cited in jaxfne/ but absent from the page: {missing}"


#: Modules whose public callables the walk below covers.
PUBLIC_MODULES = ("jaxfne", "jaxfne.emitters", "jaxfne.fields", "jaxfne.bridges", "jaxfne.presets")


def _public_callables():
    """``(dotted name, object)`` for public functions and classes, deduplicated by object."""
    seen = {}
    for mod_name in PUBLIC_MODULES:
        mod = importlib.import_module(mod_name)
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr, None)
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and id(obj) not in seen:
                seen[id(obj)] = (f"{mod_name}.{attr}", obj)
    return list(seen.values())


def test_every_citing_function_is_listed_on_its_row():
    listed = {}
    for reference, _, functions in _method_rows():
        (doi,) = _row_dois(reference)
        listed[doi] = {id(_resolve(n)) for n in CODE_SPAN.findall(functions)}
    walked = _public_callables()
    assert len(walked) > 100, "the walk must reach the public surface"
    citing = 0
    for name, obj in walked:
        doc = obj.__doc__ or ""
        for doi in {_norm(d) for d in SOURCE_DOI.findall(doc)}:
            citing += 1
            assert id(obj) in listed.get(doi, set()), f"{name} cites doi:{doi} but no row lists it"
    assert citing >= 8, "the walk must find the citing functions"
