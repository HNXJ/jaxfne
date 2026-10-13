"""Round-trip and loader-equivalence for the .lab bundle (scripts/lab_bundle.py)."""
import json

import pytest

import scripts.lab_balance as lab_balance
import scripts.lab_bundle as lab_bundle


def _write(lab_dir, name, data_bytes):
    path = lab_dir / name
    path.write_bytes(data_bytes)
    return path


def test_pack_unpack_round_trip_byte_exact(tmp_path):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    originals = {
        "a.json": json.dumps({"id": "a", "kind": "x"}, indent=1).encode("utf-8"),
        # CRLF line endings and a non-ASCII character must survive byte-exact.
        "b.json": '{"id": "b", "note": "caf\u00e9"}\r\n'.encode("utf-8"),
        "c.json": json.dumps({"id": "c", "generated": {"links": []}}).encode("utf-8"),
    }
    for name, blob in originals.items():
        _write(lab_dir, name, blob)

    lab_bundle.pack(lab_dir)
    for name in originals:
        (lab_dir / name).unlink()

    out = tmp_path / "restored"
    written = lab_bundle.unpack(out, lab_dir)
    assert sorted(p.name for p in written) == sorted(originals)
    for name, blob in originals.items():
        assert (out / name).read_bytes() == blob


def test_lab_balance_loads_same_nodes_from_bundle_and_directory(tmp_path):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    _write(lab_dir, "one.json",
           json.dumps({"id": "one", "generated": {"links": [{"to": "two"}]}}).encode("utf-8"))
    _write(lab_dir, "two.json",
           json.dumps({"id": "two", "note": "caf\u00e9"}).encode("utf-8"))

    nodes_dir, skipped_dir = lab_balance.load_nodes(lab_dir)
    assert set(nodes_dir) == {"one", "two"} and not skipped_dir

    lab_bundle.pack(lab_dir)
    nodes_bundle, skipped_bundle = lab_balance.load_nodes(lab_dir)
    assert nodes_bundle == nodes_dir
    assert skipped_bundle == skipped_dir


def test_build_edges_skips_a_prose_generated_field():
    nodes = {
        "one": {"id": "one", "generated": {"links": [{"to": "two", "relation": "uses"}]}},
        "two": {"id": "two", "generated": "2026-08-27 public stub"},
    }
    out_adj, in_adj, relations, _ = lab_balance.build_edges(nodes)
    assert out_adj["one"] == ["two"] and sum(relations.values()) == 1


def test_pack_refuses_to_shrink_the_bundle(tmp_path):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    node = _write(lab_dir, "a.json", b'{"id": "a"}')
    _write(lab_dir, "b.json", b'{"id": "b"}')
    lab_bundle.pack(lab_dir)
    node.unlink()
    before = (lab_dir / "bundle.json").read_bytes()
    with pytest.raises(ValueError, match="1 nodes over a bundle of 2"):
        lab_bundle.pack(lab_dir)
    assert (lab_dir / "bundle.json").read_bytes() == before


@pytest.mark.parametrize("name", ["../evil.json", "sub/d.json", "notes.txt", "bundle.json"])
def test_unpack_refuses_names_outside_the_directory(tmp_path, name):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "bundle.json").write_text(
        json.dumps({"format": lab_bundle.FORMAT, "files": {name: "{}"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not plain node file names"):
        lab_bundle.unpack(tmp_path / "out", lab_dir)
    assert not (tmp_path / "out").exists() and not (tmp_path / "evil.json").exists()


_MALFORMED = [
    json.dumps({"format": "lab-bundle-1", "files": ["a.json"]}).encode(),
    json.dumps({"format": "lab-bundle-1", "files": {"a.json": 42}}).encode(),
    json.dumps({}).encode(),
    b"\xff\xfe not utf-8",
]


@pytest.mark.parametrize("payload", _MALFORMED)
def test_malformed_bundle_is_reported_not_raised(tmp_path, payload):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "bundle.json").write_bytes(payload)
    nodes, skipped = lab_balance.load_nodes(lab_dir)
    assert nodes == {} and len(skipped) >= 1


@pytest.mark.parametrize("payload", _MALFORMED)
def test_malformed_bundle_is_refused_by_pack_and_unpack(tmp_path, payload):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "bundle.json").write_bytes(payload)
    with pytest.raises(ValueError):
        lab_bundle.unpack(tmp_path / "out", lab_dir)
    with pytest.raises(ValueError):
        lab_bundle.pack(lab_dir)


def test_a_non_utf8_loose_node_is_skipped(tmp_path):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    _write(lab_dir, "x.json", b"\xff\xfe")
    _write(lab_dir, "a.json", b'{"id": "a"}')
    nodes, skipped = lab_balance.load_nodes(lab_dir)
    assert set(nodes) == {"a"} and skipped == ["x.json (unreadable: UnicodeDecodeError)"]


def test_build_edges_skips_malformed_links():
    nodes = {
        "one": {"id": "one", "generated": {"links": "stub"}},
        "two": {"id": "two", "generated": {"links": ["one", {"to": "one"}]}},
    }
    out_adj, _, relations, _ = lab_balance.build_edges(nodes)
    assert dict(out_adj) == {"two": ["one"]} and sum(relations.values()) == 1


def test_tracked_bundle_holds_every_node():
    nodes, skipped = lab_balance.load_nodes(lab_bundle.DEFAULT_LAB)
    assert len(nodes) == 108 and not skipped
