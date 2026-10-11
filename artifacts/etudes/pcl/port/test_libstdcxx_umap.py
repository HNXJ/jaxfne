"""Unit tests for libstdcxx_umap: the libstdc++ unordered_map key order (port stage A).

Three checks. (1) Hand-derived cases of the rules in libstdcxx_umap.py. (2) The prime table against
the bytes of libstdc++-6.dll (skipped when the DLL is absent). (3) The g++ 13.3 libstdc++ output
(umap_gxx13_order.txt) on every key set in umap_gxx_keys.txt. The g++ program (umap_gxx_order.cpp)
was built and run by the dispatcher in WSL; this worker did not compile or run any C++.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pytest  # noqa: E402

import libstdcxx_umap as um  # noqa: E402

DLL = Path("C:/Program Files/Git/mingw64/bin/libstdc++-6.dll")
GXX_KEYS = HERE / "umap_gxx_keys.txt"  # one key set per line, in insertion order
GXX_ORDER = HERE / "umap_gxx13_order.txt"  # g++ 13.3 libstdc++ iteration order of each line


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def test_prime_table_shape_and_primality():
    assert len(um.PRIME_LIST) == 256
    assert all(a < b for a, b in zip(um.PRIME_LIST, um.PRIME_LIST[1:])), "table must increase"
    assert all(_is_prime(p) for p in um.PRIME_LIST), "every entry must be prime"
    assert um.PRIME_LIST[:7] == (2, 3, 5, 7, 11, 13, 17)
    assert um.PRIME_LIST[-1] == 4294967291


def test_prime_table_matches_libstdcxx_binary():
    if not DLL.is_file():
        pytest.skip(f"{DLL} not present on this machine")
    data = DLL.read_bytes()
    first = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pat = b"".join(struct.pack("<I", p) for p in first)
    k = data.find(pat)
    assert k >= 0, "prime-table head not found in the binary"
    run = []
    off = k
    while off + 4 <= len(data):
        v = struct.unpack_from("<I", data, off)[0]
        if not _is_prime(v) or (run and v <= run[-1]):
            break
        run.append(v)
        off += 4
    assert len(run) >= 256, f"fixture: the scan must find the 256-prime run, found {len(run)}"
    assert tuple(run[:256]) == um.PRIME_LIST


def test_policy_starts_with_no_allocation():
    pol = um.PrimeRehashPolicy()
    assert pol.next_resize == 0
    assert um.UnorderedMapKeys().bucket_count == 1


def test_next_bkt_known_values():
    pol = um.PrimeRehashPolicy()
    assert pol.next_bkt(12) == 13  # fast table, _M_next_bkt
    assert pol.next_resize == 13
    assert um.PrimeRehashPolicy().next_bkt(14) == 17  # first lower_bound from the 7th prime
    assert um.PrimeRehashPolicy().next_bkt(26) == 29


def test_bucket_count_sequence_for_distinct_inserts():
    expected = {1: 13, 13: 13, 14: 29, 29: 29, 30: 59, 59: 59, 60: 127, 127: 127,
                128: 257, 257: 257, 258: 541}
    for n, buckets in expected.items():
        m = um.UnorderedMapKeys()
        for k in range(n):
            m.insert(k)
        assert len(m) == n
        assert m.bucket_count == buckets, f"after {n} inserts"


def test_distinct_buckets_iterate_newest_first():
    # 0, 1, 2 land in buckets 0, 1, 2 of 13; each empty bucket goes to the front of the list.
    assert um.iteration_order([0, 1, 2]) == [2, 1, 0]


def test_same_bucket_newest_first():
    # 5, 18, 31 are all 5 mod 13: a non-empty bucket takes the node at the front of its run.
    assert um.iteration_order([5, 18, 31]) == [31, 18, 5]


def test_rehash_reverses_then_restores_order():
    # 1..13 in 13 buckets: list [13, ..., 1]. Key 14 rehashes to 29 buckets; re-inserting the
    # list in its order makes [1, ..., 13]; then 14 (empty bucket) goes to the front.
    assert um.iteration_order(range(1, 15)) == [14] + list(range(1, 14))


def test_insert_after_rehash_into_occupied_bucket():
    # After the rehash to 29 buckets the list is [14, 1, ..., 13]. 30 is 1 mod 29, so it goes in
    # front of the run of bucket 1, which holds only node 1.
    assert um.iteration_order(list(range(1, 15)) + [30]) == [14, 30] + list(range(1, 14))


def test_existing_key_is_not_reinserted():
    assert um.iteration_order([7, 7, 7]) == [7]


def test_random_keys_form_a_permutation_with_contiguous_buckets():
    keys = random.Random(0).sample(range(5200), 1300)
    m = um.UnorderedMapKeys()
    for k in keys:
        m.insert(k)
    assert len(keys) == 1300 and m.bucket_count == 2357, "fixture must reach the 2357-bucket table"
    order = m.order()
    assert sorted(order) == sorted(keys)
    buckets = [k % m.bucket_count for k in order]
    runs = 1 + sum(1 for a, b in zip(buckets, buckets[1:]) if a != b)
    assert runs == len(set(buckets)), "each bucket must form one contiguous run"


def test_iteration_order_matches_gxx13_on_every_key_set():
    keys_lines = GXX_KEYS.read_text(encoding="utf-8").splitlines()
    order_lines = GXX_ORDER.read_text(encoding="utf-8").splitlines()
    assert len(keys_lines) == 162 and len(order_lines) == 162, "fixture: 162 key sets"
    not_ascending = 0
    for n, (kl, ol) in enumerate(zip(keys_lines, order_lines)):
        keys = [int(t) for t in kl.split()]
        gxx = [int(t) for t in ol.split()]
        assert sorted(gxx) == sorted(keys), f"line {n}: g++ output is not a permutation of its keys"
        assert um.iteration_order(keys) == gxx, f"line {n}: emulator differs from g++ 13.3 libstdc++"
        not_ascending += gxx != sorted(keys)
    assert not_ascending > 150, f"fixture: orders must differ from ascending (found {not_ascending} of 162)"


def test_lateral_window_fixture_grows_to_10273_buckets():
    # Simple cell at (4, 4): 9x9 window minus its own column, 64 depths, keys in ascending index.
    x0, y0 = 4, 4
    keys = [x * 576 + y * 64 + z
            for x in range(9) for y in range(9) for z in range(64)
            if (x, y) != (x0, y0) and abs(x - x0) <= 4 and abs(y - y0) <= 4]
    assert len(keys) == 5120, "fixture must hold the full lateral window"
    m = um.UnorderedMapKeys()
    for k in keys[:5087]:
        m.insert(k)
    assert m.bucket_count == 5087
    m.insert(keys[5087])
    assert m.bucket_count == 10273
    order = um.iteration_order(keys)
    assert sorted(order) == keys
