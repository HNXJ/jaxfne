"""Exact emulation of the key order of a libstdc++ std::unordered_map<size_t, V> (port stage A).

Why: WeightMap keeps its lateral (<i>li.npy) and top-down (<i>tdi.npy) weights in a
std::unordered_map<size_t, double> (WeightMap.hpp:16) and saves the values in iteration order with
no keys (WeightMap.cpp:131-137). The file order is therefore the map's iteration order, which depends
on the bucket structure and on the insertion sequence. This module reproduces that order.

Provenance, stated so that each claim can be checked:
- PRIME_LIST is read from a binary, not from memory: the __prime_list of libstdc++-6.dll (Git for
  Windows, mingw64/bin, 2024), found by a read-only byte scan. It holds 256 primes; the C++ table
  keeps one more entry, a sentinel equal to the last prime. Entries are 4-byte unsigned long (LLP64).
  The Linux build that trained the network (LP64, 8-byte entries) is assumed to hold the same primes.
- The growth rule and the list rules are from memory of libstdc++ source, not read from source
  (no libstdc++ source on this machine). Names are the source names: hashtable_policy.h
  (_Prime_rehash_policy), hashtable_c++0x.cc (_M_next_bkt, _M_need_rehash), hashtable.h
  (_M_insert_unique_node, _M_insert_bucket_begin, _M_rehash_aux). Hand-derived cases in
  test_libstdcxx_umap.py check them, and so does the g++ check below.
- g++ check: umap_gxx_order.cpp (a program of ours, not the authors' code) was built and run by the
  dispatcher with g++ 13.3.0 libstdc++ under WSL. It inserts each line of umap_gxx_keys.txt (the
  lateral_keys and topdown_keys of all 81 simple positions, 162 lines) with operator[] into a fresh
  std::unordered_map<size_t, double> and prints the iteration order (umap_gxx13_order.txt).
  iteration_order equals that output on all 162 key sets
  (test_iteration_order_matches_gxx13_on_every_key_set). This worker did not compile or run C++.
  The check covers libstdc++ 13 on Linux (LP64) only, for these key counts.
- Declared assumption: the insertion order (ascending index, read from SpikingNetwork.cpp:409-442 and
  :368-400) is the order in which the authors' run inserted the keys. No log of the authors' run
  confirms it, and the g++ check does not test it.
- std::hash<size_t> is the identity, so the bucket is key % bucket_count (fast hash, no cached code).
- max_load_factor is 1.0. A default-constructed map has one bucket and _M_next_resize 0.
- The authors' maps only receive operator[] inserts of new keys, never erase, so no other path applies.
"""

from __future__ import annotations

import bisect
import math
from typing import Iterable

# __prime_list of libstdc++-6.dll: 256 primes (the sentinel copy is not stored).
PRIME_LIST: tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89,
    97, 103, 109, 113, 127, 137, 139, 149,
    157, 167, 179, 193, 199, 211, 227, 241,
    257, 277, 293, 313, 337, 359, 383, 409,
    439, 467, 503, 541, 577, 619, 661, 709,
    761, 823, 887, 953, 1031, 1109, 1193, 1289,
    1381, 1493, 1613, 1741, 1879, 2029, 2179, 2357,
    2549, 2753, 2971, 3209, 3469, 3739, 4027, 4349,
    4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123,
    8783, 9497, 10273, 11113, 12011, 12983, 14033, 15173,
    16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411,
    30727, 33223, 35933, 38873, 42043, 45481, 49201, 53201,
    57557, 62233, 67307, 72817, 78779, 85229, 92203, 99733,
    107897, 116731, 126271, 136607, 147793, 159871, 172933, 187091,
    202409, 218971, 236897, 256279, 277261, 299951, 324503, 351061,
    379787, 410857, 444487, 480881, 520241, 562841, 608903, 658753,
    712697, 771049, 834181, 902483, 976369, 1056323, 1142821, 1236397,
    1337629, 1447153, 1565659, 1693859, 1832561, 1982627, 2144977, 2320627,
    2510653, 2716249, 2938679, 3179303, 3439651, 3721303, 4026031, 4355707,
    4712381, 5098259, 5515729, 5967347, 6456007, 6984629, 7556579, 8175383,
    8844859, 9569143, 10352717, 11200489, 12117689, 13109983, 14183539, 15345007,
    16601593, 17961079, 19431899, 21023161, 22744717, 24607243, 26622317, 28802401,
    31160981, 33712729, 36473443, 39460231, 42691603, 46187573, 49969847, 54061849,
    58488943, 63278561, 68460391, 74066549, 80131819, 86693767, 93793069, 101473717,
    109783337, 118773397, 128499677, 139022417, 150406843, 162723577, 176048909, 190465427,
    206062531, 222936881, 241193053, 260944219, 282312799, 305431229, 330442829, 357502601,
    386778277, 418451333, 452718089, 489790921, 529899637, 573292817, 620239453, 671030513,
    725980837, 785430967, 849749479, 919334987, 994618837, 1076067617, 1164186217, 1259520799,
    1362662261, 1474249943, 1594975441, 1725587117, 1866894511, 2019773507, 2185171673, 2364114217,
    2557710269, 2767159799, 2993761039, 3238918481, 3504151727, 3791104843, 4101556399, 4294967291,
)

# _M_next_bkt for n < 14 (hashtable_c++0x.cc); the table holds the smallest prime bucket count for n.
_FAST_BKT: tuple[int, ...] = (2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13)
_GROWTH_FACTOR = 2  # _Prime_rehash_policy::_S_growth_factor (hashtable_policy.h)
_SIZE_MAX = 2**64 - 1  # size_t(-1), the next_resize that stops further rehashing


class PrimeRehashPolicy:
    """_Prime_rehash_policy (hashtable_policy.h, hashtable_c++0x.cc) with a fixed max load factor."""

    def __init__(self, max_load_factor: float = 1.0) -> None:
        self.max_load_factor = float(max_load_factor)
        self.next_resize = 0  # _M_next_resize; 0 means that nothing is allocated yet

    def next_bkt(self, n: int) -> int:
        """_M_next_bkt: the bucket count for a request of n; also sets next_resize."""
        if n < len(_FAST_BKT):
            if n == 0:
                return 1  # a zero hint keeps next_resize at 0
            self.next_resize = math.floor(_FAST_BKT[n] * self.max_load_factor)
            return _FAST_BKT[n]
        # lower_bound from the 7th prime (17) to the last prime, which the search excludes
        last = len(PRIME_LIST) - 1
        idx = bisect.bisect_left(PRIME_LIST, n, 6, last)
        if idx == last:
            self.next_resize = _SIZE_MAX  # largest prime reached: never rehash again
        else:
            self.next_resize = math.floor(PRIME_LIST[idx] * self.max_load_factor)
        return PRIME_LIST[idx]

    def need_rehash(self, n_bkt: int, n_elt: int, n_ins: int) -> tuple[bool, int]:
        """_M_need_rehash: (rehash now?, new bucket count if so)."""
        if n_elt + n_ins > self.next_resize:
            # An empty policy starts from 11 buckets' worth of load.
            load = max(n_elt + n_ins, 0 if self.next_resize else 11)
            min_bkts = load / self.max_load_factor
            if min_bkts >= n_bkt:
                grown = max(math.floor(min_bkts) + 1, n_bkt * _GROWTH_FACTOR)
                return True, self.next_bkt(grown)
            self.next_resize = math.floor(n_bkt * self.max_load_factor)
        return False, 0


class UnorderedMapKeys:
    """Node order of a std::unordered_map<size_t, V> built by operator[] inserts of new keys.

    The nodes form one singly linked list, head first. The nodes of a bucket are contiguous. For
    bucket b, `_first[b]` is the first node of its run, so the node before that run is `_prv[first]`;
    in C++ that node is _M_buckets[b]. Erase is not modelled because the PCL maps never erase.
    """

    def __init__(self) -> None:
        self.policy = PrimeRehashPolicy()
        self.bucket_count = 1  # default-constructed map: a single bucket (_M_single_bucket)
        self._size = 0
        self._head: int | None = None
        self._nxt: dict[int, int | None] = {}
        self._prv: dict[int, int | None] = {}
        self._first: dict[int, int] = {}

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: int) -> bool:
        return key in self._nxt

    def insert(self, key: int) -> None:
        """operator[] on a key: a new key is inserted; an existing key stays where it is."""
        if key in self._nxt:
            return
        need, new_count = self.policy.need_rehash(self.bucket_count, self._size, 1)  # _M_insert_unique_node
        if need:
            self._rehash(new_count)
        self._link(key)  # _M_insert_bucket_begin
        self._size += 1

    def order(self) -> list[int]:
        """Iteration order: the list from the head along its links (WeightMap.cpp mapToWeights)."""
        out: list[int] = []
        k = self._head
        while k is not None:
            out.append(k)
            k = self._nxt[k]
        return out

    def _link(self, key: int) -> None:
        """_M_insert_bucket_begin: a non-empty bucket takes the node at the front of its run, an
        empty bucket takes it at the front of the whole list."""
        b = key % self.bucket_count  # identity hash, _M_bucket_index
        first = self._first.get(b)
        if first is not None:
            before = self._prv[first]
            self._nxt[key] = first
            self._prv[key] = before
            self._prv[first] = key
            if before is None:
                self._head = key
            else:
                self._nxt[before] = key
        else:
            old_head = self._head
            self._nxt[key] = old_head
            self._prv[key] = None
            if old_head is not None:
                self._prv[old_head] = key
            self._head = key
        self._first[b] = key

    def _rehash(self, new_count: int) -> None:
        """_M_rehash_aux: re-insert every node in the current list order under the new bucket count."""
        old_order = self.order()
        self._head = None
        self._nxt = {}
        self._prv = {}
        self._first = {}
        self.bucket_count = new_count
        for key in old_order:
            self._link(key)


def iteration_order(keys: Iterable[int]) -> list[int]:
    """Iteration order of an unordered_map after operator[] inserts of `keys` in the given order."""
    m = UnorderedMapKeys()
    for key in keys:
        m.insert(int(key))
    return m.order()
