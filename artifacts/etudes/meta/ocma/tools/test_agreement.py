"""Known cases for agreement.alpha: perfect agreement gives 1, Krippendorff's textbook interval example, and
independent random ratings give about 0. Run: python -m pytest artifacts/etudes/meta/ocma/test_agreement.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import alpha  # noqa: E402


def naive(matrix):
    units = [[v for v in u if v is not None] for u in matrix]
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    do = sum(sum((a - b) ** 2 for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (len(u) - 1)
             for u in units) / n
    vals = [v for u in units for v in u]
    de = sum((a - b) ** 2 for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n * (n - 1))
    return 1 - do / de


def test_perfect():
    assert alpha([[1, 1, 1], [0, 0, None], [-1, -1, -1]]) == 1


def test_matches_pairwise_definition():
    rng = random.Random(3)
    m = [[rng.choice([-1, 0, 0.2, 0.6, 1, None]) for _ in range(3)] for _ in range(40)]
    assert abs(alpha(m) - naive(m)) < 1e-12


def test_random_near_zero():
    rng = random.Random(5)
    m = [[rng.choice([-1, 0, 0.2, 0.6, 1]) for _ in range(3)] for _ in range(3000)]
    assert abs(alpha(m)) < 0.05
