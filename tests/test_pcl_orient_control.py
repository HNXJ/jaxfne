"""PCL control O1: hand-set oriented kernels put the ON and OFF lobes across the bar."""
from __future__ import annotations

import sys

import numpy as np

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.pcl.pcl_column as C  # noqa: E402
import artifacts.etudes.pcl.pcl_orient_control as O  # noqa: E402


def test_oriented_kernels_lobes_and_normalization():
    k = O.oriented_kernels().reshape(C.FS, C.RF, C.RF, 2)
    # orientation 0: horizontal bar, normal along +y; q = 0 puts ON one row below centre, OFF one row above
    on, off = k[0, ..., 0], k[0, ..., 1]
    assert np.argmax(on.sum(1)) == 4 and np.argmax(off.sum(1)) == 2
    assert np.allclose(on.std(1), 0) and np.allclose(off.std(1), 0)  # flat along the bar
    assert np.allclose(k[1, ..., 0], off) and np.allclose(k[1, ..., 1], on)  # q = 1 swaps the lobes
    # orientation 4: vertical bar, lobes in columns
    assert np.argmax(k[8, ..., 0].sum(0)) in (2, 4) and np.allclose(k[8, ..., 0].std(0), 0)
    W = np.asarray(C.normalize(O.oriented_kernels(), C.MASKS["s_exc"], "s_exc"))
    assert np.allclose(W.sum(1), C.CONN["s_exc"][3] * C.RF * C.RF * 2, rtol=1e-5)
