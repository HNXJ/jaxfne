"""Does the gradient cross neurons? Jacobian of per-neuron mean V_m w.r.t. per-neuron Izhikevich a,
against a finite difference of the same map (the forward coupling through W . spikes), for the hard
spike and for the sigmoid_beta surrogate at several slopes (P-052). Also reports whether the backward
graph holds the recurrent transpose (a dot_general that contracts W on its row axis)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "perf"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaxfne as jtfne  # noqa: E402
from precision_profile import HANDLE, _model  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

N, STEPS, DT = 20, 2000, 0.1
BETAS = (None, 10.0, 1.0, 0.1)
model = _model(N)
theta0 = jnp.asarray(np.asarray(jtfne.get_parameter(model, HANDLE)), dtype=jnp.float64)


def _runtime(beta):
    sg = None if beta is None else jtfne.SurrogateConfig(method="sigmoid_beta", beta=beta)
    return jtfne.RuntimeConfig(dtype="float64", jit=True, recompilation_guard="off", surrogate=sg)


def mean_v(theta, beta):
    sig = jtfne.simulate(jtfne.set_parameter(model, HANDLE, theta), duration_ms=STEPS * DT, dt_ms=DT, seed=0,
                         runtime=_runtime(beta), record_fields=False)
    return jnp.mean(sig.V_m, axis=0)


def _dots(jaxpr, out):
    """Every dot_general in a jaxpr and its sub-jaxprs: (lhs shape, rhs shape, contracting dims)."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            (lc, rc), _ = eqn.params["dimension_numbers"]
            out.append((tuple(eqn.invars[0].aval.shape), tuple(eqn.invars[1].aval.shape), tuple(lc), tuple(rc)))
        for p in eqn.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else (p,)):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    _dots(inner, out)
    return out


def _has_recurrent_transpose(dots):
    """W (N, N) contracted on axis 0, on either side: W^T g."""
    return any((ls == (N, N) and lc == (0,)) or (rs == (N, N) and rc == (0,)) for ls, rs, lc, rc in dots)


base = jtfne.simulate(model, duration_ms=STEPS * DT, dt_ms=DT, seed=0, runtime=_runtime(None), record_fields=False)
spikes = np.asarray(base.spikes).sum(axis=0)
assert spikes.sum() > 0, "fixture must spike"
i = int(np.argmax(spikes))
print(f"V_m dtype {np.asarray(base.V_m).dtype}; spikes per neuron {spikes.tolist()}; source a[{i}]")
fds = {}
for h in (1e-2, 1e-3, 1e-4):
    fd = (np.asarray(mean_v(theta0.at[i].add(h), None)) - np.asarray(mean_v(theta0.at[i].add(-h), None))) / (2 * h)
    fds[h] = np.delete(fd, i)
    print(f"finite difference h={h:g}: |self| {abs(fd[i]):.4g}  |others| max {np.abs(fds[h]).max():.4g}"
          f"  others moved {int(np.count_nonzero(np.abs(fds[h]) > 1e-9))}/{N - 1}")
print(f"FD stability: corr(h=1e-2, 1e-3) {np.corrcoef(fds[1e-2], fds[1e-3])[0, 1]:.3f}, "
      f"corr(h=1e-3, 1e-4) {np.corrcoef(fds[1e-3], fds[1e-4])[0, 1]:.3f}")
fd_off = fds[1e-3]

rows = []
for beta in BETAS:
    J = np.asarray(jax.jacrev(mean_v)(theta0, beta))
    off = J - np.diag(np.diag(J))
    col = np.delete(J[:, i], i)
    corr = float(np.corrcoef(col, fd_off)[0, 1]) if np.abs(col).max() > 0 else None
    dots = _dots(jax.make_jaxpr(jax.grad(lambda th: mean_v(th, beta).sum()))(theta0).jaxpr, [])
    row = {"beta_per_mV": beta, "diag_max": float(np.abs(np.diag(J)).max()), "offdiag_max": float(np.abs(off).max()),
           "offdiag_nonzero": int(np.count_nonzero(off)), "col_others_max": float(np.abs(col).max()),
           "corr_with_fd": corr, "sign_agree_with_fd": int(np.sum(np.sign(col) == np.sign(fd_off))),
           "recurrent_transpose_in_backward": _has_recurrent_transpose(dots), "dot_generals": len(dots)}
    rows.append(row)
    print(json.dumps(row))
