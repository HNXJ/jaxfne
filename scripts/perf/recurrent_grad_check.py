"""Does the gradient cross neurons? Jacobian of per-neuron mean V_m w.r.t. per-neuron Izhikevich a,
against a finite difference of the same map (the forward coupling through W . spikes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "perf"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaxfne as jtfne  # noqa: E402
from precision_profile import HANDLE, _model  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

N, STEPS, DT = 20, 2000, 0.1
model = _model(N)
W = None
for name in ("W", "weights"):
    W = getattr(getattr(model, "emitter", None), name, None)
    if W is not None:
        break
theta0 = jnp.asarray(np.asarray(jtfne.get_parameter(model, HANDLE)), dtype=jnp.float64)


def mean_v(theta):
    sig = jtfne.simulate(jtfne.set_parameter(model, HANDLE, theta), duration_ms=STEPS * DT, dt_ms=DT, seed=0,
                         dtype="float64", record_fields=False)
    return jnp.mean(sig.V_m, axis=0)


base_sig = jtfne.simulate(model, duration_ms=STEPS * DT, dt_ms=DT, seed=0, dtype="float64", record_fields=False)
spikes = np.asarray(base_sig.spikes).sum(axis=0)
print("W shape", None if W is None else np.shape(W), "nonzero", None if W is None else int(np.count_nonzero(np.asarray(W))))
print("spikes per neuron", spikes.tolist())
assert spikes.sum() > 0, "fixture must spike"

J = np.asarray(jax.jacrev(mean_v)(theta0))
off = J - np.diag(np.diag(J))
print("autodiff: |diag| max", float(np.abs(np.diag(J)).max()), " |off-diag| max", float(np.abs(off).max()),
      " nonzero off-diag", int(np.count_nonzero(off)))

# Finite difference on the source neuron that spikes most: does it move other neurons' mean V?
i = int(np.argmax(spikes))
h = 1e-3
fd = (np.asarray(mean_v(theta0.at[i].add(h))) - np.asarray(mean_v(theta0.at[i].add(-h)))) / (2 * h)
fd_off = np.delete(fd, i)
print(f"finite difference on a[{i}]: |self| {abs(fd[i]):.4g}  |others| max {np.abs(fd_off).max():.4g}"
      f"  others moved {int(np.count_nonzero(np.abs(fd_off) > 1e-9))}/{N - 1}")
print(f"autodiff column {i}: |others| max {float(np.abs(np.delete(J[:, i], i)).max()):.4g}")
