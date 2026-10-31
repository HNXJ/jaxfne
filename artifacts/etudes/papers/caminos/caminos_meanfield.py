"""Mean-field models of the CAMINOS PLoS CB paper (Tahvili, Vinck, di Volo 2026, PMC13286281).

Linear rate model (Results, "Linear mean-field model", Eqs. 3-14, Fig. 6):
    tau dnu/dt = (Q - I) nu,  nu = [nu_E, nu_PV, nu_SOM],
with Q holding the effective couplings (inhibitory entries negative). The base model (Eq. 10) has
Q_EE = 5, Q_EPV = 5, Q_PVE = 5, Q_PVPV = 4, Q_ESOM = 4, Q_SOME = 4, Q_SOMPV = 2.05, tau = 20 ms.
Here ``Q_XY`` is the coupling from population X to population Y, as in the paper's Q_{X->Y}.

Usage: python caminos_meanfield.py   (prints the Fig. 6 quantities)
"""
import numpy as np

TAU_S = 0.02  # Eq. 10
BASE_Q = {"EE": 5.0, "EPV": 5.0, "PVE": 5.0, "PVPV": 4.0, "ESOM": 4.0, "SOME": 4.0, "SOMPV": 2.05,
          "PVSOM": 0.0, "SOMSOM": 0.0}


def coupling_matrix(**q):
    """A = Q - I of Eq. 4 (rows: target E, PV, SOM; columns: source), base values overridden by q."""
    unknown = set(q) - set(BASE_Q)
    if unknown:
        raise ValueError("unknown couplings %s" % sorted(unknown))
    c = {**BASE_Q, **q}
    Q = np.array([[c["EE"], -c["PVE"], -c["SOME"]],
                  [c["EPV"], -c["PVPV"], -c["SOMPV"]],
                  [c["ESOM"], -c["PVSOM"], -c["SOMSOM"]]], dtype=float)
    return Q - np.eye(3)


def red_line(q_some):
    """Q_SOM->PV on the pure-oscillation line of Fig. 6B for the base E-PV couplings (Eq. 7)."""
    return 0.6 * q_some - 0.35


def oscillatory_mode(A, tau_s=TAU_S):
    """The complex eigenmode of A / tau: frequency (Hz), real part (1/s) and the PV-to-SOM and
    E-to-SOM phase leads (rad and ms). Returns None when every eigenvalue is real."""
    lam, vec = np.linalg.eig(A / tau_s)
    k = int(np.argmax(lam.imag))
    if lam[k].imag <= 0:
        return None
    omega = float(lam[k].imag)
    v = vec[:, k]
    lead = {}
    for name, i in (("PV", 1), ("E", 0)):
        d = float(np.angle(v[i] / v[2]))  # phase of population i minus phase of SOM
        lead[name] = {"rad": d, "ms": 1000.0 * d / omega}
    return {"freq_hz": omega / (2 * np.pi), "real": float(lam[k].real), "lead_over_som": lead,
            "eigenvalues": lam}


def regime(A, tau_s=TAU_S, tol=1e-9):
    """Fig. 6B regime label from the eigenvalues of A / tau."""
    lam = np.linalg.eigvals(A / tau_s)
    cplx = lam[np.abs(lam.imag) > tol]
    real = lam[np.abs(lam.imag) <= tol].real
    if cplx.size == 0:
        return "real"
    rc = float(cplx[0].real)
    if real.size and real.max() > tol:
        return "positive real"
    if abs(rc) * tau_s <= tol:
        return "pure oscillation"
    return "growing" if rc > 0 else "damped"


def main():
    base = oscillatory_mode(coupling_matrix())
    print("Fig 6A base: f = %.2f Hz, Re = %.3g 1/s, PV lead %.2f ms, E lead %.2f ms"
          % (base["freq_hz"], base["real"], base["lead_over_som"]["PV"]["ms"],
             base["lead_over_som"]["E"]["ms"]))
    for q_se in (2.0, 4.0, 6.0):
        on = red_line(q_se)
        print("Fig 6B Q_SOME %.1f: line %.3f -> %s; +0.5 -> %s; -0.5 -> %s" % (
            q_se, on, regime(coupling_matrix(SOME=q_se, SOMPV=on)),
            regime(coupling_matrix(SOME=q_se, SOMPV=on + 0.5)),
            regime(coupling_matrix(SOME=q_se, SOMPV=on - 0.5))))
    for q in (0.5, 2.0, 220 / 59 - 0.01, 220 / 59 + 0.01, 5.0):
        A = coupling_matrix(PVSOM=q)
        print("Fig 6C PV->SOM %.3f: det(Y) %.4f, regime %s" % (q, np.linalg.det(A), regime(A)))
    for q in (0.0, 0.5, 1.0, 2.0, 4.0):
        m = oscillatory_mode(coupling_matrix(SOMSOM=q))
        print("Fig 6C SOM->SOM %.1f: Re %.2f 1/s, f %.2f Hz" % (q, m["real"], m["freq_hz"]))


if __name__ == "__main__":
    main()
