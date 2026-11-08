"""Mean-field models of the CAMINOS PLoS CB paper (Tahvili, Vinck, di Volo 2026, PMC13286281).

Linear rate model (Results, "Linear mean-field model", Eqs. 3-14, Fig. 6):
    tau dnu/dt = (Q - I) nu,  nu = [nu_E, nu_PV, nu_SOM],
with Q holding the effective couplings (inhibitory entries negative). The base model (Eq. 10) has
Q_EE = 5, Q_EPV = 5, Q_PVE = 5, Q_PVPV = 4, Q_ESOM = 4, Q_SOME = 4, Q_SOMPV = 2.05, tau = 20 ms.
Here ``Q_XY`` is the coupling from population X to population Y, as in the paper's Q_{X->Y}.

Nonlinear AdEx mean field (Eq. 1, Methods; SI units): PARAMS, moments, transfer, DelayLine,
simulate_meanfield, fixed_point. Assumed values and numerical choices are marked in the code.

Usage: python caminos_meanfield.py   (prints the Fig. 6 quantities, then the nonlinear run)
"""
import numpy as np
from scipy.optimize import root
from scipy.special import erfc

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


# ---------------------------------------------------------------------------
# Nonlinear AdEx mean field: Eq. 1 with the tau_d delay, Methods "Transfer function derivation",
# fixed point Eq. 20. SI units (V, A, S, s; rates in 1/s). Line numbers in comments refer to
# plos_text.txt as the Read tool numbers it.
# ---------------------------------------------------------------------------
TYPES = ("E", "PV", "SOM")
PRE = ("E", "PV", "SOM", "ext")  # presynaptic inputs; "ext" is the Poisson drive
T_MF = 15e-3  # mean-field time constant T (line 18)
TAU_D = 2e-3  # synaptic delay tau_d (line 19; Table 2, line 166)
W_SCALE = 1e-12  # 1 pA: scale of the adaptation residuals in fixed_point
F_FLOOR = 1e-12  # 1/s: floor on the input weights of sigma_V and T_v only (see moments)
# Start for fixed_point and simulate_meanfield: [nu_E, nu_PV, nu_SOM (1/s), w_E, w_SOM (A)].
# ASSUMED: the text asks for a "biologically plausible" start (line 285) and gives no numbers.
# Chosen inside the stated ordering: E 1-3 Hz, PV > SOM > E (line 17).
DEFAULT_START = np.array([2.0, 8.0, 4.0, 0.0, 0.0])

PARAMS = {
    # Table 1 (lines 137-156), per type E, PV, SOM
    "C_m": {"E": 200e-12, "PV": 200e-12, "SOM": 200e-12},  # 200 pF (line 150)
    "g_L": {"E": 10e-9, "PV": 10e-9, "SOM": 10e-9},  # 10 nS (line 152)
    "E_L": {"E": -65e-3, "PV": -65e-3, "SOM": -55e-3},  # -65, -65, -55 mV (line 154)
    "a": {"E": 4e-9, "PV": 0.0, "SOM": 4e-9},  # 4, 0, 4 nS (line 146)
    "b": {"E": 50e-12, "PV": 0.0, "SOM": 80e-12},  # 50, 0, 80 pA (line 148)
    "tau_w": {"E": 0.5, "PV": 0.5, "SOM": 0.5},  # 500 ms (line 144)
    # Table 2 (lines 165-182)
    "tau_E": 2e-3,  # 2 ms (line 168); also the drive's tau: DERIVED, the text gives none for the drive
    "tau_I": 8e-3,  # 8 ms (line 170)
    "E_E": 0.0,  # 0 mV (line 172)
    "E_I": -80e-3,  # -80 mV (line 174)
    # Q[(pre, post)] in S. Line 176: PV, SOM -> E, PV 5 nS; line 178: E -> E 2 nS;
    # line 180: E -> PV, SOM 3 nS; line 182: Ext -> E, PV 2.5 nS. Zero where absent (line 183)
    # and where there is no drive (line 184).
    "Q": {
        ("E", "E"): 2e-9, ("E", "PV"): 3e-9, ("E", "SOM"): 3e-9,
        ("PV", "E"): 5e-9, ("PV", "PV"): 5e-9, ("PV", "SOM"): 0.0,
        ("SOM", "E"): 5e-9, ("SOM", "PV"): 5e-9, ("SOM", "SOM"): 0.0,
        ("ext", "E"): 2.5e-9, ("ext", "PV"): 2.5e-9, ("ext", "SOM"): 0.0,
    },
    "p": 0.07,  # line 183
    "N": {"E": 8000, "PV": 750, "SOM": 750},  # line 183
    "nu_ext": 4.0,  # Hz (line 184); f_ext = nu_ext p N_E (line 192)
    # Table 3 (lines 220-252), columns P0, P_mu, P_sigma, P_tau, P_mu2, P_sigma2, P_tau2,
    # P_mu_sigma, P_mu_tau, P_sigma_tau
    "P": {
        "E": (-0.049734, 0.007999, 0.009232, -0.002773, 0.027645,
              -0.037666, 0.029874, -0.023012, -0.005805, 0.022400),  # lines 221-230
        "PV": (-0.052335, 0.005472, -0.010339, -0.014467, -0.000929,
               0.000323, 0.002562, 0.023115, -0.023466, -0.034865),  # lines 232-241
        "SOM": (-0.046763, 0.000063, -0.003225, -0.011496, 0.000968,
                0.001398, 0.009397, 0.013090, 0.022406, 0.010788),  # lines 243-252
    },
    # Table 4 (lines 261-281): mu_V0, D mu_V0, sigma_V0, D sigma_V0, T_vN0, D T_vN0.
    # ASSUMED to be SI volts: the text gives no unit line; V_T = -50 mV (line 138) sets the scale of P0.
    "base": {
        "E": (-0.06, 0.05, 0.004, 0.006, 0.5, 1.0),  # lines 262-267
        "PV": (-0.06, 0.01, 0.004, 0.006, 0.5, 1.0),  # lines 269-274
        "SOM": (-0.05, 0.01, 0.004, 0.006, 0.5, 1.0),  # lines 276-281
    },
}


def moments(ctype, nu_E, nu_PV, nu_SOM, w, params=PARAMS):
    """Input statistics of target ctype (lines 191-204): returns (mu_V [V], sigma_V [V], T_vN).

    Rates in 1/s; w in A (0 for PV). Negative rates are clipped to 0 (they occur only inside the
    fixed-point search). mu_V uses the true rates. sigma_V and T_vN weight each input by at least
    F_FLOOR, so sigma_V stays positive when an input vanishes: the transfer then takes its limit.
    Example: SOM with nu_E = 0 has no other input (Q_PV->SOM = Q_SOM->SOM = 0, no drive to SOM).
    """
    N, p = params["N"], params["p"]
    nu = {"E": max(nu_E, 0.0), "PV": max(nu_PV, 0.0), "SOM": max(nu_SOM, 0.0)}
    f = {X: nu[X] * p * N[X] for X in TYPES}  # line 192: f_X = nu_X p N_X
    f["ext"] = params["nu_ext"] * p * N["E"]  # line 192: f_ext = nu_ext p N_E
    # Each input's tau: E and drive are excitatory (tau_E), PV and SOM inhibitory (tau_I).
    tau = {"E": params["tau_E"], "PV": params["tau_I"], "SOM": params["tau_I"], "ext": params["tau_E"]}
    rev = {"E": params["E_E"], "PV": params["E_I"], "SOM": params["E_I"], "ext": params["E_E"]}
    Q = {X: params["Q"][(X, ctype)] for X in PRE}  # Q_{X->Y}, Y = ctype (line 161)
    muG_X = {X: Q[X] * tau[X] * f[X] for X in PRE}  # line 193: muG_X = Q_X tau_X f_X
    g_L, C_m = params["g_L"][ctype], params["C_m"][ctype]
    muG = g_L + sum(muG_X.values())  # line 194
    T_m = C_m / muG  # line 195
    # Line 196, whole numerator over muG (ruling: steady state of Eq. 15).
    muV = (sum(muG_X[X] * rev[X] for X in PRE) + g_L * params["E_L"][ctype] - w) / muG
    U = {X: Q[X] / muG * (rev[X] - muV) for X in PRE}  # line 200
    g = {X: max(f[X], F_FLOOR) * (U[X] * tau[X]) ** 2 for X in PRE}  # f_X (U_X tau_X)^2, floored
    sigma = np.sqrt(sum(g[X] / (2.0 * (tau[X] + T_m)) for X in PRE))  # line 202
    Tv = sum(g.values()) / sum(g[X] / (tau[X] + T_m) for X in PRE)  # line 204
    return muV, sigma, g_L * Tv / C_m  # line 204: T_vN = g_L T_v / C_m


def _rate(ctype, nu_E, nu_PV, nu_SOM, w, params=PARAMS):
    """Return (mu_V, nu_out) of target ctype (lines 206-207)."""
    muV, sigma, TvN = moments(ctype, nu_E, nu_PV, nu_SOM, w, params)
    mu0, dmu, s0, ds, t0, dt0 = params["base"][ctype]
    P = params["P"][ctype]
    x, y, z = (muV - mu0) / dmu, (sigma - s0) / ds, (TvN - t0) / dt0
    V_thr = (P[0] + P[1] * x + P[2] * y + P[3] * z + P[4] * x * x + P[5] * y * y + P[6] * z * z
             + P[7] * x * y + P[8] * x * z + P[9] * y * z)  # line 206
    Tv = TvN * params["C_m"][ctype] / params["g_L"][ctype]  # T_v = T_vN C_m / g_L
    # Line 207: g_L / (2 C_m T_vN) = 1 / (2 T_v).
    return muV, erfc((V_thr - muV) / (np.sqrt(2.0) * sigma)) / (2.0 * Tv)


def transfer(ctype, nu_E, nu_PV, nu_SOM, w, params=PARAMS):
    """Output rate (1/s) of target ctype: erfc((V_thr_eff - mu_V) / (sqrt(2) sigma_V)) / (2 T_v)."""
    return _rate(ctype, nu_E, nu_PV, nu_SOM, w, params)[1]


class DelayLine:
    """Ring buffer of the last n rate vectors. read() returns nu(t - tau_d); write() stores nu(t).

    Filled with the initial rates, so nu(t < 0) equals them. Each step calls read() before write().
    """

    def __init__(self, n, init):
        self.buf = np.tile(np.asarray(init, dtype=float), (n, 1))
        self.n = n
        self.k = 0

    def read(self):
        return self.buf[self.k].copy()

    def write(self, nu):
        self.buf[self.k] = nu
        self.k = (self.k + 1) % self.n


def _rhs(nu, w, nu_del, params):
    """Right-hand side of Eq. 1: returns (dnu/dt, dw/dt). nu_del = nu(t - tau_d) enters transfer and mu_V.

    ADAPTATION READING (ruling B). Eq. 1 and Eq. 20 as printed, dw/dt = -w/tau_w + b nu + a (mu_V - E_L),
    are dimensionally inconsistent (a (mu_V - E_L) is amperes, dw/dt is A/s). This reading follows
    Eq. 16, tau_w dw/dt = a (V - E_L) - w + (spikes), so the a term carries /tau_w.
    """
    dnu, dw = np.zeros(3), np.zeros(3)
    for i, ct in enumerate(TYPES):
        muV, F = _rate(ct, nu_del[0], nu_del[1], nu_del[2], w[i], params)
        tw = params["tau_w"][ct]
        dnu[i] = (F - nu[i]) / T_MF
        dw[i] = -w[i] / tw + params["b"][ct] * nu[i] + params["a"][ct] * (muV - params["E_L"][ct]) / tw
    return dnu, dw


def simulate_meanfield(T_s, dt_s=1e-4, params=PARAMS, nu0=None, pv_silence=0.0, som_silence=0.0):
    """Euler integration of Eq. 1 with the tau_d delay in a ring buffer.

    Silencing a fraction f scales that population's N by (1 - f) in every input term; the silenced
    rate still obeys Eq. 1 but sends no input. Rates start at nu0 (default DEFAULT_START), nu(t < 0)
    equals nu0 and w starts at 0. Returns (t, nu, w): t in s; nu and w have shape (n + 1, 3), columns
    E, PV, SOM. The PV column of w is 0 (a = b = 0).
    """
    if not (0.0 <= pv_silence <= 1.0 and 0.0 <= som_silence <= 1.0):
        raise ValueError("silenced fractions must lie in [0, 1]")
    p = dict(params)
    p["N"] = {"E": params["N"]["E"], "PV": params["N"]["PV"] * (1.0 - pv_silence),
              "SOM": params["N"]["SOM"] * (1.0 - som_silence)}
    nu = np.array(DEFAULT_START[:3] if nu0 is None else nu0, dtype=float)
    w = np.zeros(3)
    n = int(round(T_s / dt_s))
    line = DelayLine(int(round(TAU_D / dt_s)), nu)
    t = np.arange(n + 1) * dt_s
    NU, W = np.empty((n + 1, 3)), np.empty((n + 1, 3))
    NU[0], W[0] = nu, w
    for k in range(n):
        dnu, dw = _rhs(nu, w, line.read(), p)
        line.write(nu)
        nu, w = nu + dt_s * dnu, w + dt_s * dw
        NU[k + 1], W[k + 1] = nu, w
    return t, NU, W


def fixed_point(params=PARAMS, x0=None):
    """Eq. 20 solved with scipy.optimize.root (hybr) from x0 = [nu_E, nu_PV, nu_SOM, w_E, w_SOM] (SI).

    Returns (x, residual). x is in SI. residual has five entries evaluated at x: the three rate
    residuals (1/s) and the two adaptation residuals in pA (divided by W_SCALE).
    ADAPTATION READING (ruling B): w = tau_w b nu + a (mu_V - E_L). The printed Eq. 20,
    w = tau_w [b nu + a (mu_V - E_L)], is dimensionally inconsistent (see _rhs); this follows Eq. 16.
    """
    start = DEFAULT_START if x0 is None else np.asarray(x0, dtype=float)
    z0 = np.array([start[0], start[1], start[2], start[3] / W_SCALE, start[4] / W_SCALE])

    def resid(z):
        nu = z[:3]
        w = np.array([z[3], 0.0, z[4]]) * W_SCALE
        out = np.zeros(5)
        muV = {}
        for i, ct in enumerate(TYPES):
            muV[ct], F = _rate(ct, nu[0], nu[1], nu[2], w[i], params)
            out[i] = nu[i] - F
        for j, ct in ((3, "E"), (4, "SOM")):
            i = TYPES.index(ct)
            w_fp = params["tau_w"][ct] * params["b"][ct] * nu[i] + params["a"][ct] * (muV[ct] - params["E_L"][ct])
            out[j] = (w[i] - w_fp) / W_SCALE
        return out

    # xtol 1e-12 (solver default 1.5e-8 leaves residuals near the 1e-8 criterion of the packet).
    sol = root(resid, z0, method="hybr", options={"xtol": 1e-12})
    z = sol.x
    x = np.array([z[0], z[1], z[2], z[3] * W_SCALE, z[4] * W_SCALE])
    return x, resid(z)


def main_meanfield():
    t, nu, _ = simulate_meanfield(4.0)
    sel = (t >= 2.0) & (t <= 4.0)
    mean = nu[sel].mean(axis=0)
    e = nu[sel, 0]
    amp = np.abs(np.fft.rfft(e - e.mean()))
    freq = np.fft.rfftfreq(e.size, t[1] - t[0])
    peak = freq[1 + int(np.argmax(amp[1:]))]
    x, res = fixed_point()
    print("Nonlinear mean field, 4 s run, mean rates over 2-4 s: E %.3f  PV %.3f  SOM %.3f Hz" % tuple(mean))
    print("E rate over 2-4 s: min %.3f  max %.3f Hz; spectral peak %.2f Hz (mean removed)"
          % (e.min(), e.max(), peak))
    print("Fixed point: nu_E %.4f  nu_PV %.4f  nu_SOM %.4f Hz; w_E %.4e A  w_SOM %.4e A" % tuple(x))
    print("Residual: rates (1/s) %.2e %.2e %.2e; adaptation (pA) %.2e %.2e" % tuple(res))


if __name__ == "__main__":
    main()
    main_meanfield()
