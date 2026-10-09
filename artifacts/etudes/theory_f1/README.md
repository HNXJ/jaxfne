# F1: state reduction for HDP traces (declaration, 2026-10-08)

Family F1 of `artifacts/programme/theory_programme_proposal.md`, opened by Hamm on
2026-10-08. Declared before any check runs. Source evidence: K1h and K1h-nf
(`artifacts/etudes/pcl/`), where `pcl_stdp_h` (`artifacts/etudes/pcl_h/pcl_h_rule.py`)
rewrites the per-edge reset trace of `pcl_stdp` as a per-neuron H trace minus a
per-edge snapshot.

## Setting

Discrete time t = 0, 1, …, T on the declared `dt` grid. Edge e = (i → j) belongs to a
class c(e). Presynaptic input s_i(t) is real (spikes are the case s ∈ {0, 1}); the
postsynaptic reset r_j(t) ∈ {0, 1}. A class carries M_c ∈ ℝ^{d×d} and b_c ∈ ℝ^d; for a
scalar trace M_c = a_c = exp(−dt/τ_c). All states start at 0.

Direct per-edge trace, with the pre-reset value Â read by the rule at step t:

    Â_e(t) = M_c A_e(t−1) + b_c s_i(t),    A_e(t) = (1 − r_j(t)) Â_e(t).

Factored form: one H trace per neuron and class, never reset, and one per-edge
snapshot:

    x_{i,c}(t) = M_c x_{i,c}(t−1) + b_c s_i(t),
    S_e(t) = (1 − r_j(t)) M_c S_e(t−1) + r_j(t) x_{i,c}(t).

## Claims

**L1 (factorization, exact arithmetic).** For every t: Â_e(t) = x_{i,c}(t) − M_c S_e(t−1)
and A_e(t) = x_{i,c}(t) − S_e(t).

Proof. Induction on t; at t = −1 all terms are 0. Assume A_e(t−1) = x(t−1) − S_e(t−1).
Then Â_e(t) = M_c x(t−1) − M_c S_e(t−1) + b_c s_i(t) = x(t) − M_c S_e(t−1). If r_j(t) = 0,
A_e(t) = Â_e(t) and S_e(t) = M_c S_e(t−1), so A_e(t) = x(t) − S_e(t). If r_j(t) = 1,
A_e(t) = 0 and S_e(t) = x(t), so x(t) − S_e(t) = 0. ∎ Tier T2 (second-reviewer, 2026-10-08).

**L2 (lazy snapshot).** Let τ_j(t) be the last step ≤ t with r_j = 1. Then
S_e(t) = M_c^{t−τ_j(t)} x_{i,c}(τ_j(t)), and S_e(t) = 0 if j has not reset. So S_e is
written only at resets of j, and its value at any t follows from the stored
x_{i,c}(τ_j) and τ_j(t), which is per neuron. Memory stays O(E). For this trace alone,
per-step work falls from O(E) for the direct form to O(N·K), plus O(fan-in of j) at
each reset of j, where K is the number of classes.

Proof. Unroll the S recurrence from τ_j(t): S_e(τ_j) = x(τ_j), and each later step
without a reset multiplies by M_c. ∎ Tier T2 (second-reviewer, 2026-10-08).

**L3 (one trace per decay is necessary).** Let i's outgoing plastic edges carry scalar
classes with K distinct nonzero decays a_1, …, a_K and nonzero gains. Classes that
share a decay can share one trace, since the trace for gain b′ is (b′/b) times the
trace for gain b. Conversely, no linear per-neuron state of dimension m < K, read
linearly by each edge, reproduces the direct traces. This K is the one `pcl_stdp_h`
requires and refuses to average.

Proof. With no resets, one spike at t₀ gives the direct trace b_k a_k^{n}, n = t − t₀ ≥ 0.
A state h(t) = G h(t−1) + g s_i(t) read as c_k·h gives c_k G^{n} g. By Cayley–Hamilton
every such sequence solves the order-m linear recurrence of G's characteristic
polynomial, whose solutions form a space of dimension m. The K sequences a_k^{n} with
distinct nonzero a_k are linearly independent (Vandermonde), so m ≥ K. ∎ Tier T2
(second-reviewer, 2026-10-08).

**L4 (floating point, measured).** In float32 the two forms are both exact only to
rounding and differ by evaluation order. Expected order of the difference:
|A_direct − A_factored| ≤ C·u/(1 − a)², with u the unit roundoff and C a small
constant, at most one spike per step. This is a first-order bound, not proved here.
Tier T0: the check measures the maximum difference against the bound.

**L5 (conjecture, the family's boundary).** The `pcl_stdp` LTD accumulator,
B_e(t) = B_e(t−1) + a·s_i(t)·Y_j(t−1), reset at postsynaptic spikes, is bilinear in pre and post histories and does
not factor into finite per-neuron states of i and j. Not proved; listed so that F1
does not claim more than L1–L3.

## Checks (to be built, all float64 unless stated)

1. L1: random spike and reset trains, scalar and d = 2 classes; max |direct − factored|
   ≤ 1e-12 relative to max |A|.
2. L2: lazy and eager snapshots agree to 1e-12.
3. L3, a negative control: two decays forced onto one trace give an error above 1e-3;
   the check passes only when the error exceeds it.
4. L4: float32 difference measured against C·u/(1 − a)² at a = exp(−0.5/40) and
   exp(−0.5/7); report C needed.

## Result (2026-10-08, `f1_check.py`, seeds 0–4, T = 20000)

| check | measured | verdict |
|---|---|---|
| 1 L1, float64 | max relative difference 1.1e-15 (scalar and d = 2) | PASS |
| 2 L2, float64 | max relative difference 1.3e-15 | PASS |
| 3 L3 negative control | min over seeds of max difference 5.0 | PASS |
| 4 L4, float32 | C = 0.012 (τ = 40 ms), 0.057 (τ = 7 ms) | measured |

Check 4 shows the first-order bound is loose by a factor of about 17 to 81 on these trains,
consistent with rounding errors that partly cancel rather than add. The script
flags C ≤ 10 and the test C ≤ 1; neither threshold was declared before the run.

Falsifier for the family: any case meeting the Setting's hypotheses where check 1 or 2
fails in float64.
