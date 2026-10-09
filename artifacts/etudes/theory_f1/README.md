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
A_e(t) = 0 and S_e(t) = x(t), so x(t) − S_e(t) = 0. ∎ Tier T2 (pending second-reviewer).

**L2 (lazy snapshot).** Let τ_j(t) be the last step ≤ t with r_j = 1. Then
S_e(t) = M_c^{t−τ_j(t)} x_{i,c}(τ_j(t)), and S_e(t) = 0 if j has not reset. So S_e is
written only at resets of j, and with scalar M_c its value at any t needs only
τ_j(t), which is per neuron. For this trace alone, per-step work falls from O(E) for the direct form to
O(N·K), plus O(fan-in of j) at each reset of j, where K is the number of classes.

Proof. Unroll the S recurrence from τ_j(t): S_e(τ_j) = x(τ_j), and each later step
without a reset multiplies by M_c. ∎ Tier T2 (pending).

**L3 (classes are necessary).** If two plastic edges from neuron i have scalar decays
a ≠ a′, no single trace of i reproduces both direct traces. So the factored form
needs one H trace per distinct (M_c, b_c) among i's outgoing plastic edges; this is
the `K` that `pcl_stdp_h` requires and refuses to average.

Proof. One spike at t₀ and no resets give a^{t−t₀} and a′^{t−t₀}, which differ for
every t > t₀; one shared sequence equals at most one of them. ∎ Tier T2 (pending).

**L4 (floating point, measured).** In float32 the two forms are both exact only to
rounding and differ by evaluation order. Expected order of the difference:
|A_direct − A_factored| ≤ C·u/(1 − a)², with u the unit roundoff and C a small
constant, at most one spike per step. This is a first-order bound, not proved here.
Tier T0: the check measures the maximum difference against the bound.

**L5 (conjecture, the family's boundary).** The `pcl_stdp` LTD accumulator,
B_e(t) = a B_e(t−1) + s_i(t)·Y_j(t−1), is bilinear in pre and post histories and does
not factor into finite per-neuron states of i and j. Not proved; listed so that F1
does not claim more than L1–L3.

## Checks (to be built, all float64 unless stated)

1. L1: random spike and reset trains, scalar and d = 2 classes; max |direct − factored|
   ≤ 1e-12 relative to max |A|.
2. L2: lazy and eager snapshots agree to 1e-12.
3. L3, a negative control: two decays forced onto one trace give an error above 1e-3.
   A check that passes here has failed.
4. L4: float32 difference measured against C·u/(1 − a)² at a = exp(−0.5/40) and
   exp(−0.5/7); report C needed.

Falsifier for the family: any case meeting the Setting's hypotheses where check 1 or 2
fails in float64.
