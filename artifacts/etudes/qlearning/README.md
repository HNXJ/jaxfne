# Q-learning on a jaxfne E/I network (lane QL)

Task: the registered `chain5` task (`jaxfne.tasks`): 5 states, start 0, right from 3
pays 1 and ends the episode, gamma 0.9. Analytic `Q*(s,right) = 0.9**(3-s)`.

State code: a 40 E + 10 I network. The deterministic all-to-all base graph is
unchanged; one `connections()` rule adds E-to-E edges (probability 0.5, weight 1.0),
and `connectivity(edge_seed=seed)` fixes that random subset per seed. State s drives
two neighbouring 8-cell E groups, s and (s+1) mod 5, at amplitude 10 for 200 ms.
`phi(s)` = E spike counts / max, then scaled to unit L2 norm; cached per (seed, s).

Agent: linear `Q(s,a) = w_a . phi(s)`, epsilon-greedy (0.1), alpha 0.1. The same loop
with one-hot features is the tabular baseline.

Seeds 0-4 fix the network realization (`edge_seed`) and the simulation noise.

Why these settings:

- Dense backend: the rule is ignored there. `jaxfne/_runtime_config.py:76` (default
  `recurrent_backend="dense"`); edge dispatch at `jaxfne/_model_simulate.py:650`.
  Measured: edge counts 3250 (seed 0) vs 3219 (seed 1), spike counts identical on the
  dense backend. Not fixed here (integration hub).
- Recurrent weight, probe at drive 10 on the edge_list backend (seeds 0 and 1; cells
  whose E spike count differs): default 1/sqrt(50) = 0.14: 0 cells (no effect);
  0.5: 10 cells; 1.0: 16 cells (chosen); 4.0: 181 cells, but the mean E cell fires
  34 times and the state contrast is gone.
- Unit norm: max-normalised features gave `alpha * ||phi||^2` of 2.07 to 2.12 for every
  seed (stability limit 2), so the learner diverged. Unit norm gives
  `alpha * ||phi||^2 = alpha`; alpha stays 0.1.

Episodes-to-policy: the first episode E such that the greedy policy is correct after E
and after every later episode up to the end of training. Correct means the greedy
action (ties go left) is optimal under `Q*` in every non-terminal state; where `Q*`
ties (corridor7 state 3), either action counts. The check runs after every episode.
"never" if the last episode is wrong.

Tests (`test_qlearn_jaxfne.py`):

1. Tabular Q within 1e-2 of `Q*` (2000 episodes, seed 0).
2. jaxfne features learn all-right in 0..3 for seeds 0-4 (<=1000 episodes); prints
   episodes-to-policy, final greedy policy and max `|Q-Q*|` per seed.
3. Flat control (state-mean features) takes one action everywhere.
4. Seeds 0/1 give different `phi`; the top cells of each state lie in its driven cells.
5. `alpha = 0` leaves `w` at zero after 50 episodes.
6. The max-normalised features put `alpha * ||phi||^2` above 2 (precondition); the
   unit-norm features keep every state at alpha or below.
7. Episodes-to-policy on constructed histories (correct, wrong, correct, correct ...),
   including "never"; the wiring records one entry per episode.
8. corridor7: optimal actions are left in 1-2, right in 4-5, either in 3 (precondition);
   the flat control is constant, never reaches the policy and has one greedy action
   over states 1-5 (seeds 0-4).
9. corridor7: network features reach and hold the policy for seeds 0-4; the tabular
   control does so in at least 4 of 5 (seed 3 does not, see below). Every row is printed
   before the assertion.

Results (seeds 0-4, 1000 episodes; from the test run):

| seed | episodes-to-policy | final greedy correct | max abs(Q - Q*) |
|---|---|---|---|
| 0 | 1 | yes | 0.016768 |
| 1 | 1 | yes | 0.022400 |
| 2 | 1 | yes | 0.039509 |
| 3 | 1 | yes | 0.046058 |
| 4 | 1 | yes | 0.021577 |

Episodes-to-policy is 1 for every seed because the greedy policy is correct after
episode 1 and stays correct through episode 1000 (checked: 0 wrong episodes out of
1000 for each seed).

chain5 cannot discriminate: right is optimal in every non-terminal state (`chain.py`,
`q_star_table`), so a state-blind policy that always goes right is already correct,
and the metric is 1 for every seed.

corridor7 (`jaxfne.tasks`): states 0..6, terminal 0 and 6 (absorbing, reward 1 on
entry), start drawn uniformly from 1..5, gamma 0.9. Optimal action: left in 1-2, right
in 4-5, either in 3. Feature sets: network (the same E/I code; state s drives groups
(s mod 5, (s+1) mod 5), so policy states 1-5 drive five distinct group pairs; the
terminals reuse the drive of states 5 and 1 but never bootstrap), flat (one constant
unit vector of length 40 for every state), tabular (one-hot). alpha 0.1, epsilon 0.1,
1000 episodes, seeds 0-4.

Results (corridor7, seeds 0-4, 1000 episodes; from the test run, `pytest -s`):

| features | seed | episodes-to-policy | final greedy, states 1..5 | final correct | max abs(Q - Q*), states 1..5 |
|---|---|---|---|---|---|
| flat | 0 | never | 1,1,1,1,1 | no | 0.253429 |
| flat | 1 | never | 0,0,0,0,0 | no | 0.273178 |
| flat | 2 | never | 0,0,0,0,0 | no | 0.214857 |
| flat | 3 | never | 0,0,0,0,0 | no | 0.356668 |
| flat | 4 | never | 0,0,0,0,0 | no | 0.364214 |
| tabular | 0 | 2 | 0,0,1,1,1 | yes | 0.315541 |
| tabular | 1 | 3 | 0,0,1,1,1 | yes | 0.340128 |
| tabular | 2 | 23 | 0,0,0,1,1 | yes | 0.395384 |
| tabular | 3 | never | 0,1,1,1,1 | no | 0.350974 |
| tabular | 4 | 6 | 0,0,1,1,1 | yes | 0.228817 |
| network | 0 | 486 | 0,0,1,1,1 | yes | 0.112915 |
| network | 1 | 454 | 0,0,1,1,1 | yes | 0.109080 |
| network | 2 | 222 | 0,0,0,1,1 | yes | 0.102670 |
| network | 3 | 235 | 0,0,1,1,1 | yes | 0.082039 |
| network | 4 | 313 | 0,0,1,1,1 | yes | 0.094881 |

Reading: the flat control never learns the corridor, so the task needs state
information. The network code learns it in every seed, later than tabular (222-486
episodes against 2-23) but with a third of the Q error (0.08-0.11 against 0.23-0.40),
because overlapping codes share each update across neighbouring states.

Tabular seed 3, the only "never" outside the flat control: in
state 2 the greedy action is right, but the optimal action is left: `Q*(2, left) = 0.9`
and `Q*(2, right) = 0.729`. The learned `Q(2, left)` is 0.7221, and that entry was
visited 19 times in 1000 episodes (non-greedy entries are sampled only by epsilon
exploration). The policy is wrong in 993 of 1000 episodes.
This is a data-limited result under alpha 0.1, epsilon 0.1 and 1000 episodes; those
values are unchanged here. A change to them is a decision for the owner.

Run: `pytest -q -p no:cacheprovider artifacts/etudes/qlearning/` (from the repo root).
