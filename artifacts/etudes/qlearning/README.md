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

Episodes-to-policy: the first episode E such that the greedy policy (right in states
0..3) is correct after E and after every later episode up to the end of training. The
check runs after every episode. "never" if the last episode is wrong.

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

Run: `pytest -q -p no:cacheprovider artifacts/etudes/qlearning/` (from the repo root).
