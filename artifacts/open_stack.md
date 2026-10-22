# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.

## In flight
- O-41 remainder (P-038): Read the Docs regenerates atlases only, so étude pages publish stand-ins (RTD itself is strict since 2026-10-19) | keeper | since 2026-10-18 | closes when RTD builds the étude pages or they leave the RTD nav
- O-59 The HDP étude rerun (`rerun_etude_figures.py` hdp_mcc3) reproduces the frozen AGSDR theta_hat on this machine (WSL, jax 0.11.2) but tunes to a different optimum on GitHub's ubuntu runner with the same jax/numpy (m_EE 0.306, m_EI 2.759, m_IE 0.383, m_II 0.1 against frozen 1.433, 1.111, 0.108, 0.960; Dashboard runs 39693463744, 39698962390), so its figure pages are not on the figure site | Hamm (whether the site may build the étude panels from the frozen theta) | since 2026-10-21 | closes when the figures job publishes the HDP étude and the tune's platform sensitivity is recorded as a known limit or bounded

## Waiting on others
- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL (the 0.5.7 sdist and package families ran on Windows Python) | Hamm (system setting) | since 2026-10-21 | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
