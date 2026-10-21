# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.

## In flight
- O-41 remainder (P-038): Read the Docs regenerates atlases only, so étude pages publish stand-ins (RTD itself is strict since 2026-10-19) | keeper | since 2026-10-18 | closes when RTD builds the étude pages or they leave the RTD nav


## Waiting on others
- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL (the 0.5.7 sdist and package families ran on Windows Python) | Hamm (system setting) | since 2026-10-21 | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
