# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.

## In flight
- O-41 remainder (P-038): Read the Docs regenerates atlases only, so étude pages publish stand-ins (RTD itself is strict since 2026-10-19) | keeper | since 2026-10-18 | closes when RTD builds the étude pages or they leave the RTD nav

- O-56 Validation backbone (Hamm 2026-10-21): six new core-method rows on the dashboard Validation tab (emitters, plasticity, numerics in build; objectives+optimizer, fields, ports queued), then a grammar-stage column | keeper, Muse and Haiku workers | since 2026-10-21 | closes when seven core-method rows score on dev with CI green

## Waiting on others
- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls
- O-57 colab-mcp does not connect (CONNECT_TIMEOUT 30 s in the keeper and fork sessions); heavy compute waits on it (Hamm 2026-10-21 rule: local runs stay small). The fork found the entry in `~/.claude.json` and waits for Hamm's yes to edit it | Hamm | since 2026-10-21 | closes when a session lists colab-mcp tools
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL (the 0.5.7 sdist and package families ran on Windows Python) | Hamm (system setting) | since 2026-10-21 | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
