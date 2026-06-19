# Objective Grammar Skill

## Purpose
Force every workflow to follow the repo's object-transform grammar.

## Mandatory chain
`Configuration -> Model -> Signals -> Probe -> Objective -> Optimizer -> Manifest`

## Rules
- Start from `Configuration` or the documented builder, not from ad hoc local state.
- Build once, reuse the model, and simulate many times on the same structure when possible.
- Keep simulation output in `Signals`.
- Keep probing and objective logic separate from simulation code.
- Keep export and manifest logic separate from analysis and plotting.
- Do not add local scientific engines when a package API already exists.

## Acceptance checks
- A task can be mapped to one stage of the chain.
- The code path stays inspectable from config to manifest.
- Notebook helper code does not replace package-level APIs.
