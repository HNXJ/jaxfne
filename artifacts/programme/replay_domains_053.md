# Receipt — 0.5.3 ENGINE item 4: deterministic replay + RNG domains (w1-53)

Parent: `865e74b` (item 3).

## Declared RNG domains

| Consumer | Derivation (seed, t) | Owner |
|---|---|---|
| membrane noise | per-step `split(chain_key(t))[1]` → normal; plain path consumes the identical `continuation_noise_schedule` | item 3 chain contract |
| per-rule noise | per-step `fold_in(split(chain_key(t))[0], t_global)`; plain registered path re-derives from the same chain | item 3 fix |
| drive/paradigm | explicit schedule arrays, no RNG | `_signals.StimulusSchedule` |
| poisson_drive | own seed (`sim.seed + 7919` default); rejected under continuation (cursor unambiguous) | `_model_simulate.simulate` |
| shuffled_timing | own key (`sim.seed + 12345`); rejected under continuation (ablations unsupported) | same |
| batch | `vmap` over `split(PRNGKey(base_seed), n_seeds)` | `simulate_batch` |
| construction | seeded builders | construct path |

Isolation argument: membrane and rule branch independently off the
carried chain (`split[1]` vs `split[0]`+`fold_in(t)`); neither
consumer's draws depend on the other's consumption (JAX keys are
values). Same-seed cross-rule stream sharing is common-random-numbers
(by design, good for rule comparisons), not interference: separate
runs never share mutable RNG state.

## Code change

None in `jaxfne/`. Item 3 built the domains; item 4 pins them with
tests + this declaration. (`HDPRuleContext.key` docstring and the
kernel split comments already declare the contract.)

## Tests

`tests/test_replay_domains_053.py`: 10 passed (13.0 s, CPU).

- Replay bit-identical: baseline stochastic, HDP stochastic (V/spikes/
  sources/H/w traces + finals + carried H/W/prng_key), registered
  stochastic rule (V + aux), paradigm-drive runs, construction
  (weights + v0), batch (V/spikes).
- Seed sensitivity (live streams, not constants).
- Rule-stream isolation: sigma 0.5 → 2.0 with k_w=0 moves aux while
  spikes/V stay bit-identical (membrane untouched).
- Membrane/rule separation: rule-key derivation is a pure function of
  (seed, t) — identical across noise_scales — while the membrane
  trajectory moves (both streams live).
- Per-rule determinism: same rule + same seed replays bit-identically.

## Invariants held

- Same seed + same inputs → bit-identical on every covered path.
- K_HDP=0 null unchanged; H != HDP preserved; full recording default
  untouched; no state containers restructured.
