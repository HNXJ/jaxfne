# Lanes and registries: parallel development of jaxfne

Status: design, 2026-10-11, approved for a prototype by Hamm. It extends
`release_plan_0.5.6-0.5.10.md` §7 (structure) and §8 (roles).

## Problem

Parallel work collides in a few hub files, which every extension edits:

| Hub | Size at 6dde18d5 | Edited by |
|---|---|---|
| `jaxfne/emitters.py` | 5,077 lines, 24 step variants | every new neuron family |
| `jaxfne/_model.py` | 1,070 lines | every new `Model` method |
| `jaxfne/__init__.py`, `public_surface.py` | 482 lines + surface pins | every new public name |
| `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/run_test_gate.py` | | every new gate or extra |

Two lanes that touch one hub conflict, and every change runs the whole dev gate
(about 4 minutes).

## Rule

A lane adds files. It does not edit a hub. Edits to hubs go through one integrator,
at merge time.

## 1. Registries at the seams

`jaxfne/hdp_rule.py` already has the pattern: a frozen descriptor, a step function,
`register_*` that validates at registration, `get_*` that refuses an unknown name.
Three more registries follow it:

| Registry | Entry | First users |
|---|---|---|
| tasks | environment: `reset(rng)`, `step(state, action)`, sizes, optional `q_star` | Q-learning chain; omission and oddball paradigms |
| families | neuron family: parameters, state layout, step function | AdEx (0.5.6), LIF (0.5.7); Izhikevich variants moved one at a time |
| analyses | readout operation on simulated signals; the jnwb seam | calibrated oscillation estimator; spectra |

Each registry has one contract test that every entry must pass:
- descriptor validation at registration;
- the P4 perturbation test (B3): each accepted parameter at 0, None, boundary
  and a non-default value gives a realized change or an explicit refusal;
- for families, outputs bit-equal to the code they replace.

The core never imports an entry by name. A lane registers its file, and the
registry's contract test picks it up.

## 2. Lanes

| Lane | Owns | Changes | Integrator checks |
|---|---|---|---|
| Core | kernel, `simulate`, the registries, `public_surface` | slow; bit-equality tests | second review, full gate |
| Models | `jaxfne/families/<name>.py`; replications as examples | AdEx, LIF, PCL, CAMINOS (P6 library path) | family contract test |
| Analysis | `jaxfne/analysis/<op>.py` | estimator, metrics, jnwb items 2–4 | analysis contract test, calibration (B7) |
| Tasks and learning | `jaxfne/tasks/<name>.py`, `jaxfne/learning/` | Q-learning, paradigms, plasticity rules | task contract test |
| Integration (Cursor) | hubs, CI, docs, release | surface pins, exports, gates | rc gate |

Each lane has its own branch, worktree, tests directory, docs page and size budget.

## 3. CI per lane

Path-filtered jobs run a lane's tests on each push, which takes seconds. The full
gate runs at integration. A lane's job also runs the contract tests of the registry it
uses.

## 4. Contracts per release

Each registry's interface is frozen for a release. A change is a Core-lane item and
needs Hamm's sign-off (B3). New entries need no hub edit, so they need no sign-off
beyond their own review. Public exposure (adding to `__all__`) is an integration
step, gated by goal 4: new abilities are switched on explicitly.

## Order

1. Task registry, private (`jaxfne/tasks/`, not in `__all__`), with the Q-learning
   chain as its first entry. Check: the contract test passes for the chain and fails
   on a mutant task that ignores its seed.
2. Family registry, with AdEx as its first entry and the 0.5.6 emitter (B3). The
   Izhikevich variants move one at a time, each behind bit-equality tests.
3. Analysis registry, with the CAMINOS estimator after its preregistration is signed.
4. Path-filtered CI jobs.
