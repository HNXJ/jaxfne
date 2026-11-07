`construct()` on the homeostatic_ei route now passes the `x` soft-bound range to `bound_mode="stable"`: canonical
-10/10 (the `make_minimal_ei_params` values), settable by `set_emitter(x_min=, x_max=)`. Before, the route dropped
it and "stable" clamped `x` at the dataclass default of ±1e6. `x_min`/`x_max` under `bound_mode="minimal"` are
refused (x is unbounded there). "minimal" output is unchanged.
