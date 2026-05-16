import jaxfne as jtfne

# Build paradigm scaffold (v0.0.2: API skeleton, v0.0.4: execution engine)
paradigm = jtfne.paradigm("global_local_oddball")
paradigm = paradigm.habituation(sequence=["x", "x", "x", "y"], n_trials=2000)
paradigm = paradigm.main_block(
    standard=["x", "x", "x", "y"],
    global_oddball=["x", "x", "x", "x"],
    p_global=0.2,
)

# Get batch specification (placeholder: does not execute trials yet)
batch_spec = paradigm.batch(n_trials=128, seed=0)
print("=== Paradigm batch spec (v0.0.2: API scaffold, not executable) ===")
print(batch_spec)
print("\nNote: Paradigm.batch() returns a specification dict in v0.0.2.")
print("Runtime execution via Model.simulate(paradigm=batch_spec) is planned for v0.0.4.")
