import jaxfne as jtfne

# Build paradigm scaffold
paradigm = jtfne.paradigm("global_local_oddball")
paradigm = paradigm.habituation(sequence=["x", "x", "x", "y"], n_trials=2000)
paradigm = paradigm.main_block(
    standard=["x", "x", "x", "y"],
    global_oddball=["x", "x", "x", "x"],
    p_global=0.2,
)

# Get batch specification
batch_spec = paradigm.batch(n_trials=128, seed=0)
print("=== Paradigm batch spec ===")
print(batch_spec)
print("\nNote: Paradigm.batch() returns a specification dict.")
print("Batch specifications are consumed by trial-driven simulators.")
