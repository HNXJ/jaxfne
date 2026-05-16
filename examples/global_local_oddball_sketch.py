import jaxfne as jtfne

paradigm = jtfne.paradigm("global_local_oddball")
paradigm = paradigm.habituation(sequence=["x", "x", "x", "y"], n_trials=2000)
paradigm = paradigm.main_block(
    standard=["x", "x", "x", "y"],
    global_oddball=["x", "x", "x", "x"],
    p_global=0.2,
)

batch = paradigm.batch(n_trials=128, seed=0)
print(batch)
