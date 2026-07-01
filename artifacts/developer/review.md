<!-- auto-generated from review.json by scripts/prp_to_markdown.py — do not hand-edit -->

| path | score | review_status | moved_from_progress_on | review_command |
|---|---|---|---|---|
| jaxfne/core.py::RuntimeConfig.dtype + Model.with_hdp_initial_state | 90 | pending | 2026-07-01 | python3 -c "import jaxfne as jtfne; jtfne.RuntimeConfig(dtype='bfloat16')" && python3 -m pytest tests/ -q |
| jaxfne/core.py::Configuration.population | 90 | pending | 2026-07-01 | python3 -c "import jaxfne as jtfne; cfg = jtfne.Configuration().geometry(layer_thickness={'L4':1.0}).population(N=2, neu |
