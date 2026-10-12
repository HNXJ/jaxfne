"""Q1: how the authors' notebook computes sf tuning and any sf selectivity number.

Reads tuningCurves.ipynb, its sibling notebooks and the src analysis modules as
TEXT ONLY. Nothing is executed. Lines longer than 300 characters (embedded
images) are counted and skipped. Run with -I.

Prints raw file line numbers (file:line evidence) and a per-cell index.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_Python_Analysis")
NB = ROOT / "tuningCurves.ipynb"

KEY = re.compile(
    r"neuron_id = |neur_ids = |^\s*\"start = |end = 2001000|simulation_max = |surr_angles = |"
    r"layer_id = |folder = source_path|thresh=10|thresh = |\bdef \w+\(|real_fqcyval = |"
    r"idex_\.append|to_plot = |spikes_ag_0 = np|if\(spikes_ag_0 == 0|thresh_std = |thresh_num = |"
    r"timestamps = range|tuningSelectivities|tuningCurvesOutputSine\(spinet|sineTimeResponse\(spinet|"
    r"np\.argmax|fq_max\.append|ang_max\.append|cell_fqcy\.append|cell_r\.append|max_val = 1|"
    r"windows_upperBound|simple_phase\[it\]|for phase in surr_angles|np\.mean\(np\.array\(spikes_ag\),axis=1\)|"
    r"neuron_z = 0|depth = 64|depth = spinet"
)
SF_WORDS = re.compile(r"F1|F0|doubl|harmonic|fft|FFT|Fourier|selectiv|bandwidth|fqcy|spatial|frequenc|\bsf\b|\bSF\b")

print("=== A. tuningCurves.ipynb: cells (index, id, first code line) ===")
with open(NB, encoding="utf-8", errors="replace") as fh:
    text = fh.read()
nb = json.loads(text)
print("n_cells", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first = src.strip().splitlines()[0][:70] if src.strip() else ""
    print(f"cell[{i}] id={c.get('id')} type={c.get('cell_type')} :: {first}")

print("")
print("=== B. tuningCurves.ipynb: raw-file lines matching analysis keys ===")
with open(NB, encoding="utf-8", errors="replace") as fh:
    lines = fh.read().splitlines()
for n, line in enumerate(lines, start=1):
    if len(line) > 300:
        continue
    if KEY.search(line):
        print(f"{NB.name}:{n}: {line.strip()[:170]}")

print("")
print("=== C. sf / frequency / doubling words in sibling notebooks (text, long lines skipped) ===")
for nb_path in sorted(ROOT.glob("*.ipynb")):
    if nb_path.name == NB.name:
        continue
    with open(nb_path, encoding="utf-8", errors="replace") as fh:
        ls = fh.read().splitlines()
    long_skipped = sum(1 for l in ls if len(l) > 300)
    hits = [(n, l) for n, l in enumerate(ls, start=1) if len(l) <= 300 and SF_WORDS.search(l)]
    print(f"{nb_path.name}: lines={len(ls)} long_skipped={long_skipped} sf_word_hits={len(hits)}")
    for n, l in hits[:6]:
        print(f"    {nb_path.name}:{n}: {l.strip()[:150]}")

print("")
print("=== D. F1/F0/doubling/selectivity words in src analysis modules (text) ===")
src_files = sorted(
    list((ROOT / "src" / "spiking_network" / "analysis").glob("*.py"))
    + list((ROOT / "src" / "spiking_network" / "gabor2").glob("*.py"))
)
for p in src_files:
    with open(p, encoding="utf-8", errors="replace") as fh:
        ls = fh.read().splitlines()
    hits = [(n, l) for n, l in enumerate(ls, start=1) if re.search(r"F1|F0|doubl|harmonic|fft|FFT|Fourier|selectiv|bandwidth", l)]
    print(f"{p.relative_to(ROOT).as_posix()}: hits={len(hits)}")
    for n, l in hits[:4]:
        print(f"    {p.name}:{n}: {l.strip()[:150]}")

print("")
print("q1_notebook_scan DONE")
sys.exit(0)
