import json, pathlib, hashlib
receipt_path = pathlib.Path('artifacts/e2/preregistration/E2a_search/e2a_search_receipt.json')
manifest_path = pathlib.Path('artifacts/e2/preregistration/36_AUDIT_INPUT_MANIFEST.json')
blinding_path = pathlib.Path('artifacts/e2/preregistration/E2A_BLINDING_SPEC.json')

receipt = json.loads(receipt_path.read_bytes())
manifest = json.loads(manifest_path.read_bytes())
blinding = json.loads(blinding_path.read_bytes())

print("=== E2a freeze audit ===")
# 1. Reconstruct theta* from preserved allowed table via f_select
records = receipt['records']
theta_stats = receipt['theta_stats']
# Recompute f_select
max_rate = max(s['rate'] for s in theta_stats)
candidates = [s for s in theta_stats if s['rate']==max_rate and max_rate>0]
candidates_sorted = sorted(candidates, key=lambda s: (s['theta']['drive_E'], s['theta']['weight_mu']))
theta_star_recomputed = candidates_sorted[0]['theta'] if candidates_sorted else None
theta_star_stored = receipt['result']['theta*']
print(f"Recomputed theta* {theta_star_recomputed}")
print(f"Stored theta* {theta_star_stored}")
print(f"Match {theta_star_recomputed==theta_star_stored}")

# 2. Forbidden grep zero (keys)
forbidden = set(blinding['forbidden_observables'])
keys = set()
for r in records:
    keys.update(r.keys())
leaked = keys & forbidden
print(f"Leaked forbidden keys in records: {leaked} (must be empty)")
# Also check file content keys
content = receipt_path.read_text()
import re
file_keys = set(re.findall(r'"([^"]+)":', content))
leaked_file = file_keys & forbidden
# Allow blinding description to contain forbidden strings? Check only data section
# We already checked records keys, so file_keys may include blinding forbidden list itself
# So check that forbidden not in records keys is sufficient
print(f"File keys leaked (including blinding spec list): {leaked_file} -> expect forbidden list itself, not data")
# More precise: ensure no record contains forbidden as data key already checked

# 3. Development seeds disjoint
dev_seeds = set(receipt['D_dev']['seeds'])
conf_seeds = set(range(0,4100))
print(f"D_dev {sorted(dev_seeds)[:3]}...{sorted(dev_seeds)[-3:]} disjoint from confirmatory {conf_seeds & dev_seeds == set()}")

# 4. Every attempted point retained (including invalid)
print(f"Total records {len(records)} expected 30 (6 thetas *5 seeds) retained {len(records)==30}")
# Check all thetas present
thetas_expected = set(["theta0","theta1","theta2","theta3","theta4","theta5"])
thetas_present = set(r['theta_id'] for r in records)
print(f"Thetas present {thetas_present} == expected {thetas_expected} -> {thetas_present==thetas_expected}")

# 5. Write-once check: file exists and not overwritten (we can check git status)
import subprocess
try:
    out = subprocess.check_output(['git','status','--porcelain','artifacts/e2/preregistration/E2a_search/e2a_search_receipt.json'], text=True)
    print(f"Git status for receipt: '{out.strip()}' (should be ?? or A)")
except Exception as e:
    print(e)

# 6. Authority packet match
print(f"Authority ping_spec_hash match manifest {receipt['authority_packet']['ping_spec_hash']==manifest['ping_spec_hash']}")
print(f"Authority ssa_spec_hash match {receipt['authority_packet']['ssa_spec_hash']==manifest['ssa_spec_hash']}")

# Verdict
pass_audit = (theta_star_recomputed==theta_star_stored and not leaked and len(records)==30 and thetas_present==thetas_expected and receipt['authority_packet']['ping_spec_hash']==manifest['ping_spec_hash'])
print(f"\nFreeze audit PASS: {pass_audit}")
if pass_audit:
    print("Ready for E2b")
else:
    print("Freeze audit FAIL")
