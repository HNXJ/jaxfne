import subprocess
import sys

def test_root_import_does_not_load_heavy_optional_modules():
    """Verify in a clean subprocess that importing jaxfne does not load heavy optional packages."""
    check_code = (
        "import sys\n"
        "import jaxfne\n"
        "forbidden = {'matplotlib', 'plotly', 'pandas', 'optax', 'jaxley'}\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "import json\n"
        "print(json.dumps(list(forbidden & loaded)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
        check=True
    )
    
    import json
    loaded_heavy = json.loads(result.stdout.strip())
    assert not loaded_heavy, f"Forbidden heavy packages loaded by root import: {loaded_heavy}"
