import json
from pathlib import Path
import jaxfne as jtfne

def test_public_api_matches_snapshot():
    snapshot_path = Path("artifacts/public_api_before.json")
    assert snapshot_path.exists(), "Public API snapshot file missing"
    
    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)
        
    expected_names = set(snapshot["public_names"])
    
    current_names = set(list(jtfne.__all__)) if hasattr(jtfne, "__all__") else set([name for name in dir(jtfne) if not name.startswith("_")])
    
    # Require that all snapshot names match exactly
    assert current_names == expected_names, (
        f"Public API changed!\n"
        f"Added: {current_names - expected_names}\n"
        f"Removed: {expected_names - current_names}"
    )

def test_essential_facade_symbols_present():
    """Double check that the main facade classes/functions are exposed."""
    for name in ["Configuration", "Config", "Model", "Simulation", "Signals", "RuntimeConfig", "construct", "simulate"]:
        assert hasattr(jtfne, name), f"jtfne is missing essential facade symbol: {name}"
