"""E2E Smoke example for JaxleyBridge feature.

Demonstrates extracting transmembrane currents using JaxleyBridge, handling optional
dependencies gracefully, and checking the dependency guard.
"""

import sys
import json
import numpy as np
import jaxfne as jtfne

def run_mock_bridge():
    print("Running fake Jaxley fixture simulation smoke...")
    
    # 1. Simulate a mock Jaxley-like model response
    t_steps = 1000
    n_cells = 3
    n_compartments = 5
    
    # Mock Vm and transmembrane currents
    v = np.random.randn(t_steps, n_cells, n_compartments).astype(np.float32) * 5.0 - 65.0
    i_membrane = np.random.randn(t_steps, n_cells, n_compartments).astype(np.float32) * 0.1
    # xyz positions for each cell/compartment
    pos = np.random.rand(n_cells, n_compartments, 3).astype(np.float32) * 100.0
    
    # Mock jaxley model/simulation result object
    class FakeJaxleyModel:
        pass
        
    class FakeSimulationResult:
        def __init__(self):
            self.v = v
            self.i_membrane = i_membrane
            self.pos = pos
            
    mock_model = FakeJaxleyModel()
    mock_res = FakeSimulationResult()
    
    # 2. Instantiate JaxleyBridge (using mock mode since jaxley isn't loaded)
    bridge = jtfne.bridges.JaxleyBridge(
        model=mock_model,
        source_mode="transmembrane_current",
        compartment_axis="last"
    )
    
    # Extract sources from mock results
    # Flatten across compartments or represent as raw compartment current density proxies
    extracted_sources = bridge.extract_sources(mock_res)
    
    print(f"Mock bridge extraction successful! Extracted sources shape: {extracted_sources.shape}")
    assert extracted_sources.shape == (t_steps, n_cells * n_compartments)
    assert np.all(np.isfinite(extracted_sources))
    
    # Save a manifest
    manifest = {
        "manifest_version": "1.0",
        "mock_bridge": True,
        "source_mode": bridge.source_mode,
        "compartment_axis": bridge.compartment_axis,
        "extracted_sources_shape": list(extracted_sources.shape),
        "metadata": {
            "source_decomposition": "transmembrane_current",
            "source_calibration_status": "uncalibrated_simulated_proxy"
        }
    }
    with open("outputs/jaxley_bridge_smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Mock bridge manifest exported successfully!")


def main():
    print("Running Jaxley optional dependency check smoke...")
    
    # Test our requirement function check
    try:
        jtfne.bridges.require_jaxley()
        print("Jaxley is installed in this environment. Proceeding with real dependency checks...")
        # Since Jaxley is installed, let's try importing it
        import jaxley as jx
        print(f"Successfully imported real Jaxley version {jx.__version__ if hasattr(jx, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"Optional dependency check caught expected ImportError: {e}")
        print("This environment does not have Jaxley installed. Proceeding with mock fixture verification.")
        
    # Run mock bridge to ensure functional paths of the class are correct and logic is validated
    run_mock_bridge()

if __name__ == "__main__":
    main()
