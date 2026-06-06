import jaxfne as jtfne
import json
import sys

# Ensure list of public names is generated based on jtfne.__all__ and dir(jtfne)
public_names = sorted(list(set(jtfne.__all__))) if hasattr(jtfne, "__all__") else sorted([name for name in dir(jtfne) if not name.startswith("_")])

# Build API snapshot including key classes/functions
snapshot = {
    "version": jtfne.__version__,
    "public_names": public_names,
    "has_Configuration": hasattr(jtfne, "Configuration"),
    "has_Config": hasattr(jtfne, "Config"),
    "has_Model": hasattr(jtfne, "Model"),
    "has_Simulation": hasattr(jtfne, "Simulation"),
    "has_Signals": hasattr(jtfne, "Signals"),
    "has_RuntimeConfig": hasattr(jtfne, "RuntimeConfig"),
    "has_construct": hasattr(jtfne, "construct"),
    "has_simulate": hasattr(jtfne, "simulate"),
}

print(json.dumps(snapshot, indent=2))
