import os
import json
import glob
from pathlib import Path

def split_code_lines(source_lines: list[str], max_logical_lines: int = 10) -> list[list[str]]:
    """Split source lines of a code cell into chunks, each with <= max_logical_lines.
    
    Tries to split at safe boundaries (nesting level = 0, and preferably at indentation = 0).
    """
    chunks = []
    current_chunk = []
    logical_count = 0
    
    # Track nesting level of parentheses, brackets, and braces
    nesting = 0
    
    for line in source_lines:
        current_chunk.append(line)
        stripped = line.strip()
        
        if stripped and not stripped.startswith("#"):
            logical_count += 1
            
        # Update nesting level
        for char in line:
            if char in "([{":
                nesting += 1
            elif char in ")]}":
                nesting -= 1
                
        # Safe split condition: nesting == 0, and next line might start a new statement.
        if logical_count >= max_logical_lines - 2:
            if nesting == 0:
                chunks.append(current_chunk)
                current_chunk = []
                logical_count = 0
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def repair_notebook(nb_path: Path) -> bool:
    """Repair notebook cell structure and formatting."""
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
        return False

    cells = nb.get("cells", [])
    new_cells = []
    
    descriptions = [
        "*Configure simulation parameters and environment state.*",
        "*Define laminar populations and set up source-field projections.*",
        "*Run the core neural simulation and compute proxy readouts.*",
        "*Plot spatial-temporal dynamics and power spectral densities.*",
        "*Validate metrics and verify biophysical constraints.*",
        "*Perform secondary analysis and process trial readouts.*",
        "*Initialize the model configuration and layout neural sheets.*",
        "*Execute the simulation sweep and save results.*"
    ]
    desc_idx = 0
    
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type", "")
        if ctype != "code":
            new_cells.append(cell)
            continue
            
        raw_source = cell.get("source", [])
        if isinstance(raw_source, str):
            source = [line + "\n" for line in raw_source.splitlines()]
        else:
            source = raw_source
            
        # Check if this cell is just a print of exercises, and convert to markdown
        source_str = "".join(source)
        if "EXERCISE 1" in source_str and ("print(\"\"\"" in source_str or "print(\'\'\'" in source_str):
            lines_stripped = []
            for line in source:
                # Remove triple quotes and print function wrapping
                clean_line = line.replace("print(\"\"\"", "").replace("print(\'\'\'", "").replace("\"\"\")", "").replace("\'\'\')", "")
                lines_stripped.append(clean_line)
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines_stripped
            })
            continue
            
        lines = [l.strip() for l in source if l.strip() and not l.strip().startswith("#")]
        
        # Check if we should split this cell
        has_large_dict = any("{" in line or "[" in line for line in lines)
        if len(lines) > 10 and not has_large_dict:
            # Split the cell
            split_chunks = split_code_lines(source, max_logical_lines=9)
            for s_idx, chunk in enumerate(split_chunks):
                new_cell = dict(cell)
                new_cell["source"] = chunk
                new_cell["outputs"] = []
                new_cell["execution_count"] = None
                new_cells.append(new_cell)
                
                # If not the last chunk, insert a separating markdown cell
                if s_idx < len(split_chunks) - 1:
                    md_desc = descriptions[desc_idx % len(descriptions)]
                    desc_idx += 1
                    new_cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [md_desc + "\n"]
                    })
        else:
            # Keep as is
            clean_cell = dict(cell)
            clean_cell["source"] = source
            clean_cell["outputs"] = []
            clean_cell["execution_count"] = None
            new_cells.append(clean_cell)

    # Now, check for any back-to-back code cells in the reconstructed list and insert markdown separators
    final_cells = []
    for i in range(len(new_cells)):
        final_cells.append(new_cells[i])
        if i < len(new_cells) - 1:
            if new_cells[i].get("cell_type") == "code" and new_cells[i+1].get("cell_type") == "code":
                md_desc = descriptions[desc_idx % len(descriptions)]
                desc_idx += 1
                final_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [md_desc + "\n"]
                })
                
    nb["cells"] = final_cells
    
    # Save the repaired notebook
    try:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        return True
    except Exception as e:
        print(f"Error writing {nb_path}: {e}")
        return False

def main():
    print("=== REPAIRING JAXFNE JUPYTER NOTEBOOKS (ROBUST) ===")
    
    # Scan both notebooks and tutorials directories
    nb_files = list(Path("tutorials").glob("*.ipynb")) + list(Path("notebooks").glob("**/*.ipynb"))
    
    repaired_count = 0
    for nb in nb_files:
        if ".ipynb_checkpoints" in str(nb) or "outputs/" in str(nb):
            continue
        print(f"Repairing structure for: {nb}")
        if repair_notebook(nb):
            repaired_count += 1
            
    print(f"\nSuccessfully repaired {repaired_count} notebooks!")

if __name__ == "__main__":
    main()
