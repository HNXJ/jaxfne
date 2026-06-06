#!/usr/bin/env python3
import ast
import os
import json
from pathlib import Path

def score_node(file_name, node, code_lines):
    # Get function source code
    func_lines = code_lines[node.lineno - 1 : node.end_lineno]
    func_code = "".join(func_lines)
    
    # 1. Docstring check (20 points)
    docstring = ast.get_docstring(node)
    doc_score = 20 if docstring else 0
    
    # 2. Type hints check (20 points)
    hints_present = 0
    total_args = len(node.args.args)
    if total_args > 0:
        annotated = sum(1 for arg in node.args.args if arg.annotation is not None)
        hints_present += (annotated / total_args) * 10
    else:
        hints_present += 10 # No args is automatically fully annotated
    if node.returns is not None:
        hints_present += 10
    
    # 3. Code length and complexity (30 points)
    length = len(func_lines)
    # Deduct points if too long (optimal length <= 30 lines)
    len_score = max(0, 30 - max(0, length - 30) * 0.5)
    
    # AST Complexity: count loops, branches, nested functions
    branches = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
            branches += 1
    comp_score = max(0, 20 - branches * 1.5)
    
    # 4. JAX best practices and safety check (30 points)
    # Deduct if there are raw print statements, file I/O, or raw loops over timesteps inside simulations
    jax_score = 30
    func_lower = func_code.lower()
    if "print(" in func_lower and not file_name.startswith("vis"):
        jax_score -= 5
    if "open(" in func_lower and not file_name.startswith("io") and not file_name.startswith("vis"):
        jax_score -= 5
    if "for " in func_lower and ("step" in func_lower or "timestep" in func_lower) and "lax.scan" not in func_lower:
        # Potential non-JIT-friendly loop over timesteps
        jax_score -= 10
    if "plt." in func_lower and not file_name.startswith("vis") and not "tutorial_utils" in file_name:
        # Plotting inside core computation
        jax_score -= 10
        
    total_score = int(doc_score + hints_present + len_score + comp_score + jax_score)
    # Cap total score at 100
    total_score = min(max(total_score, 0), 100)
    
    return {
        "file": file_name,
        "name": node.name,
        "line": node.lineno,
        "score": total_score,
        "length": length,
        "branches": branches,
        "has_doc": bool(docstring),
        "annotated_args": sum(1 for arg in node.args.args if arg.annotation is not None) if total_args > 0 else 0,
        "total_args": total_args,
        "has_return_hint": node.returns is not None
    }

def main():
    scored_funcs = []
    
    for p in Path('jaxfne').rglob('*.py'):
        if '__pycache__' in str(p) or p.name == '__init__.py':
            continue
        try:
            code = p.read_text()
            lines = code.splitlines(keepends=True)
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info = score_node(p.name, node, lines)
                    info["path"] = str(p)
                    scored_funcs.append(info)
        except Exception as e:
            print(f"Error parsing {p}: {e}")
            
    # Sort functions by score ascending
    scored_funcs.sort(key=lambda x: (x["score"], x["file"], x["name"]))
    
    print(f"Scored {len(scored_funcs)} functions.")
    
    # Save all scores to a report file
    with open("outputs/all_functions_scores.json", "w") as f:
        json.dump(scored_funcs, f, indent=2)
        
    # Print bottom 15
    print("\n--- BOTTOM 15 SCORED FUNCTIONS ---")
    for idx, item in enumerate(scored_funcs[:15]):
        print(f"{idx+1}. {item['file']}:{item['name']} (Line {item['line']}) - Score: {item['score']}/100")
        print(f"   Length: {item['length']} lines, Branches: {item['branches']}, Has Doc: {item['has_doc']}, Typed Args: {item['annotated_args']}/{item['total_args']}")

if __name__ == "__main__":
    main()
