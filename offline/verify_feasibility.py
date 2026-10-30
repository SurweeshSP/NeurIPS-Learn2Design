import h5py
import pandas as pd
import jax
import jax.numpy as jnp
from dfbench import Objective
from dfbench.problems import UIFOProblem

def verify_feasibility(h5_path="dataset/dataset.h5", csv_path="analysis/results/feasibility_analysis.csv"):
    df = pd.read_csv(csv_path)
    feasible_df = df[df["is_feasible_heuristic"] == True]
    
    print(f"Total heuristic feasible designs: {len(feasible_df)}")
    
    # Just test first 10
    test_indices = feasible_df["index"].values[:10]
    
    with h5py.File(h5_path, 'r') as f:
        entries = f['entries']
        bounded_params = f['bounded_params']
        
        for idx in test_indices:
            entry = entries[idx]
            top = entry['topology_string'].decode('utf-8')
            size = entry['size']
            
            p_off = entry['param_offset']
            p_len = entry['param_length']
            params = bounded_params[p_off:p_off+p_len]
            
            problem = UIFOProblem(size=size, topology=top, n_frequencies=50)
            obj = Objective(problem)
            
            obj.start_logging()
            _, _, aux = obj.value_and_grad_aux(params)
            is_feasible = aux.get("is_feasible", False)
            
            print(f"Index {idx} | Heuristic Feasible: True | Official Feasible: {is_feasible} | Raw loss: {aux.get('raw_loss', 'N/A')}")

if __name__ == "__main__":
    verify_feasibility()
