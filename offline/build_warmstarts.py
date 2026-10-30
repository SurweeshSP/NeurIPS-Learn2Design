import h5py
import numpy as np
import pandas as pd

def build_warmstarts(h5_path="dataset/dataset.h5", csv_path="analysis/results/feasibility_analysis.csv", out_path="submission/bundle/warmstarts.npz", k_per_topo=10):
    df = pd.read_csv(csv_path)
    feasible_df = df[df["is_feasible_heuristic"] == True]
    
    # Sort by loss to get elites
    feasible_df = feasible_df.sort_values(by="loss", ascending=True)
    
    # Group by topology and take top K
    elites = feasible_df.groupby("topology").head(k_per_topo)
    
    topology_codes = []
    params_unit = []
    losses = []
    grid_sizes = []
    
    with h5py.File(h5_path, 'r') as f:
        entries = f['entries']
        bounded_params = f['bounded_params']
        
        for _, row in elites.iterrows():
            idx = int(row["index"])
            entry = entries[idx]
            
            top = entry['topology_string'].decode('utf-8') if isinstance(entry['topology_string'], bytes) else entry['topology_string']
            size = int(entry['size'])
            loss = float(row["loss"])
            
            p_off = int(entry['param_offset'])
            p_len = int(entry['param_length'])
            params = bounded_params[p_off:p_off+p_len]
            
            # Since we don't have problem.bounds in this script to scale to [0,1], 
            # and problem init might OOM, we store the raw bounded params and 
            # expect submission to use bounded params.
            
            topology_codes.append(top)
            params_unit.append(params) # raw params for now
            losses.append(loss)
            grid_sizes.append(size)
            
    # Pad params_unit to max length
    max_len = max(len(p) for p in params_unit) if params_unit else 0
    padded_params = np.full((len(params_unit), max_len), np.nan)
    for i, p in enumerate(params_unit):
        padded_params[i, :len(p)] = p
        
    np.savez_compressed(
        out_path,
        topology_codes=np.array(topology_codes),
        params_unit=padded_params,
        loss=np.array(losses),
        grid_size=np.array(grid_sizes)
    )
    print(f"Saved {len(topology_codes)} warm-start entries to {out_path}")

if __name__ == "__main__":
    import os
    os.makedirs("submission/bundle", exist_ok=True)
    build_warmstarts()
