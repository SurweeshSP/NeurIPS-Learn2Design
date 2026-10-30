import os
import numpy as np
import pandas as pd
import h5py
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    
    print("Reading dataset for power and feasibility analysis...")
    
    feasibility_data = []
    
    # In Learn2Design, we typically have soft and hard power limits for components.
    # We will compute the max power in each matrix and summarize.
    # Note: Exact feasibility logic (like hard=10kW, soft=2kW) depends on the challenge specs.
    # Here we perform an exploratory analysis.
    
    with DatasetReader(dataset_path) as reader:
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            power = entry['power']
            
            if power is not None and power.size > 0:
                max_power = np.max(power)
                mean_power = np.mean(power)
                p99_power = np.percentile(power, 99)
                
                # A heuristic check based on typical thresholds (e.g., max_power > 10000 could be hard violation)
                hard_violation = bool(max_power > 10000) 
                
                feasibility_data.append({
                    'index': entry['index'],
                    'hash': entry['unique_hash'],
                    'topology': entry['topology_string'],
                    'max_power': max_power,
                    'mean_power': mean_power,
                    'p99_power': p99_power,
                    'is_feasible_heuristic': not hard_violation,
                    'loss': entry['loss']
                })
            
            if i % 5000 == 0:
                print(f"Processed {i} entries for power analysis.")
                
    df = pd.DataFrame(feasibility_data)
    
    # Save the raw feasibility reconstruction
    feasibility_csv = os.path.join(results_dir, "feasibility_analysis.csv")
    df.to_csv(feasibility_csv, index=False)
    
    # Summary report
    total = len(df)
    feasible = df['is_feasible_heuristic'].sum()
    infeasible = total - feasible
    
    report = f"""# Feasibility & Power Analysis

**Disclaimer:** Feasibility is reconstructed based on physical constraints (e.g., hard power thresholds) since an explicit `is_feasible` flag is not present. This must be validated using the official `Objective` API.

- **Total Designs Analysed**: {total}
- **Heuristically Feasible**: {feasible} ({(feasible/total)*100:.2f}%)
- **Heuristically Infeasible**: {infeasible} ({(infeasible/total)*100:.2f}%)

## Power Statistics (Max Power across components)
- **Global Max Power**: {df['max_power'].max():.2f} W
- **Median Max Power**: {df['max_power'].median():.2f} W
- **Mean Max Power**: {df['max_power'].mean():.2f} W
"""
    
    with open(os.path.join(results_dir, "reports", "feasibility_report.md"), "w") as f:
        f.write(report)
        
    print(f"Feasibility analysis complete. Saved to {feasibility_csv}")

if __name__ == '__main__':
    main()
