import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    plots_dir = os.path.join(results_dir, "parameter_distributions")
    os.makedirs(plots_dir, exist_ok=True)
    
    # We will accumulate values for each parameter index up to the maximum parameter length.
    param_lists = {}
    
    print("Reading dataset for parameter statistics...")
    with DatasetReader(dataset_path) as reader:
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            params = entry['params']
            for idx, val in enumerate(params):
                if idx not in param_lists:
                    param_lists[idx] = []
                param_lists[idx].append(val)
                
    print(f"Collected data for {len(param_lists)} parameter indices.")
    
    stats = []
    
    # Calculate statistics and generate plots
    print("Calculating statistics and generating plots...")
    for idx, vals in param_lists.items():
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
            
        vmin, vmax = np.min(arr), np.max(arr)
        stats.append({
            'parameter_index': idx,
            'count': len(arr),
            'min': vmin,
            'max': vmax,
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr),
            'var': np.var(arr),
            'Q1': np.percentile(arr, 25),
            'Q3': np.percentile(arr, 75),
            'P1': np.percentile(arr, 1),
            'P5': np.percentile(arr, 5),
            'P95': np.percentile(arr, 95),
            'P99': np.percentile(arr, 99),
            'unique_values': len(np.unique(arr)),
            'nan_count': len(vals) - len(arr),
            'inf_count': np.sum(np.isinf(arr))
        })
        
        # Plot distribution
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Parameter {idx} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'param_{idx}_dist.png'))
        plt.close()
        
        if idx % 50 == 0:
            print(f"Processed {idx} parameters...")
            
    # Save CSV
    df = pd.DataFrame(stats)
    csv_path = os.path.join(results_dir, "parameter_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

if __name__ == '__main__':
    main()
