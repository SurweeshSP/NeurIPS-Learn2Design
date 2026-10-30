import os
import numpy as np
import pandas as pd
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    
    print("Reading dataset for topology statistics...")
    data = []
    with DatasetReader(dataset_path) as reader:
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            data.append({
                'topology_string': entry['topology_string'],
                'loss': entry['loss'],
                'complexity': entry['complexity'],
                'param_count': len(entry['params'])
            })
            
    df = pd.DataFrame(data)
    
    # Group by topology
    print("Aggregating statistics...")
    stats = df.groupby('topology_string').agg(
        frequency=('topology_string', 'count'),
        param_count=('param_count', 'first'),
        best_loss=('loss', 'min'),
        median_loss=('loss', 'median'),
        mean_loss=('loss', 'mean'),
        p90_loss=('loss', lambda x: np.percentile(x, 90)),
        p99_loss=('loss', lambda x: np.percentile(x, 99)),
        median_complexity=('complexity', 'median')
    ).reset_index()
    
    stats = stats.sort_values(by='frequency', ascending=False)
    
    # Add a topology ID for easier reference
    stats.insert(0, 'topology_id', range(1, len(stats) + 1))
    
    csv_path = os.path.join(results_dir, "topology_statistics.csv")
    stats.to_csv(csv_path, index=False)
    print(f"Saved {len(stats)} unique topologies to {csv_path}")

if __name__ == '__main__':
    main()
