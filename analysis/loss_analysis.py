import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    
    print("Reading dataset for loss analysis...")
    losses = []
    param_counts = []
    complexities = []
    
    with DatasetReader(dataset_path) as reader:
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            losses.append(entry['loss'])
            param_counts.append(len(entry['params']))
            complexities.append(entry['complexity'])
            
    losses = np.array(losses)
    param_counts = np.array(param_counts)
    complexities = np.array(complexities)
    
    # Calculate statistics
    stats = {
        'count': len(losses),
        'min': np.min(losses),
        'max': np.max(losses),
        'mean': np.mean(losses),
        'median': np.median(losses),
        'std': np.std(losses),
        'P1': np.percentile(losses, 1),
        'P5': np.percentile(losses, 5),
        'P25': np.percentile(losses, 25),
        'P75': np.percentile(losses, 75),
        'P95': np.percentile(losses, 95),
        'P99': np.percentile(losses, 99)
    }
    
    df = pd.DataFrame([stats])
    df.to_csv(os.path.join(results_dir, "loss_statistics.csv"), index=False)
    
    # Generate Plots
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(losses, bins=50, color='purple', edgecolor='black')
    plt.title('Loss Histogram')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    import seaborn as sns
    sns.kdeplot(losses, fill=True, color='purple')
    plt.title('Loss KDE')
    plt.xlabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_distribution.png"))
    plt.close()
    
    # Complexity vs Loss
    plt.figure(figsize=(6, 5))
    plt.scatter(complexities, losses, alpha=0.1, color='orange')
    plt.title('Complexity vs Loss')
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "complexity_vs_loss.png"))
    plt.close()
    
    print("Loss analysis complete.")

if __name__ == '__main__':
    main()
