import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    sens_dir = os.path.join(results_dir, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)
    
    print("Reading dataset for sensitivity analysis...")
    sensitivities = []
    losses = []
    
    with DatasetReader(dataset_path) as reader:
        frequencies = reader.frequency_values
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            sensitivities.append(entry['sensitivity'])
            losses.append(entry['loss'])
            
    sensitivities = np.array(sensitivities)
    losses = np.array(losses)
    
    # Sort indices by loss
    sorted_idx = np.argsort(losses)
    num_entries = len(losses)
    
    idx_top01 = sorted_idx[:int(num_entries * 0.001)]
    idx_top1 = sorted_idx[:int(num_entries * 0.01)]
    idx_top5 = sorted_idx[:int(num_entries * 0.05)]
    idx_median = sorted_idx[int(num_entries * 0.45):int(num_entries * 0.55)]
    idx_worst10 = sorted_idx[int(num_entries * 0.9):]
    
    def plot_group(indices, label, color):
        mean_sens = np.mean(sensitivities[indices], axis=0)
        std_sens = np.std(sensitivities[indices], axis=0)
        plt.plot(frequencies, mean_sens, label=label, color=color)
        plt.fill_between(frequencies, mean_sens - std_sens, mean_sens + std_sens, alpha=0.2, color=color)

    # Plot comparisons
    plt.figure(figsize=(10, 6))
    plot_group(idx_top01, 'Top 0.1%', 'blue')
    plot_group(idx_median, 'Median', 'green')
    plot_group(idx_worst10, 'Worst 10%', 'red')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Sensitivity (Strain/rtHz)')
    plt.title('Sensitivity Curves by Performance Tier')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(sens_dir, 'top_vs_median_sensitivity.png'))
    plt.close()
    
    # Analyze frequency of minimum/maximum
    min_freq_indices = np.argmin(sensitivities, axis=1)
    max_freq_indices = np.argmax(sensitivities, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(frequencies[min_freq_indices], bins=20, color='blue', edgecolor='black')
    plt.title('Freq of Minimum Sensitivity')
    plt.xscale('log')
    
    plt.subplot(1, 2, 2)
    plt.hist(frequencies[max_freq_indices], bins=20, color='red', edgecolor='black')
    plt.title('Freq of Maximum Sensitivity')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(sens_dir, 'sensitivity_distribution.png'))
    plt.close()
    
    print("Sensitivity analysis complete.")

if __name__ == '__main__':
    main()
