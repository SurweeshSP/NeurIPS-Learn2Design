import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_reader import DatasetReader

def main():
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    
    print("Reading dataset for correlation analysis...")
    
    # We will build a matrix: rows = designs, columns = parameters
    # To handle varying lengths, we will use a fixed max size (e.g. 195) based on common lengths,
    # or just pad with NaNs.
    
    param_list = []
    loss_list = []
    max_params = 195 # Based on typical UIFO parameters, we analyze the first 195.
    
    with DatasetReader(dataset_path) as reader:
        # Sample 10000 designs to keep correlation matrix computationally feasible and memory low
        num_samples = min(10000, reader.num_entries)
        indices = np.random.choice(reader.num_entries, num_samples, replace=False)
        for i in indices:
            entry = reader.get_entry(i)
            params = entry['params']
            
            # Pad or truncate
            if len(params) > max_params:
                params = params[:max_params]
            elif len(params) < max_params:
                params = np.pad(params, (0, max_params - len(params)), constant_values=np.nan)
                
            param_list.append(params)
            loss_list.append(entry['loss'])
            
    param_matrix = np.array(param_list) # shape: (10000, 195)
    losses = np.array(loss_list) # shape: (10000,)
    
    # Create DataFrame for easy correlation
    df = pd.DataFrame(param_matrix, columns=[f'P{i}' for i in range(max_params)])
    df['Loss'] = losses
    
    print("Computing Parameter vs Loss correlations...")
    # Calculate Spearman correlation with Loss
    corr_with_loss = df.corr(method='spearman')['Loss'].drop('Loss')
    
    # Save parameter-loss stats
    corr_df = pd.DataFrame({
        'parameter_index': range(max_params),
        'spearman_corr_with_loss': corr_with_loss.values
    })
    corr_df.to_csv(os.path.join(results_dir, "parameter_loss_statistics.csv"), index=False)
    
    # Plot top correlated parameters
    top_indices = corr_with_loss.abs().nlargest(5).index
    
    param_loss_dir = os.path.join(results_dir, "parameter_loss")
    os.makedirs(param_loss_dir, exist_ok=True)
    
    for col in top_indices:
        plt.figure(figsize=(6, 5))
        # Hexbin plot to handle large density of points
        plt.hexbin(df[col], df['Loss'], gridsize=50, cmap='inferno', mincnt=1)
        plt.colorbar(label='Count')
        plt.title(f'{col} vs Loss (Spearman: {corr_with_loss[col]:.3f})')
        plt.xlabel(col)
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(param_loss_dir, f'{col}_vs_loss.png'))
        plt.close()
        
    print("Computing Parameter-Parameter correlation matrix...")
    # Compute correlation matrix between parameters (excluding Loss)
    param_corr_matrix = df.drop(columns=['Loss']).corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(param_corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False)
    plt.title('Parameter-Parameter Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "parameter_correlation_matrix.png"))
    plt.close()
    
    print("Correlation analysis complete.")

if __name__ == '__main__':
    main()
