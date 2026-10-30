import pandas as pd
import os

def generate_report():
    csv_path = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results\parameter_statistics.csv"
    report_path = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results\parameter_distributions_report.md"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    with open(report_path, "w") as f:
        f.write("# Parameter Distributions Report\n\n")
        f.write("This report provides a text-based summary of the parameter distributions based on the calculated statistics from the dataset.\n\n")
        
        for _, row in df.iterrows():
            idx = int(row['parameter_index'])
            mean = row['mean']
            p50 = row['median']
            p25 = row['Q1']
            p75 = row['Q3']
            min_val = row['min']
            max_val = row['max']
            std = row['std']
            
            # Determine skewness based on mean vs median
            skew = "roughly symmetric"
            if mean > p50 + 0.1 * std:
                skew = "right-skewed (higher frequency of lower values)"
            elif mean < p50 - 0.1 * std:
                skew = "left-skewed (higher frequency of higher values)"
                
            # Determine spread
            iqr = p75 - p25
            
            f.write(f"### Parameter {idx}\n")
            f.write(f"- **Range**: [{min_val:.4f}, {max_val:.4f}]\n")
            f.write(f"- **Mean**: {mean:.4f}, **Median**: {p50:.4f}\n")
            f.write(f"- **Distribution Shape**: The distribution is {skew}. 50% of the values fall tightly between {p25:.4f} and {p75:.4f} (IQR: {iqr:.4f}).\n")
            f.write(f"- **Observation**: The highest concentration of values is around {p50:.4f}.\n\n")
            
    print(f"Report successfully generated at {report_path}")

if __name__ == "__main__":
    generate_report()
