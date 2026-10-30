import h5py
import json
import os

def get_schema(name, obj):
    schema = {}
    if isinstance(obj, h5py.Group):
        schema['type'] = 'Group'
        schema['children'] = {}
        for key, val in obj.items():
            schema['children'][key] = get_schema(key, val)
    elif isinstance(obj, h5py.Dataset):
        schema['type'] = 'Dataset'
        schema['shape'] = list(obj.shape)
        schema['dtype'] = str(obj.dtype)
    return schema

def main():
    file_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    results_dir = r"c:\Users\surwe\Project\Learn2Design-2026\analysis\results"
    os.makedirs(results_dir, exist_ok=True)
    
    schema = {}
    with h5py.File(file_path, 'r') as f:
        schema = get_schema('/', f)
        
        # Additional summary statistics
        num_entries = len(f['entries'])
        total_params = f['bounded_params'].size
        total_power_values = f['power_values'].size
        
        md_content = f"""# Dataset Summary

- **Total Entries (Designs)**: {num_entries}
- **Total Continuous Parameters Stored**: {total_params}
- **Total Power Values Stored**: {total_power_values}

## Schema Details
The dataset uses a compound dataset for metadata (`entries`) and flattened 1D arrays for dynamically-sized physical parameters (`bounded_params`) and power matrices (`power_values`). This layout is highly optimized for fast I/O and minimal overhead.

## Datasets
"""
        for key, val in schema.get('children', {}).items():
            if val['type'] == 'Dataset':
                md_content += f"- **{key}**: Shape `{val['shape']}`, Dtype `{val['dtype']}`\n"
        
        with open(os.path.join(results_dir, "dataset_summary.md"), "w") as md_file:
            md_file.write(md_content)

    with open(os.path.join(results_dir, "dataset_schema.json"), "w") as json_file:
        json.dump(schema, json_file, indent=2)
        
    print("Dataset schema and summary generated in analysis/results/")

if __name__ == '__main__':
    main()
