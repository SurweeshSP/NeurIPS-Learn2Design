import h5py
import numpy as np

class DatasetReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._file = None

    def __enter__(self):
        self._file = h5py.File(self.file_path, 'r')
        self.entries = self._file['entries']
        self.bounded_params = self._file['bounded_params']
        self.sensitivities = self._file['sensitivities']
        self.frequency_values = self._file['frequency_values'][:]
        self.power_values = self._file['power_values']
        self.power_port_map_indices = self._file['power_port_map_indices']
        self.power_port_map_names = self._file['power_port_map_names']
        self.power_port_map_offsets = self._file['power_port_map_offsets']
        self.num_entries = len(self.entries)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def get_entry(self, index):
        if index < 0 or index >= self.num_entries:
            raise IndexError("Index out of bounds")
        
        entry = self.entries[index]
        
        param_offset = entry['param_offset']
        param_length = entry['param_length']
        params = self.bounded_params[param_offset : param_offset + param_length]
        
        power_offset = entry['power_offset']
        power_length = entry['power_length']
        power = self.power_values[power_offset : power_offset + power_length]
        
        # Power matrix reshaping based on rows and cols
        power_rows = entry['power_rows']
        power_cols = entry['power_cols']
        if power_length == power_rows * power_cols:
            power = power.reshape((power_rows, power_cols))
            
        power_map_id = entry['power_map_id']
        power_metadata = None # Could be extracted using power_map_id and power_port_* datasets if needed
        
        return {
            "index": index,
            "unique_hash": entry['unique_hash'].decode('utf-8') if isinstance(entry['unique_hash'], bytes) else entry['unique_hash'],
            "initialized_from": entry['initialized_from'].decode('utf-8') if isinstance(entry['initialized_from'], bytes) else entry['initialized_from'],
            "topology_string": entry['topology_string'].decode('utf-8') if isinstance(entry['topology_string'], bytes) else entry['topology_string'],
            "params": params,
            "loss": entry['loss'],
            "complexity": entry['complexity'],
            "sensitivity": self.sensitivities[index],
            "frequencies": self.frequency_values,
            "power": power,
            "power_metadata": power_metadata,
            "source_file": entry['source_file'].decode('utf-8') if isinstance(entry['source_file'], bytes) else entry['source_file'],
            "run_id": entry['run_id'].decode('utf-8') if isinstance(entry['run_id'], bytes) else entry['run_id']
        }

if __name__ == '__main__':
    dataset_path = r"c:\Users\surwe\Project\Learn2Design-2026\dataset\dataset.h5"
    with DatasetReader(dataset_path) as reader:
        # First 10
        print("--- First 10 ---")
        for i in range(10):
            e = reader.get_entry(i)
            print(f"Index {i}: Hash {e['unique_hash']}, Loss {e['loss']}, Param len {len(e['params'])}, Power shape {e['power'].shape}")
            assert len(e['params']) == reader.entries[i]['param_length']
            
        # Random 10
        print("\n--- Random 10 ---")
        indices = np.random.choice(reader.num_entries, 10, replace=False)
        for i in indices:
            e = reader.get_entry(i)
            print(f"Index {i}: Hash {e['unique_hash']}, Loss {e['loss']}, Param len {len(e['params'])}, Power shape {e['power'].shape}")
