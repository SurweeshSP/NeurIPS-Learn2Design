import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(r"c:\Users\surwe\Project\Learn2Design-2026\analysis")
from multi_start_naadam import MultiStartNAAdam
from dataset_reader import DatasetReader

def get_warm_start_params(dataset_path, topology_hash, num_samples):
    """
    Finds the best historical configurations for a given topology hash.
    If fewer than num_samples exist, pads with mean or random values.
    """
    best_params = []
    
    with DatasetReader(dataset_path) as reader:
        # Optimization: Since we just want top N for a specific topology,
        # we can iterate through the dataset and collect them.
        # In a real scenario, building an index of hash -> indices is better.
        
        matches = []
        for i in range(reader.num_entries):
            entry = reader.get_entry(i)
            if entry['hash'] == topology_hash:
                matches.append((entry['loss'], entry['params']))
                
        # Sort by loss (ascending)
        matches.sort(key=lambda x: x[0])
        
        # Take the best ones
        for _, params in matches[:num_samples]:
            best_params.append(params)
            
        # Pad if not enough matches
        while len(best_params) < num_samples:
            # Fallback to some random noise around the best if we have at least 1,
            # otherwise just return what we have (and let the optimizer handle random init).
            if len(matches) > 0:
                noise = np.random.normal(0, 0.1, matches[0][1].shape)
                best_params.append(matches[0][1] + noise)
            else:
                break
                
    if not best_params:
        return None
        
    return jnp.array(best_params)

class WarmStartMultiStartNAAdam(MultiStartNAAdam):
    def __init__(self, dataset_path, batch_size=4):
        super().__init__(batch_size=batch_size)
        self.dataset_path = dataset_path
        
    def optimize(
        self,
        objective,
        init_params=None, # This is ignored, we fetch from dataset
        random_seed=None,
        learning_rate=0.1,
        **kwargs
    ):
        print("Fetching warm start parameters from dataset...")
        
        # Check if objective problem has a topology string/hash
        # ConstrainedVoyagerProblem doesn't have a topology hash in the same way,
        # but UIFOProblem does.
        topology_hash = None
        if hasattr(objective.problem, 'topology_hash'):
            topology_hash = objective.problem.topology_hash
            
        # For testing purposes, if it's Voyager we just use a dummy hash or ignore
        if topology_hash:
            warm_params = get_warm_start_params(self.dataset_path, topology_hash, self.batch_size)
        else:
            print("No topology hash found, falling back to random initialization.")
            warm_params = None
            
        if warm_params is not None:
            if len(warm_params) < self.batch_size:
                print(f"Warning: Only found {len(warm_params)} warm start samples, padding with random.")
                # The base class takes a single init_params, but we modified it to accept a batch slot 0.
                # Let's adjust the base class to take a batch of init_params if provided.
                
        # Since our current base class `MultiStartNAAdam` takes a single `init_params`,
        # we will just pass the absolute best one for now.
        best_single_init = warm_params[0] if warm_params is not None else None
        
        print(f"Starting optimization with warm_start={best_single_init is not None}")
        
        super().optimize(
            objective,
            init_params=best_single_init,
            random_seed=random_seed,
            learning_rate=learning_rate,
            **kwargs
        )
