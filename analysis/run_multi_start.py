import argparse
import time
import jax
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench import Objective

import sys
sys.path.append(r"c:\Users\surwe\Project\Learn2Design-2026\analysis")
from multi_start_naadam import MultiStartNAAdam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-time", type=int, default=120)
    args = parser.parse_args()
    
    SEED = 42
    print(f"Starting MultiStartNAAdam with batch_size={args.batch_size} for {args.max_time}s")
    
    problem = ConstrainedVoyagerProblem()
    obj = Objective(
        problem,
        verbose=1,
        max_time=args.max_time,
        print_every=5,
        save_params_history=False,
        save_to_file_every=1000,
        display_mode="log",
    )
    
    optimizer = MultiStartNAAdam(batch_size=args.batch_size)
    
    t0 = time.time()
    optimizer.optimize(
        obj,
        learning_rate=0.1,
        noise_std_start=0.3,
        random_seed=SEED,
    )
    t1 = time.time()
    
    dt = t1 - t0
    evals = obj.eval_count
    
    print("\n--- RESULTS ---")
    print(f"Batch Size: {args.batch_size}")
    print(f"Time Taken: {dt:.2f}s")
    print(f"Evals: {evals}")
    if evals > 0:
        print(f"Throughput: {evals / dt:.2f} evals/sec")
    if obj.best_loss is not None:
        print(f"Best Loss: {obj.best_loss:.6f}")
    else:
        print("Best Loss: None")

if __name__ == '__main__':
    main()
