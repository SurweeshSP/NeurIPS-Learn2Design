import os
import time
import argparse
import pandas as pd
import jax
import jax.numpy as jnp
from dfbench import Objective
from dfbench.problems import UIFOProblem

from learn2design.example_algorithms.na_adam_gd import NAAdamGD
import sys
sys.path.append('submission')
try:
    from submission import NAAdamGDBaseline
except ImportError:
    pass

DEV_TOPOLOGIES = [
    ("ABBGDGBEA-SLLLSDSLSLSL", 3),
    ("CDCCHDAFD-LDLSLSLLSLSL", 3),
]

def run_variant(variant_name, optimizer_class, topology, size, seed, max_time, kwargs):
    problem = UIFOProblem(size=size, topology=topology, n_frequencies=50)
    obj = Objective(problem, max_time=max_time)
    optimizer = optimizer_class()
    
    start_time = time.time()
    optimizer.optimize(obj, random_seed=seed, **kwargs)
    elapsed = time.time() - start_time
    
    df = obj.log_df
    if df is not None and not df.empty:
        best_loss = df['loss'].min()
        feasible_df = df[df['is_feasible'] == True] if 'is_feasible' in df.columns else pd.DataFrame()
        best_feasible = feasible_df['loss'].min() if not feasible_df.empty else float('inf')
        evals = len(df)
    else:
        best_loss = float('inf')
        best_feasible = float('inf')
        evals = 0
        
    return {
        "variant": variant_name,
        "topology": topology,
        "seed": seed,
        "raw_loss": float(best_loss),
        "feasible_loss": float(best_feasible),
        "evals": evals,
        "runtime": elapsed,
        "throughput": evals / max(elapsed, 1e-6)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    
    results = []
    
    variants = [
        ("Official_NAAdamGD", NAAdamGD, {}),
    ]
    if 'NAAdamGDBaseline' in globals():
        variants.append(("Safety_Net_Baseline", NAAdamGDBaseline, {}))
        
    os.makedirs("harness/results", exist_ok=True)
        
    for topology, size in DEV_TOPOLOGIES:
        for seed in range(args.seeds):
            for var_name, opt_cls, opt_kwargs in variants:
                print(f"Running {var_name} on {topology} (seed={seed})...")
                try:
                    res = run_variant(var_name, opt_cls, topology, size, seed, args.max_time, opt_kwargs)
                    results.append(res)
                    print(f"  -> Feasible loss: {res['feasible_loss']:.4f} (Raw: {res['raw_loss']:.4f})")
                except Exception as e:
                    print(f"  -> Failed: {e}")
                    
    df = pd.DataFrame(results)
    df.to_csv("harness/results/eval_results.csv", index=False)
    print("\nSummary:")
    if not df.empty:
        print(df.groupby("variant")[["raw_loss", "feasible_loss", "throughput"]].mean())

if __name__ == "__main__":
    main()
