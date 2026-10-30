import time
import os
import sys

t0 = time.time()
print(f"[{time.time()-t0:.2f}s] Process started")

try:
    import dfbench
    print(f"[{time.time()-t0:.2f}s] dfbench import complete")
    from dfbench.problems import UIFOProblem
    from dfbench import Objective
    print(f"[{time.time()-t0:.2f}s] UIFOProblem import complete")
    
    # We skip dataset reader import as this is a minimal test
    
    # Topology loaded / Problem constructed
    problem = UIFOProblem(topology_seed=42)
    print(f"[{time.time()-t0:.2f}s] topology loaded and problem constructed")
    
    objective = Objective(problem, max_time=30.0, save=["is_feasible"])
    print(f"[{time.time()-t0:.2f}s] Objective constructed")
    
    params = objective.random_params_unbounded(n_samples=1)
    if params.ndim == 2: params = params[0]
    print(f"[{time.time()-t0:.2f}s] random_params generated")
    
    print(f"[{time.time()-t0:.2f}s] value warmup started")
    objective.warmup_value()
    print(f"[{time.time()-t0:.2f}s] value warmup completed")
    
    print(f"[{time.time()-t0:.2f}s] value_and_grad warmup started")
    objective.warmup_value_and_grad()
    print(f"[{time.time()-t0:.2f}s] value_and_grad warmup completed")
    
    print(f"[{time.time()-t0:.2f}s] vmap warmup started")
    # For speed we just warmup batch=2
    objective.warmup_vmap_value_and_grad(2)
    print(f"[{time.time()-t0:.2f}s] vmap warmup completed")
    
    objective.start_logging()
    print(f"[{time.time()-t0:.2f}s] start_logging()")
    
    t_exec_start = time.time()
    objective.value(params)
    print(f"[{time.time()-t0:.2f}s] first timed value evaluation (took {time.time()-t_exec_start:.4f}s)")
    
    t_exec_start = time.time()
    objective.value_and_grad(params)
    print(f"[{time.time()-t0:.2f}s] first timed value_and_grad evaluation (took {time.time()-t_exec_start:.4f}s)")

except Exception as e:
    print(f"FAILED: {e}")
