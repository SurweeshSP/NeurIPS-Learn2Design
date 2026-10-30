import time
from dfbench import Objective
from dfbench.problems import UIFOProblem
from dfbench.algorithms.gradient_based.na_adam_gd import NAAdamGD

def main():
    print("Initializing UIFOProblem...")
    start_time = time.time()
    try:
        problem = UIFOProblem(seed=42)
        print(f"Problem init took: {time.time() - start_time:.2f}s")
        
        # 30 seconds budget
        objective = Objective(problem, max_time=30.0, save=["is_feasible"])
        
        algo = NAAdamGD()
        print("Starting optimization (compilation will happen first)...")
        opt_start = time.time()
        algo.optimize(objective)
        print(f"Optimization finished in {time.time() - opt_start:.2f}s")
        print(f"Best Feasible Loss: {objective.best_feasible_loss}")
        print(f"Best Loss: {objective.best_loss}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
