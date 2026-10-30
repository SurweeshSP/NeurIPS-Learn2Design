"""Test smoke execution of the submitted TopologyAwareEliteHybridOptimizer."""

import os
import sys
import time

sys.path.insert(0, "submission")

from dfbench import Objective
from dfbench.problems import UIFOProblem
from submission import TopologyAwareEliteHybridOptimizer


def test_optimizer_smoke():
    print("Testing TopologyAwareEliteHybridOptimizer smoke run...")
    problem = UIFOProblem(size=2, n_frequencies=5, topology_seed=42)
    # 10 second budget
    objective = Objective(
        problem,
        max_time=10.0,
        save=["is_feasible", "batched_loss", "batched_is_feasible"],
        display_mode="log",
    )

    optimizer = TopologyAwareEliteHybridOptimizer(batch_size=2)
    start_time = time.time()
    optimizer.optimize(objective, random_seed=123)
    elapsed = time.time() - start_time

    print(f"Run completed in {elapsed:.2f} seconds.")
    print(f"Evaluations logged: {objective.eval_count}")
    print(f"Best loss: {objective.best_loss}")
    print(f"Best is feasible: {getattr(objective, 'best_is_feasible', None)}")
    assert objective.eval_count > 0, "No evaluations recorded"
    print("SMOKE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_optimizer_smoke()
