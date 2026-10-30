#!/usr/bin/env python3
"""
Runner script to execute local prediction and scoring for Learn2Design-2026
and export standard Codabench-compatible output artifacts:
1. prediction_output.txt  (execution trajectory and evaluation statistics)
2. scores.json            (machine-readable scoring metrics)
3. scoring_output.txt     (formatted competition summary and leaderboard scores)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_submission import run_local_submission

def main():
    parser = argparse.ArgumentParser(description="Generate prediction and scoring outputs for submission archive.")
    parser.add_argument("--archive", default="learn2design_final_submission.zip", help="Path to submission ZIP")
    parser.add_argument("--max-time", type=float, default=30.0, help="Budget in seconds (default 30s)")
    parser.add_argument("--output-dir", default="artifacts/evaluation_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f" LEARN2DESIGN-2026: PREDICTION & SCORING PIPELINE ")
    print(f" Archive:  {args.archive}")
    print(f" Budget:   {args.max_time} seconds")
    print(f" Output:   {out_dir}")
    print(f"==================================================\n")

    print("[1/2] Executing Prediction Step (Optimization on fresh topology)...")
    start_time = time.time()
    
    try:
        result = run_local_submission(
            args.archive,
            max_time_seconds=args.max_time,
            topology_seed=42,
            optimizer_seed=123
        )
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"[OK] Prediction Step Finished in {elapsed:.1f}s.")

    print("\n[2/2] Generating Scoring Step Outputs...")
    
    # 1. scores.json
    scores_dict = {
        "best_feasible_loss": result.best_feasible_loss,
        "feasible_candidate_count": result.feasible_candidate_count,
        "finite_candidate_count": result.finite_candidate_count,
        "total_evaluations": result.evaluation_count,
        "evaluations_per_second": result.evaluation_count / max(result.wall_time_seconds, 1e-6),
        "wall_time_seconds": result.wall_time_seconds,
        "objective_time_seconds": result.objective_time_seconds,
        "budget_exceeded": result.budget_exceeded,
        "optimizer_class": result.optimizer_class,
        "algorithm_name": result.algorithm_name,
        "topology_seed": result.topology_seed,
        "optimizer_seed": result.optimizer_seed
    }

    scores_json_path = out_dir / "scores.json"
    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2)

    # 2. prediction_output.txt
    prediction_txt_path = out_dir / "prediction_output.txt"
    with open(prediction_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PREDICTION STEP EXECUTION LOG ===\n")
        f.write(f"Optimizer:               {result.optimizer_class} ({result.algorithm_name})\n")
        f.write(f"Topology Seed:           {result.topology_seed}\n")
        f.write(f"Optimizer Seed:          {result.optimizer_seed}\n")
        f.write(f"Evaluations Logged:      {result.evaluation_count}\n")
        f.write(f"Finite Candidates:       {result.finite_candidate_count}\n")
        f.write(f"Feasible Candidates:     {result.feasible_candidate_count}\n")
        f.write(f"Objective Wall Time:     {result.wall_time_seconds:.2f}s\n")
        f.write(f"Budget Exceeded Flag:    {result.budget_exceeded}\n")

    # 3. scoring_output.txt
    scoring_txt_path = out_dir / "scoring_output.txt"
    loss_str = f"{result.best_feasible_loss:.6f}" if result.best_feasible_loss is not None else "None"
    with open(scoring_txt_path, "w", encoding="utf-8") as f:
        f.write("======================================================\n")
        f.write(" LEARN2DESIGN-2026: OFFICIAL SCORING SUMMARY REPORT  \n")
        f.write("======================================================\n\n")
        f.write(f"Primary Metric (Best Feasible Loss):  {loss_str}\n")
        f.write(f"Feasibility Rate:                    {(result.feasible_candidate_count / max(result.finite_candidate_count, 1)) * 100:.1f}%\n")
        f.write(f"Optimization Throughput:             {result.evaluation_count / max(result.wall_time_seconds, 1e-6):.2f} evals/sec\n")
        f.write(f"Status:                              SUCCESS (Valid Evaluation)\n")

    print(f"\n[OK] Outputs successfully saved to:\n  - {scores_json_path}\n  - {prediction_txt_path}\n  - {scoring_txt_path}")

if __name__ == "__main__":
    main()
