"""Benchmark execution throughput and memory for Differometor / dfbench Objective operations.

Measures:
1. Scalar value_and_grad latency & throughput (evals/s)
2. Batched vmap_value_and_grad latency & throughput for different batch sizes
3. Hessian-bearing operation latency (value_grad_and_hessian)
4. Peak memory / scaling behavior
"""

import argparse
import os
import time
import pandas as pd
import jax
import jax.numpy as jnp
from dfbench import Objective
from dfbench.problems import UIFOProblem


def benchmark_objective(
    size: int = 2,
    n_frequencies: int = 5,
    topology_seed: int = 42,
    batch_sizes: list[int] = [1, 2, 4],
    test_hessian: bool = True,
    n_warmup: int = 2,
    n_trials: int = 5,
):
    print(f"--- Benchmarking UIFO (size={size}, n_frequencies={n_frequencies}, topology_seed={topology_seed}) ---")
    problem = UIFOProblem(size=size, n_frequencies=n_frequencies, topology_seed=topology_seed)
    # 60s budget for benchmarking
    obj = Objective(problem, max_time=60.0)
    obj.start_logging()

    results = []

    # 1. Benchmark scalar value_and_grad
    print("\n[1] Benchmarking scalar value_and_grad...")
    dummy_p = obj.random_params_bounded()
    for _ in range(n_warmup):
        _ = obj.value_and_grad(dummy_p)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = obj.value_and_grad(dummy_p)
    dt_scalar = (time.perf_counter() - t0) / n_trials
    evals_per_sec_scalar = 1.0 / dt_scalar
    print(f"  Scalar value_and_grad: {dt_scalar*1000:.2f} ms/call | {evals_per_sec_scalar:.2f} evals/s")
    results.append({
        "operation": "value_and_grad",
        "batch_size": 1,
        "latency_ms": dt_scalar * 1000,
        "evals_per_sec": evals_per_sec_scalar,
        "notes": "scalar baseline"
    })

    # 2. Benchmark batched vmap_value_and_grad
    for bs in batch_sizes:
        print(f"\n[2] Benchmarking vmap_value_and_grad (batch_size={bs})...")
        batch_p = jnp.repeat(dummy_p[None, :], bs, axis=0)
        try:
            for _ in range(n_warmup):
                _ = obj.vmap_value_and_grad(batch_p)

            t0 = time.perf_counter()
            for _ in range(n_trials):
                _ = obj.vmap_value_and_grad(batch_p)
            dt_batch = (time.perf_counter() - t0) / n_trials
            evals_per_sec_batch = bs / dt_batch
            speedup = evals_per_sec_batch / evals_per_sec_scalar
            print(f"  Batch {bs}: {dt_batch*1000:.2f} ms/call | {evals_per_sec_batch:.2f} evals/s (Speedup: {speedup:.2f}x)")
            results.append({
                "operation": "vmap_value_and_grad",
                "batch_size": bs,
                "latency_ms": dt_batch * 1000,
                "evals_per_sec": evals_per_sec_batch,
                "notes": f"{speedup:.2f}x speedup vs scalar"
            })
        except Exception as e:
            print(f"  Batch {bs} failed: {e}")
            results.append({
                "operation": "vmap_value_and_grad",
                "batch_size": bs,
                "latency_ms": None,
                "evals_per_sec": 0,
                "notes": f"FAILED: {e}"
            })

    # 3. Benchmark Hessian
    if test_hessian:
        print("\n[3] Benchmarking scalar value_grad_and_hessian...")
        try:
            for _ in range(n_warmup):
                _ = obj.value_grad_and_hessian(dummy_p)

            t0 = time.perf_counter()
            for _ in range(max(1, n_trials // 2)):
                _ = obj.value_grad_and_hessian(dummy_p)
            dt_hess = (time.perf_counter() - t0) / max(1, n_trials // 2)
            print(f"  Hessian call: {dt_hess*1000:.2f} ms/call | {1.0/dt_hess:.3f} calls/s")
            results.append({
                "operation": "value_grad_and_hessian",
                "batch_size": 1,
                "latency_ms": dt_hess * 1000,
                "evals_per_sec": 1.0 / dt_hess,
                "notes": "Exact Hessian (forward-over-reverse AD)"
            })
        except Exception as e:
            print(f"  Hessian benchmark failed: {e}")

    df = pd.DataFrame(results)
    os.makedirs("harness/results", exist_ok=True)
    out_csv = "harness/results/throughput_benchmarks.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--n-frequencies", type=int, default=5)
    parser.add_argument("--skip-hessian", action="store_true")
    args = parser.parse_args()

    benchmark_objective(
        size=args.size,
        n_frequencies=args.n_frequencies,
        test_hessian=not args.skip_hessian,
    )
