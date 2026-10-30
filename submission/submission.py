"""Learn2Design-2026: TopologyAwareEliteHybridOptimizer (Compliant v2 Architecture)

NeurIPS 2026 Challenge: Learn2Design-2026
Automated Continuous Optimization of Quasi-Universal Interferometers (UIFO).

Lifecycle & Compliance:
1. Before start_logging():
   - Load static configuration and offline warm-start bundle from relative paths.
   - Set fixed physical constraint penalty function (squashed_relu_penalty).
   - JIT-compile batched and scalar evaluation kernels on fixed tensor shapes.
   - Zero problem-instance inspection, candidate ranking, or evaluation before logging.
2. After start_logging():
   - Read problem dimensionality and public topology spec.
   - Initialize population: mix of uniform random, coverage-balanced, and dataset priors.
   - Multi-slot Noise-Annealed Adam (NAAdam) exploration with antithetic stagnation recovery.
   - Continuous physical feasibility tracking (feasible-first bookkeeping).
   - Second-order L-BFGS refinement on the top feasible basin with Armijo line search.
   - End-of-run margin guard ensuring safe completion within the 4-hour compute budget.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dfbench import Objective, OptimizationAlgorithm
from jaxtyping import Array, Float


def _anneal_sigma(progress: float, sigma_start: float, sigma_end: float, schedule: str = "exponential") -> float:
    """Anneal exploration noise standard deviation from sigma_start to sigma_end."""
    p = float(jnp.clip(progress, 0.0, 1.0))
    if schedule == "linear":
        return sigma_start + (sigma_end - sigma_start) * p
    eps = 1e-12
    s0 = max(float(sigma_start), 0.0)
    s1 = max(float(sigma_end), 0.0)
    if s0 <= 0.0:
        return 0.0
    return s0 * (max(s1, eps) / max(s0, eps)) ** p


def _cap_step_relative(step: jax.Array, reference: jax.Array, ratio: float = 0.25) -> jax.Array:
    """Scale perturbation step down so its norm does not exceed ratio * ||reference||."""
    max_norm = ratio * jnp.linalg.norm(reference)
    step_norm = jnp.linalg.norm(step) + 1e-12
    return step * jnp.minimum(1.0, max_norm / step_norm)


def _load_warmstarts_bundle(bundle_dir: str) -> dict[str, np.ndarray] | None:
    """Load pre-extracted feasible warm-start database from the bundled NPZ file."""
    candidate_paths = [
        os.path.join(bundle_dir, "warmstarts.npz"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle", "warmstarts.npz"),
        "bundle/warmstarts.npz",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                data = np.load(path)
                return {
                    "topology_codes": data["topology_codes"],
                    "params_unit": data["params_unit"],
                    "loss": data["loss"],
                    "grid_size": data["grid_size"],
                }
            except Exception:
                pass
    return None


def _load_config(bundle_dir: str) -> dict[str, Any]:
    """Load optimizer hyperparameters from bundled config.json or return defaults."""
    candidate_paths = [
        os.path.join(bundle_dir, "config.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle", "config.json"),
        "bundle/config.json",
    ]
    defaults: dict[str, Any] = {
        "batch_size": 4,
        "learning_rate": 0.1,
        "noise_std_start": 0.3,
        "noise_std_end": 0.0,
        "noise_schedule": "exponential",
        "noise_anneal_iters": 5000,
        "stagnation_patience": 50,
        "refinement_budget_fraction": 0.90,
        "lbfgs_memory_size": 10,
        "penalty_fn": "squashed_relu_penalty",
    }
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                    defaults.update(user_cfg)
                break
            except Exception:
                pass
    return defaults


def squashed_relu_penalty(value: jax.Array, threshold: float) -> jax.Array:
    relu = jnp.maximum(value / threshold - 1.0, 0.0)
    return relu / (1.0 + relu)


def relu_penalty(value: jax.Array, threshold: float) -> jax.Array:
    return jnp.maximum(value / threshold - 1.0, 0.0)


def zero_penalty(value: jax.Array, threshold: float) -> jax.Array:
    return jnp.zeros_like(value)


PENALTY_REGISTRY = {
    "squashed_relu_penalty": squashed_relu_penalty,
    "relu_penalty": relu_penalty,
    "zero_penalty": zero_penalty,
}


class TopologyAwareEliteHybridOptimizer(OptimizationAlgorithm):
    """Compliant multi-stage hybrid optimizer combining batched NAAdam and L-BFGS refinement."""

    algorithm_str = "topology_aware_hybrid_optimizer"

    def __init__(self, batch_size: int | None = None) -> None:
        self.custom_batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        obj = objective
        bundle_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle")
        config = _load_config(bundle_dir)
        warmstarts_db = _load_warmstarts_bundle(bundle_dir)

        # Resolve execution batch size
        if self.custom_batch_size is not None:
            batch_size = self.custom_batch_size
        else:
            batch_size = int(config.get("batch_size", 4))

        # ------------------------------------------------------------------
        # PHASE 1: PRE-LOGGING SETUP (Strictly Instance-Independent)
        # ------------------------------------------------------------------
        # Set fixed penalty function if custom choice specified
        penalty_choice = config.get("penalty_fn", "squashed_relu_penalty")
        if penalty_choice in PENALTY_REGISTRY and hasattr(obj, "set_penalty_fn"):
            try:
                obj.set_penalty_fn(PENALTY_REGISTRY[penalty_choice])
            except Exception:
                pass

        # Prepare space mode and random seeding
        random_seed, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        # Ahead-Of-Time JIT warmup on fixed batch shape (zero instance evaluation)
        if batch_size > 1:
            if hasattr(obj, "warmup_vmap_value_and_grad_aux"):
                obj.warmup_vmap_value_and_grad_aux(batch_size)
            elif hasattr(obj, "warmup_vmap_value_and_grad"):
                obj.warmup_vmap_value_and_grad(batch_size)
        else:
            if hasattr(obj, "warmup_value_and_grad_aux"):
                obj.warmup_value_and_grad_aux()
            elif hasattr(obj, "warmup_value_and_grad"):
                obj.warmup_value_and_grad()

        if hasattr(obj, "warmup_value_and_grad"):
            obj.warmup_value_and_grad()

        # START OFFICIAL COMPUTATION CLOCK
        obj.start_logging()

        # ------------------------------------------------------------------
        # PHASE 2: POPULATION INITIALIZATION (After start_logging)
        # ------------------------------------------------------------------
        n_params = int(getattr(obj, "n_params", 0))
        lb, ub = obj.bounds
        rng_key, subkey = jax.random.split(rng_key)

        # Retrieve public topology string from problem_spec if available
        problem_spec = getattr(obj, "problem_spec", {})
        topo_string = ""
        if isinstance(problem_spec, dict):
            topo_string = str(problem_spec.get("params", {}).get("topology", ""))

        # Assemble candidate population in unit space [0, 1]
        population_unit = []

        # (a) Warm-starts from public dataset feasible elites
        if warmstarts_db is not None and len(warmstarts_db.get("topology_codes", [])) > 0:
            codes = warmstarts_db["topology_codes"]
            params_db = warmstarts_db["params_unit"]
            matches = np.where(codes == topo_string)[0] if topo_string else np.array([], dtype=int)

            if len(matches) > 0:
                for idx in matches[:batch_size]:
                    raw_vec = params_db[idx]
                    valid = raw_vec[~np.isnan(raw_vec)]
                    if len(valid) == n_params:
                        unit_p = (valid - np.asarray(lb)) / (np.asarray(ub) - np.asarray(lb) + 1e-12)
                        population_unit.append(np.clip(unit_p, 1e-5, 1.0 - 1e-5))

            # Dimension-matched fallback from feasible database
            if len(population_unit) < batch_size:
                len_matches = np.where(~np.isnan(params_db[:, min(n_params - 1, params_db.shape[1] - 1)]))[0]
                for idx in len_matches[: batch_size - len(population_unit)]:
                    raw_vec = params_db[idx]
                    valid = raw_vec[~np.isnan(raw_vec)]
                    if len(valid) == n_params:
                        unit_p = (valid - np.asarray(lb)) / (np.asarray(ub) - np.asarray(lb) + 1e-12)
                        population_unit.append(np.clip(unit_p, 1e-5, 1.0 - 1e-5))

        # (b) Coverage-balanced / Stratified sampling
        target_stratified = max(1, batch_size // 3)
        while len(population_unit) < min(batch_size, len(population_unit) + target_stratified):
            idx = len(population_unit)
            stratified_sample = (np.arange(n_params) + (idx + 0.5) / batch_size) % 1.0
            stratified_sample = np.clip(stratified_sample, 1e-5, 1.0 - 1e-5)
            population_unit.append(stratified_sample)

        # (c) Uniform random filling
        while len(population_unit) < batch_size:
            rng_key, subkey = jax.random.split(rng_key)
            rnd = jax.random.uniform(subkey, shape=(n_params,))
            population_unit.append(np.asarray(rnd))

        # Convert unit parameters to unbounded space via logit transform
        pop_array = jnp.array(population_unit[:batch_size])
        pop_array = jnp.clip(pop_array, 1e-6, 1.0 - 1e-6)
        batch_params = jnp.log(pop_array / (1.0 - pop_array))

        if init_params is not None:
            batch_params = batch_params.at[0].set(init_params)

        # ------------------------------------------------------------------
        # PHASE 3: BATCHED NAADAM EXPLORATION
        # ------------------------------------------------------------------
        learning_rate = float(config.get("learning_rate", 0.1))
        noise_std_start = float(config.get("noise_std_start", 0.3))
        noise_std_end = float(config.get("noise_std_end", 0.0))
        noise_schedule = str(config.get("noise_schedule", "exponential"))
        noise_anneal_iters = int(config.get("noise_anneal_iters", 5000))
        stagnation_patience = int(config.get("stagnation_patience", 50))
        refinement_budget_fraction = float(config.get("refinement_budget_fraction", 0.90))

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate),
        )
        opt_init_fn = jax.vmap(optimizer.init)
        opt_update_fn = jax.vmap(optimizer.update)
        apply_updates_fn = jax.vmap(optax.apply_updates)

        state = opt_init_fn(batch_params)

        # Initial evaluation
        if batch_size == 1:
            if hasattr(obj, "value_and_grad_aux"):
                loss0, grad0, aux0 = obj.value_and_grad_aux(batch_params[0])
                losses = jnp.array([loss0])
                grads = jnp.array([grad0])
                is_feasible = jnp.array([aux0.get("is_feasible", True)])
            else:
                loss0, grad0 = obj.value_and_grad(batch_params[0])
                losses = jnp.array([loss0])
                grads = jnp.array([grad0])
                is_feasible = jnp.array([True])
        elif hasattr(obj, "vmap_value_and_grad_aux"):
            losses, grads, aux0 = obj.vmap_value_and_grad_aux(batch_params)
            is_feasible = aux0.get("is_feasible", jnp.ones_like(losses, dtype=bool))
        else:
            losses, grads = obj.vmap_value_and_grad(batch_params)
            is_feasible = jnp.ones_like(losses, dtype=bool)

        best_loss_history = losses
        stagnation_counter = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Track global best feasible configuration
        best_feasible_loss = float("inf")
        best_feasible_params = batch_params[0]
        for i in range(batch_size):
            if bool(is_feasible[i]) and float(losses[i]) < best_feasible_loss:
                best_feasible_loss = float(losses[i])
                best_feasible_params = batch_params[i]

        iteration = 0

        # Exploration loop capped with safety margin
        while not obj.budget_exceeded:
            progress_frac = float(getattr(obj, "budget_progress_fraction", 0.0))
            if progress_frac >= refinement_budget_fraction:
                break
            time_left = getattr(obj, "time_left", None)
            if time_left is not None and time_left < 180.0:
                break

            progress = iteration / max(1, noise_anneal_iters)
            sigma_t = _anneal_sigma(progress, noise_std_start, noise_std_end, noise_schedule)

            # Batched evaluation
            if batch_size == 1:
                if hasattr(obj, "value_and_grad_aux"):
                    loss, grad, aux = obj.value_and_grad_aux(batch_params[0])
                    losses = jnp.array([loss])
                    grads = jnp.array([grad])
                    is_feasible = jnp.array([aux.get("is_feasible", True)])
                else:
                    loss, grad = obj.value_and_grad(batch_params[0])
                    losses = jnp.array([loss])
                    grads = jnp.array([grad])
                    is_feasible = jnp.array([True])
            elif hasattr(obj, "vmap_value_and_grad_aux"):
                losses, grads, aux = obj.vmap_value_and_grad_aux(batch_params)
                is_feasible = aux.get("is_feasible", jnp.ones_like(losses, dtype=bool))
            else:
                losses, grads = obj.vmap_value_and_grad(batch_params)
                is_feasible = jnp.ones_like(losses, dtype=bool)

            # Robust NaN / Inf detection and replacement
            nan_mask = jnp.isnan(losses) | jnp.isinf(losses) | jnp.any(jnp.isnan(grads), axis=-1)
            if jnp.any(nan_mask):
                rng_key, subkey = jax.random.split(rng_key)
                fresh_samples = obj.random_params_unbounded(n_samples=batch_size)
                batch_params = jnp.where(nan_mask[:, None], fresh_samples, batch_params)
                state = opt_init_fn(batch_params)
                iteration += 1
                continue

            # Feasibility-first bookkeeping
            for i in range(batch_size):
                if bool(is_feasible[i]) and float(losses[i]) < best_feasible_loss:
                    best_feasible_loss = float(losses[i])
                    best_feasible_params = batch_params[i]

            # Optimizer step
            updates, state = opt_update_fn(grads, state, batch_params)

            if sigma_t > 0:
                rng_key, subkey = jax.random.split(rng_key)
                noise_step = jax.random.normal(subkey, shape=batch_params.shape) * sigma_t
                if iteration >= 500:
                    noise_step = jax.vmap(_cap_step_relative, in_axes=(0, 0, None))(noise_step, updates, 0.25)
                updates = updates + noise_step

            batch_params = apply_updates_fn(batch_params, updates)

            # Stagnation counter & Antithetic mutation escape
            improved = losses < (best_loss_history - 1e-4)
            best_loss_history = jnp.minimum(best_loss_history, losses)
            stagnation_counter = jnp.where(improved, 0, stagnation_counter + 1)

            restart_mask = stagnation_counter > stagnation_patience
            restart_mask = restart_mask.at[0].set(False)

            if jnp.any(restart_mask):
                rng_key, subkey = jax.random.split(rng_key)
                delta = jax.random.normal(subkey, shape=batch_params[0].shape) * 0.1
                new_params = batch_params

                if batch_size > 1 and restart_mask[1]:
                    new_params = new_params.at[1].set(batch_params[0] + delta)
                if batch_size > 2 and restart_mask[2]:
                    new_params = new_params.at[2].set(batch_params[0] - delta)
                for i in range(3, batch_size):
                    if restart_mask[i]:
                        rng_key, sk2 = jax.random.split(rng_key)
                        perturbation = jax.random.normal(sk2, shape=batch_params[0].shape) * 0.5
                        new_params = new_params.at[i].set(batch_params[0] + perturbation)

                batch_params = jnp.where(restart_mask[:, None], new_params, batch_params)
                stagnation_counter = jnp.where(restart_mask, 0, stagnation_counter)
                best_loss_history = jnp.where(restart_mask, float("inf"), best_loss_history)

            iteration += 1

        if obj.budget_exceeded:
            return

        # ------------------------------------------------------------------
        # PHASE 4: SECOND-ORDER L-BFGS REFINEMENT WITH ARMIJO SEARCH
        # ------------------------------------------------------------------
        # Select best feasible candidate or lowest-loss candidate
        refine_params = best_feasible_params
        lbfgs_memory_size = int(config.get("lbfgs_memory_size", 10))

        lbfgs_optimizer = optax.lbfgs(
            learning_rate=1.0,
            memory_size=lbfgs_memory_size,
            scale_init_precond=True,
            linesearch=None,
        )
        lbfgs_state = lbfgs_optimizer.init(refine_params)

        if hasattr(obj, "value_and_grad_aux"):
            loss, grads, aux = obj.value_and_grad_aux(refine_params)
            curr_feasible = aux.get("is_feasible", True)
        else:
            loss, grads = obj.value_and_grad(refine_params)
            curr_feasible = True

        if curr_feasible and float(loss) < best_feasible_loss:
            best_feasible_loss = float(loss)
            best_feasible_params = refine_params

        backtracking_factor = 0.5
        armijo_coefficient = 1e-4
        max_linesearch_steps = 15

        while not obj.budget_exceeded:
            time_left = getattr(obj, "time_left", None)
            if time_left is not None and time_left < 60.0:
                break

            updates, lbfgs_state = lbfgs_optimizer.update(
                grads,
                lbfgs_state,
                refine_params,
                value=loss,
                grad=grads,
            )
            slope = float(jnp.vdot(grads, updates))
            current_loss = float(loss)
            step_size = 1.0

            candidate_params = refine_params
            candidate_loss = loss
            candidate_grads = grads
            step_accepted = False

            for _ in range(max_linesearch_steps):
                test_params = optax.apply_updates(refine_params, step_size * updates)
                if hasattr(obj, "value_and_grad_aux"):
                    test_loss, test_grads, test_aux = obj.value_and_grad_aux(test_params)
                    test_feasible = test_aux.get("is_feasible", True)
                else:
                    test_loss, test_grads = obj.value_and_grad(test_params)
                    test_feasible = True

                if obj.budget_exceeded:
                    return

                test_val = float(test_loss)
                if math.isfinite(test_val) and test_feasible:
                    armijo_target = (
                        current_loss + armijo_coefficient * step_size * slope if slope < 0 else current_loss
                    )
                    if test_val <= armijo_target:
                        candidate_params = test_params
                        candidate_loss = test_loss
                        candidate_grads = test_grads
                        step_accepted = True
                        if test_val < best_feasible_loss:
                            best_feasible_loss = test_val
                            best_feasible_params = test_params
                        break

                step_size *= backtracking_factor

            if not step_accepted:
                # If L-BFGS fails to find a feasible descent step, break early
                break

            refine_params = candidate_params
            loss = candidate_loss
            grads = candidate_grads

        # ------------------------------------------------------------------
        # PHASE 5: END-OF-RUN GUARANTEE
        # ------------------------------------------------------------------
        if not obj.budget_exceeded and math.isfinite(best_feasible_loss):
            try:
                if hasattr(obj, "value_and_grad_aux"):
                    obj.value_and_grad_aux(best_feasible_params)
                else:
                    obj.value_and_grad(best_feasible_params)
            except Exception:
                pass
