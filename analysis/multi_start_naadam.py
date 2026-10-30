import jax
import jax.numpy as jnp
import optax
from typing import Literal

from dfbench import Objective, OptimizationAlgorithm
from learn2design.example_algorithms.na_adam_gd import _anneal_sigma, _clip_step_by_global_norm, _cap_step_relative

class MultiStartNAAdam(OptimizationAlgorithm):
    algorithm_str = "multi_start_na_adam"

    def __init__(self, batch_size=4):
        self.batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params=None,
        random_seed=None,
        learning_rate=0.1,
        noise_std_start=0.3,
        noise_std_end=0.0,
        noise_schedule="exponential",
        noise_anneal_iters=5000,
        **adam_kwargs,
    ) -> None:
        obj = objective
        # Initialize
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)
        
        # We need a batch of initial parameters. 
        # If init_params is provided, we use it for one of the slots, and random for the rest.
        rng_key, subkey = jax.random.split(rng_key)
        
        # Random initializations for the batch
        batch_params = jax.random.normal(subkey, (self.batch_size, obj.n_params))
        
        # If init_params given, insert into the first batch slot
        if init_params is not None:
            batch_params = batch_params.at[0].set(init_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), 
            optax.adam(learning_rate, **adam_kwargs)
        )
        
        # Optax operates on the batch directly if we vmap the optimizer update, 
        # but since optax state is just pytrees, applying it to a batch works out of the box mostly.
        # But to be safe and clean, we vmap the optimizer.
        opt_init_fn = jax.vmap(optimizer.init)
        opt_update_fn = jax.vmap(optimizer.update)
        apply_updates_fn = jax.vmap(optax.apply_updates)
        
        state = opt_init_fn(batch_params)
        
        obj.warmup_vmap_value_and_grad(self.batch_size)
        obj.start_logging()
        
        iteration = 0

        while not obj.budget_exceeded:
            progress = iteration / max(1, noise_anneal_iters)
            sigma_t = _anneal_sigma(progress, noise_std_start, noise_std_end, noise_schedule)

            losses, grads = obj.vmap_value_and_grad(batch_params)

            updates, state = opt_update_fn(grads, state, batch_params)

            if sigma_t > 0:
                rng_key, subkey = jax.random.split(rng_key)
                noise_step = jax.random.normal(subkey, shape=batch_params.shape) * sigma_t
                
                # noise cap
                updates = updates + noise_step

            batch_params = apply_updates_fn(batch_params, updates)
            iteration += 1
            
            # Simple heuristic: occasionally check if we should reset poor performers 
            # (To be refined into full elite retention)
            if iteration % 500 == 0:
                pass
