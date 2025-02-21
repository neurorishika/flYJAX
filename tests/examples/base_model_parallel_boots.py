"""
tests/examples/base_model_parallel_cv_boot.py

Test script for bootstrapping routines in flyjax/fitting/bootstrapping.py.
This script simulates datasets for single-group, joint, and hierarchical
model fitting, then runs bootstrapping replications using process-based parallelization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import our test agent and simulation utilities.
from flyjax.agent.base import test_agent
from flyjax.simulation.parse import parse_reward_matrix
from flyjax.simulation.simulate import simulate_dataset_jit

# Import bootstrapping routines.
from flyjax.fitting.bootstrapping import (
    bootstrap_train_single,
    bootstrap_train_joint,
    bootstrap_train_hierarchical,
)
# Import samplers.
from flyjax.fitting.samplers import base_randn_sampler, make_base_randn_sampler

def main():
    # --------------------------
    # Setup common parameters.
    # --------------------------
    rng_key = jax.random.PRNGKey(0)
    true_params = jnp.array([2.0, -1.0, 1.0, 0.1])
    print("True parameters:", true_params)

    # --------------------------
    # Single-Group Bootstrapping
    # --------------------------
    n_replicates = 8
    reward_str = "[0.0,0.0]x30;[0.33,0.17]x40;[0.33,0.67]x40;[0.89,0.11]x40"
    reward_matrices = [parse_reward_matrix(reward_str) for _ in range(n_replicates)]
    rng_key, subkey = jax.random.split(rng_key)
    choices, rewards = simulate_dataset_jit(true_params, jnp.stack(reward_matrices), test_agent, subkey, baiting=True)
    experiments = [(choices[i], rewards[i]) for i in range(n_replicates)]
    print(f"Simulated {len(experiments)} experiments for single-group training.")

    # Setup sampler for single-group model.
    init_param_sampler = partial(base_randn_sampler, n_params=4)

    bootstrap_single = bootstrap_train_single(
        training_experiments=experiments,
        n_bootstrap=5,
        init_param_sampler=init_param_sampler,
        agent=test_agent,
        learning_rate=5e-2,
        num_steps=1000,
        n_restarts=3,
        min_num_converged=1,
        early_stopping={"min_delta": 1e-4},
        use_parallel=True,
        get_history=False,
    )
    print("\n--- Single-Group Bootstrap Results ---")
    for idx, params in bootstrap_single.items():
        print(f"Bootstrap {idx}: params = {params}")

    # --------------------------
    # Joint Model Bootstrapping
    # --------------------------
    experiments_control = experiments
    perturbed_params = true_params + 0.5
    rng_key, subkey = jax.random.split(rng_key)
    p_choices, p_rewards = simulate_dataset_jit(perturbed_params, jnp.stack(reward_matrices), test_agent, subkey, baiting=True)
    experiments_treatment = [(p_choices[i], p_rewards[i]) for i in range(n_replicates)]
    print("\nSimulated treatment group with perturbed parameters.")

    init_theta_sampler = partial(base_randn_sampler, n_params=4)
    init_delta_sampler = partial(base_randn_sampler, n_params=4, std=0.1)

    bootstrap_joint = bootstrap_train_joint(
        experiments_control=experiments_control,
        experiments_treatment=experiments_treatment,
        n_bootstrap=5,
        n_params=4,
        init_theta_sampler=init_theta_sampler,
        init_delta_sampler=init_delta_sampler,
        agent=test_agent,
        learning_rate=5e-2,
        num_steps=1000,
        n_restarts=3,
        min_num_converged=1,
        early_stopping={"min_delta": 1e-4},
        delta_penalty_sigma=1.0,
        use_parallel=True,
        get_history=False,
    )
    print("\n--- Joint Model Bootstrap Results ---")
    for idx, (theta, delta) in bootstrap_joint.items():
        print(f"Bootstrap {idx}: theta = {theta}, delta = {delta}")

    # --------------------------
    # Hierarchical Bootstrapping
    # --------------------------
    subject_experiments = [[exp] for exp in experiments]
    print(f"\nPrepared {len(subject_experiments)} subjects for hierarchical bootstrapping.")

    init_theta_pop_sampler = partial(base_randn_sampler, n_params=4)
    make_sample_init_theta_subjects = partial(make_base_randn_sampler, n_params=4)

    bootstrap_hier = bootstrap_train_hierarchical(
        experiments_by_subject=subject_experiments,
        n_bootstrap=5,
        n_params=4,
        init_theta_pop_sampler=init_theta_pop_sampler,
        make_sample_init_theta_subjects=make_sample_init_theta_subjects,
        agent=test_agent,
        sigma_prior=1.0,
        learning_rate=5e-2,
        num_steps=1000,
        n_restarts=3,
        min_num_converged=1,
        early_stopping={"min_delta": 1e-4},
        use_parallel=True,
        get_history=False,
    )
    print("\n--- Hierarchical Bootstrap Results ---")
    for idx, (theta_pop, theta_subjects) in bootstrap_hier.items():
        print(f"Bootstrap {idx}: theta_pop = {theta_pop}, theta_subjects = {theta_subjects}")

if __name__ == "__main__":
    main()
