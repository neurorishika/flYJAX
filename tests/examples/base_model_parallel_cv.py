import jax
import jax.numpy as jnp
import numpy as np
import chex
from flyjax.agent.model import test_agent
from flyjax.simulation.simulate import simulate_experiment_jit, simulate_dataset_jit, simulate_dataset_jit_different_params
from flyjax.simulation.parse import parse_reward_matrix
from flyjax.fitting.cv import k_fold_cross_validation_train, \
    k_fold_cross_validation_train_joint, \
    k_fold_cross_validation_train_hierarchical, \
    parallel_k_fold_cross_validation_train, \
    parallel_k_fold_cross_validation_train_joint, \
    parallel_k_fold_cross_validation_train_hierarchical
from flyjax.utils.plotting import plot_single_experiment_data
from typing import Tuple, Dict, Any, List, Callable, Optional
from flyjax.fitting.samplers import base_randn_sampler
from functools import partial

if __name__ == "__main__":
    # 0. Set up random number generator
    rng_key = jax.random.PRNGKey(0)
        
    # 1. Define "true" parameters
    true_params = jnp.array([2.0, -1.0, 1.0, 0.1])
    print("True params:", true_params)

    # --- Set true parameters for simulation ---
    # params = [alpha_learn_logit, alpha_forget_logit, kappa_reward, kappa_omission]
    true_params = jnp.array([2.0, -1.0, 1.0, 0.1])
    print("True parameters:", true_params)

    # --- Simulate a dataset ---
    # Define the reward probabilities for each experiment.
    n_replicates = 8
    reward_matrices = [
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.33,0.67]x40;[0.17,0.33]x40;[0.89,0.11]x40;[0.33,0.17]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40;[0.17,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.11,0.89]x40;[0.17,0.33]x40;[0.67,0.33]x40;[0.33,0.17]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.33,0.17]x40;[0.11,0.89]x40;[0.67,0.33]x40;[0.89,0.11]x40;[0.17,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.17,0.33]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.17,0.33]x40;[0.89,0.11]x40;[0.33,0.17]x40;[0.33,0.67]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.11,0.89]x40;[0.33,0.17]x40;[0.89,0.11]x40;[0.33,0.67]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.67,0.33]x40;[0.11,0.89]x40;[0.33,0.67]x40;[0.89,0.11]x40;[0.67,0.33]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.17,0.33]x40;[0.11,0.89]x40;[0.67,0.33]x40;[0.89,0.11]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.17,0.33]x40;[0.67,0.33]x40;[0.33,0.17]x40;[0.11,0.89]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.33,0.67]x40;[0.33,0.17]x40;[0.67,0.33]x40;[0.11,0.89]x40"),
        parse_reward_matrix("[0.0,0.0]x30;[0.89,0.11]x40;[0.33,0.67]x40;[0.11,0.89]x40;[0.33,0.17]x40;[0.89,0.11]x40")
    ]*n_replicates

    rng_key = jax.random.PRNGKey(0)
    rng_key, subkey = jax.random.split(rng_key)
    choices, rewards = simulate_dataset_jit(true_params, jnp.stack(reward_matrices), test_agent, subkey, baiting=True)
    n_experiments = len(reward_matrices)
    # convert to experiment data format
    experiments = [(choices[i], rewards[i]) for i in range(n_experiments)]

    init_param_sampler = partial(base_randn_sampler, n_params=4)

    total_pred_ll, per_experiment_ll, params = parallel_k_fold_cross_validation_train(
        experiments=experiments,
        k=5,
        init_param_sampler=init_param_sampler,
        agent=test_agent,
        learning_rate=5e-2,
        num_steps=10000,
        n_restarts=10,
        min_num_converged=3,
        early_stopping={"min_delta": 1e-4}
    )

    print("Total Predictive Log Likelihood:", total_pred_ll)
    print("Per Experiment Log Likelihoods:", per_experiment_ll)

    param_vals = np.array(list(params.values())).mean(axis=0)
    print("Estimated parameters:", param_vals)
    print("True parameters:", true_params)
